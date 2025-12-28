"""
Connection Pooling and Performance Optimization Module.

This module provides:
- Connection pooling for Kusto clients
- Async query execution utilities
- Batch query processing
- Schema preloading capabilities
- Health monitoring for connections

Author: Arjun Trivedi
Version: 2.2.0
"""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

DEFAULT_POOL_SIZE = 5
DEFAULT_MAX_POOL_SIZE = 20
DEFAULT_CONNECTION_TIMEOUT = 30  # seconds
DEFAULT_IDLE_TIMEOUT = 1800  # 30 minutes
DEFAULT_MAX_CONNECTION_AGE = 3600  # 1 hour
DEFAULT_HEALTH_CHECK_INTERVAL = 60  # seconds
DEFAULT_BATCH_SIZE = 10
DEFAULT_BATCH_TIMEOUT = 300  # 5 minutes


# =============================================================================
# Connection Pool Data Classes
# =============================================================================

@dataclass
class ConnectionInfo:
    """Information about a pooled connection."""
    client: Any
    cluster_url: str
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    is_healthy: bool = True
    last_health_check: Optional[datetime] = None

    def mark_used(self) -> None:
        """Mark the connection as recently used."""
        self.last_used = datetime.now()
        self.use_count += 1

    def is_expired(self, max_age: int = DEFAULT_MAX_CONNECTION_AGE) -> bool:
        """Check if connection has exceeded maximum age."""
        return (datetime.now() - self.created_at).total_seconds() > max_age

    def is_idle(self, idle_timeout: int = DEFAULT_IDLE_TIMEOUT) -> bool:
        """Check if connection has been idle too long."""
        return (datetime.now() - self.last_used).total_seconds() > idle_timeout


@dataclass
class PoolStatistics:
    """Statistics for connection pool monitoring."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    failed_connections: int = 0
    average_wait_time_ms: float = 0.0
    connections_created: int = 0
    connections_recycled: int = 0

    def hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0


@dataclass
class BatchQueryResult:
    """Result from a batch query execution."""
    query_id: str
    query: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    row_count: int = 0


# =============================================================================
# Connection Pool Implementation
# =============================================================================

class KustoConnectionPool:
    """
    Thread-safe connection pool for Kusto clients.

    Features:
    - Automatic connection reuse and recycling
    - Configurable pool size and timeouts
    - Health checking for connections
    - Statistics tracking

    Example:
        pool = KustoConnectionPool(max_size=10)
        client = pool.get_client("https://cluster.kusto.windows.net")
        # Use client...
        pool.release_client("https://cluster.kusto.windows.net")
    """

    _instance: Optional['KustoConnectionPool'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for connection pool."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_POOL_SIZE,
        min_size: int = DEFAULT_POOL_SIZE,
        connection_timeout: int = DEFAULT_CONNECTION_TIMEOUT,
        idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
        max_connection_age: int = DEFAULT_MAX_CONNECTION_AGE,
        health_check_interval: int = DEFAULT_HEALTH_CHECK_INTERVAL
    ):
        """Initialize the connection pool."""
        if getattr(self, '_initialized', False):
            return

        self._max_size = max_size
        self._min_size = min_size
        self._connection_timeout = connection_timeout
        self._idle_timeout = idle_timeout
        self._max_connection_age = max_connection_age
        self._health_check_interval = health_check_interval

        # Connection storage: cluster_url -> OrderedDict of ConnectionInfo
        self._pools: Dict[str, OrderedDict[str, ConnectionInfo]] = {}
        self._pool_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.RLock()

        # Statistics
        self._stats = PoolStatistics()

        # Background health checker
        self._health_checker_running = False
        self._health_checker_thread: Optional[threading.Thread] = None

        self._initialized = True
        logger.info("[POOL] Connection pool initialized (max=%d, min=%d)", max_size, min_size)

    def _get_pool_lock(self, cluster_url: str) -> threading.Lock:
        """Get or create a lock for a specific cluster pool."""
        with self._global_lock:
            if cluster_url not in self._pool_locks:
                self._pool_locks[cluster_url] = threading.Lock()
            return self._pool_locks[cluster_url]

    def _create_client(self, cluster_url: str) -> Any:
        """Create a new Kusto client."""
        try:
            from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
            from .utils import normalize_cluster_uri

            normalized_url = normalize_cluster_uri(cluster_url)
            kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(normalized_url)
            client = KustoClient(kcsb)

            self._stats.connections_created += 1
            logger.info("[POOL] Created new connection for %s", normalized_url)
            return client

        except Exception as e:
            self._stats.failed_connections += 1
            logger.error("[POOL] Failed to create connection for %s: %s", cluster_url, e)
            raise

    def get_client(self, cluster_url: str) -> Any:
        """
        Get a Kusto client from the pool.

        Args:
            cluster_url: The Kusto cluster URL

        Returns:
            A Kusto client instance

        Raises:
            ConnectionError: If unable to get or create a connection
        """
        from .utils import normalize_cluster_uri
        normalized_url = normalize_cluster_uri(cluster_url)
        pool_lock = self._get_pool_lock(normalized_url)
        start_time = time.time()

        with pool_lock:
            self._stats.total_requests += 1

            # Initialize pool for this cluster if needed
            if normalized_url not in self._pools:
                self._pools[normalized_url] = OrderedDict()

            pool = self._pools[normalized_url]

            # Try to find an available healthy connection
            for conn_id, conn_info in list(pool.items()):
                # Skip expired or unhealthy connections
                if conn_info.is_expired(self._max_connection_age):
                    self._recycle_connection(normalized_url, conn_id)
                    continue

                if not conn_info.is_healthy:
                    continue

                # Found a usable connection
                conn_info.mark_used()
                self._stats.cache_hits += 1
                wait_time = (time.time() - start_time) * 1000
                self._update_average_wait_time(wait_time)
                logger.debug(
                    "[POOL] Reusing connection for %s (hits=%d)",
                    normalized_url, self._stats.cache_hits
                )
                return conn_info.client

            # No available connection, create new one if under limit
            self._stats.cache_misses += 1
            if len(pool) < self._max_size:
                client = self._create_client(normalized_url)
                conn_id = f"conn_{len(pool)}_{int(time.time())}"
                pool[conn_id] = ConnectionInfo(
                    client=client,
                    cluster_url=normalized_url
                )
                wait_time = (time.time() - start_time) * 1000
                self._update_average_wait_time(wait_time)
                return client

            # Pool is full, reuse oldest connection
            if pool:
                _, oldest_conn = next(iter(pool.items()))
                oldest_conn.mark_used()
                logger.warning("[POOL] Pool full, reusing oldest connection for %s", normalized_url)
                return oldest_conn.client

            raise ConnectionError(f"Unable to get connection for {normalized_url}")

    def release_client(self, cluster_url: str) -> None:
        """
        Release a client back to the pool.

        For this implementation, connections are kept in the pool
        until they expire or are explicitly closed.
        """
        # In this implementation, connections stay in pool
        # This method is provided for API compatibility
        _ = cluster_url  # Acknowledge parameter for API compatibility

    def _recycle_connection(self, cluster_url: str, conn_id: str) -> None:
        """Recycle (close and remove) a connection."""
        pool = self._pools.get(cluster_url, {})
        if conn_id in pool:
            conn_info = pool.pop(conn_id)
            try:
                if hasattr(conn_info.client, 'close'):
                    conn_info.client.close()
            except (OSError, RuntimeError, AttributeError) as e:
                logger.warning("[POOL] Error closing connection: %s", e)
            self._stats.connections_recycled += 1
            logger.info("[POOL] Recycled connection %s for %s", conn_id, cluster_url)

    def _update_average_wait_time(self, wait_time_ms: float) -> None:
        """Update rolling average wait time."""
        n = self._stats.total_requests
        if n == 1:
            self._stats.average_wait_time_ms = wait_time_ms
        else:
            # Rolling average
            self._stats.average_wait_time_ms = (
                (self._stats.average_wait_time_ms * (n - 1) + wait_time_ms) / n
            )

    def cleanup_idle_connections(self) -> int:
        """Remove idle connections from all pools."""
        removed = 0
        with self._global_lock:
            for cluster_url in list(self._pools.keys()):
                pool_lock = self._get_pool_lock(cluster_url)
                with pool_lock:
                    pool = self._pools[cluster_url]
                    for conn_id in list(pool.keys()):
                        conn_info = pool[conn_id]
                        is_idle = conn_info.is_idle(self._idle_timeout)
                        is_expired = conn_info.is_expired(self._max_connection_age)
                        if is_idle or is_expired:
                            self._recycle_connection(cluster_url, conn_id)
                            removed += 1
        logger.info("[POOL] Cleaned up %d idle connections", removed)
        return removed

    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._global_lock:
            total_conns = sum(len(pool) for pool in self._pools.values())
            self._stats.total_connections = total_conns

            return {
                "total_connections": self._stats.total_connections,
                "pools": len(self._pools),
                "total_requests": self._stats.total_requests,
                "cache_hits": self._stats.cache_hits,
                "cache_misses": self._stats.cache_misses,
                "hit_rate_percent": round(self._stats.hit_rate(), 2),
                "connections_created": self._stats.connections_created,
                "connections_recycled": self._stats.connections_recycled,
                "failed_connections": self._stats.failed_connections,
                "average_wait_time_ms": round(self._stats.average_wait_time_ms, 2),
                "max_pool_size": self._max_size,
                "idle_timeout_seconds": self._idle_timeout
            }

    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._global_lock:
            for cluster_url in list(self._pools.keys()):
                pool_lock = self._get_pool_lock(cluster_url)
                with pool_lock:
                    pool = self._pools[cluster_url]
                    for conn_id in list(pool.keys()):
                        self._recycle_connection(cluster_url, conn_id)
            self._pools.clear()
        logger.info("[POOL] All connections closed")


# =============================================================================
# Global Pool Instance
# =============================================================================

def get_connection_pool() -> KustoConnectionPool:
    """Get the global connection pool instance."""
    return KustoConnectionPool()


# =============================================================================
# Batch Query Execution
# =============================================================================

class BatchQueryExecutor:
    """
    Execute multiple KQL queries in parallel.

    Features:
    - Parallel execution with configurable concurrency
    - Progress tracking
    - Error handling per query
    - Aggregated results

    Example:
        executor = BatchQueryExecutor(max_workers=5)
        queries = [
            ("q1", "StormEvents | take 10"),
            ("q2", "StormEvents | count"),
        ]
        results = executor.execute_batch(queries, cluster, database)
    """

    def __init__(
        self,
        max_workers: int = DEFAULT_BATCH_SIZE,
        timeout: int = DEFAULT_BATCH_TIMEOUT
    ):
        """Initialize batch executor."""
        self.max_workers = max_workers
        self.timeout = timeout
        self._pool = get_connection_pool()

    def execute_batch(
        self,
        queries: List[Tuple[str, str]],  # List of (query_id, query_text)
        cluster_url: str,
        database: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[BatchQueryResult]:
        """
        Execute multiple queries in parallel.

        Args:
            queries: List of (query_id, query_text) tuples
            cluster_url: Kusto cluster URL
            database: Database name
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            List of BatchQueryResult objects
        """
        results: List[BatchQueryResult] = []
        total = len(queries)
        completed = 0

        def execute_single(query_id: str, query: str) -> BatchQueryResult:
            """Execute a single query and return result."""
            start_time = time.time()
            try:
                from .execute_kql import _execute_kusto_query_sync  # pylint: disable=C0415
                df = _execute_kusto_query_sync(query, cluster_url, database)
                execution_time = (time.time() - start_time) * 1000

                return BatchQueryResult(
                    query_id=query_id,
                    query=query,
                    success=True,
                    data=df.to_dict('records') if not df.empty else [],
                    execution_time_ms=execution_time,
                    row_count=len(df)
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Must catch all exceptions to return proper BatchQueryResult
                execution_time = (time.time() - start_time) * 1000
                return BatchQueryResult(
                    query_id=query_id,
                    query=query,
                    success=False,
                    error=str(e),
                    execution_time_ms=execution_time
                )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(execute_single, qid, q): qid
                for qid, q in queries
            }

            for future in as_completed(futures, timeout=self.timeout):
                result = future.result()
                results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        logger.info("[BATCH] Executed %d queries: %d succeeded, %d failed",
                    total,
                    sum(1 for r in results if r.success),
                    sum(1 for r in results if not r.success))

        return results

    async def execute_batch_async(
        self,
        queries: List[Tuple[str, str]],
        cluster_url: str,
        database: str
    ) -> List[BatchQueryResult]:
        """
        Execute batch queries using asyncio.

        This is a convenience wrapper for async contexts.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.execute_batch(queries, cluster_url, database)
        )


# =============================================================================
# Schema Preloader
# =============================================================================

class SchemaPreloader:
    """
    Preload and cache database schemas at startup.

    Features:
    - Background schema loading
    - Configurable tables to preload
    - Progress tracking
    - Automatic refresh

    Example:
        preloader = SchemaPreloader()
        preloader.preload_schemas(cluster, database, ["Table1", "Table2"])
    """

    def __init__(self):
        """Initialize schema preloader."""
        self._preloaded: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def preload_schemas(
        self,
        cluster_url: str,
        database: str,
        tables: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Preload schemas for specified tables.

        Args:
            cluster_url: Kusto cluster URL
            database: Database name
            tables: List of table names (None = discover all)
            progress_callback: Optional callback(table, completed, total)

        Returns:
            Dictionary of preloaded schemas
        """
        from .memory import get_memory_manager  # pylint: disable=C0415

        memory = get_memory_manager()
        results = {}

        # Discover tables if not specified
        if tables is None:
            try:
                # Get all tables from database
                from .execute_kql import _execute_kusto_query_sync  # pylint: disable=C0415
                df = _execute_kusto_query_sync(
                    ".show tables | project TableName",
                    cluster_url,
                    database
                )
                tables = df['TableName'].tolist() if 'TableName' in df.columns else []
            except (ValueError, RuntimeError, OSError) as e:
                logger.warning("[PRELOAD] Could not discover tables: %s", e)
                tables = []

        total = len(tables)
        for idx, table in enumerate(tables):
            try:
                # Use memory manager to cache schema
                # store_schema expects schema dict, not individual columns/source args
                memory.store_schema(
                    cluster=cluster_url,
                    database=database,
                    table=table,
                    schema={"columns": {}},  # Placeholder, will be discovered
                    description="Preloaded at startup"
                )
                results[table] = {"success": True, "preloaded": True}

                if progress_callback:
                    progress_callback(table, idx + 1, total)

            except (ValueError, RuntimeError, OSError) as e:
                results[table] = {"success": False, "error": str(e)}
                logger.warning("[PRELOAD] Failed to preload schema for %s: %s", table, e)

        with self._lock:
            cache_key = f"{cluster_url}:{database}"
            self._preloaded[cache_key] = results

        logger.info("[PRELOAD] Preloaded %d/%d table schemas for %s/%s",
                    sum(1 for r in results.values() if r.get("success")),
                    total, cluster_url, database)

        return results

    def get_preloaded_status(self) -> Dict[str, Any]:
        """Get status of preloaded schemas."""
        with self._lock:
            return {
                "databases_preloaded": len(self._preloaded),
                "details": {
                    key: {
                        "tables": len(schemas),
                        "successful": sum(1 for s in schemas.values() if s.get("success"))
                    }
                    for key, schemas in self._preloaded.items()
                }
            }


# =============================================================================
# Async Utilities
# =============================================================================

async def execute_query_async(
    query: str,
    cluster_url: str,
    database: str,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Execute a KQL query asynchronously.

    This wraps the synchronous execution in an executor
    to avoid blocking the event loop.

    Args:
        query: KQL query string
        cluster_url: Kusto cluster URL
        database: Database name
        timeout: Query timeout in seconds

    Returns:
        Query result dictionary
    """
    loop = asyncio.get_event_loop()

    def _run_query():
        from .execute_kql import _execute_kusto_query_sync  # pylint: disable=C0415
        try:
            df = _execute_kusto_query_sync(query, cluster_url, database, timeout)
            return {
                "success": True,
                "data": df.to_dict('records') if not df.empty else [],
                "row_count": len(df)
            }
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Must catch all exceptions to return proper error dict
            return {
                "success": False,
                "error": str(e)
            }

    return await loop.run_in_executor(None, _run_query)


async def execute_queries_async(
    queries: List[Tuple[str, str]],
    cluster_url: str,
    database: str,
    max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """
    Execute multiple queries concurrently with async/await.

    Args:
        queries: List of (query_id, query_text) tuples
        cluster_url: Kusto cluster URL
        database: Database name
        max_concurrent: Maximum concurrent queries

    Returns:
        List of result dictionaries
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _execute_with_semaphore(query_id: str, query: str) -> Dict[str, Any]:
        async with semaphore:
            result = await execute_query_async(query, cluster_url, database)
            result["query_id"] = query_id
            return result

    tasks = [
        _execute_with_semaphore(qid, q)
        for qid, q in queries
    ]

    # Gather results, converting any exceptions to error dicts
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results: List[Dict[str, Any]] = []
    for i, res in enumerate(raw_results):
        if isinstance(res, BaseException):
            query_id = queries[i][0] if i < len(queries) else f"query_{i}"
            results.append({
                "query_id": query_id,
                "success": False,
                "error": str(res)
            })
        else:
            results.append(res)
    return results


# =============================================================================
# Health Check
# =============================================================================

def check_connection_health(cluster_url: str, database: str) -> Dict[str, Any]:
    """
    Check the health of a Kusto connection.

    Args:
        cluster_url: Kusto cluster URL
        database: Database name

    Returns:
        Health check result dictionary
    """
    start_time = time.time()
    try:
        from .execute_kql import _execute_kusto_query_sync  # pylint: disable=C0415

        # Simple query to test connection
        _df = _execute_kusto_query_sync("print now()", cluster_url, database)
        latency = (time.time() - start_time) * 1000

        return {
            "healthy": True,
            "cluster": cluster_url,
            "database": database,
            "latency_ms": round(latency, 2),
            "checked_at": datetime.now().isoformat()
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Must catch all exceptions to return health status
        latency = (time.time() - start_time) * 1000
        return {
            "healthy": False,
            "cluster": cluster_url,
            "database": database,
            "error": str(e),
            "latency_ms": round(latency, 2),
            "checked_at": datetime.now().isoformat()
        }


# =============================================================================
# Performance Metrics
# =============================================================================

class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        """Initialize performance monitor."""
        self._metrics: Dict[str, List[float]] = {
            "query_times": [],
            "schema_lookup_times": [],
            "connection_times": []
        }
        self._lock = threading.Lock()
        self._max_samples = 1000

    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a performance metric."""
        with self._lock:
            if metric_name not in self._metrics:
                self._metrics[metric_name] = []
            self._metrics[metric_name].append(value)
            # Keep only last N samples
            if len(self._metrics[metric_name]) > self._max_samples:
                self._metrics[metric_name] = self._metrics[metric_name][-self._max_samples:]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            summary = {}
            for name, values in self._metrics.items():
                if values:
                    sorted_vals = sorted(values)
                    p95_idx = int(len(values) * 0.95)
                    p95_val = sorted_vals[p95_idx] if len(values) > 20 else max(values)
                    summary[name] = {
                        "count": len(values),
                        "avg_ms": round(sum(values) / len(values), 2),
                        "min_ms": round(min(values), 2),
                        "max_ms": round(max(values), 2),
                        "p95_ms": round(p95_val, 2)
                    }
            return summary


# Global performance monitor
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _performance_monitor


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Classes
    "KustoConnectionPool",
    "BatchQueryExecutor",
    "SchemaPreloader",
    "PerformanceMonitor",
    "ConnectionInfo",
    "PoolStatistics",
    "BatchQueryResult",
    # Functions
    "get_connection_pool",
    "get_performance_monitor",
    "execute_query_async",
    "execute_queries_async",
    "check_connection_health",
    # Constants
    "DEFAULT_POOL_SIZE",
    "DEFAULT_MAX_POOL_SIZE",
    "DEFAULT_BATCH_SIZE",
]
