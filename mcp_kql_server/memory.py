"""
Unified Schema Memory System for MCP KQL Server (PostgreSQL + pgvector + CAG + TOON)

This module provides a high-performance memory system that:
- Uses PostgreSQL with pgvector for persistent, shared storage of schemas and queries.
- Implements Context Augmented Generation (CAG) to load full schemas into LLM context.
- Uses TOON (Token-Oriented Object Notation) for compact schema representation.
- Supports Semantic Search (using Azure OpenAI embeddings) for Few-Shot prompting.
- Uses pgvector for native cosine similarity search in PostgreSQL.

Author: Arjun Trivedi
"""

import json
import hashlib
import logging
import threading
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    import psycopg2
    import psycopg2.pool
    import psycopg2.extras
    from pgvector.psycopg2 import register_vector
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

from .llm_client import generate_embedding

logger = logging.getLogger(__name__)

# TOON Type Mapping for compression
TOON_TYPE_MAP = {
    'string': 's',
    'int': 'i',
    'long': 'l',
    'real': 'r',
    'double': 'd',
    'decimal': 'd',
    'datetime': 'dt',
    'timespan': 'ts',
    'bool': 'b',
    'boolean': 'b',
    'dynamic': 'dyn',
    'guid': 'g',
    'array': 'arr',
    'object': 'obj'
}

@dataclass
class ValidationResult:
    """Result of query validation against schema."""
    is_valid: bool
    validated_query: str
    errors: List[str]

class MemoryManager:
    """
    PostgreSQL-backed Memory Manager for KQL Schemas and Queries.
    Implements CAG (Context Augmented Generation) with TOON formatting.
    Uses pgvector for native vector similarity search.
    """
    def __init__(self):
        from .constants import POSTGRES_CONFIG
        self._pg_config = POSTGRES_CONFIG
        self._schema_cache: Dict[str, Any] = {}
        self._pool = None
        self._pool_lock = threading.Lock()
        self._prefix = POSTGRES_CONFIG["table_prefix"]
        self._db_available = False
        self._init_db()

    @property
    def memory_path(self) -> str:
        """Return connection info string for compatibility."""
        return f"postgresql://{self._pg_config['host']}:{self._pg_config['port']}/{self._pg_config['database']}"

    def _get_pool(self):
        """Lazily create and return the connection pool."""
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    self._pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=self._pg_config["min_connections"],
                        maxconn=self._pg_config["max_connections"],
                        host=self._pg_config["host"],
                        port=self._pg_config["port"],
                        user=self._pg_config["user"],
                        password=self._pg_config["password"],
                        dbname=self._pg_config["database"],
                        connect_timeout=self._pg_config["connect_timeout"],
                        sslmode=self._pg_config["sslmode"],
                    )
                    # Register pgvector type on a test connection
                    conn = self._pool.getconn()
                    try:
                        conn.rollback()  # End any implicit transaction
                        register_vector(conn)
                        conn.rollback()  # End transaction started by register_vector
                    finally:
                        self._pool.putconn(conn)
                    logger.info("PostgreSQL connection pool created (%s:%s/%s)",
                                self._pg_config["host"],
                                self._pg_config["port"],
                                self._pg_config["database"])
        return self._pool

    @contextmanager
    def _get_conn(self):
        """Get a connection from the pool, yield cursor, commit on success."""
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            conn.rollback()  # End any implicit transaction
            register_vector(conn)
            conn.rollback()  # End transaction started by register_vector
            cur = conn.cursor()
            try:
                yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.close()
        finally:
            pool.putconn(conn)

    def _init_db(self):
        """Initialize PostgreSQL database schema with pgvector."""
        if not HAS_PSYCOPG2:
            logger.error("psycopg2 not installed. Memory features will be unavailable.")
            return

        try:
            with self._get_conn() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Schema table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._prefix}schemas (
                        cluster TEXT NOT NULL,
                        database TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        columns_json TEXT,
                        embedding vector(1536),
                        description TEXT,
                        last_updated TIMESTAMP DEFAULT NOW(),
                        PRIMARY KEY (cluster, database, table_name)
                    )
                """)

                # Queries table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._prefix}queries (
                        id SERIAL PRIMARY KEY,
                        cluster TEXT,
                        database TEXT,
                        query TEXT,
                        description TEXT,
                        embedding vector(1536),
                        timestamp TIMESTAMP DEFAULT NOW(),
                        execution_time_ms REAL
                    )
                """)

                # Join hints table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._prefix}join_hints (
                        id SERIAL PRIMARY KEY,
                        table1 TEXT,
                        table2 TEXT,
                        join_condition TEXT,
                        confidence REAL,
                        last_used TIMESTAMP DEFAULT NOW(),
                        UNIQUE(table1, table2, join_condition)
                    )
                """)

                # Query cache table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._prefix}query_cache (
                        query_hash TEXT PRIMARY KEY,
                        result_json TEXT,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        row_count INTEGER
                    )
                """)

                # Learning events table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._prefix}learning_events (
                        id SERIAL PRIMARY KEY,
                        query TEXT,
                        execution_type TEXT,
                        result_json TEXT,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        execution_time_ms REAL
                    )
                """)

                # Standard indexes
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}schemas_db ON {self._prefix}schemas(cluster, database)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}queries_db ON {self._prefix}queries(cluster, database)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}join_hints ON {self._prefix}join_hints(table1, table2)")

                # HNSW indexes for vector similarity search
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._prefix}schemas_embedding
                    ON {self._prefix}schemas USING hnsw (embedding vector_cosine_ops)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._prefix}queries_embedding
                    ON {self._prefix}queries USING hnsw (embedding vector_cosine_ops)
                """)

            self._db_available = True
            logger.info("PostgreSQL memory tables initialized")
        except Exception as e:
            logger.error("Failed to initialize PostgreSQL memory: %s. Memory features will be unavailable.", e)
            self._db_available = False

    def store_schema(self, cluster: str, database: str, table: str,
                     schema: Dict[str, Any], description: Optional[str] = None):
        """Store or update a table schema with embedding and description."""
        if not self._db_available:
            return

        columns = schema.get("columns", {})

        # Normalize columns to dict format if it's a list
        if isinstance(columns, list):
            normalized_cols = {}
            for col in columns:
                if isinstance(col, dict):
                    name = col.get("name") or col.get("column")
                    if name:
                        normalized_cols[name] = col
                elif isinstance(col, str):
                    normalized_cols[col] = {"data_type": "string"}
            columns = normalized_cols

        # Generate embedding for table (name + column names + dynamic sub-field names)
        col_names = " ".join(columns.keys())
        embedding_text = f"Table {table} contains columns: {col_names}"

        # Include dynamic sub-field names for richer semantic search
        dynamic_paths = []
        for col_name, col_def in columns.items():
            if isinstance(col_def, dict) and col_def.get("dynamic_fields"):
                for field_name in col_def["dynamic_fields"]:
                    dynamic_paths.append(f"{col_name}.{field_name}")
        if dynamic_paths:
            embedding_text += f". Dynamic fields: {' '.join(dynamic_paths)}"

        embedding = generate_embedding(embedding_text)

        try:
            # If description is not provided, try to preserve existing one
            if description is None:
                with self._get_conn() as cur:
                    cur.execute(
                        f"SELECT description FROM {self._prefix}schemas WHERE cluster=%s AND database=%s AND table_name=%s",
                        (cluster, database, table)
                    )
                    row = cur.fetchone()
                    if row:
                        description = row[0]

            with self._get_conn() as cur:
                cur.execute(f"""
                    INSERT INTO {self._prefix}schemas
                    (cluster, database, table_name, columns_json, embedding, description, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (cluster, database, table_name)
                    DO UPDATE SET
                        columns_json = EXCLUDED.columns_json,
                        embedding = EXCLUDED.embedding,
                        description = EXCLUDED.description,
                        last_updated = EXCLUDED.last_updated
                """, (cluster, database, table, json.dumps(columns), embedding, description))

            logger.debug("Stored schema for %s in %s", table, database)
        except Exception as e:
            logger.error("Failed to store schema: %s", e)

    def add_successful_query(self, cluster: str, database: str, query: str,
                             description: str, execution_time_ms: float = 0.0):
        """Store a successful query with its description and embedding."""
        if not self._db_available:
            return

        embedding = generate_embedding(f"{description} {query}")

        try:
            with self._get_conn() as cur:
                cur.execute(f"""
                    INSERT INTO {self._prefix}queries
                    (cluster, database, query, description, embedding, timestamp, execution_time_ms)
                    VALUES (%s, %s, %s, %s, %s, NOW(), %s)
                """, (cluster, database, query, description, embedding, execution_time_ms))
        except Exception as e:
            logger.error("Failed to store query: %s", e)

    def add_global_successful_query(self, cluster: str, database: str, query: str,
                                    description: str, execution_time_ms: float = 0.0):
        """Store a successful query globally (alias for add_successful_query for now)."""
        self.add_successful_query(cluster, database, query, description, execution_time_ms)

    def store_learning_result(self, query: str, result_data: Dict[str, Any],
                              execution_type: str, execution_time_ms: float = 0.0):
        """Store learning result from query execution."""
        if not self._db_available:
            return

        try:
            with self._get_conn() as cur:
                cur.execute(f"""
                    INSERT INTO {self._prefix}learning_events
                    (query, execution_type, result_json, timestamp, execution_time_ms)
                    VALUES (%s, %s, %s, NOW(), %s)
                """, (query, execution_type, json.dumps(result_data), execution_time_ms))
        except Exception as e:
            logger.error("Failed to store learning result: %s", e)

    def find_relevant_tables(self, cluster: str, database: str,
                             query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find tables semantically related to the query using pgvector."""
        if not self._db_available:
            return []

        query_embedding = generate_embedding(query)
        if query_embedding is None:
            return []

        try:
            with self._get_conn() as cur:
                cur.execute(f"""
                    SELECT table_name, columns_json,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM {self._prefix}schemas
                    WHERE cluster = %s AND database = %s
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, cluster, database, query_embedding, limit))

                results = []
                for row in cur.fetchall():
                    results.append({
                        "table": row[0],
                        "columns": json.loads(row[1]),
                        "score": float(row[2])
                    })

            return results
        except Exception as e:
            logger.error("Failed to find relevant tables: %s", e)
            return []

    def find_similar_queries(self, cluster: str, database: str,
                               query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find similar past queries using pgvector similarity search."""
        if not self._db_available:
            return []

        query_embedding = generate_embedding(query)
        if query_embedding is None:
            return []

        try:
            with self._get_conn() as cur:
                cur.execute(f"""
                    SELECT query, description,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM {self._prefix}queries
                    WHERE cluster = %s AND database = %s
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, cluster, database, query_embedding, limit))

                return [
                    {"query": row[0], "description": row[1], "score": float(row[2])}
                    for row in cur.fetchall()
                ]
        except Exception as e:
            logger.error("Failed to find similar queries: %s", e)
            return []

    def cache_query_result(self, query: str, result_json: str, row_count: int):
        """Cache query result."""
        if not self._db_available:
            return

        query_hash = hashlib.sha256(query.encode()).hexdigest()

        try:
            with self._get_conn() as cur:
                cur.execute(f"""
                    INSERT INTO {self._prefix}query_cache (query_hash, result_json, timestamp, row_count)
                    VALUES (%s, %s, NOW(), %s)
                    ON CONFLICT (query_hash)
                    DO UPDATE SET result_json = EXCLUDED.result_json,
                                  timestamp = EXCLUDED.timestamp,
                                  row_count = EXCLUDED.row_count
                """, (query_hash, result_json, row_count))
        except Exception as e:
            logger.error("Failed to cache query result: %s", e)

    def get_cached_result(self, query: str, ttl_seconds: int = 300) -> Optional[str]:
        """Get cached result if valid."""
        if not self._db_available:
            return None

        query_hash = hashlib.sha256(query.encode()).hexdigest()

        try:
            with self._get_conn() as cur:
                cur.execute(f"""
                    SELECT result_json FROM {self._prefix}query_cache
                    WHERE query_hash = %s
                      AND timestamp > NOW() - INTERVAL '%s seconds'
                """, (query_hash, ttl_seconds))
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.error("Failed to get cached result: %s", e)
            return None

    def store_join_hint(self, table1: str, table2: str, condition: str, confidence: float = 1.0):
        """Store a discovered join relationship."""
        if not self._db_available:
            return

        try:
            with self._get_conn() as cur:
                cur.execute(f"""
                    INSERT INTO {self._prefix}join_hints (table1, table2, join_condition, confidence, last_used)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (table1, table2, join_condition)
                    DO UPDATE SET confidence = EXCLUDED.confidence,
                                  last_used = EXCLUDED.last_used
                """, (table1, table2, condition, confidence))
        except Exception as e:
            logger.error("Failed to store join hint: %s", e)

    def get_join_hints(self, tables: List[str]) -> List[str]:
        """Get join hints relevant to the provided tables."""
        if not tables or not self._db_available:
            return []

        try:
            placeholders = ','.join(['%s'] * len(tables))
            with self._get_conn() as cur:
                cur.execute(f"""
                    SELECT table1, table2, join_condition FROM {self._prefix}join_hints
                    WHERE table1 IN ({placeholders}) OR table2 IN ({placeholders})
                """, tables + tables)

                hints = [f"{row[0]} joins with {row[1]} on {row[2]}" for row in cur.fetchall()]

            return list(set(hints))
        except Exception as e:
            logger.error("Failed to get join hints: %s", e)
            return []

    def _get_database_schema(self, cluster: str, database: str) -> List[Dict[str, Any]]:
        """Get schema from PostgreSQL with in-memory caching."""
        cache_key = f"db_schema_{cluster}_{database}"
        # Simple in-memory cache check
        if cache_key in self._schema_cache:
            cached = self._schema_cache[cache_key]
            if (datetime.now() - cached['ts']).seconds < 300:  # 5 min TTL
                return cached['data']

        if not self._db_available:
            return []

        try:
            with self._get_conn() as cur:
                cur.execute(
                    f"SELECT table_name, columns_json FROM {self._prefix}schemas WHERE cluster = %s AND database = %s",
                    (cluster, database)
                )
                schemas = [{"table": row[0], "columns": json.loads(row[1])} for row in cur.fetchall()]
        except Exception as e:
            logger.error("Failed to get database schema: %s", e)
            schemas = []

        # Cache result
        self._schema_cache[cache_key] = {'data': schemas, 'ts': datetime.now()}
        return schemas

    def get_relevant_context(self, cluster: str, database: str, user_query: str, max_tables: int = 20) -> str:
        """
        Optimized CAG: Get schema + similar queries + join hints in TOON format.
        Limited to max_tables to prevent token overflow.
        """
        # 1. Get schemas (limited)
        schemas = self._get_database_schema(cluster, database)[:max_tables]
        table_names = [s["table"] for s in schemas]

        # 2. Get similar queries (parallel-safe)
        similar_queries = self.find_similar_queries(cluster, database, user_query, limit=3)

        # 3. Get join hints
        join_hints = self.get_join_hints(table_names) if table_names else []

        # 4. Format as compact TOON
        return self._to_toon(schemas, similar_queries, join_hints)

    def _to_toon(self, schemas: List[Dict], similar_queries: List[Dict],
                 join_hints: Optional[List[str]] = None) -> str:
        """Optimized TOON formatting with size limits."""
        lines = ["<CAG_CONTEXT>"]

        # Compact syntax guidance
        lines.append("# KQL Rules: Use != (not ! =), !contains, !in, !has. No spaces in negation.")

        # Schema Section (compact)
        if schemas:
            lines.append("# Schema (TOON)")
            for schema in schemas:
                table = schema["table"]
                cols = []
                for col_name, col_def in schema["columns"].items():
                    # Handle different column definition formats
                    col_type = "string"
                    if isinstance(col_def, dict):
                        col_type = col_def.get("data_type") or col_def.get("type") or "string"
                    elif isinstance(col_def, str): # simple key-value
                        col_type = col_def

                    # Map to short type
                    short_type = TOON_TYPE_MAP.get(col_type.lower(), 's')

                    # For dynamic columns with known sub-fields, append them
                    dynamic_fields = col_def.get("dynamic_fields") if isinstance(col_def, dict) else None
                    if short_type == 'dyn' and dynamic_fields:
                        sub_parts = []
                        for sf_name, sf_info in list(dynamic_fields.items())[:10]:  # Limit to 10 sub-fields
                            sf_type = sf_info.get("type", "s") if isinstance(sf_info, dict) else "s"
                            sf_short = TOON_TYPE_MAP.get(sf_type, sf_type[:3])
                            sub_parts.append(f"{sf_name}:{sf_short}")
                        cols.append(f"{col_name}:dyn{{{','.join(sub_parts)}}}")
                    else:
                        cols.append(f"{col_name}:{short_type}")

                lines.append(f"{table}({', '.join(cols)})")
        else:
            lines.append("# No Schema Found (Run queries to discover)")

        # Join Hints Section
        if join_hints:
            lines.append("\n# Join Hints")
            for hint in join_hints:
                lines.append(f"// {hint}")

        # Few-Shot Section
        if similar_queries:
            lines.append("\n# Similar Queries")
            for q in similar_queries:
                lines.append(f"// {q['description']}")
                lines.append(q['query'])

        lines.append("</CAG_CONTEXT>")
        return "\n".join(lines)

    def clear_memory(self) -> bool:
        """Clear all data from the database."""
        if not self._db_available:
            return False

        try:
            with self._get_conn() as cur:
                cur.execute(f"DELETE FROM {self._prefix}schemas")
                cur.execute(f"DELETE FROM {self._prefix}queries")
                cur.execute(f"DELETE FROM {self._prefix}learning_events")
            return True
        except Exception as e:
            logger.error("Failed to clear memory: %s", e)
            return False

    # Use centralized normalize_cluster_uri from utils.py
    # Import at method level to avoid circular imports
    def normalize_cluster_uri(self, uri: str) -> str:
        """Normalize cluster URI - delegates to utils."""
        from .utils import normalize_cluster_uri as _normalize
        return _normalize(uri) if uri else ""

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory database."""
        if not self._db_available:
            return {
                "schema_count": 0,
                "query_count": 0,
                "learning_count": 0,
                "db_size_bytes": 0,
                "db_path": self.memory_path
            }

        try:
            with self._get_conn() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._prefix}schemas")
                schema_count = cur.fetchone()[0]
                cur.execute(f"SELECT COUNT(*) FROM {self._prefix}queries")
                query_count = cur.fetchone()[0]
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {self._prefix}learning_events")
                    learning_count = cur.fetchone()[0]
                except Exception:
                    learning_count = 0

            return {
                "schema_count": schema_count,
                "query_count": query_count,
                "learning_count": learning_count,
                "db_size_bytes": 0,
                "db_path": self.memory_path
            }
        except Exception as e:
            logger.error("Failed to get memory stats: %s", e)
            return {
                "schema_count": 0,
                "query_count": 0,
                "learning_count": 0,
                "db_size_bytes": 0,
                "db_path": self.memory_path
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics (execution time, success rate)."""
        if not self._db_available:
            return {"average_execution_time_ms": 0.0, "total_successful_queries": 0}

        try:
            with self._get_conn() as cur:
                try:
                    cur.execute(
                        f"SELECT AVG(execution_time_ms) FROM {self._prefix}queries WHERE execution_time_ms > 0"
                    )
                    avg_time = cur.fetchone()[0]
                    avg_time = avg_time if avg_time is not None else 0.0
                except Exception:
                    avg_time = 0.0

                cur.execute(f"SELECT COUNT(*) FROM {self._prefix}queries")
                query_count = cur.fetchone()[0]

            return {
                "average_execution_time_ms": round(avg_time, 2),
                "total_successful_queries": query_count
            }
        except Exception as e:
            logger.error("Failed to get performance metrics: %s", e)
            return {"average_execution_time_ms": 0.0, "total_successful_queries": 0}

    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent successful queries."""
        if not self._db_available:
            return []

        try:
            with self._get_conn() as cur:
                cur.execute(
                    f"SELECT query, description, cluster, database, timestamp FROM {self._prefix}queries ORDER BY id DESC LIMIT %s",
                    (limit,)
                )
                return [
                    {
                        "query": row[0],
                        "description": row[1],
                        "cluster": row[2],
                        "database": row[3],
                        "timestamp": row[4].isoformat() if row[4] else None,
                        "result_metadata": {"success": True}
                    }
                    for row in cur.fetchall()
                ]
        except Exception as e:
            logger.error("Failed to get recent queries: %s", e)
            return []

    def get_ai_context_for_tables(self, cluster: str, database: str, tables: List[str]) -> str:
        """Wrapper for get_relevant_context to support list of tables."""
        table_str = ", ".join(tables)
        dummy_query = f"Querying tables: {table_str}"
        return self.get_relevant_context(cluster, database, dummy_query)

    def validate_query(self, query: str, cluster: str, database: str) -> ValidationResult:  # pylint: disable=unused-argument
        """
        Validate query against schema.
        Returns an object with is_valid, validated_query, errors.
        """
        return ValidationResult(
            is_valid=True,
            validated_query=query,
            errors=[]
        )

    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """Get session data (stub for compatibility)."""
        return {
            "sessions": {},
            "active_session": session_id
        }

    def get_database_schema(self, cluster: str, database: str) -> Dict[str, Any]:
        """Get database schema in the format expected by utils.py."""
        schemas = self._get_database_schema(cluster, database)
        table_names = [s["table"] for s in schemas]
        return {
            "database_name": database,
            "tables": table_names,
            "cluster": cluster
        }

    @property
    def corpus(self) -> Dict[str, Any]:
        """Compatibility property for legacy corpus access."""
        return {"clusters": {}}

    def save_corpus(self):
        """Compatibility method for legacy save_corpus calls (no-op)."""
        return None

# Global instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get the singleton MemoryManager instance."""
    global _memory_manager  # pylint: disable=global-statement
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager

def get_kql_operator_syntax_guidance() -> str:
    """
    Get KQL operator syntax guidance for AI query generation.
    """
    return """
=== KQL GENERATION RULES (STRICT) ===
1. SCHEMA COMPLIANCE:
   - You MUST ONLY use columns that explicitly appear in the provided schema.
   - Do NOT hallucinate column names (e.g., do not assume 'EntityType', 'Target', 'Source' exist unless shown).
   - If a column is missing, use 'find' or 'search' instead of specific column references, or ask the user to refresh schema.

2. OPERATOR SYNTAX (CRITICAL):
   - Negation: Use '!=' (not '! ='), '!contains', '!in', '!has'. NO SPACES in negation operators.

   ✓ CORRECT Negation Syntax:
   - where Status != 'Active' (no space between ! and =)
   - where Name !contains 'test' (no space between ! and contains)
   - where Category !in ('A', 'B') (no space between ! and in)
   - where Title !has 'error' (no space between ! and has)

   ✗ WRONG Negation Syntax (DO NOT USE):
   - where Status ! = 'Active' (space between ! and =)
   - where Name ! contains 'test' (space between ! and contains)
   - where Category ! in ('A', 'B') (space between ! and in)
   - where Category !has_any ('A', 'B') (!has_any does not exist)

   List Operations:
   - Use 'in' for membership: where RuleName in ('Rule1', 'Rule2')
   - Use '!in' for exclusion: where RuleName !in ('Rule1', 'Rule2')
   - NEVER use '!has_any': !has_any does not exist in KQL

   Alternative Negation (using 'not' keyword):
   - where not (Status == 'Active')
   - where not (Name contains 'test')

   String Operators:
   - has: whole word/term matching (e.g., 'error' matches 'error log' but not 'errors')
   - contains: substring matching (e.g., 'test' matches 'testing')
   - startswith: prefix matching
   - endswith: suffix matching
   - All can be negated with ! prefix (NO SPACE): !has, !contains, !startswith, !endswith

3. BEST PRACTICES:
   - Always verify column names against the schema before generating the query.
   - Use 'take 10' for initial exploration if unsure about data volume.
   - Prefer 'where Column has "Value"' over 'where Column == "Value"' for text search unless exact match is required.
"""
