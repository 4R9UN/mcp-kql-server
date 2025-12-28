# API Reference - MCP KQL Server

**Version**: 2.2.0
**Last Updated**: December 2025

---

## Table of Contents

1. [MCP Tools](#1-mcp-tools)
2. [Memory Manager API](#2-memory-manager-api)
3. [Performance API](#3-performance-api)
4. [Utility Functions](#4-utility-functions)
5. [Error Handling](#5-error-handling)
6. [Configuration](#6-configuration)

---

## 1. MCP Tools

### 1.1 execute_kql_query

Execute KQL queries or generate queries from natural language descriptions.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | KQL query or natural language description |
| `cluster_url` | string | Yes | - | Azure Data Explorer cluster URL |
| `database` | string | Yes | - | Database name |
| `output_format` | string | No | `"json"` | Output format: `"json"`, `"csv"`, `"table"` |
| `generate_query` | boolean | No | `false` | If true, treat query as natural language |
| `table_name` | string | No | - | Target table for NL2KQL |
| `use_live_schema` | boolean | No | `true` | Use live schema for validation |

#### Returns

```json
{
    "success": true,
    "query": "StormEvents | take 10",
    "generated_query": "StormEvents | take 10",  // If generate_query=true
    "row_count": 10,
    "columns": ["StartTime", "EndTime", "EventType", ...],
    "data": [...],
    "execution_time_ms": 234.5,
    "cluster": "https://help.kusto.windows.net",
    "database": "Samples"
}
```

#### Error Response

```json
{
    "success": false,
    "error": "Error message",
    "error_type": "SemanticError",
    "suggestions": [
        "Check column name spelling",
        "Available columns: Col1, Col2, Col3"
    ],
    "schema_context": {
        "table": "StormEvents",
        "columns": {...}
    }
}
```

#### Examples

**Direct KQL Execution:**
```python
result = await execute_kql_query(
    query="StormEvents | where State == 'TEXAS' | take 100",
    cluster_url="https://help.kusto.windows.net",
    database="Samples"
)
```

**Natural Language to KQL:**
```python
result = await execute_kql_query(
    query="Show me the top 10 storm events in Texas",
    cluster_url="https://help.kusto.windows.net",
    database="Samples",
    generate_query=True
)
```

---

### 1.2 schema_memory

Manage schema discovery, caching, and AI context generation.

#### Operations

| Operation | Description |
|-----------|-------------|
| `discover` | Discover and cache schema for a table |
| `list_tables` | List all tables in a database |
| `get_context` | Get AI-friendly context for tables |
| `generate_report` | Generate analysis report with visualizations |
| `clear_cache` | Clear cached schema data |
| `get_stats` | Get memory usage statistics |
| `refresh_schema` | Force refresh schema for a database |

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `operation` | string | Yes | Operation to perform |
| `cluster_url` | string | Varies | Kusto cluster URL |
| `database` | string | Varies | Database name |
| `table_name` | string | Varies | Table name (for discover) |
| `natural_language_query` | string | No | Query for context operations |
| `session_id` | string | No | Session ID for reports |
| `include_visualizations` | boolean | No | Include visualizations in reports |

#### Returns (by operation)

**discover:**
```json
{
    "success": true,
    "table": "StormEvents",
    "columns": {
        "StartTime": "datetime",
        "EndTime": "datetime",
        "EventType": "string",
        "State": "string",
        "DamageProperty": "long"
    },
    "row_count": 59066,
    "ai_context": "Storm events data with temporal and geographic information..."
}
```

**list_tables:**
```json
{
    "success": true,
    "tables": ["StormEvents", "PopulationData", "US_States"],
    "count": 3,
    "database": "Samples"
}
```

**get_stats:**
```json
{
    "success": true,
    "total_schemas": 15,
    "total_queries": 234,
    "cache_size_mb": 2.4,
    "oldest_schema": "2024-01-15T10:30:00Z",
    "newest_schema": "2024-12-28T15:45:00Z"
}
```

---

## 2. Memory Manager API

### 2.1 MemoryManager Class

```python
from mcp_kql_server.memory import get_memory_manager, MemoryManager

# Get singleton instance
memory = get_memory_manager()

# Or create with custom path
memory = MemoryManager(db_path="/custom/path/memory.db")
```

### 2.2 Methods

#### store_table_schema

Store schema information for a table.

```python
def store_table_schema(
    self,
    cluster: str,
    database: str,
    table: str,
    columns: Dict[str, str],
    row_count: Optional[int] = None,
    source: str = "discovery"
) -> Dict[str, Any]
```

**Parameters:**
- `cluster`: Cluster URL
- `database`: Database name
- `table`: Table name
- `columns`: Dictionary of column_name -> data_type
- `row_count`: Optional row count
- `source`: Source of schema ("discovery", "manual", "preload")

**Returns:**
```python
{
    "stored": True,
    "table": "StormEvents",
    "columns_count": 25
}
```

---

#### get_table_schema

Retrieve cached schema for a table.

```python
def get_table_schema(
    self,
    cluster: str,
    database: str,
    table: str
) -> Optional[Dict[str, Any]]
```

**Returns:**
```python
{
    "table": "StormEvents",
    "columns": {"StartTime": "datetime", ...},
    "row_count": 59066,
    "discovered_at": "2024-12-28T10:00:00Z",
    "ai_context": "..."
}
# Or None if not cached
```

---

#### find_relevant_tables

Find tables relevant to a query using semantic search.

```python
def find_relevant_tables(
    self,
    cluster: str,
    database: str,
    user_query: str,
    max_results: int = 5
) -> List[Dict[str, Any]]
```

**Parameters:**
- `cluster`: Cluster URL
- `database`: Database name
- `user_query`: Natural language query
- `max_results`: Maximum number of tables to return

**Returns:**
```python
[
    {
        "table": "StormEvents",
        "relevance_score": 0.95,
        "columns": {...},
        "description": "Storm event data..."
    },
    ...
]
```

---

#### get_relevant_context

Get AI-friendly context for NL2KQL generation.

```python
def get_relevant_context(
    self,
    cluster: str,
    database: str,
    user_query: str,
    max_tables: int = 20
) -> str
```

**Returns:** Formatted string with schema context suitable for AI prompts.

---

#### get_memory_stats

Get statistics about memory usage.

```python
def get_memory_stats(self) -> Dict[str, Any]
```

**Returns:**
```python
{
    "total_schemas": 15,
    "total_queries": 234,
    "total_embeddings": 15,
    "db_size_bytes": 2516992,
    "db_path": "/path/to/kql_memory.db"
}
```

---

#### clear_cache

Clear cached data.

```python
def clear_cache(
    self,
    cluster: Optional[str] = None,
    database: Optional[str] = None
) -> Dict[str, int]
```

**Returns:**
```python
{
    "schemas_cleared": 15,
    "queries_cleared": 234,
    "embeddings_cleared": 15
}
```

---

## 3. Performance API

### 3.1 KustoConnectionPool

```python
from mcp_kql_server.performance import get_connection_pool, KustoConnectionPool

# Get singleton instance
pool = get_connection_pool()

# Or create with custom config
pool = KustoConnectionPool(
    max_size=20,
    min_size=5,
    idle_timeout=1800,  # 30 minutes
    max_connection_age=3600  # 1 hour
)
```

#### Methods

**get_client:**
```python
def get_client(self, cluster_url: str) -> KustoClient
```

**get_statistics:**
```python
def get_statistics(self) -> Dict[str, Any]
```

Returns:
```python
{
    "total_connections": 5,
    "pools": 2,
    "total_requests": 1500,
    "cache_hits": 1400,
    "cache_misses": 100,
    "hit_rate_percent": 93.33,
    "connections_created": 5,
    "connections_recycled": 2,
    "average_wait_time_ms": 2.3
}
```

**cleanup_idle_connections:**
```python
def cleanup_idle_connections(self) -> int  # Returns count removed
```

**close_all:**
```python
def close_all(self) -> None
```

---

### 3.2 BatchQueryExecutor

```python
from mcp_kql_server.performance import BatchQueryExecutor

executor = BatchQueryExecutor(max_workers=5, timeout=300)

queries = [
    ("q1", "StormEvents | count"),
    ("q2", "StormEvents | take 10"),
    ("q3", "StormEvents | summarize count() by State")
]

results = executor.execute_batch(
    queries=queries,
    cluster_url="https://help.kusto.windows.net",
    database="Samples",
    progress_callback=lambda done, total: print(f"{done}/{total}")
)
```

**Result:**
```python
[
    BatchQueryResult(
        query_id="q1",
        query="StormEvents | count",
        success=True,
        data=[{"Count": 59066}],
        execution_time_ms=145.2,
        row_count=1
    ),
    ...
]
```

---

### 3.3 Async Utilities

```python
from mcp_kql_server.performance import (
    execute_query_async,
    execute_queries_async,
    check_connection_health
)

# Single async query
result = await execute_query_async(
    query="StormEvents | take 10",
    cluster_url="https://help.kusto.windows.net",
    database="Samples"
)

# Multiple concurrent queries
results = await execute_queries_async(
    queries=[("q1", "StormEvents | count"), ("q2", "StormEvents | take 10")],
    cluster_url="https://help.kusto.windows.net",
    database="Samples",
    max_concurrent=5
)

# Health check
health = check_connection_health(
    cluster_url="https://help.kusto.windows.net",
    database="Samples"
)
# {"healthy": True, "latency_ms": 45.2, ...}
```

---

### 3.4 PerformanceMonitor

```python
from mcp_kql_server.performance import get_performance_monitor

monitor = get_performance_monitor()

# Record metrics
monitor.record_metric("query_times", 234.5)
monitor.record_metric("schema_lookup_times", 12.3)

# Get summary
summary = monitor.get_metrics_summary()
# {
#     "query_times": {"count": 100, "avg_ms": 245.2, "p95_ms": 512.3, ...},
#     "schema_lookup_times": {...}
# }
```

---

## 4. Utility Functions

### 4.1 Query Processing

```python
from mcp_kql_server.utils import QueryProcessor

processor = QueryProcessor()

# Clean query
cleaned = processor.clean_query("  StormEvents | take 10  ")

# Extract entities
entities = processor.extract_entities("StormEvents | where State == 'TX'")
# {"tables": ["StormEvents"], "columns": ["State"], ...}
```

### 4.2 Error Handling

```python
from mcp_kql_server.utils import ErrorHandler

handler = ErrorHandler()

try:
    # ... execute query ...
except Exception as e:
    error_response = handler.handle_error(e, context={"query": query})
    # {
    #     "error": "...",
    #     "error_type": "SemanticError",
    #     "suggestions": [...],
    #     "recoverable": True
    # }
```

### 4.3 Schema Utilities

```python
from mcp_kql_server.utils import SchemaManager, normalize_cluster_uri

# Normalize cluster URL
url = normalize_cluster_uri("help.kusto.windows.net")
# "https://help.kusto.windows.net"

# Schema manager
schema_mgr = SchemaManager()
formatted = schema_mgr.format_schema_for_display(schema_dict)
```

---

## 5. Error Handling

### 5.1 Error Types

| Error Type | Description | HTTP-like Code |
|------------|-------------|----------------|
| `AuthenticationError` | Azure authentication failed | 401 |
| `NotFoundError` | Table/database not found | 404 |
| `SyntaxError` | KQL syntax error | 400 |
| `SemanticError` | Invalid column/function reference | 400 |
| `TimeoutError` | Query execution timeout | 504 |
| `RateLimitError` | Too many requests | 429 |
| `InternalError` | Unexpected server error | 500 |

### 5.2 Error Response Structure

```json
{
    "success": false,
    "error": "Human-readable error message",
    "error_type": "SemanticError",
    "error_code": "SEM0100",
    "suggestions": [
        "Suggestion 1",
        "Suggestion 2"
    ],
    "schema_context": {
        "available_columns": ["Col1", "Col2"],
        "similar_columns": ["ColA", "ColB"]
    },
    "recoverable": true,
    "retry_after_seconds": null
}
```

---

## 6. Configuration

### 6.1 Environment Variables

> **Authentication**: This server uses **Azure CLI authentication**. Run `az login` before starting the server. No service principal credentials are required.

| Variable | Description | Default |
|----------|-------------|--------|
| `KQL_QUERY_TIMEOUT` | Query timeout (seconds) | 300 |
| `KQL_MAX_ROWS` | Maximum rows returned | 10000 |
| `KQL_MEMORY_PATH` | Custom memory DB path | Auto |
| `KQL_LOG_LEVEL` | Logging level | INFO |
| `KQL_POOL_SIZE` | Connection pool size | 10 |

### 6.2 Constants

```python
from mcp_kql_server.constants import (
    SERVER_NAME,
    __version__,
    DEFAULT_QUERY_TIMEOUT,
    MAX_ROWS_DEFAULT,
    CONNECTION_CONFIG,
    ERROR_HANDLING_CONFIG
)
```

### 6.3 MCP Configuration (mcp.json)

```json
{
    "mcpServers": {
        "kql-server": {
            "command": "python",
            "args": ["-m", "mcp_kql_server"],
            "env": {
                "KQL_QUERY_TIMEOUT": "300",
                "KQL_LOG_LEVEL": "INFO"
            }
        }
    }
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.2.0 | Dec 2025 | Added performance module, connection pooling |
| 2.1.0 | Dec 2025 | Schema-only NL2KQL, version checker |
| 2.0.9 | Nov 2025 | CAG updates, SQLite migration |

---

For more details, see [Architecture Documentation](architecture.md) or [Troubleshooting Guide](troubleshooting.md).
