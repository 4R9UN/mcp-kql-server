# MCP KQL Server Architecture

**Version**: 2.2.0
**Author**: Arjun Trivedi
**Email**: arjuntrivedi42@yahoo.com
**Last Updated**: December 2025

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Core Components](#3-core-components)
4. [Data Flow Diagrams](#4-data-flow-diagrams)
5. [Performance Architecture](#5-performance-architecture)
6. [Memory & Caching Strategy](#6-memory--caching-strategy)
7. [Security Model](#7-security-model)
8. [Module Reference](#8-module-reference)

---

## 1. System Overview

The MCP KQL Server is an AI-augmented service for executing Kusto Query Language (KQL) queries against Azure Data Explorer. It leverages the Model Context Protocol (MCP) to expose capabilities as tools that AI models can consume.

### Key Features

| Feature | Description |
|---------|-------------|
| **Natural Language to KQL** | Convert natural language questions to KQL queries |
| **Schema Memory** | Intelligent caching with SQLite backend |
| **Connection Pooling** | Efficient client reuse and management |
| **Batch Execution** | Execute multiple queries in parallel |
| **Auto-Update** | Automatic version checking from PyPI |

### Design Principles

1. **Zero Configuration**: Works out of the box with Azure CLI authentication
2. **Schema-Driven**: All NL2KQL uses discovered schema, no hardcoded values
3. **Resilient**: Graceful error handling with actionable suggestions
4. **Performant**: Connection pooling, caching, and async operations

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MCP Client Environment                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Claude AI   │  │   VS Code    │  │  Custom App  │  │  Other MCP   │     │
│  │   Desktop    │  │   Copilot    │  │    Client    │  │   Clients    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────┘
          │                 │                 │                 │
          └─────────────────┴────────┬────────┴─────────────────┘
                                     │
                            MCP Protocol (stdio)
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MCP KQL Server (v2.2.0)                               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         API Layer (mcp_server.py)                       │ │
│  │  ┌─────────────────────────┐  ┌─────────────────────────────────────┐  │ │
│  │  │  execute_kql_query      │  │  schema_memory                      │  │ │
│  │  │  - Direct KQL execution │  │  - discover, list_tables, get_context│ │ │
│  │  │  - NL2KQL generation    │  │  - generate_report, clear_cache     │  │ │
│  │  └─────────────────────────┘  └─────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│  ┌──────────────────────────────────┼──────────────────────────────────────┐│
│  │                     Core Processing Layer                                ││
│  │                                                                          ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    ││
│  │  │ QueryProcessor│ │ ErrorHandler │ │SchemaManager │ │ KQLValidator │    ││
│  │  │  (utils.py)  │ │  (utils.py)  │ │  (utils.py)  │ │(kql_validator)│    ││
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘    ││
│  │                                                                          ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    ││
│  │  │execute_kql.py│ │ ai_prompts.py│ │ kql_auth.py  │ │version_checker│   ││
│  │  │(KQL Executor)│ │(AI Templates)│ │(Azure Auth)  │ │ (Updates)    │    ││
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘    ││
│  └──────────────────────────────────────────────────────────────────────────┘│
│                                     │                                        │
│  ┌──────────────────────────────────┼──────────────────────────────────────┐│
│  │                     Performance Layer (NEW in v2.2.0)                    ││
│  │                                                                          ││
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐ ││
│  │  │KustoConnectionPool │  │BatchQueryExecutor  │  │  SchemaPreloader   │ ││
│  │  │ - Client reuse     │  │ - Parallel queries │  │ - Startup loading  │ ││
│  │  │ - Health checks    │  │ - Progress tracking│  │ - Auto refresh     │ ││
│  │  │ - Auto recycling   │  │ - Error aggregation│  │                    │ ││
│  │  └────────────────────┘  └────────────────────┘  └────────────────────┘ ││
│  │                                                                          ││
│  │  ┌────────────────────┐  ┌────────────────────┐                         ││
│  │  │PerformanceMonitor  │  │   Async Utilities  │                         ││
│  │  │ - Metrics tracking │  │ - execute_query_async │                      ││
│  │  │ - P95 latencies    │  │ - execute_queries_async│                     ││
│  │  └────────────────────┘  └────────────────────┘                         ││
│  └──────────────────────────────────────────────────────────────────────────┘│
│                                     │                                        │
│  ┌──────────────────────────────────┼──────────────────────────────────────┐│
│  │                       Data Layer (memory.py)                             ││
│  │                                                                          ││
│  │  ┌────────────────────────────────────────────────────────────────────┐ ││
│  │  │                    MemoryManager (SQLite Backend)                   │ ││
│  │  │                                                                     │ ││
│  │  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │ ││
│  │  │  │ Schema Cache  │  │ Query History │  │ AI Embeddings │           │ ││
│  │  │  │  - Tables     │  │  - Successful │  │ - Semantic    │           │ ││
│  │  │  │  - Columns    │  │  - Patterns   │  │   Search      │           │ ││
│  │  │  └───────────────┘  └───────────────┘  └───────────────┘           │ ││
│  │  └────────────────────────────────────────────────────────────────────┘ ││
│  │                                                                          ││
│  │  ┌───────────────────────────────────────────────────────────────────┐  ││
│  │  │                 SemanticSearch (sentence-transformers)             │  ││
│  │  │  - Table similarity matching                                       │  ││
│  │  │  - Context-aware table selection                                   │  ││
│  │  └───────────────────────────────────────────────────────────────────┘  ││
│  └──────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                             Azure SDK for Python
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Azure Data Explorer (Kusto)                            │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐           │
│  │    Cluster A     │  │    Cluster B     │  │    Cluster N     │           │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐  │           │
│  │  │ Database 1 │  │  │  │ Database 1 │  │  │  │ Database 1 │  │           │
│  │  │ Database 2 │  │  │  │ Database 2 │  │  │  │ Database 2 │  │           │
│  │  └────────────┘  │  │  └────────────┘  │  │  └────────────┘  │           │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 Module Responsibilities

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `mcp_server.py` | MCP tool registration & routing | `execute_kql_query()`, `schema_memory()` |
| `execute_kql.py` | KQL execution engine | `kql_execute_tool()`, `_get_kusto_client()` |
| `memory.py` | Persistent schema storage | `MemoryManager`, `SemanticSearch` |
| `utils.py` | Utilities & helpers | `QueryProcessor`, `ErrorHandler`, `SchemaManager` |
| `kql_auth.py` | Azure authentication | `authenticate_kusto()` |
| `kql_validator.py` | Query validation | `KQLValidator` |
| `ai_prompts.py` | NL2KQL prompt templates | `build_nl2kql_prompt()` |
| `constants.py` | Configuration & analyzers | `DynamicSchemaAnalyzer`, `DynamicColumnAnalyzer` |
| `performance.py` | Performance optimization | `KustoConnectionPool`, `BatchQueryExecutor` |
| `version_checker.py` | Auto-update detection | `check_for_updates()`, `startup_version_check()` |

### 3.2 Component Interaction

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Component Dependencies                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  mcp_server.py ──────┬────► execute_kql.py ────► Azure Kusto        │
│        │             │            │                                  │
│        │             │            ▼                                  │
│        │             ├────► memory.py ◄──── utils.py                │
│        │             │            │              │                   │
│        │             │            ▼              ▼                   │
│        │             │      SQLite DB      kql_validator.py          │
│        │             │                                               │
│        │             └────► kql_auth.py ────► Azure Identity         │
│        │                                                             │
│        └────► ai_prompts.py ────► constants.py                       │
│                                                                      │
│        ▼                                                             │
│  performance.py (Connection Pooling, Batch Execution)                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow Diagrams

### 4.1 Query Execution Flow

```mermaid
sequenceDiagram
    participant Client as MCP Client
    participant Server as mcp_server.py
    participant Pool as ConnectionPool
    participant Auth as kql_auth.py
    participant Processor as QueryProcessor
    participant Memory as MemoryManager
    participant Executor as execute_kql.py
    participant Azure as Azure Data Explorer
    participant Error as ErrorHandler

    Client->>Server: execute_kql_query(query, cluster, db)
    
    Note over Server: Phase 1: Authentication
    Server->>Pool: get_client(cluster)
    Pool->>Pool: Check pool for existing connection
    alt Connection exists
        Pool->>Server: Return cached client
    else No connection
        Pool->>Auth: Create new client
        Auth->>Auth: Azure CLI authentication
        Auth->>Pool: Return new client
        Pool->>Pool: Add to pool
        Pool->>Server: Return new client
    end
    
    Note over Server: Phase 2: Query Processing
    Server->>Processor: process_query(query)
    Processor->>Processor: Clean, validate, extract params
    Processor->>Server: Return processed query
    
    Note over Server: Phase 3: Schema Context
    Server->>Memory: get_relevant_context(cluster, db, query)
    alt Schema cached
        Memory->>Server: Return cached schema
    else Not cached
        Memory->>Azure: Discover schema
        Azure->>Memory: Schema metadata
        Memory->>Memory: Cache schema
        Memory->>Server: Return schema
    end
    
    Note over Server: Phase 4: Execution
    Server->>Executor: execute_query(query, client, context)
    Executor->>Azure: Execute KQL
    
    alt Success
        Azure->>Executor: Query results
        Executor->>Memory: Update query history (async)
        Executor->>Server: Return results
        Server->>Client: Return success response
    else Failure
        Azure->>Executor: KustoServiceError
        Executor->>Error: Handle error
        Error->>Error: Classify & suggest fix
        Error->>Server: Structured error
        Server->>Client: Return error with suggestions
    end
```

### 4.2 NL2KQL Flow

```mermaid
flowchart TD
    A[User Natural Language Query] --> B{Has Schema Context?}
    
    B -->|No| C[Discover Schema]
    C --> D[Cache Schema in Memory]
    D --> E[Find Relevant Tables]
    
    B -->|Yes| E
    
    E --> F[Build NL2KQL Prompt]
    F --> G[Include Schema Context]
    G --> H[LLM Generates KQL]
    
    H --> I{Validate Generated KQL}
    I -->|Valid| J[Execute Query]
    I -->|Invalid| K[Return Validation Error]
    
    J --> L{Query Success?}
    L -->|Yes| M[Return Results]
    L -->|No| N[Enhanced Error + Schema Suggestions]
    
    style A fill:#0b102a,stroke:#1a9fd6,stroke-width:3px,color:#d5efff
    style H fill:#141a34,stroke:#ff8244,stroke-width:2px,color:#ffd8ba
    style M fill:#1a1f3b,stroke:#26ffa4,stroke-width:2px,color:#ccfff3
    style K fill:#1a1f3b,stroke:#ff5c8c,stroke-width:2px,color:#ffe0ec
    style N fill:#1a1f3b,stroke:#ff5c8c,stroke-width:2px,color:#ffe0ec
```

### 4.3 Schema Memory Operations

```mermaid
flowchart LR
    subgraph Operations
        A[discover] --> DB[(SQLite)]
        B[list_tables] --> DB
        C[get_context] --> DB
        D[clear_cache] --> DB
        E[get_stats] --> DB
        F[refresh_schema] --> DB
        G[generate_report] --> DB
    end
    
    subgraph Storage
        DB --> T[schemas table]
        DB --> Q[queries table]
        DB --> E2[embeddings table]
    end
    
    subgraph Output
        T --> J[JSON Schema]
        Q --> H[Query History]
        E2 --> S[Semantic Search]
    end
    
    style DB fill:#141a34,stroke:#1a9fd6,stroke-width:2px,color:#d5efff
```

---

## 5. Performance Architecture

### 5.1 Connection Pooling

```
┌─────────────────────────────────────────────────────────────────┐
│                    KustoConnectionPool                           │
│                                                                  │
│  Configuration:                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ max_size: 20          │ idle_timeout: 30 min               │ │
│  │ min_size: 5           │ max_connection_age: 60 min         │ │
│  │ connection_timeout: 30s│ health_check_interval: 60s        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pool Structure:                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ cluster_url_1:                                              │ │
│  │   ├── conn_0: {client, created, last_used, healthy}        │ │
│  │   ├── conn_1: {client, created, last_used, healthy}        │ │
│  │   └── conn_2: {client, created, last_used, healthy}        │ │
│  │                                                             │ │
│  │ cluster_url_2:                                              │ │
│  │   ├── conn_0: {client, created, last_used, healthy}        │ │
│  │   └── conn_1: {client, created, last_used, healthy}        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Statistics:                                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ total_requests: 1500  │ cache_hits: 1400    │ hit_rate: 93% │ │
│  │ connections_created: 5│ connections_recycled: 2            │ │
│  │ average_wait_time: 2.3ms                                   │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Batch Query Execution

```mermaid
flowchart TD
    A[Batch Query Request] --> B[BatchQueryExecutor]
    B --> C[ThreadPoolExecutor]
    
    C --> D1[Worker 1: Query A]
    C --> D2[Worker 2: Query B]
    C --> D3[Worker 3: Query C]
    C --> D4[Worker N: Query N]
    
    D1 --> E[Aggregate Results]
    D2 --> E
    D3 --> E
    D4 --> E
    
    E --> F[Return BatchQueryResults]
    
    style A fill:#0b102a,stroke:#1a9fd6,stroke-width:2px,color:#d5efff
    style C fill:#141a34,stroke:#ff8244,stroke-width:2px,color:#ffd8ba
    style F fill:#1a1f3b,stroke:#26ffa4,stroke-width:2px,color:#ccfff3
```

### 5.3 Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Query Latency (P50) | Median query execution time | < 500ms |
| Query Latency (P95) | 95th percentile latency | < 2s |
| Connection Pool Hit Rate | Reused vs new connections | > 90% |
| Schema Cache Hit Rate | Cached vs discovered schemas | > 95% |
| Error Rate | Failed queries / total queries | < 1% |

---

## 6. Memory & Caching Strategy

### 6.1 SQLite Schema

```sql
-- Schema storage
CREATE TABLE schemas (
    id INTEGER PRIMARY KEY,
    cluster TEXT NOT NULL,
    database TEXT NOT NULL,
    table_name TEXT NOT NULL,
    columns_json TEXT,
    ai_context TEXT,
    discovered_at TIMESTAMP,
    updated_at TIMESTAMP,
    UNIQUE(cluster, database, table_name)
);

-- Query history
CREATE TABLE queries (
    id INTEGER PRIMARY KEY,
    cluster TEXT,
    database TEXT,
    query_text TEXT,
    success BOOLEAN,
    row_count INTEGER,
    execution_time_ms REAL,
    executed_at TIMESTAMP
);

-- Semantic embeddings
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    table_name TEXT,
    embedding BLOB,
    model_version TEXT,
    created_at TIMESTAMP
);
```

### 6.2 Caching Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                       Caching Layers                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: In-Memory (fastest)                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ - Connection pool clients                                    ││
│  │ - Recent schema lookups (5 min TTL)                          ││
│  │ - Semantic search results                                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                          ▼                                       │
│  Layer 2: SQLite (persistent)                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ - All discovered schemas                                     ││
│  │ - Query history                                              ││
│  │ - AI embeddings                                              ││
│  │ - Location: ~/.local/share/KQL_MCP/kql_memory.db             ││
│  └─────────────────────────────────────────────────────────────┘│
│                          ▼                                       │
│  Layer 3: Azure (source of truth)                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ - Live schema discovery via .show commands                   ││
│  │ - Query execution                                            ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Security Model

### 7.1 Authentication Flow

```mermaid
flowchart TD
    A[Start Authentication] --> B{Azure CLI Logged In?}
    
    B -->|Yes| C[Use CLI Token]
    B -->|No| D[Initiate Device Code Flow]
    
    D --> E[Display Device Code URL]
    E --> F[User Authenticates in Browser]
    F --> G[Receive Token]
    
    C --> H[Create KustoClient]
    G --> H
    
    H --> I[Cache Client in Pool]
    I --> J[Ready for Queries]
    
    style A fill:#0b102a,stroke:#1a9fd6,stroke-width:2px,color:#d5efff
    style J fill:#1a1f3b,stroke:#26ffa4,stroke-width:2px,color:#ccfff3
```

### 7.2 Security Principles

| Principle | Implementation |
|-----------|----------------|
| **No Credential Storage** | Uses Azure CLI tokens, never stores passwords |
| **Encryption in Transit** | All connections use TLS 1.2+ |
| **Local Cache Protection** | SQLite in user profile directory |
| **Query Logging** | Sanitized logs without sensitive data |
| **Timeout Protection** | Configurable query timeouts prevent runaway queries |

---

## 8. Module Reference

### File Structure

```
mcp_kql_server/
├── __init__.py          # Package initialization, exports
├── __main__.py          # Entry point for python -m
├── mcp_server.py        # MCP tool definitions
├── execute_kql.py       # Query execution engine
├── memory.py            # SQLite-backed memory manager
├── utils.py             # Utilities (QueryProcessor, ErrorHandler)
├── kql_auth.py          # Azure authentication
├── kql_validator.py     # KQL syntax validation
├── ai_prompts.py        # NL2KQL prompt templates
├── constants.py         # Configuration, analyzers
├── performance.py       # Connection pooling, batch execution
├── version_checker.py   # Auto-update functionality
└── py.typed             # PEP 561 marker
```

### Key Exports

```python
# From mcp_kql_server
from mcp_kql_server import (
    # Core
    __version__,
    SERVER_NAME,
    
    # Tools
    execute_kql_query,
    schema_memory,
    
    # Memory
    get_memory_manager,
    MemoryManager,
    
    # Performance
    get_connection_pool,
    BatchQueryExecutor,
    
    # Version
    check_for_updates,
    get_current_version,
)
```

---

## Conclusion

The MCP KQL Server v2.2.0 architecture provides a robust, performant, and secure foundation for AI-augmented KQL query execution. The addition of connection pooling, batch execution, and performance monitoring in this version significantly improves throughput and resource utilization.

For questions or contributions, please refer to [CONTRIBUTING.md](../CONTRIBUTING.md).
