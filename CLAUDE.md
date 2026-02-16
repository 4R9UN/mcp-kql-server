# MCP KQL Server - Deep Dive Analysis

## What Is This Project?

MCP KQL Server (v2.1.0) is an AI-powered Model Context Protocol server for executing Kusto Query Language (KQL) queries against Azure Data Explorer. It goes well beyond a basic "pass-through" KQL MCP server by adding an intelligent memory layer, semantic search, schema validation, natural language to KQL conversion, and a learning loop that improves over time.

**Author:** Arjun Trivedi
**License:** MIT
**Python:** 3.10+
**Transport:** STDIO (via FastMCP)

---

## Architecture Overview

```
MCP Client (Claude Desktop / VSCode / Custom)
    |  (STDIO, MCP Protocol)
    v
FastMCP Server Framework
    |
    +-- execute_kql_query   (main query tool)
    +-- schema_memory       (schema discovery tool)
    |
    +-- KQLValidator        (pre-execution query validation)
    +-- SchemaManager       (live schema discovery, multi-strategy)
    +-- MemoryManager       (PostgreSQL + pgvector + CAG)
    +-- SemanticSearch      (sentence-transformers, all-MiniLM-L6-v2)
    |
    v
Azure Data Explorer (Kusto)
    (via azure-kusto-data + AzureCliCredential)
```

---

## Project Structure

| File | Purpose |
|---|---|
| `mcp_server.py` | FastMCP server, defines the 2 MCP tools, NL2KQL generation, report generation |
| `memory.py` | PostgreSQL + pgvector memory system with semantic search, CAG, TOON formatting |
| `utils.py` | SchemaManager (live discovery with 3 strategies), SchemaDiscovery, ErrorHandler, retry logic, query entity extraction |
| `execute_kql.py` | Core KQL execution against Kusto, background learning loop, client caching |
| `kql_validator.py` | Pre-execution query validation (tables, columns, operator syntax) |
| `kql_auth.py` | Azure CLI authentication with retry and device-code fallback |
| `ai_prompts.py` | System prompts, few-shot examples, prompt builders for KQL generation |
| `performance.py` | Connection pooling (singleton), batch execution, schema preloading, health checks |
| `constants.py` | All configuration: reserved words, operator syntax rules, special tokens, network config, error patterns |
| `version_checker.py` | PyPI version checking at startup |
| `__init__.py` | Package initialization (UTF-8, logging, Azure SDK suppression, memory dir setup) |
| `__main__.py` | CLI entry point |

### Supporting Files

| File/Directory | Purpose |
|---|---|
| `pyproject.toml` | Modern Python package configuration (name: mcp-kql-server, >=3.10) |
| `server.json` | MCP server metadata and registry info |
| `requirements.txt` | Python dependencies |
| `uv.lock` | UV package manager lock file |
| `tests/` | Comprehensive test suite (constants, execution, auth, server, memory, utils, package) |
| `docs/` | API reference, architecture docs, troubleshooting |
| `Example/` | Usage examples |
| `deployment/` | Deployment resources |
| `.github/` | CI/CD workflows, version bump automation |

---

## The Two MCP Tools

### 1. `execute_kql_query`

**Location:** `mcp_server.py:50-300`

The primary tool. Accepts either a raw KQL query or a natural language description. Its execution flow:

1. **Auth check** - Verifies startup authentication succeeded
2. **NL2KQL generation** (if `generate_query=True`) - Converts natural language to KQL using schema memory
3. **Result cache check** - SHA256 hash lookup with 2-minute TTL (`memory.py:377-403`)
4. **Pre-execution validation** - Validates tables, columns, and operator syntax against cached schemas (`kql_validator.py:34-121`)
5. **Query execution** - Sends query to Kusto via `azure-kusto-data` with retry logic (`execute_kql.py:143-197`)
6. **Error recovery** - Context-aware error handling: SEM0100 triggers schema refresh, database/table errors produce targeted suggestions
7. **Result formatting** - DataFrame to JSON with special tokens for LLM semantic parsing
8. **Background learning** - Asynchronously stores successful queries and triggers schema discovery for newly-seen tables (`execute_kql.py:316-383`)

### 2. `schema_memory`

**Location:** `mcp_server.py:463-566`

The schema management and intelligence tool. Operations:

| Operation | What It Does |
|---|---|
| `list_tables` | Lists all tables in a database via `.show tables` |
| `discover` | Discovers and caches a table's schema (columns, types, sample data) |
| `get_context` | Returns AI context based on a natural language query (schema + similar queries + join hints) |
| `refresh_schema` | Iterates all tables in a database and refreshes their schemas |
| `get_stats` | Returns memory statistics (schema count, query count, DB size) |
| `clear_cache` | Clears schema caches |
| `generate_report` | Produces a markdown analysis report with Mermaid diagrams |

---

## Key Subsystems

### PostgreSQL + pgvector Memory System (`memory.py`)

The memory manager stores everything in PostgreSQL with pgvector for native vector similarity search. Tables are prefixed with `kql_mcp_` to avoid collision with other applications sharing the same database. Connection is managed via `psycopg2.pool.ThreadedConnectionPool`. Configuration is read from environment variables (`DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_DATABASE`) via `POSTGRES_CONFIG` in `constants.py`.

**5 Tables (all prefixed `kql_mcp_`):**

| Table | Purpose |
|---|---|
| `kql_mcp_schemas` | Table definitions with column JSON and `vector(384)` embeddings, composite PK (cluster, database, table_name) |
| `kql_mcp_queries` | Successful queries with `vector(384)` embeddings for similarity search |
| `kql_mcp_join_hints` | Discovered table relationships (table1 joins table2 on condition) |
| `kql_mcp_query_cache` | Result caching with SHA256 hash keys and TTL via SQL interval |
| `kql_mcp_learning_events` | Execution learning data for analytics |

**Semantic Search** (`memory.py:60-115`): Uses `sentence-transformers` with the `all-MiniLM-L6-v2` model (singleton, lazy-loaded, preloaded in background thread). Generates 384-dimensional embeddings stored as `vector(384)` columns in PostgreSQL. Similarity search uses pgvector's `<=>` cosine distance operator with HNSW indexes for fast server-side ranking.

**CAG (Context Augmented Generation)** (`memory.py:561-634`): Builds compact context strings in TOON (Token-Oriented Object Notation) format combining schemas, similar queries, and join hints. TOON compresses data types (`string` -> `s`, `datetime` -> `dt`, etc.) to minimize token usage.

**In-memory schema cache** (`memory.py:534-559`): 5-minute TTL dictionary cache on top of PostgreSQL to avoid repeated DB reads.

**Graceful degradation**: If PostgreSQL is unavailable, the server starts with `_db_available = False` and all memory operations return empty results. Query execution still works against Kusto.

### Schema Discovery (`utils.py:275-766`)

The `SchemaManager` class is the single source of truth for live schema discovery. It uses a **3-strategy fallback approach**:

1. **Strategy 1** - `.show table <name> schema as json` (most detailed, includes ordinals and CSL types)
2. **Strategy 2** - `<table> | getschema` (backup, works on more table types)
3. **Strategy 3** - `<table> | take 2` then infer schema from sample data (last resort)

Each strategy also fetches sample data (`| take 2`) for enhanced column descriptions. After discovery, the schema is stored in PostgreSQL via `MemoryManager.store_schema()` with a `vector(384)` embedding generated from the table name and column names.

**Column enrichment** (`utils.py:907-1120`): Each discovered column gets:
- A semantic description generated from data patterns (not hardcoded keywords)
- Tags based on data type analysis (DATETIME, NUMERIC, TEXT, etc.)
- Sample values (up to 3 unique values)
- An AI token string for LLM consumption

**Connection validation** (`utils.py:300-495`): Before executing discovery queries, the manager can validate Azure auth, test TCP connectivity, and verify cluster access with `.show version`.

### Pre-Execution Validation (`kql_validator.py`)

Validates queries before they reach Kusto to catch errors early:

1. **Table extraction** (`kql_validator.py:123-147`): 3 regex patterns for tables at start/pipe boundaries, join/union/lookup operations, and bracketed names.

2. **Auto-discovery** (`kql_validator.py:153-174`): If a table lacks a cached schema, triggers live discovery before validating.

3. **Column validation** (`kql_validator.py:176-312`): Extracts column references from `where`, `summarize by`, aggregation functions, and `join on` clauses. Distinguishes between columns read from tables vs aliases created by the query (extend, summarize aliases, project renames). Only table-source columns are validated.

4. **Operator syntax** (`kql_validator.py:360-377`): Detects common KQL mistakes like `! =` (should be `!=`), `! contains` (should be `!contains`), and `!has_any` (doesn't exist in KQL).

### NL2KQL Generation (`mcp_server.py:302-460`)

The natural language to KQL pipeline uses **only schema-validated data** - no hardcoded table or column names:

1. Find relevant tables via semantic search (embedding similarity)
2. Determine target table: explicit param > semantic match > regex extraction from NL
3. Fetch schema from memory (not hardcoded)
4. Extract candidate columns from the NL query text
5. Filter out KQL reserved words
6. Also extract columns from similar past queries (high confidence matches)
7. Match candidates against schema columns (case-insensitive)
8. Build query with only validated columns, or fall back to first N schema columns
9. Return structured response with special tokens

### Connection Pooling (`performance.py:106-351`)

`KustoConnectionPool` is a thread-safe singleton that manages connections per cluster URL:

- OrderedDict for FIFO ordering
- Automatic recycling of expired connections (1-hour max age)
- Cleanup of idle connections (30-minute timeout)
- Health checks before reuse
- Statistics tracking (hit rate, wait times, created/recycled counts)
- Configurable pool sizes (default 5 min, 20 max per cluster)

Also provides `BatchQueryExecutor` for parallel query execution via `ThreadPoolExecutor`, and `SchemaPreloader` for background schema loading at startup.

### Authentication (`kql_auth.py`)

- Uses Azure CLI (`az account get-access-token`) with LRU-cached result
- Falls back to `az login --use-device-code` if not authenticated
- Retry logic via `tenacity` (exponential backoff, max 3 attempts for check, 2 for login)
- Runs once at server startup; subsequent operations use `AzureCliCredential`

### Special Tokens (`constants.py`)

The server wraps metadata in structured tokens to help LLMs parse responses semantically:

```
<CLUSTER>cluster_url</CLUSTER>
<DATABASE>db_name</DATABASE>
<TABLE>table_name</TABLE>
##COLUMN##:column_name
>>TYPE<<:data_type
<QUERY>kql_query</QUERY>
<RESULT>summary</RESULT>
<AI_NOTE>guidance</AI_NOTE>
```

---

## Background Learning Loop

After every successful query execution (`execute_kql.py:316-383`):

1. Extract table names from the executed query
2. Store the query + description + embedding in the `queries` table
3. For each table referenced, check if a schema exists in memory
4. If not, trigger async schema discovery (non-blocking)
5. Store learning events for analytics

This means the server gets smarter over time without explicit user action. The more queries run, the richer the memory becomes - more schemas cached, more query patterns available for few-shot learning, more join hints discovered.

---

## Advantages Over a Simple Kusto MCP Server

A "simple" Kusto MCP server would typically just proxy KQL queries to Azure Data Explorer and return results. Here's what this project adds:

### 1. Schema Memory + Semantic Search

A simple server has no memory between queries. This server persists schemas, queries, and join hints in PostgreSQL with pgvector embeddings. When a user asks about data, it can semantically match their intent to the right tables without the user knowing exact table names.

### 2. Pre-Execution Validation

A simple server sends every query to Kusto and lets it fail. This server catches column typos, invalid operator syntax (`! =` vs `!=`), and missing tables *before* making a network call. This saves round-trip time and provides better error messages with suggestions.

### 3. Natural Language to KQL

A simple server requires the user to write KQL. This server can take natural language like "find failed login attempts" and generate a schema-validated KQL query. Critically, it only uses columns that actually exist in the table schema, preventing hallucinated column names.

### 4. Context Augmented Generation (CAG)

When generating queries, the server builds compact context that includes the full table schema, similar past queries (few-shot examples), and join hints. This CAG context dramatically improves query generation accuracy compared to generating from scratch.

### 5. Background Learning

A simple server is stateless. This server learns from every successful query execution: it discovers new schemas, stores query patterns, and builds a knowledge base that improves future interactions. First-time queries discover and cache schemas; subsequent queries benefit from the cache.

### 6. Multi-Strategy Schema Discovery

A simple server might use one method to get a table schema. This server tries 3 strategies in sequence (JSON schema, getschema, sample inference), with sample data extraction for enriched column metadata. Each column gets semantic descriptions, tags, and AI tokens.

### 7. Connection Pooling + Health Checks

A simple server creates a new connection per query. This server maintains a thread-safe connection pool with automatic recycling, idle cleanup, health checks, and hit-rate statistics. This reduces connection overhead significantly for frequent queries.

### 8. Structured Output for LLMs

A simple server returns raw data. This server wraps responses in special tokens (`<CLUSTER>`, `<TABLE>`, `##COLUMN##`) that help LLMs parse and understand the structure of results, schemas, and errors. This enables more reliable tool-use chains.

### 9. Error Recovery

A simple server returns raw Kusto errors. This server classifies errors (SEM0100, database not found, unknown table) and generates context-aware suggestions. For column errors, it automatically refreshes the table schema to fix stale caches.

### 10. Result Caching

A simple server re-executes every query. This server caches results with SHA256 hashing and a 2-minute TTL, returning instant results for repeated queries.

### 11. Query Reports + Visualizations

The `generate_report` operation produces markdown analytics reports with Mermaid diagrams (data flow, schema relationships, timeline), executive summaries, and actionable recommendations.

### 12. TOON Format for Token Efficiency

Schema context is compressed using TOON (Token-Oriented Object Notation) where `string` becomes `s`, `datetime` becomes `dt`, etc. This minimizes token usage when loading schemas into LLM context windows.

---

## Execution Flow Diagram

```
User Query (NL or KQL)
        |
        v
[Auth Check] ----fail----> "Run az login"
        |
        v (pass)
[NL2KQL?] ----yes----> [Semantic Table Search]
        |                       |
        |               [Schema Lookup]
        |                       |
        |               [Column Validation]
        |                       |
        |               [Generate KQL]
        |                       |
        v                       v
[Cache Check] ---hit---> Return Cached Result
        |
        v (miss)
[Pre-Execution Validation]
   |         |
   |     [Table Extraction]
   |     [Column Validation vs Schema]
   |     [Operator Syntax Check]
   |         |
   v         v
[Valid?] ---no----> Return Errors + Suggestions
   |
   v (yes)
[Execute on Kusto] (with retry, client pool)
   |         |
   |     [Error?] ---yes---> [Classify Error]
   |                              |
   |                         [SEM0100?] -> Refresh Schema
   |                              |
   |                         Return Error + Suggestions
   v
[Format Results] (DataFrame -> JSON + Special Tokens)
   |
   v
[Cache Result] (SHA256 hash, 2min TTL)
   |
   v
[Background Learning] (async, non-blocking)
   |-- Store successful query + embedding
   |-- Discover schemas for new tables
   |-- Store learning events
   |
   v
Return Results to Client
```

---

## Configuration Highlights

| Setting | Value | Location |
|---|---|---|
| Query timeout | 600s | `constants.py` |
| Connection timeout | 30s | `performance.py:33` |
| Result cache TTL | 120s | `mcp_server.py:125` |
| Schema memory cache TTL | 300s | `memory.py:438` |
| Connection pool max | 20 per cluster | `performance.py:31` |
| Connection max age | 3600s (1hr) | `performance.py:35` |
| Idle connection timeout | 1800s (30min) | `performance.py:34` |
| Retry attempts | 3 (queries), 5 (connections) | `utils.py:45`, `constants.py` |
| Embedding model | all-MiniLM-L6-v2 | `memory.py:54` |
| Max columns in NL2KQL | 12 projected, 20 displayed | `mcp_server.py:409,451` |
| Max tables in CAG context | 20 | `memory.py:454` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastmcp` | MCP server framework |
| `azure-kusto-data` | Kusto client SDK |
| `azure-identity` | Azure CLI credential |
| `azure-cli` | Authentication backend |
| `psycopg2-binary` | PostgreSQL adapter for Python |
| `pgvector` | pgvector support for psycopg2 (vector type registration) |
| `sentence-transformers` | Embedding generation for semantic search |
| `scikit-learn` | ML utilities |
| `numpy` | Vector operations for similarity |
| `pandas` | DataFrame handling for query results |
| `tenacity` | Retry logic with exponential backoff |
| `httpx` / `requests` | HTTP clients |
| `pydantic` | Data validation |

---

## Summary

This is not a simple Kusto proxy. It is an intelligent data access layer that:

- **Remembers** schemas and query patterns in PostgreSQL with pgvector embeddings
- **Validates** queries against real schemas before execution
- **Generates** KQL from natural language using only schema-validated columns
- **Learns** from every query to improve future interactions
- **Recovers** from errors with targeted suggestions and automatic schema refresh
- **Optimizes** with connection pooling, result caching, and token-efficient formatting

The core design philosophy is: make the LLM-to-Kusto pipeline as reliable and intelligent as possible, so that users can query Azure Data Explorer effectively even without deep KQL expertise.
