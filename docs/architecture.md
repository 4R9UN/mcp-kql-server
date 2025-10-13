# MCP KQL Server Architecture

**Version**: 2.0.7
**Author**: Arjun Trivedi
**Email**: arjuntrivedi42@yahoo.com

## 1. System Overview

The MCP KQL Server is an intelligent, AI-augmented service designed to execute Kusto Query Language (KQL) queries against Azure Data Explorer. It leverages the Model Context Protocol (MCP) to expose its capabilities as tools that AI models can consume. The server's core mission is to provide a seamless, zero-configuration experience for developers and AI agents, enhanced by intelligent schema caching, robust error handling, and a sophisticated query processing pipeline.

This document details the server's architecture, data flow, and key components, reflecting the significant refactoring in version 2.0.7.

## 2. High-Level Architecture

The architecture is designed around a central processing pipeline that validates, enriches, executes, and learns from every KQL query.

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Client Environment                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Claude AI     │  │   Custom App    │  │   VS Code       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ MCP Protocol (Tools: execute_kql_query, schema_memory)
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MCP KQL Server (v2.0.7)                      │
│                                                                 │
│ ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐ │
│ │  `mcp_server`   │   │  `kql_auth`     │   │   `memory`      │ │
│ │  (Tool Handler) │   │  (Azure Auth)   │   │ (Unified Cache) │ │
│ └───────┬─────────┘   └────────┬────────┘   └────────┬────────┘ │
│         │                      │                     │          │
│         └───────────┬──────────┴─────────────────────┘          │
│                     │                                           │
│                     ▼                                           │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                 Core Processing Pipeline                    │ │
│ │                                                             │ │
│ │ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │ │
│ │ │`QueryProcessor`│ │`ErrorHandler`│ │`SchemaManager` │          │ │
│ │ │  (utils.py)  │ │  (utils.py)  │ │  (utils.py)  │          │ │
│ │ └───────┬──────┘ └───────┬──────┘ └───────┬──────┘          │ │
│ │         │                │                │                 │ │
│ │         └────────────────┼────────────────┴─────────────────┘ │
│ │                          │                                    │
│ │                          ▼                                    │
│ │ ┌───────────────────────────────────────────────────────────┐ │
│ │ │                  `execute_kql.py`                         │ │
│ │ │                (Low-Level Executor)                       │ │
│ │ └───────────────────────────────────────────────────────────┘ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ Azure SDK for Python
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Azure Data Explorer (Kusto)                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Cluster 1     │  │   Cluster 2     │  │   Cluster N     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Core Components & Logic Flow

The server's logic is modularized into several key Python files, each with a distinct responsibility.

### 3.1. `mcp_server.py` - The MCP Entrypoint
- **Purpose**: Acts as the main entrypoint and tool handler for the MCP server.
- **Responsibilities**:
    - Initializes the `FastMCP` server.
    - Registers and exposes the two primary tools: `execute_kql_query` and `schema_memory`.
    - Handles incoming MCP requests and routes them to the appropriate tool.
    - Orchestrates the high-level interaction between authentication, memory, and the core query pipeline.

### 3.2. `kql_auth.py` - Authentication Manager
- **Purpose**: Manages all aspects of Azure authentication.
- **Responsibilities**:
    - **Azure CLI Integration**: Checks for an active Azure CLI login session.
    - **Device Code Flow**: If not logged in, it automatically triggers a device code authentication flow, guiding the user to log in.
    - **Token Caching**: Caches the authenticated client to minimize repeated authentication checks.
    - **Client Provision**: Provides a ready-to-use `KustoTrustedClient` for query execution.

### 3.3. `memory.py` - The Intelligence Layer
- **Purpose**: Provides the server's "brain" through an intelligent, persistent caching system.
- **Key Class**: `MemoryManager`.
- **Responsibilities**:
    - **Unified Memory**: Manages a single `unified_memory.json` file to store discovered schemas and successful query history.
    - **AI-Friendly Schema**: Stores schemas in a format optimized for AI consumption, including table descriptions, key columns, and common usage patterns.
    - **Dynamic Schema Analysis**: Uses `DynamicSchemaAnalyzer` and `DynamicColumnAnalyzer` (from `constants.py`) to generate rich, semantic context for tables and columns, moving beyond simple keyword matching.
    - **Persistence**: Ensures that learned schemas and query history are persisted across server restarts.

### 3.4. `utils.py` - The Central Processing Pipeline
This module, introduced in v2.0.6, centralizes the core business logic into a set of cohesive helper classes.

- **`QueryProcessor`**:
    - **Purpose**: Standardizes the pre-execution query pipeline.
    - **Responsibilities**:
        - **Query Cleaning**: Removes extraneous characters, comments, and formatting.
        - **Cluster & Database Parsing**: Intelligently extracts the target cluster and database from the query string.
        - **Validation**: Performs syntax and semantic checks before execution.

- **`ErrorHandler`**:
    - **Purpose**: Provides a centralized and structured error handling mechanism.
    - **Responsibilities**:
        - **Error Classification**: Catches exceptions, particularly `KustoServiceError`, and classifies them into meaningful categories (e.g., `Authentication`, `Syntax`, `NotFound`).
        - **Structured Responses**: Formats error messages into a consistent JSON structure.
        - **Actionable Suggestions**: Provides clear, user-friendly suggestions for how to resolve the error.

- **`SchemaManager`**:
    - **Purpose**: A utility class to assist with schema-related operations.
    - **Responsibilities**:
        - Provides helper functions for formatting and presenting schema information.

### 3.5. `execute_kql.py` - The Low-Level Executor
- **Purpose**: Handles the direct interaction with the Azure Kusto SDK.
- **Responsibilities**:
    - **Query Execution**: Takes a cleaned query and an authenticated client and executes it against the target Azure Data Explorer cluster.
    - **Result Formatting**: Processes the raw results from Azure into a structured format (columns, rows).
    - **Background Learning**: Spawns an asynchronous background task (`post_query_learning`) after a successful query to update the `MemoryManager` with the newly executed query and any discovered schema information.

### 3.6. `constants.py` - Configuration & Dynamic Intelligence
- **Purpose**: Centralizes all static configuration, feature descriptions, and the logic for dynamic schema analysis.
- **Key Classes**:
    - **`DynamicSchemaAnalyzer`**: Analyzes table names and properties to infer their purpose and generate AI-friendly descriptions.
    - **`DynamicColumnAnalyzer`**: Analyzes column names and data types to infer their meaning, potential use cases, and relationships.
- **Responsibilities**:
    - Provides a single source of truth for configuration values.
    - Encapsulates the "intelligence" used by `memory.py` to enrich raw schemas.

## 4. Data Flow: `execute_kql_query` Tool

The primary workflow is initiated when the `execute_kql_query` tool is called.

```mermaid
sequenceDiagram
    participant Client as MCP Client
    participant Server as MCP Server (`mcp_server.py`)
    participant Auth as Auth Manager (`kql_auth.py`)
    participant Processor as Query Processor (`utils.py`)
    participant Memory as Memory Manager (`memory.py`)
    participant Executor as KQL Executor (`execute_kql.py`)
    participant Azure as Azure Data Explorer
    participant ErrorHandler as Error Handler (`utils.py`)

    Client->>Server: execute_kql_query(query)

    Note over Server: Phase 1: Authentication
    Server->>Auth: Get authenticated client
    alt Not Authenticated
        Auth->>Client: Initiate Device Code Flow
        Client-->>Auth: User authenticates
    end
    Auth->>Server: Return KustoTrustedClient

    Note over Server: Phase 2: Query Processing
    Server->>Processor: Process query
    Processor->>Processor: Clean and validate query
    Processor->>Server: Return cleaned query, cluster, db

    Note over Server: Phase 3: Schema Enrichment
    Server->>Memory: Get schema for cluster/db
    alt Schema in Cache
        Memory->>Server: Return cached schema
    else Schema not in Cache
        Server->>Executor: Discover schema from Azure
        Executor->>Azure: .show schema commands
        Azure->>Executor: Schema metadata
        Executor->>Memory: Update cache with new schema
        Memory->>Server: Return new schema
    end

    Note over Server: Phase 4: Execution
    Server->>Executor: Execute query with context
    Executor->>Azure: Run KQL query
    alt Query Fails
        Azure-->>Executor: KustoServiceError
        Executor->>ErrorHandler: Handle error
        ErrorHandler->>Server: Formatted error response
        Server-->>Client: Return structured error
    else Query Succeeds
        Azure-->>Executor: Query results
        Executor->>Server: Return processed results

        Note over Server: Phase 5: Post-Execution Learning (Async)
        Executor-)-Memory: Run post_query_learning task
        Memory-)-Memory: Update unified_memory.json
    end

    Server-->>Client: Return successful result
```

## 5. Security Model

- **Authentication**: Relies entirely on the user's existing Azure CLI session. The server never stores passwords, secrets, or long-lived tokens.
- **Query Sanitization**: The `QueryProcessor` performs basic cleaning, but the primary defense is the parameterized nature of the Azure Kusto SDK, which prevents classical injection attacks.
- **Local Caching**: All sensitive schema information is stored in a local JSON file (`unified_memory.json`) within the user's profile directory, never transmitted elsewhere.

## 6. Conclusion

The v2.0.7 architecture of the MCP KQL Server represents a mature, robust, and intelligent system. By centralizing logic in the `utils.py` pipeline and introducing dynamic schema analysis via `constants.py`, the server is more maintainable, extensible, and powerful. The clear separation of concerns—from authentication (`kql_auth.py`) to memory (`memory.py`) to execution (`execute_kql.py`)—creates a solid foundation for future enhancements.
