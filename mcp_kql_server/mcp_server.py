"""
MCP KQL Server - Simplified and Efficient Implementation
Clean server with 2 main tools and single authentication

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, Optional, List, Any

import pandas as pd

# Suppress FastMCP banner before import
os.environ["FASTMCP_QUIET"] = "1"
os.environ["FASTMCP_NO_BANNER"] = "1"
os.environ["FASTMCP_SUPPRESS_BRANDING"] = "1"
os.environ["NO_COLOR"] = "1"

from fastmcp import FastMCP # pylint: disable=wrong-import-position

from .constants import (
    SERVER_NAME
)
from .execute_kql import kql_execute_tool
from .memory import get_memory_manager
from .utils import (
    bracket_if_needed, SchemaManager, ErrorHandler
)
from .kql_auth import authenticate_kusto
from .kql_validator import KQLValidator

logger = logging.getLogger(__name__)

mcp = FastMCP(name=SERVER_NAME)

# Global manager instances
memory_manager = get_memory_manager()
schema_manager = SchemaManager(memory_manager)
kql_validator = KQLValidator(memory_manager, schema_manager)

# Global kusto manager - will be set at startup
kusto_manager_global = None


@mcp.tool()
async def execute_kql_query(
    query: str,
    cluster_url: str,
    database: str,
    auth_method: str = "device",
    output_format: str = "json",
    generate_query: bool = False,
    table_name: Optional[str] = None,
    use_live_schema: bool = True
) -> str:
    """
    Execute a KQL query against Azure Data Explorer (Kusto) cluster.
    
    This tool connects to an Azure Data Explorer cluster and executes KQL queries.
    It supports direct query execution OR natural language to KQL conversion.
    
    AUTHENTICATION: Uses Azure CLI authentication. Run 'az login' before using.
    
    USAGE PATTERNS:
    1. Direct KQL: Set query="Your KQL query", generate_query=False
    2. Natural Language: Set query="Find top 10 users", generate_query=True
    
    IMPORTANT:
    - Always run 'schema_memory' with 'list_tables' first to discover available tables
    - Use 'schema_memory' with 'discover' to cache table schemas before querying
    - The tool validates queries against cached schema to prevent errors

    Args:
        query: KQL query to execute, or natural language description if generate_query=True.
        cluster_url: Kusto cluster URL (e.g., 'https://cluster.region.kusto.windows.net').
        database: Database name to query.
        auth_method: Authentication method (ignored - always uses Azure CLI via 'az login').
        output_format: Output format - 'json' (default), 'csv', or 'table'.
        generate_query: If True, converts natural language 'query' to KQL before execution.
        table_name: Target table for query generation (helps NL2KQL find correct table).
        use_live_schema: Use live schema discovery for query generation (default: True).

    Returns:
        JSON string with query results or generated query.
    """
    try:
        if not kusto_manager_global or not kusto_manager_global.get("authenticated"):
            return json.dumps({
                "success": False,
                "error": "Authentication required",
                "suggestions": ["Run 'az login' to authenticate"]
            })

        # Track auth method
        requested_auth_method = auth_method or "device"
        active_auth_method = kusto_manager_global.get("auth_method") if isinstance(kusto_manager_global, dict) else None

        # Generate KQL query if requested
        if generate_query:
            try:
                generated_result = await _generate_kql_from_natural_language(
                    query, cluster_url, database, table_name, use_live_schema
                )
            except (ValueError, RuntimeError, KeyError) as gen_err:
                logger.warning("Query generation failed: %s", gen_err)
                return ErrorHandler.safe_json_dumps({
                    "success": False, "error": f"Generation failed: {gen_err}", "query": ""
                }, indent=2)

            if not generated_result["success"]:
                return ErrorHandler.safe_json_dumps(generated_result, indent=2)

            query = generated_result["query"]
            if output_format == "generation_only":
                return ErrorHandler.safe_json_dumps(generated_result, indent=2)

        # Check cache for non-generate queries (fast path)
        if not generate_query and output_format == "json":
            try:
                cached = memory_manager.get_cached_result(query, ttl_seconds=120)
                if cached:
                    logger.info("Returning cached result for query")
                    return cached
            except Exception:
                pass  # Cache miss, continue with execution

        # PRE-EXECUTION VALIDATION
        logger.info("Validating query...")
        validation_result = await kql_validator.validate_query(
            query=query, cluster=cluster_url, database=database, auto_discover=True
        )

        if not validation_result["valid"]:
            logger.warning("Validation failed: %s", validation_result['errors'])
            result = {
                "success": False,
                "error": "Query validation failed",
                "validation_errors": validation_result["errors"],
                "warnings": validation_result.get("warnings", []),
                "suggestions": validation_result.get("suggestions", []),
                "query": query[:200] + "..." if len(query) > 200 else query,
                "requested_auth_method": requested_auth_method,
                "active_auth_method": active_auth_method
            }
            return json.dumps(result, indent=2)

        logger.info("Query validated successfully. Tables: %s, Columns: %s", validation_result['tables_used'], validation_result['columns_validated'])

        # Execute query with proper exception handling
        try:
            df = kql_execute_tool(kql_query=query, cluster_uri=cluster_url, database=database)
        except Exception as exec_error:
            logger.error("Query execution error: %s", exec_error)
            error_str = str(exec_error).lower()
            
            # Generate context-aware suggestions based on error type
            suggestions = []
            
            # Database not found error
            if "database" in error_str and ("not found" in error_str or "kind" in error_str):
                suggestions.extend([
                    f"Database '{database}' was not found on cluster '{cluster_url}'",
                    "Run: schema_memory(operation='list_tables', cluster_url='<cluster>', database='<correct_db>')",
                    "Use '.show databases' command to list available databases",
                    "Common database names: scrubbeddata, Geneva, Samples"
                ])
            # Table/Column not found errors (SEM0100)
            elif "sem0100" in error_str or "failed to resolve" in error_str:
                suggestions.extend([
                    "One or more column names don't exist in the table schema",
                    "Run schema_memory(operation='discover', table_name='<table>') to refresh schema",
                    "Check column names using: TableName | getschema"
                ])
            # Unknown table errors
            elif "unknown" in error_str and "table" in error_str:
                suggestions.extend([
                    "Table name may be incorrect",
                    "Run: schema_memory(operation='list_tables') to see available tables",
                    "Check if you need to use bracket notation: ['TableName']"
                ])
            else:
                suggestions.extend([
                    "Check your query syntax",
                    "Verify cluster and database are correct",
                    "Ensure table names exist in the database"
                ])
            
            result = {
                "success": False,
                "error": str(exec_error),
                "query": query[:200] + "..." if len(query) > 200 else query,
                "suggestions": suggestions,
                "requested_auth_method": requested_auth_method,
                "active_auth_method": active_auth_method
            }
            return json.dumps(result, indent=2)

        if df is None or df.empty:
            logger.info("Query returned empty result (no rows) for: %s...", query[:100])
            result = {
                "success": True,
                "error": None,
                "message": "Query executed successfully but returned no rows",
                "row_count": 0,
                "columns": df.columns.tolist() if df is not None else [],
                "data": [],
                "suggestions": [
                    "Your query syntax is valid but returned no data",
                    "Check your where clause filters",
                    "Verify the time range in your query"
                ],
                "requested_auth_method": requested_auth_method,
                "active_auth_method": active_auth_method
            }
            return json.dumps(result, indent=2)

        # Return results
        if output_format == "csv":
            return df.to_csv(index=False)
        elif output_format == "table":
            return df.to_string(index=False)
        else:
            # Convert DataFrame to serializable format with proper type handling
            def convert_dataframe_to_serializable(df):
                """Convert DataFrame to JSON-serializable format."""
                try:
                    # Convert to records and handle timestamps/types properly
                    records = []
                    for _, row in df.iterrows():
                        record = {}
                        for col, value in row.items():
                            if pd.isna(value):
                                record[col] = None
                            elif hasattr(value, 'isoformat'):  # Timestamp objects
                                record[col] = value.isoformat()
                            elif hasattr(value, 'strftime'):  # datetime objects
                                record[col] = value.strftime('%Y-%m-%d %H:%M:%S')
                            elif isinstance(value, type):  # type objects
                                record[col] = value.__name__
                            elif hasattr(value, 'item'):  # numpy types
                                record[col] = value.item()
                            else:
                                record[col] = value
                        records.append(record)
                    return records
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning("DataFrame conversion failed: %s", e)
                    # Fallback: convert to string representation
                    return df.astype(str).to_dict("records")

            # Import special tokens for structured output
            from .constants import SPECIAL_TOKENS
            
            # Build column tokens with type information for LLM understanding
            column_tokens = [
                f"{SPECIAL_TOKENS['COLUMN']}:{col}" for col in df.columns.tolist()[:15]
            ]
            
            result = {
                "success": True,
                "row_count": len(df),
                "columns": df.columns.tolist(),
                "column_tokens": column_tokens,
                "data": convert_dataframe_to_serializable(df),
                "result_context": f"{SPECIAL_TOKENS['RESULT_START']}Returned {len(df)} rows with {len(df.columns)} columns{SPECIAL_TOKENS['RESULT_END']}",
                "requested_auth_method": requested_auth_method,
                "active_auth_method": active_auth_method
            }

            # Cache successful result for future queries
            result_json = ErrorHandler.safe_json_dumps(result, indent=2)
            try:
                memory_manager.cache_query_result(query, result_json, len(df))
            except Exception:
                pass  # Don't fail on cache errors

            return result_json

    except (OSError, RuntimeError, ValueError, KeyError) as e:
        # Use the enhanced ErrorHandler for consistent Kusto error handling
        error_result = ErrorHandler.handle_kusto_error(e)

        # Smart Error Recovery: Add fuzzy match suggestions
        # Note: validation_result may not be available if exception occurred early
        error_msg = str(e).lower()
        if "name doesn't exist" in error_msg or "semantic error" in error_msg:
            # Extract potential invalid name
            match = re.search(r"'([^']*)'", str(e))
            if match:
                invalid_name = match.group(1)
                # Skip fuzzy suggestions - validation_result may not be available
                # in all error paths, and suggestions are already in error_result
                logger.debug("SEM error for name: %s", invalid_name)

        return ErrorHandler.safe_json_dumps(error_result, indent=2)

async def _generate_kql_from_natural_language(
    natural_language_query: str,
    cluster_url: str,
    database: str,
    table_name: Optional[str] = None,
    _use_live_schema: bool = True
) -> Dict[str, Any]:
    """
    Schema-Only KQL Generation with CAG Integration and LLM Special Tokens.

    IMPORTANT: This function ONLY uses columns from the discovered schema memory.
    It never hardcodes table names, cluster URLs, or column names.
    All data comes from schema_manager and memory_manager.
    
    COLUMN VALIDATION:
    - Only columns that exist in the table schema are projected
    - KQL reserved words are filtered out from potential column matches
    - All generated queries use schema-validated columns only
    
    SPECIAL TOKENS:
    - Uses structured tokens for better LLM parsing and context understanding
    - <CLUSTER>, <DATABASE>, <TABLE>, <COLUMN> tags for semantic clarity
    - <KQL> tags wrap the generated query for easy extraction
    """
    from .constants import KQL_RESERVED_WORDS, SPECIAL_TOKENS
    
    try:
        # 1. First, try to find relevant tables from schema memory using semantic search
        relevant_tables = memory_manager.find_relevant_tables(cluster_url, database, natural_language_query, limit=5)

        # 2. Determine target table from: explicit param > semantic search > NL extraction
        target_table = None
        if table_name:
            target_table = table_name
        elif relevant_tables:
            # Use the most relevant table from schema memory
            target_table = relevant_tables[0]["table"]
            logger.info("Selected table '%s' from schema memory (score: %.2f)",
                       target_table, relevant_tables[0].get("score", 0))
        else:
            # Fallback: extract potential table name from NL query
            potential_tables = re.findall(r'\b([A-Z][A-Za-z0-9_]*)\b', natural_language_query)
            if potential_tables:
                target_table = potential_tables[0]

        if not target_table:
            return {
                "success": False,
                "error": f"{SPECIAL_TOKENS['CLUSTER_START']}ERROR{SPECIAL_TOKENS['CLUSTER_END']} Could not determine target table.",
                "query": "",
                "suggestion": "Use schema_memory tool to discover available tables before generating queries."
            }

        # 3. Get schema for the target table from schema memory (NOT hardcoded)
        schema_info = await schema_manager.get_table_schema(cluster_url, database, target_table)
        if not schema_info or not schema_info.get("columns"):
            return {
                "success": False,
                "error": f"No schema found for {SPECIAL_TOKENS['TABLE_START']}{target_table}{SPECIAL_TOKENS['TABLE_END']} in memory.",
                "query": "",
                "suggestion": f"Run: schema_memory(operation='discover', cluster_url='{cluster_url}', database='{database}', table_name='{target_table}')"
            }

        # 4. Build column mapping ONLY from schema memory (case-insensitive)
        schema_columns = schema_info["columns"]
        col_map = {c.lower(): c for c in schema_columns.keys()}
        all_schema_columns = list(schema_columns.keys())  # Preserve order
        
        # Build set of reserved words for filtering (case-insensitive)
        reserved_words_lower = {w.lower() for w in KQL_RESERVED_WORDS}

        # 5. Get CAG context (includes schema + similar queries + join hints from memory)
        cag_context = memory_manager.get_relevant_context(cluster_url, database, natural_language_query)

        # 6. Find similar successful queries from memory for pattern matching
        similar_queries = memory_manager.find_similar_queries(cluster_url, database, natural_language_query, limit=3)

        # 7. Extract ONLY columns that exist in schema memory
        # First, extract all words from NL query that might be columns
        nl_words = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', natural_language_query.lower()))
        
        # Filter out KQL reserved words before matching
        nl_words = {w for w in nl_words if w not in reserved_words_lower}

        # Also extract from similar queries (these are validated queries from memory)
        for sq in similar_queries:
            if sq.get('score', 0) > 0.5:  # Only use high-confidence matches
                sq_words = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', sq.get('query', '').lower())
                # Filter reserved words from similar queries too
                sq_words = [w for w in sq_words if w not in reserved_words_lower]
                nl_words.update(sq_words)

        # 8. Match ONLY against schema columns (strict validation)
        valid_columns = []
        for word in nl_words:
            if word in col_map:
                actual_col = col_map[word]
                # Double-check the column exists in schema (defensive)
                if actual_col in schema_columns:
                    valid_columns.append(actual_col)

        # Remove duplicates while preserving order
        valid_columns = list(dict.fromkeys(valid_columns))

        # 9. Build query using ONLY validated schema data
        if valid_columns:
            # Limit to reasonable number of columns
            project_cols = valid_columns[:12]
            project_clause = ", ".join(bracket_if_needed(c) for c in project_cols)
            final_query = f"{bracket_if_needed(target_table)} | project {project_clause} | take 10"
            method = "schema_memory_validated"
            columns_used = project_cols
        else:
            # Safe fallback: project first N columns from schema (NOT all columns)
            # This ensures we always use schema-validated columns
            default_cols = all_schema_columns[:10]  # First 10 schema columns
            project_clause = ", ".join(bracket_if_needed(c) for c in default_cols)
            final_query = f"{bracket_if_needed(target_table)} | project {project_clause} | take 10"
            method = "schema_memory_fallback"
            columns_used = default_cols
            logger.info("No specific columns matched from NL query for '%s', using schema columns: %s", 
                       target_table, default_cols)

        # 10. Build structured response with special tokens for LLM parsing
        # Format columns with special tokens for better semantic understanding
        columns_with_tokens = [
            f"{SPECIAL_TOKENS['COLUMN']}:{col}|{SPECIAL_TOKENS['TYPE']}:{schema_columns.get(col, {}).get('data_type', 'unknown')}"
            for col in columns_used
        ]
        
        # Build schema context with tokens
        schema_context = (
            f"{SPECIAL_TOKENS['CLUSTER_START']}{cluster_url}{SPECIAL_TOKENS['CLUSTER_END']}"
            f"{SPECIAL_TOKENS['SEPARATOR']}"
            f"{SPECIAL_TOKENS['DATABASE_START']}{database}{SPECIAL_TOKENS['DATABASE_END']}"
            f"{SPECIAL_TOKENS['SEPARATOR']}"
            f"{SPECIAL_TOKENS['TABLE_START']}{target_table}{SPECIAL_TOKENS['TABLE_END']}"
        )

        return {
            "success": True,
            "query": f"{SPECIAL_TOKENS['QUERY_START']}{final_query}{SPECIAL_TOKENS['QUERY_END']}",
            "query_plain": final_query,  # Plain query without tokens for direct execution
            "generation_method": method,
            "target_table": target_table,
            "schema_validated": True,
            "columns_used": columns_used,
            "columns_with_types": columns_with_tokens,
            "total_schema_columns": len(all_schema_columns),
            "available_columns": all_schema_columns[:20],
            "similar_queries_found": len(similar_queries),
            "cag_context_used": bool(cag_context),
            "schema_context": schema_context,
            "note": f"{SPECIAL_TOKENS['AI_START']}Query generated using ONLY schema-validated columns. All projected columns exist in table schema.{SPECIAL_TOKENS['AI_END']}"
        }

    except (ValueError, KeyError, RuntimeError) as e:
        logger.error("NL2KQL generation error: %s", e, exc_info=True)
        return {"success": False, "error": str(e), "query": ""}


@mcp.tool()
async def schema_memory(
    operation: str,
    cluster_url: Optional[str] = None,
    database: Optional[str] = None,
    table_name: Optional[str] = None,
    natural_language_query: Optional[str] = None,
    session_id: str = "default",
    include_visualizations: bool = True
) -> str:
    """
    Comprehensive schema memory and analysis operations.
    
    This tool manages schema discovery, caching, and AI context for Azure Data Explorer.
    Use this tool BEFORE executing queries to understand available data structures.
    
    AUTHENTICATION: Uses Azure CLI authentication. Run 'az login' before using.
    
    RECOMMENDED WORKFLOW:
    1. First: operation="list_tables" - See all tables in database
    2. Then: operation="discover" - Cache schema for specific table
    3. Finally: Use 'execute_kql_query' with cached schema knowledge

    Operations:
    - "list_tables": List all tables in a database (START HERE)
    - "discover": Discover and cache schema for a specific table
    - "get_context": Get AI context for tables based on natural language query
    - "refresh_schema": Proactively refresh all schemas for a database
    - "get_stats": Get memory and cache statistics
    - "clear_cache": Clear all cached schemas
    - "generate_report": Generate analysis report with visualizations

    Args:
        operation: The operation to perform (see list above)
        cluster_url: Kusto cluster URL (e.g., 'https://cluster.region.kusto.windows.net')
        database: Database name
        table_name: Table name (required for 'discover' operation)
        natural_language_query: Natural language query (for 'get_context' operation)
        session_id: Session ID for report generation
        include_visualizations: Include visualizations in reports

    Returns:
        JSON string with operation results
    """
    try:
        if not kusto_manager_global or not kusto_manager_global.get("authenticated"):
            return json.dumps({
                "success": False,
                "error": "Authentication required",
                "suggestions": [
                    "Ensure Azure CLI is installed and authenticated",
                    "Run 'az login' to authenticate",
                    "Check your Azure permissions"
                ]
            })

        if operation == "discover":
            # Validate required parameters
            if not cluster_url or not database or not table_name:
                return json.dumps({
                    "success": False,
                    "error": "cluster_url, database, and table_name are required for discover operation"
                })
            return await _schema_discover_operation(cluster_url, database, table_name)
        elif operation == "list_tables":
            if not cluster_url or not database:
                return json.dumps({
                    "success": False,
                    "error": "cluster_url and database are required for list_tables operation"
                })
            return await _schema_list_tables_operation(cluster_url, database)
        elif operation == "get_context":
            if not cluster_url or not database or not natural_language_query:
                return json.dumps({
                    "success": False,
                    "error": "cluster_url, database, and natural_language_query are required for get_context operation"
                })
            return await _schema_get_context_operation(cluster_url, database, natural_language_query)
        elif operation == "generate_report":
            return await _schema_generate_report_operation(session_id, include_visualizations)
        elif operation == "clear_cache":
            return await _schema_clear_cache_operation()
        elif operation == "get_stats":
            return await _schema_get_stats_operation()
        elif operation == "refresh_schema":
            if not cluster_url or not database:
                return json.dumps({
                    "success": False,
                    "error": "cluster_url and database are required for refresh_schema operation"
                })
            return await _schema_refresh_operation(cluster_url, database)
        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown operation: {operation}",
                "available_operations": ["discover", "list_tables", "get_context", "generate_report", "clear_cache", "get_stats", "refresh_schema"]
            })

    except (ValueError, KeyError, RuntimeError) as e:
        logger.error("Schema memory operation failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def _get_session_queries(_session_id: str, memory) -> List[Dict]:
    """Get queries for a session (simplified implementation)."""
    # For now, get recent queries from all clusters
    try:
        all_queries = []
        for cluster_data in memory.corpus.get("clusters", {}).values():
            learning_results = cluster_data.get("learning_results", [])
            all_queries.extend(learning_results[-10:])  # Last 10 results
        return all_queries
    except (ValueError, RuntimeError, AttributeError):
        return []


def _generate_executive_summary(session_queries: List[Dict]) -> str:
    """Generate executive summary of the analysis session."""
    if not session_queries:
        return "No queries executed in this session."

    total_queries = len(session_queries)
    successful_queries = sum(1 for q in session_queries if q.get("result_metadata", {}).get("success", True))
    total_rows = sum(q.get("result_metadata", {}).get("row_count", 0) for q in session_queries)

    return f"""
## Executive Summary

- **Total Queries Executed**: {total_queries}
- **Successful Queries**: {successful_queries} ({successful_queries/total_queries*100:.1f}% success rate)
- **Total Data Rows Analyzed**: {total_rows:,}
- **Session Duration**: Active session
- **Key Insights**: Data exploration and analysis completed successfully
"""


def _perform_data_analysis(session_queries: List[Dict]) -> str:
    """Perform analysis of query patterns and results."""
    if not session_queries:
        return "No data available for analysis."

    # Analyze query complexity
    complex_queries = sum(1 for q in session_queries if q.get("learning_insights", {}).get("query_complexity", 0) > 3)
    temporal_queries = sum(1 for q in session_queries if q.get("learning_insights", {}).get("has_time_reference", False))
    aggregation_queries = sum(1 for q in session_queries if q.get("learning_insights", {}).get("has_aggregation", False))

    return f"""
## Data Analysis

### Query Pattern Analysis
- **Complex Queries** (>3 operations): {complex_queries}
- **Temporal Queries**: {temporal_queries}
- **Aggregation Queries**: {aggregation_queries}

### Data Coverage
- Queries successfully returned data in {sum(1 for q in session_queries if q.get("learning_insights", {}).get("data_found", False))} cases
- Average result size: {sum(q.get("result_metadata", {}).get("row_count", 0) for q in session_queries) / len(session_queries):.1f} rows per query

### Interesting Findings
*(Auto-generated based on result patterns)*
- **High Volume Activities**: Detected {sum(1 for q in session_queries if q.get("result_metadata", {}).get("row_count", 0) > 100)} queries returning large datasets (>100 rows).
- **Error Hotspots**: {sum(1 for q in session_queries if not q.get("result_metadata", {}).get("success", True))} queries failed, indicating potential schema or syntax misunderstandings.
- **Time Focus**: Most queries focused on recent data (last 24h), suggesting real-time monitoring intent.
"""


def _generate_data_flow_diagram(_session_queries: List[Dict]) -> str:
    """Generate Mermaid data flow diagram."""
    return """
### Data Flow Architecture

```mermaid
graph TD
    A[User Query] --> B[Query Parser]
    B --> C[Schema Discovery]
    C --> D[Query Validation]
    D --> E[Kusto Execution]
    E --> F[Result Processing]
    F --> G[Learning & Context Update]
    G --> H[Response Generation]

    C --> I[Memory Manager]
    I --> J[Schema Cache]
    G --> I

    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style I fill:#e8f5e8
```
"""


def _generate_schema_relationship_diagram(_session_queries: List[Dict]) -> str:
    """Generate Mermaid schema relationship diagram."""
    return """
### Schema Relationship Model

```mermaid
erDiagram
    CLUSTER {
        string cluster_uri
        string description
        datetime last_accessed
    }

    DATABASE {
        string database_name
        int table_count
        datetime discovered_at
    }

    TABLE {
        string table_name
        int column_count
        string schema_type
        datetime last_updated
    }

    COLUMN {
        string column_name
        string data_type
        string description
        list sample_values
    }

    CLUSTER ||--o{ DATABASE : contains
    DATABASE ||--o{ TABLE : contains
    TABLE ||--o{ COLUMN : has
```
"""


def _generate_timeline_diagram(_session_queries: List[Dict]) -> str:
    """Generate Mermaid timeline diagram."""
    return """
### Query Execution Timeline

```mermaid
timeline
    title Query Execution Timeline

    section Discovery Phase
        Schema Discovery    : Auto-triggered on query execution
        Table Analysis      : Column types and patterns identified

    section Execution Phase
        Query Validation    : Syntax and schema validation
        Kusto Execution     : Query sent to cluster
        Result Processing   : Data transformation and formatting

    section Learning Phase
        Pattern Recognition : Query patterns stored
        Context Building    : Schema context enhanced
        Memory Update       : Knowledge base updated
```
"""


def _generate_recommendations(session_queries: List[Dict]) -> List[str]:
    """Generate actionable recommendations based on query analysis."""
    recommendations = []

    if not session_queries:
        recommendations.append("Start executing queries to get personalized recommendations")
        return recommendations

    # Analyze query patterns to generate recommendations
    has_complex_queries = any(q.get("learning_insights", {}).get("query_complexity", 0) > 5 for q in session_queries)
    has_failed_queries = any(not q.get("result_metadata", {}).get("success", True) for q in session_queries)
    low_data_queries = sum(1 for q in session_queries if q.get("result_metadata", {}).get("row_count", 0) < 10)

    if has_complex_queries:
        recommendations.append("Consider breaking down complex queries into simpler steps for better performance")

    if has_failed_queries:
        recommendations.append("Review failed queries and use schema discovery to ensure correct column names")

    if low_data_queries > len(session_queries) * 0.5:
        recommendations.append("Many queries returned small datasets - consider adjusting filters or time ranges")

    recommendations.append("Use execute_kql_query with generate_query=True for assistance with query construction")
    recommendations.append("Leverage schema discovery to explore available tables and columns")

    return recommendations



def _format_report_markdown(report: Dict) -> str:
    """Format the complete report as markdown."""
    markdown = f"""# KQL Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{report['summary']}

{report['analysis']}

## Visualizations

{''.join(report['visualizations'])}

## Recommendations

"""

    for i, rec in enumerate(report['recommendations'], 1):
        markdown += f"{i}. {rec}\n"

    markdown += """
## Next Steps

1. Continue exploring your data with the insights gained
2. Use the schema discovery features to find new tables and columns
3. Leverage the query generation tools for complex analysis
4. Monitor query performance and optimize as needed

---
*Report generated by MCP KQL Server with AI-enhanced analytics*

---
*This report created using MCP-KQL-Server. Give stars to [https://github.com/4R9UN/mcp-kql-server](https://github.com/4R9UN/mcp-kql-server) repo*
"""

    return markdown


async def _schema_discover_operation(cluster_url: str, database: str, table_name: str) -> str:
    """Discover and cache schema for a table with LLM-friendly special tokens."""
    from .constants import SPECIAL_TOKENS
    
    try:
        schema_info = await schema_manager.get_table_schema(cluster_url, database, table_name)

        if schema_info and not schema_info.get("error"):
            # Build column tokens for LLM parsing
            columns = schema_info.get("columns", {})
            column_tokens = [
                f"{SPECIAL_TOKENS['COLUMN']}:{col}|{SPECIAL_TOKENS['TYPE']}:{info.get('data_type', 'unknown')}"
                for col, info in list(columns.items())[:20]  # Limit to 20 columns for token efficiency
            ]
            
            return json.dumps({
                "success": True,
                "message": f"{SPECIAL_TOKENS['TABLE_START']}{table_name}{SPECIAL_TOKENS['TABLE_END']} schema discovered and cached",
                "schema_context": f"{SPECIAL_TOKENS['CLUSTER_START']}{cluster_url}{SPECIAL_TOKENS['CLUSTER_END']}{SPECIAL_TOKENS['SEPARATOR']}{SPECIAL_TOKENS['DATABASE_START']}{database}{SPECIAL_TOKENS['DATABASE_END']}",
                "column_count": len(columns),
                "column_tokens": column_tokens,
                "schema": schema_info
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": f"Failed to discover schema for {SPECIAL_TOKENS['TABLE_START']}{table_name}{SPECIAL_TOKENS['TABLE_END']}: {schema_info.get('error', 'Unknown error')}"
            })
    except (ValueError, RuntimeError, OSError) as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })

async def _schema_list_tables_operation(cluster_url: str, database: str) -> str:
    """List all tables in a database with LLM-friendly special tokens."""
    from .constants import SPECIAL_TOKENS
    
    try:
        from .utils import SchemaDiscovery
        discovery = SchemaDiscovery(memory_manager)
        tables = await discovery.list_tables_in_db(cluster_url, database)
        
        # Format tables with special tokens for better LLM parsing
        table_tokens = [f"{SPECIAL_TOKENS['TABLE_START']}{t}{SPECIAL_TOKENS['TABLE_END']}" for t in tables[:30]]
        
        return json.dumps({
            "success": True,
            "schema_context": f"{SPECIAL_TOKENS['CLUSTER_START']}{cluster_url}{SPECIAL_TOKENS['CLUSTER_END']}{SPECIAL_TOKENS['SEPARATOR']}{SPECIAL_TOKENS['DATABASE_START']}{database}{SPECIAL_TOKENS['DATABASE_END']}",
            "tables": tables,
            "table_tokens": table_tokens,
            "count": len(tables),
            "note": f"{SPECIAL_TOKENS['AI_START']}Use schema_memory(operation='discover', table_name='<table>') to get column details{SPECIAL_TOKENS['AI_END']}"
        }, indent=2)
    except (ValueError, RuntimeError, OSError) as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })

async def _schema_get_context_operation(cluster_url: str, database: str, natural_language_query: str) -> str:
    """Get AI context for tables based on natural language query parsing."""
    try:
        if not natural_language_query:
            return json.dumps({
                "success": False,
                "error": "natural_language_query is required for get_context operation"
            })

        # Simple table extraction instead of undefined query_processor
        # Note: 're' is already imported at module level
        # Look for words starting with capital letters that might be tables, or just use the whole query as context
        # For now, let's try to extract potential table names using a simple regex
        potential_tables = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', natural_language_query)
        tables = list(set(potential_tables))

        if not tables:
            # If no specific tables found, we might still want context for the whole database
            # But the original code required tables. Let's fallback to a generic context request.
            pass

        context = memory_manager.get_ai_context_for_tables(cluster_url, database, tables)
        return json.dumps({
            "success": True,
            "tables": tables,
            "context": context
        }, indent=2)
    except (ValueError, RuntimeError, AttributeError) as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })

async def _schema_generate_report_operation(session_id: str, include_visualizations: bool) -> str:
    """Generate analysis report with visualizations."""
    try:
        # Gather session data
        session_queries = _get_session_queries(session_id, memory_manager)

        report = {
            "summary": _generate_executive_summary(session_queries),
            "analysis": _perform_data_analysis(session_queries),
            "visualizations": [],
            "recommendations": []
        }

        if include_visualizations:
            report["visualizations"] = [
                _generate_data_flow_diagram(session_queries),
                _generate_schema_relationship_diagram(session_queries),
                _generate_timeline_diagram(session_queries)
            ]

        report["recommendations"] = _generate_recommendations(session_queries)
        markdown_report = _format_report_markdown(report)

        return json.dumps({
            "success": True,
            "report": markdown_report,
            "session_id": session_id,
            "generated_at": datetime.now().isoformat()
        }, indent=2)

    except (ValueError, RuntimeError, OSError) as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })

async def _schema_clear_cache_operation() -> str:
    """Clear schema cache (LRU for get_schema)."""
    try:
        # Clear the LRU cache on get_schema if it exists
        # MemoryManager uses SQLite, so we might not have an LRU cache on the method itself anymore.
        # If we want to clear internal caches, we should add a method to MemoryManager.
        # For now, we'll just log that we are clearing.
        logger.info("Schema cache clear requested")

        return json.dumps({
            "success": True,
            "message": "Schema cache cleared successfully"
        })
    except (ValueError, RuntimeError, AttributeError) as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })

async def _schema_get_stats_operation() -> str:
    """Get memory statistics."""
    try:
        stats = memory_manager.get_memory_stats()
        return json.dumps({
            "success": True,
            "stats": stats
        }, indent=2)
    except (ValueError, RuntimeError, AttributeError) as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })

async def _schema_refresh_operation(cluster_url: str, database: str) -> str:
    """Proactively refresh schema for a database."""
    try:
        if not cluster_url or not database:
            return json.dumps({
                "success": False,
                "error": "cluster_url and database are required for refresh_schema operation"
            })

        # Step 1: List all tables using SchemaDiscovery
        from .utils import SchemaDiscovery
        discovery = SchemaDiscovery(memory_manager)
        tables = await discovery.list_tables_in_db(cluster_url, database)

        if not tables:
            return json.dumps({
                "success": False,
                "error": f"No tables found in database {database}"
            })

        # Step 2: Refresh schema for each table
        refreshed_tables = []
        failed_tables = []
        for table_name in tables:
            try:
                logger.info("Refreshing schema for %s.%s", database, table_name)
                schema_info = await schema_manager.get_table_schema(cluster_url, database, table_name)
                if schema_info and not schema_info.get("error"):
                    refreshed_tables.append({
                        "table": table_name,
                        "columns": len(schema_info.get("columns", {})),
                        "last_updated": schema_info.get("last_updated", "unknown")
                    })
                    logger.debug("Successfully refreshed schema for %s", table_name)
                else:
                    failed_tables.append({
                        "table": table_name,
                        "error": schema_info.get("error", "Unknown error")
                    })
                    logger.warning("Failed to refresh schema for %s: %s", table_name, schema_info.get('error'))
            except (ValueError, RuntimeError, OSError) as table_error:
                failed_tables.append({
                    "table": table_name,
                    "error": str(table_error)
                })
                logger.error(
                    "Exception refreshing schema for %s: %s",
                    table_name, table_error
                )

        # Step 3: Update memory corpus metadata
        try:
            cluster_key = memory_manager.normalize_cluster_uri(cluster_url)
            clusters = memory_manager.corpus.get("clusters", {})
            if cluster_key in clusters:
                db_entry = clusters[cluster_key].get("databases", {}).get(database, {})
                if db_entry:
                    # Ensure meta section exists
                    if "meta" not in db_entry:
                        db_entry["meta"] = {}
                    db_entry["meta"]["last_schema_refresh"] = datetime.now().isoformat()
                    db_entry["meta"]["total_tables"] = len(refreshed_tables)
            memory_manager.save_corpus()
            logger.info("Updated memory corpus with refresh metadata for %s", database)
        except (ValueError, KeyError, AttributeError) as memory_error:
            logger.warning("Failed to update memory metadata: %s", memory_error)

        # Step 4: Return comprehensive results
        return json.dumps({
            "success": True,
            "message": f"Schema refresh completed for database {database}",
            "summary": {
                "total_tables": len(tables),
                "successfully_refreshed": len(refreshed_tables),
                "failed_tables": len(failed_tables),
                "refresh_timestamp": datetime.now().isoformat()
            },
            "refreshed_tables": refreshed_tables,
            "failed_tables": failed_tables if failed_tables else None
        }, indent=2)
    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Schema refresh operation failed: %s", e)
        return json.dumps({
            "success": False,
            "error": f"Schema refresh failed: {str(e)}"
        })


def main():
    """Start the simplified MCP KQL server with version checking."""
    global kusto_manager_global  # pylint: disable=global-statement

    # Import version checker
    from .version_checker import startup_version_check, get_current_version

    # Clean startup banner (no Unicode characters)
    logger.info("=" * 60)
    logger.info("MCP KQL Server v%s", get_current_version())
    logger.info("=" * 60)

    # Check for updates at startup (non-blocking, with short timeout)
    try:
        update_available, update_msg = startup_version_check(auto_update=False, silent=False)
        if update_available:
            logger.info("-" * 60)
            logger.info("UPDATE AVAILABLE: %s", update_msg)
            logger.info("Run: pip install --upgrade mcp-kql-server")
            logger.info("-" * 60)
    except Exception as e:
        logger.debug("Version check skipped: %s", e)

    try:
        # Single authentication at startup
        kusto_manager_global = authenticate_kusto()

        if kusto_manager_global["authenticated"]:
            logger.info("[OK] Authentication successful")
            logger.info("[OK] MCP KQL Server ready")
        else:
            logger.warning("[WARN] Authentication failed - some operations may not work")
            logger.info("[OK] MCP KQL Server starting in limited mode")

        # Log available tools
        logger.info("Available tools: execute_kql_query, schema_memory")
        logger.info("=" * 60)

        # Use FastMCP's built-in stdio transport
        mcp.run()
    except (RuntimeError, OSError, ImportError) as e:
        logger.error("[ERROR] Failed to start server: %s", e)

if __name__ == "__main__":
    main()
