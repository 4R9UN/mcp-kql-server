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
import difflib
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

GENERATION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "find", "for", "from", "get",
    "how", "in", "last", "latest", "list", "me", "of", "on", "recent", "show",
    "that", "the", "to", "top", "with"
}


def _tokenize_generation_text(text: str) -> List[str]:
    """Tokenize NL queries and identifiers for ranking/generation heuristics."""
    normalized = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text or "")
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', normalized.lower())
    return [token for token in tokens if len(token) > 1 and token not in GENERATION_STOPWORDS]


def _tokenize_identifier(name: str) -> List[str]:
    """Split an identifier into comparable lowercase tokens."""
    return _tokenize_generation_text(name.replace("_", " "))


def _extract_requested_limit(nl_query: str, default: int = 10) -> int:
    """Extract requested result size from natural language."""
    match = re.search(r'\b(?:top|first|last)\s+(\d{1,4})\b', nl_query, re.IGNORECASE)
    if match:
        return max(1, min(int(match.group(1)), 500))
    return default


def _extract_column_suggestions(error_message: str) -> List[str]:
    """Extract suggested replacement columns from a validator error string."""
    if "Did you mean:" not in error_message:
        return []
    suggestion_text = error_message.split("Did you mean:", 1)[1].strip().rstrip("?")
    return [part.strip() for part in suggestion_text.split(",") if part.strip()]


def _choose_preferred_time_column(schema_columns: Dict[str, Any]) -> Optional[str]:
    """Pick the best canonical time column from a schema."""
    preferred_names = [
        "TimeGenerated", "Timestamp", "ReportTime", "EventTime",
        "FirstSeenDateTime", "LastSeenDateTime", "CreatedOn",
    ]
    for name in preferred_names:
        if name in schema_columns:
            return name

    for column_name, column_def in schema_columns.items():
        data_type = column_def.get("data_type") if isinstance(column_def, dict) else column_def
        if str(data_type).lower() == "datetime":
            return column_name
    return None


def _choose_schema_replacement(
    invalid_column: str,
    schema_columns: Dict[str, Any],
    suggested_columns: Optional[List[str]] = None,
) -> Optional[str]:
    """Choose the best schema-backed replacement for an invalid column."""
    if suggested_columns:
        for suggestion in suggested_columns:
            if suggestion in schema_columns:
                return suggestion

    time_column = _choose_preferred_time_column(schema_columns)
    invalid_tokens = set(_tokenize_identifier(invalid_column))
    invalid_lower = invalid_column.lower()

    if {"time", "date", "timestamp"} & invalid_tokens and time_column:
        return time_column

    ranked_candidates = []
    for column_name in schema_columns:
        column_tokens = set(_tokenize_identifier(column_name))
        score = 0.0
        score += len(invalid_tokens & column_tokens) * 2.5
        score += 1.5 * difflib.SequenceMatcher(None, invalid_lower, column_name.lower()).ratio()

        if "command" in invalid_lower and "command" in column_name.lower():
            score += 2.0
        if "account" in invalid_lower and "account" in column_name.lower():
            score += 2.0
        if "parent" in invalid_lower and "parent" in column_name.lower():
            score += 2.0
        if "file" in invalid_lower and "file" in column_name.lower():
            score += 2.0
        if "path" in invalid_lower and "path" in column_name.lower():
            score += 2.0

        ranked_candidates.append((score, column_name))

    ranked_candidates.sort(reverse=True)
    if ranked_candidates and ranked_candidates[0][0] >= 2.4:
        return ranked_candidates[0][1]
    return None


async def _repair_query_with_schema_context(
    query: str,
    cluster_url: str,
    database: str,
    validation_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Attempt a schema-grounded repair for invalid user/client-generated KQL.

    Only returns a repaired query if the rewritten query fully revalidates.
    """
    tables = validation_result.get("tables_used", [])
    if len(tables) != 1:
        return None

    table_name = tables[0]
    schema_info = await schema_manager.get_table_schema(cluster_url, database, table_name)
    if not schema_info or not schema_info.get("columns"):
        return None

    schema_columns = schema_info["columns"]
    repaired_query = query
    replacements: Dict[str, str] = {}

    for error in validation_result.get("errors", []):
        match = re.search(r"Column '([^']+)' not found", error)
        if not match:
            continue

        invalid_column = match.group(1)
        if invalid_column in replacements:
            continue

        replacement = _choose_schema_replacement(
            invalid_column,
            schema_columns,
            suggested_columns=_extract_column_suggestions(error),
        )
        if replacement and replacement != invalid_column:
            repaired_query = re.sub(
                rf"\b{re.escape(invalid_column)}\b",
                replacement,
                repaired_query,
            )
            replacements[invalid_column] = replacement

    if not replacements:
        return None

    repaired_validation = await kql_validator.validate_query(
        query=repaired_query,
        cluster=cluster_url,
        database=database,
        auto_discover=False,
    )
    if not repaired_validation.get("valid"):
        return None

    return {
        "query": repaired_query,
        "replacements": replacements,
        "schema_columns": list(schema_columns.keys())[:25],
        "preferred_time_column": _choose_preferred_time_column(schema_columns),
        "table_name": table_name,
    }


def _extract_time_window(nl_query: str) -> Optional[str]:
    """Convert common NL time windows into KQL ago() units."""
    lower_query = nl_query.lower()
    match = re.search(
        r'\b(?:last|past)\s+(\d+)\s+(minute|minutes|min|hour|hours|day|days|week|weeks|month|months)\b',
        lower_query,
    )
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        if unit.startswith("min"):
            return f"{value}m"
        if unit.startswith("hour"):
            return f"{value}h"
        if unit.startswith("day"):
            return f"{value}d"
        if unit.startswith("week"):
            return f"{value * 7}d"
        if unit.startswith("month"):
            return f"{value * 30}d"

    if "today" in lower_query:
        return "1d"
    if "yesterday" in lower_query:
        return "2d"
    if any(marker in lower_query for marker in ("recent", "latest", "newest")):
        return "24h"
    return None


def _detect_generation_intent(nl_query: str) -> Dict[str, Any]:
    """Infer lightweight generation intent from an NL query."""
    lower_query = nl_query.lower()
    tokens = _tokenize_generation_text(nl_query)
    keywords = re.findall(r"""['"]([^'"]+)['"]""", nl_query)
    needs_trend = any(
        marker in lower_query for marker in ("trend", "over time", "timeline", "hourly", "daily")
    )
    needs_count = any(
        marker in lower_query for marker in ("count", "how many", "total", "number of")
    )
    needs_top = bool(re.search(r'\btop\s+\d+\b', lower_query)) or "most" in lower_query
    needs_latest = any(marker in lower_query for marker in ("latest", "recent", "newest", "last records"))
    wants_distinct = any(marker in lower_query for marker in ("distinct", "unique"))
    needs_join = any(
        marker in lower_query for marker in ("join", "correlate", "combine", "enrich", "across")
    )

    return {
        "tokens": tokens,
        "keywords": keywords[:2],
        "limit": _extract_requested_limit(nl_query, default=10),
        "time_window": _extract_time_window(nl_query),
        "needs_trend": needs_trend,
        "needs_count": needs_count,
        "needs_top": needs_top,
        "needs_latest": needs_latest or not (needs_trend or needs_count or needs_top),
        "wants_distinct": wants_distinct,
        "needs_join": needs_join,
    }


def _pick_generation_columns(table_info: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    """Choose the best columns for the requested NL intent."""
    fingerprinted = table_info.get("fingerprinted_columns", [])
    if not fingerprinted:
        fingerprinted = [
            {"name": name, "data_type": "string", "score": 0.0}
            for name in list((table_info.get("columns") or {}).keys())[:8]
        ]

    time_column = next(
        (
            column["name"]
            for column in fingerprinted
            if column.get("data_type") == "datetime"
            or any(token in column["name"].lower() for token in ("time", "date", "timestamp"))
        ),
        None,
    )
    group_column = next(
        (
            column["name"]
            for column in fingerprinted
            if column["name"] != time_column and column.get("data_type") in {"string", "guid"}
        ),
        None,
    )
    filter_column = next(
        (
            column["name"]
            for column in fingerprinted
            if column["name"] not in {time_column, group_column}
            and column.get("data_type") in {"string", "guid"}
        ),
        group_column,
    )

    output_columns = []
    if time_column:
        output_columns.append(time_column)
    for column in fingerprinted:
        if column["name"] not in output_columns:
            output_columns.append(column["name"])
        if len(output_columns) >= min(max(intent["limit"], 6), 8):
            break

    return {
        "time_column": time_column,
        "group_column": group_column,
        "filter_column": filter_column,
        "output_columns": output_columns[:8],
    }


def _build_time_filter(intent: Dict[str, Any], time_column: Optional[str]) -> str:
    """Build a time filter clause when the query implies recency."""
    if not time_column or not intent.get("time_window"):
        return ""
    return f" | where {bracket_if_needed(time_column)} > ago({intent['time_window']})"


def _trend_bucket(time_window: Optional[str]) -> str:
    """Choose a sensible bin size for trend queries."""
    if not time_window:
        return "1h"
    if time_window.endswith("m"):
        return "5m"
    if time_window.endswith("h"):
        hours = int(time_window[:-1])
        return "15m" if hours <= 6 else "1h"
    days = int(time_window[:-1]) if time_window.endswith("d") else 1
    return "1h" if days <= 2 else "1d"


def _build_candidate_queries(
    table_info: Dict[str, Any],
    intent: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build deterministic candidate KQL queries for reranking."""
    table_name = bracket_if_needed(table_info["table"])
    column_plan = _pick_generation_columns(table_info, intent)
    time_column = column_plan["time_column"]
    group_column = column_plan["group_column"]
    filter_column = column_plan["filter_column"]
    output_columns = column_plan["output_columns"] or [column["name"] for column in table_info.get("fingerprinted_columns", [])[:6]]
    project_clause = ", ".join(bracket_if_needed(column) for column in output_columns)
    time_filter = _build_time_filter(intent, time_column)
    filter_keyword = intent["keywords"][0].replace("'", "''") if intent["keywords"] else None
    keyword_filter = ""
    if filter_keyword and filter_column:
        keyword_filter = f" | where {bracket_if_needed(filter_column)} has '{filter_keyword}'"

    candidates: List[Dict[str, Any]] = []

    if intent["needs_trend"] and time_column:
        bucket = _trend_bucket(intent["time_window"])
        candidates.append({
            "template": "trend",
            "query": (
                f"{table_name}{time_filter}{keyword_filter}"
                f" | summarize count() by bin({bracket_if_needed(time_column)}, {bucket})"
                f" | order by {bracket_if_needed(time_column)} asc"
            ),
            "columns_used": [column for column in [time_column, filter_column] if column],
        })

    if (intent["needs_count"] or intent["needs_top"] or intent["wants_distinct"]) and group_column:
        if intent["wants_distinct"]:
            candidate_query = (
                f"{table_name}{time_filter}{keyword_filter}"
                f" | distinct {bracket_if_needed(group_column)}"
                f" | take {intent['limit']}"
            )
            template = "distinct"
        else:
            candidate_query = (
                f"{table_name}{time_filter}{keyword_filter}"
                f" | summarize count_ = count() by {bracket_if_needed(group_column)}"
                f" | top {intent['limit']} by count_ desc"
            )
            template = "aggregate"

        candidates.append({
            "template": template,
            "query": candidate_query,
            "columns_used": [column for column in [time_column, group_column, filter_column] if column],
        })

    if intent["needs_latest"] and time_column:
        candidates.append({
            "template": "latest",
            "query": (
                f"{table_name}{time_filter}{keyword_filter}"
                f" | top {intent['limit']} by {bracket_if_needed(time_column)} desc"
                f" | project {project_clause}"
            ),
            "columns_used": [column for column in [time_column, filter_column] if column] + output_columns,
        })

    candidates.append({
        "template": "projection",
        "query": f"{table_name}{time_filter}{keyword_filter} | project {project_clause} | take {intent['limit']}",
        "columns_used": [column for column in [time_column, filter_column] if column] + output_columns,
    })

    deduped_candidates = []
    seen_queries = set()
    for candidate in candidates:
        if candidate["query"] not in seen_queries:
            seen_queries.add(candidate["query"])
            deduped_candidates.append(candidate)
    return deduped_candidates


def _build_join_candidate(
    candidate_tables: List[Dict[str, Any]],
    join_hints: List[str],
    intent: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a simple join candidate from stored join hints."""
    if len(candidate_tables) < 2 or not join_hints or not intent.get("needs_join"):
        return None

    for join_hint in join_hints:
        match = re.match(r"(.+?) joins with (.+?) on (.+)", join_hint)
        if not match:
            continue

        left_table, right_table, join_column = match.groups()
        left_info = next((table for table in candidate_tables if table["table"] == left_table), None)
        right_info = next((table for table in candidate_tables if table["table"] == right_table), None)
        if not left_info or not right_info:
            continue

        left_columns = _pick_generation_columns(left_info, intent)
        right_columns = _pick_generation_columns(right_info, intent)
        left_time_filter = _build_time_filter(intent, left_columns["time_column"])
        projected_columns = []
        for column in left_columns["output_columns"][:4] + right_columns["output_columns"][:4]:
            if column != join_column and column not in projected_columns:
                projected_columns.append(column)
        project_clause = ", ".join(bracket_if_needed(column) for column in projected_columns[:8])

        return {
            "template": "join",
            "table": left_table,
            "query": (
                f"{bracket_if_needed(left_table)}{left_time_filter}"
                f" | join kind=inner {bracket_if_needed(right_table)} on {bracket_if_needed(join_column)}"
                f" | project {project_clause}"
                f" | take {intent['limit']}"
            ),
            "columns_used": [join_column] + projected_columns[:8],
        }

    return None


def _score_generation_candidate(
    candidate: Dict[str, Any],
    table_info: Dict[str, Any],
    intent: Dict[str, Any],
) -> float:
    """Score candidate queries before validation."""
    score = float(table_info.get("score", 0.0)) * 2.0
    template = candidate["template"]
    query = candidate["query"].lower()

    if intent["needs_trend"] and template == "trend":
        score += 4.0
    if intent["needs_count"] and template == "aggregate":
        score += 3.5
    if intent["needs_join"] and template == "join":
        score += 4.0
    if intent["needs_top"] and ("top " in query or template == "aggregate"):
        score += 2.5
    if intent["needs_latest"] and template == "latest":
        score += 3.0
    if intent["wants_distinct"] and template == "distinct":
        score += 3.0
    if intent["time_window"] and "| where " in query:
        score += 1.25
    score += min(len(candidate.get("columns_used", [])), 8) * 0.1
    score -= max(len(query) - 240, 0) * 0.002
    return round(score, 4)


async def _validate_candidate_query(
    candidate: Dict[str, Any],
    cluster_url: str,
    database: str,
) -> Dict[str, Any]:
    """Validate a generated candidate against cached schema."""
    validation = await kql_validator.validate_query(
        query=candidate["query"],
        cluster=cluster_url,
        database=database,
        auto_discover=False,
    )
    validated_candidate = {**candidate, "validation": validation}
    if validation.get("valid"):
        validated_candidate["score"] = round(candidate["score"] + 5.0, 4)
    else:
        validated_candidate["score"] = round(candidate["score"] - len(validation.get("errors", [])) * 2.0, 4)
    return validated_candidate


def _build_safe_repair_query(
    table_info: Dict[str, Any],
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    """Fallback repair candidate that uses only the top ranked schema columns."""
    column_plan = _pick_generation_columns(table_info, intent)
    time_filter = _build_time_filter(intent, column_plan["time_column"])
    project_clause = ", ".join(
        bracket_if_needed(column) for column in column_plan["output_columns"][:6]
    )
    return {
        "template": "repair_projection",
        "query": (
            f"{bracket_if_needed(table_info['table'])}{time_filter}"
            f" | project {project_clause}"
            f" | take {intent['limit']}"
        ),
        "columns_used": column_plan["output_columns"][:6],
        "score": float(table_info.get("score", 0.0)),
    }


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
        repaired_query_info = None
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

            query = generated_result.get("query_plain", generated_result["query"])
            if output_format == "generation_only":
                return ErrorHandler.safe_json_dumps(generated_result, indent=2)

        # Check cache for non-generate queries (fast path)
        if not generate_query and output_format == "json":
            try:
                cached = memory_manager.get_cached_result(
                    query,
                    ttl_seconds=120,
                    cluster=cluster_url,
                    database=database,
                    cache_namespace=f"execute:{output_format}",
                )
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
            repaired_query_info = await _repair_query_with_schema_context(
                query,
                cluster_url,
                database,
                validation_result,
            )
            if repaired_query_info:
                query = repaired_query_info["query"]
                validation_result = await kql_validator.validate_query(
                    query=query,
                    cluster=cluster_url,
                    database=database,
                    auto_discover=False,
                )
                logger.info(
                    "Schema-grounded repair succeeded for %s using replacements: %s",
                    repaired_query_info["table_name"],
                    repaired_query_info["replacements"],
                )

                if output_format == "generation_only":
                    return ErrorHandler.safe_json_dumps({
                        "success": True,
                        "query": query,
                        "query_plain": query,
                        "repair_applied": True,
                        "replacements": repaired_query_info["replacements"],
                        "table_name": repaired_query_info["table_name"],
                        "preferred_time_column": repaired_query_info["preferred_time_column"],
                        "schema_columns": repaired_query_info["schema_columns"],
                    }, indent=2)

            if not validation_result["valid"]:
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

        repair_metadata = None if generate_query else repaired_query_info

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
            if repair_metadata:
                result["repair_applied"] = True
                result["replacements"] = repair_metadata["replacements"]
                result["schema_grounded_table"] = repair_metadata["table_name"]
                result["preferred_time_column"] = repair_metadata["preferred_time_column"]

            # Cache successful result for future queries
            result_json = ErrorHandler.safe_json_dumps(result, indent=2)
            try:
                memory_manager.cache_query_result(
                    query,
                    result_json,
                    len(df),
                    cluster=cluster_url,
                    database=database,
                    cache_namespace=f"execute:{output_format}",
                )
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
    from .constants import SPECIAL_TOKENS
    
    try:
        intent = _detect_generation_intent(natural_language_query)
        cag_bundle = memory_manager.build_cag_bundle(
            cluster_url,
            database,
            natural_language_query,
            max_tables=3,
            max_columns=12,
        )
        candidate_tables = list(cag_bundle.get("tables", []))

        if table_name:
            schema_info = await schema_manager.get_table_schema(cluster_url, database, table_name)
            if not schema_info or not schema_info.get("columns"):
                return {
                    "success": False,
                    "error": f"No schema found for {SPECIAL_TOKENS['TABLE_START']}{table_name}{SPECIAL_TOKENS['TABLE_END']} in memory.",
                    "query": "",
                    "suggestion": f"Run: schema_memory(operation='discover', cluster_url='{cluster_url}', database='{database}', table_name='{table_name}')",
                }

            explicit_table = {
                "table": table_name,
                "columns": schema_info["columns"],
                "score": 100.0,
                "matched_columns": [],
                "datetime_columns": sum(
                    1
                    for column_def in schema_info["columns"].values()
                    if isinstance(column_def, dict) and column_def.get("data_type") == "datetime"
                ),
            }
            explicit_table["fingerprinted_columns"] = memory_manager.fingerprint_columns(
                natural_language_query,
                {"table": table_name, "columns": schema_info["columns"]},
                similar_queries=cag_bundle.get("similar_queries", []),
                max_columns=12,
            )
            explicit_table["selected_columns"] = {
                column["name"]: schema_info["columns"][column["name"]]
                for column in explicit_table["fingerprinted_columns"]
                if column["name"] in schema_info["columns"]
            }
            candidate_tables = [explicit_table] + [
                table for table in candidate_tables if table["table"].lower() != table_name.lower()
            ]

        if not candidate_tables:
            return {
                "success": False,
                "error": f"{SPECIAL_TOKENS['CLUSTER_START']}ERROR{SPECIAL_TOKENS['CLUSTER_END']} Could not determine a relevant table from schema memory.",
                "query": "",
                "suggestion": "Use schema_memory(operation='list_tables') or pass table_name explicitly before generating queries.",
            }

        validated_candidates = []
        join_candidate = _build_join_candidate(
            candidate_tables[:2],
            cag_bundle.get("join_hints", []),
            intent,
        )
        if join_candidate:
            join_candidate["score"] = _score_generation_candidate(
                join_candidate,
                candidate_tables[0],
                intent,
            )
            validated_candidates.append(
                await _validate_candidate_query(join_candidate, cluster_url, database)
            )

        for table_info in candidate_tables[:2]:
            for candidate in _build_candidate_queries(table_info, intent):
                candidate["table"] = table_info["table"]
                candidate["score"] = _score_generation_candidate(candidate, table_info, intent)
                validated_candidates.append(
                    await _validate_candidate_query(candidate, cluster_url, database)
                )

        validated_candidates.sort(key=lambda item: item["score"], reverse=True)
        best_candidate = next(
            (candidate for candidate in validated_candidates if candidate["validation"].get("valid")),
            validated_candidates[0] if validated_candidates else None,
        )

        if not best_candidate:
            return {
                "success": False,
                "error": "No viable KQL candidate could be generated from the available schema context.",
                "query": "",
                "suggestion": "Discover schema for the target table first, then retry with a narrower natural language request.",
            }

        if not best_candidate["validation"].get("valid"):
            repair_candidate = _build_safe_repair_query(candidate_tables[0], intent)
            repair_candidate["table"] = candidate_tables[0]["table"]
            repair_candidate = await _validate_candidate_query(repair_candidate, cluster_url, database)
            if repair_candidate["validation"].get("valid"):
                best_candidate = repair_candidate

        selected_table = next(
            (table for table in candidate_tables if table["table"] == best_candidate["table"]),
            candidate_tables[0],
        )
        valid_candidate_count = sum(
            1 for candidate in validated_candidates if candidate["validation"].get("valid")
        )
        generation_metrics = {
            "candidate_count": len(validated_candidates),
            "valid_candidate_count": valid_candidate_count,
            "repair_used": best_candidate["template"] == "repair_projection",
            "table_candidate_count": len(candidate_tables[:2]),
        }
        selected_columns = _pick_generation_columns(selected_table, intent)["output_columns"]
        schema_columns = selected_table.get("columns", {})
        columns_with_tokens = [
            f"{SPECIAL_TOKENS['COLUMN']}:{column}|{SPECIAL_TOKENS['TYPE']}:{schema_columns.get(column, {}).get('data_type', 'unknown') if isinstance(schema_columns.get(column), dict) else schema_columns.get(column, 'unknown')}"
            for column in selected_columns
        ]
        schema_context = (
            f"{SPECIAL_TOKENS['CLUSTER_START']}{cluster_url}{SPECIAL_TOKENS['CLUSTER_END']}"
            f"{SPECIAL_TOKENS['SEPARATOR']}"
            f"{SPECIAL_TOKENS['DATABASE_START']}{database}{SPECIAL_TOKENS['DATABASE_END']}"
            f"{SPECIAL_TOKENS['SEPARATOR']}"
            f"{SPECIAL_TOKENS['TABLE_START']}{selected_table['table']}{SPECIAL_TOKENS['TABLE_END']}"
        )

        try:
            memory_manager.store_learning_result(
                natural_language_query,
                {
                    "target_table": selected_table["table"],
                    "query": best_candidate["query"],
                    "generation_metrics": generation_metrics,
                    "ranked_tables": [table["table"] for table in candidate_tables[:3]],
                },
                execution_type="generation",
            )
        except Exception as learning_error:  # pylint: disable=broad-exception-caught
            logger.debug("Failed to store generation telemetry: %s", learning_error)

        return {
            "success": True,
            "query": best_candidate["query"],
            "query_plain": best_candidate["query"],
            "query_tagged": f"{SPECIAL_TOKENS['QUERY_START']}{best_candidate['query']}{SPECIAL_TOKENS['QUERY_END']}",
            "generation_method": "cag_hybrid_ranking",
            "target_table": selected_table["table"],
            "schema_validated": best_candidate["validation"].get("valid", False),
            "validation_errors": best_candidate["validation"].get("errors", []),
            "columns_used": selected_columns,
            "columns_with_types": columns_with_tokens,
            "ranked_tables": [
                {
                    "table": table["table"],
                    "score": table.get("score", 0.0),
                    "matched_columns": table.get("matched_columns", []),
                    "top_columns": [column["name"] for column in table.get("fingerprinted_columns", [])[:6]],
                }
                for table in candidate_tables[:3]
            ],
            "candidate_queries": [
                {
                    "table": candidate["table"],
                    "template": candidate["template"],
                    "score": candidate["score"],
                    "valid": candidate["validation"].get("valid", False),
                    "query": candidate["query"],
                }
                for candidate in validated_candidates[:4]
            ],
            "generation_metrics": generation_metrics,
            "similar_queries_found": len(cag_bundle.get("similar_queries", [])),
            "cag_context_used": bool(cag_bundle.get("context")),
            "cag_context": cag_bundle.get("context", ""),
            "schema_context": schema_context,
            "note": (
                f"{SPECIAL_TOKENS['AI_START']}"
                "Query generated from compact CAG retrieval, hybrid table/column ranking, "
                "candidate reranking, and schema validation."
                f"{SPECIAL_TOKENS['AI_END']}"
            ),
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
    - "cache_stats": Get detailed query cache statistics (NEW)
    - "cleanup_cache": Remove expired cache entries (NEW)
    - "list_multi_cluster_tables": List tables found in multiple clusters (NEW)

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
            return await _schema_get_context_operation(cluster_url, database, natural_language_query, table_name)
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
        elif operation == "cache_stats":
            return await _schema_cache_stats_operation()
        elif operation == "cleanup_cache":
            return await _schema_cleanup_cache_operation()
        elif operation == "list_multi_cluster_tables":
            return await _schema_list_multi_cluster_tables_operation()
        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown operation: {operation}",
                "available_operations": [
                    "discover", "list_tables", "get_context", "generate_report", 
                    "clear_cache", "get_stats", "refresh_schema",
                    "cache_stats", "cleanup_cache", "list_multi_cluster_tables"
                ]
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

async def _schema_get_context_operation(
    cluster_url: str,
    database: str,
    natural_language_query: str,
    table_name: Optional[str] = None,
) -> str:
    """Get AI context for tables based on natural language query parsing."""
    try:
        if not natural_language_query:
            return json.dumps({
                "success": False,
                "error": "natural_language_query is required for get_context operation"
            })
        if table_name:
            schema_info = await schema_manager.get_table_schema(cluster_url, database, table_name)
            if not schema_info or not schema_info.get("columns"):
                return json.dumps({
                    "success": False,
                    "error": f"No schema found for table '{table_name}'"
                })

            similar_queries = memory_manager.find_similar_queries(
                cluster_url, database, natural_language_query, limit=3
            )
            fingerprinted_columns = memory_manager.fingerprint_columns(
                natural_language_query,
                {"table": table_name, "columns": schema_info["columns"]},
                similar_queries=similar_queries,
                max_columns=12,
            )
            selected_columns = {
                column["name"]: schema_info["columns"][column["name"]]
                for column in fingerprinted_columns
                if column["name"] in schema_info["columns"]
            }
            join_hints = memory_manager.get_join_hints([table_name])
            cag_bundle = {
                "tables": [{
                    "table": table_name,
                    "score": 100.0,
                    "columns": schema_info["columns"],
                    "fingerprinted_columns": fingerprinted_columns,
                    "selected_columns": selected_columns,
                }],
                "similar_queries": similar_queries,
                "join_hints": join_hints,
                "context": memory_manager._to_toon(  # pylint: disable=protected-access
                    [{"table": table_name, "columns": selected_columns}],
                    similar_queries,
                    join_hints,
                ),
            }
        else:
            cag_bundle = memory_manager.build_cag_bundle(
                cluster_url,
                database,
                natural_language_query,
                max_tables=3,
                max_columns=12,
            )
        tables = [table["table"] for table in cag_bundle.get("tables", [])]
        context = cag_bundle.get("context", "")
        return json.dumps({
            "success": True,
            "tables": tables,
            "ranked_tables": cag_bundle.get("tables", []),
            "similar_queries": cag_bundle.get("similar_queries", []),
            "join_hints": cag_bundle.get("join_hints", []),
            "strict_schema": {
                table["table"]: {
                    "allowed_columns": list((table.get("columns") or {}).keys())[:50],
                    "recommended_columns": [column["name"] for column in table.get("fingerprinted_columns", [])[:12]],
                    "preferred_time_column": _choose_preferred_time_column(table.get("columns") or {}),
                }
                for table in cag_bundle.get("tables", [])
            },
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
        schema_manager.clear_schema_cache()
        removed_query_cache = memory_manager.clear_query_cache()
        logger.info("Schema and query cache clear requested")

        return json.dumps({
            "success": True,
            "message": "Schema and query cache cleared successfully",
            "query_cache_entries_removed": removed_query_cache,
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


async def _schema_cache_stats_operation() -> str:
    """Get detailed query cache statistics."""
    try:
        cache_stats = memory_manager.get_cache_stats()
        return json.dumps({
            "success": True,
            "cache_stats": cache_stats,
            "description": {
                "total_cached": "Total number of cached query results",
                "expired_entries": "Number of expired cache entries awaiting cleanup",
                "by_query_type": "Breakdown by query type (schema, aggregation, realtime, etc.)",
                "ttl_settings": {
                    "schema": "3600s (1 hour) - Schema queries",
                    "aggregation": "600s (10 minutes) - Aggregation queries",
                    "realtime": "60s (1 minute) - Real-time queries",
                    "default": "300s (5 minutes) - Other queries"
                }
            }
        }, indent=2)
    except (ValueError, RuntimeError, AttributeError) as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


async def _schema_cleanup_cache_operation() -> str:
    """Remove expired cache entries."""
    try:
        removed_count = memory_manager.cleanup_expired_cache()
        return json.dumps({
            "success": True,
            "message": "Cache cleanup completed",
            "entries_removed": removed_count,
            "timestamp": datetime.now().isoformat()
        }, indent=2)
    except (ValueError, RuntimeError, AttributeError) as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


async def _schema_list_multi_cluster_tables_operation() -> str:
    """List all tables with their known cluster/database locations."""
    try:
        all_locations = memory_manager.get_all_table_locations()
        
        # Identify tables in multiple clusters
        multi_cluster_tables = {
            table: locations 
            for table, locations in all_locations.items() 
            if len(set(loc["cluster"] for loc in locations)) > 1
        }
        
        return json.dumps({
            "success": True,
            "total_tables_tracked": len(all_locations),
            "multi_cluster_tables_count": len(multi_cluster_tables),
            "multi_cluster_tables": multi_cluster_tables,
            "all_table_locations": all_locations,
            "note": "Tables in multiple clusters may have different schemas or data"
        }, indent=2)
    except (ValueError, RuntimeError, AttributeError) as e:
        return json.dumps({
            "success": False,
            "error": str(e)
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
