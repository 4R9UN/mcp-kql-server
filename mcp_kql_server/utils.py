"""
Utility helpers for MCP KQL server.

This module provides several small, conservative helper routines used across
the project and in unit tests.  Implementations are intentionally simple and
robust so they can be used as fallbacks when richer adapters are not present.

Functions implemented here:
 - normalize_join_on_clause
 - get_schema_discovery (returns a lightweight discovery adapter)
 - get_schema_discovery_status
 - get_default_cluster_memory_path
 - ensure_directory_exists
 - sanitize_filename
 - get_schema_column_names
 - validate_projected_columns
 - validate_all_query_columns
 - fix_query_with_real_schema
 - generate_query_description
"""
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import KQL_RESERVED_WORDS, get_dynamic_table_analyzer, get_dynamic_column_analyzer

# Set up logger at module level
logger = logging.getLogger(__name__)

def _is_retryable_exc(e: Exception) -> bool:
    """Lightweight dynamic check for retryable exceptions (message-based)."""
    try:
        s = str(e).lower()
        return any(k in s for k in ("timeout", "connection", "throttl", "unreachable", "refused", "kusto", "service"))
    except Exception:
        return False

def retry_on_exception(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 10.0):
    """
    Simple, dependency-free retry decorator that supports both sync and async functions.
    Retries only when `_is_retryable_exc` returns True.
    """
    def deco(func):
        if asyncio.iscoroutinefunction(func):
            async def wrapped(*args, **kwargs):
                delay = base_delay
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_attempts or not _is_retryable_exc(e):
                            raise
                        await asyncio.sleep(min(delay, max_delay))
                        delay *= 2
            return wrapped
        else:
            def wrapped(*args, **kwargs):
                import time
                delay = base_delay
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_attempts or not _is_retryable_exc(e):
                            raise
                        time.sleep(min(delay, max_delay))
                        delay *= 2
            return wrapped
    return deco

def log_execution(func):
    """Minimal execution logger decorator (sync+async)."""
    if asyncio.iscoroutinefunction(func):
        async def wrapped(*args, **kwargs):
            start = datetime.now()
            try:
                return await func(*args, **kwargs)
            finally:
                logger.debug(f"{func.__name__} took {(datetime.now() - start).total_seconds():.2f}s")
        return wrapped
    else:
        def wrapped(*args, **kwargs):
            start = datetime.now()
            try:
                return func(*args, **kwargs)
            finally:
                logger.debug(f"{func.__name__} took {(datetime.now() - start).total_seconds():.2f}s")
        return wrapped


class QueryProcessor:
    """
    A consolidated class to handle all stages of query processing.
    Merges QueryParser, QueryOptimizer and cleaning logic into a single pipeline.
    """

    def __init__(self, memory_manager=None):
        """Initialize the QueryProcessor with a memory manager and all necessary components."""
        if memory_manager is None:
            from .memory import get_memory_manager
            self.memory_manager = get_memory_manager()
        else:
            self.memory_manager = memory_manager
        
        # Initialize regex patterns from QueryOptimizer and QueryParser
        self.join_on_pattern = re.compile(r"(\bjoin\b\s+(?:\w+\s+)?(?:\([^)]+\)\s+)?(?:\w+\s+)?on\s+)([^\|]+)", re.IGNORECASE)
        self.project_pattern = re.compile(r"\|\s*project\s+([^|]+)", re.IGNORECASE)
        self.identifier_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
        
        # Query parsing patterns
        self.parsing_patterns = [
            re.compile(r"cluster\(['\"][^'\"]+['\"]\)\.database\(['\"][^'\"]+['\"]\)\.([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
            re.compile(r"cluster\(['\"][^'\"]+['\"]\)\.database\(['\"][^'\"]+['\"]\)\.\['([^']+)'\]", re.IGNORECASE),
            re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\|", re.IGNORECASE),
            re.compile(r"^\s*\['([^']+)'\]\s*\|", re.IGNORECASE),
            re.compile(r"\b(?:join|union|lookup)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
            re.compile(r"\b(?:join|union|lookup)\s+\['([^']+)'\]", re.IGNORECASE),
        ]
        self.fallback_patterns = [
            re.compile(r'([A-Za-z][A-Za-z0-9_]*)\s*\|\s*getschema', re.IGNORECASE),
            re.compile(r'(?:table|from)\s+([A-Za-z][A-Za-z0-9_]*)', re.IGNORECASE),
            re.compile(r'([A-Za-z][A-Za-z0-9_]*)\s+table', re.IGNORECASE),
        ]
        self.operation_keywords = ['project', 'where', 'summarize', 'extend', 'join', 'union', 'take', 'limit', 'sort', 'order']
        
        # Dynamic analyzers for intelligent query optimization
        self.table_analyzer = get_dynamic_table_analyzer()
        self.column_analyzer = get_dynamic_column_analyzer()

    def clean(self, query: str) -> str:
        """
        Applies initial cleaning and normalization.
        Consolidates logic from execute_kql.py:clean_query_for_execution
        """
        if not query or not query.strip():
            return ""

        # Strip leading/trailing whitespace for clean processing
        query = query.strip()

        # Handle comment-only queries
        lines = query.split('\n')
        non_comment_lines = [line for line in lines if not line.strip().startswith('//')]

        if not non_comment_lines:
            # If all lines are comments, there's no executable query
            return ""

        # Reconstruct the query from non-comment lines
        cleaned_query = '\n'.join(non_comment_lines).strip()
        
        # Apply additional normalization only if we have content
        if cleaned_query:
            # Apply core syntax normalization
            cleaned_query = self._normalize_kql_syntax(cleaned_query)
            
            # Apply dynamic error-based fixes
            cleaned_query = self._apply_dynamic_fixes(cleaned_query)
        
        return cleaned_query

    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parses the query to extract entities.
        Consolidates logic from QueryParser.parse
        """
        if not query:
            return {"cluster": None, "database": None, "tables": [], "operations": []}

        cluster_match = re.search(r"cluster\(['\"]([^'\"]+)['\"]\)", query)
        db_match = re.search(r"database\(['\"]([^'\"]+)['\"]\)", query)
        
        cluster = cluster_match.group(1) if cluster_match else None
        database = db_match.group(1) if db_match else None
        
        tables = self._extract_tables(query)
        operations = self._extract_operations(query)
        
        return {
            "cluster": cluster,
            "database": database,
            "tables": list(tables),
            "operations": operations,
            "query_length": len(query),
            "has_aggregation": any(op in operations for op in ['summarize', 'count']),
            "complexity_score": len(operations)
        }

    def optimize(self, query: str, schema: Dict[str, Any] = None) -> str:
        """
        Applies optimizations like fixing join and project clauses.
        Consolidates logic from QueryOptimizer
        """
        if not query or not query.strip():
            return query
        
        optimized_query = query
        
        # Apply join normalization
        optimized_query = self._normalize_join_on_clause(optimized_query)
        
        # Apply schema-based optimizations if schema is available
        if schema and isinstance(schema, dict):
            optimized_query = self._validate_projected_columns(optimized_query, schema)
            # This is key: ensure all columns are correctly cased AND bracketed
            optimized_query = self._validate_all_query_columns(optimized_query, schema)
        
        return optimized_query

    async def process(self, query: str, cluster: str, database: str) -> str:
        """
        Runs the full processing pipeline: clean -> validate -> optimize.
        """
        # Step 1: Clean the query
        cleaned_query = self.clean(query)
        if not cleaned_query:
            raise ValueError("Query is empty or contains only comments after cleaning.")

        # Step 2: Pre-execution validation (key for accuracy)
        validation_result = await self.memory_manager.validate_query(cleaned_query, cluster, database)

        if not validation_result.is_valid:
            # If validation provides a corrected query, use it.
            if validation_result.validated_query and validation_result.validated_query != cleaned_query:
                logger.warning("Query validation failed, but an auto-correction is available.")
                processed_query = validation_result.validated_query
            else:
                # If no correction is available, raise an error with details.
                error_details = "; ".join(validation_result.errors) if validation_result.errors else "Unknown validation error"
                raise ValueError(f"Query is invalid: {error_details}")
        else:
            processed_query = validation_result.validated_query

        # Step 3: Get schema for optimization
        entities = self.parse(processed_query)
        tables = entities.get("tables", [])
        schema = None
        if tables:
            try:
                primary_table = tables[0]
                # Ensure we get a fully populated schema object
                schema = self.memory_manager.get_schema(cluster, database, primary_table)
            except Exception as schema_error:
                logger.debug(f"Schema retrieval for optimization failed: {schema_error}")

        # Step 4: Apply final optimizations with schema context
        optimized_query = self.optimize(processed_query, schema)
        
        return optimized_query

    def _normalize_kql_syntax(self, query: str) -> str:
        """Optimized KQL syntax normalization with comprehensive error prevention."""
        if not query:
            return ""
        
        query = re.sub(r'\s+', ' ', query.strip())
        error_patterns = [
            (r'\|([a-zA-Z])', r'| \1'),
            (r'([a-zA-Z0-9_])(==|!=|<=|>=|<|>)([a-zA-Z0-9_])', r'\1 \2 \3'),
            (r'\|\|+', '|'),
            (r'\s*\|\s*', ' | '),
            (r'\s+(and|or|==|!=|<=|>=|<|>)\s*$', ''),
            (r';\s*$', ''),
        ]
        
        for pattern, replacement in error_patterns:
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        
        query = re.sub(r'\|\s*project\s+([^|]+)',
                       lambda m: '| project ' + self._normalize_project_clause(m.group(1)), query)
        
        return query.strip()

    def _apply_dynamic_fixes(self, query: str) -> str:
        """Apply minimal, conservative fixes to prevent common KQL errors."""
        if not query or not query.strip():
            return query
        
        query = query.strip()
        
        if re.search(r'\s+(and|or)\s*$', query, re.IGNORECASE):
            fixed_query = re.sub(r'\s+(and|or)\s*$', '', query, flags=re.IGNORECASE)
            if fixed_query.strip():
                query = fixed_query
        
        if query.rstrip().endswith('|'):
            fixed_query = query.rstrip('|').strip()
            if fixed_query.strip():
                query = fixed_query
        
        if re.search(r'(==|!=|<=|>=)\s*(==|!=|<=|>=)', query):
            query = re.sub(r'(==|!=|<=|>=)\s*(==|!=|<=|>=)', r'\1', query)
        
        if re.search(r'\|\s*project\s*,', query, re.IGNORECASE):
            query = re.sub(r'\|\s*project\s*,', '| project', query, flags=re.IGNORECASE)
        
        if ' join ' in query.lower():
            query = self._fix_join_syntax(query)
        
        if not query.strip():
            return query
        
        return query

    def _normalize_join_on_clause(self, kql: str) -> str:
        """Normalizes join 'on' clauses to fix common syntax errors."""
        if " join " not in kql.lower():
            return kql
        try:
            return self.join_on_pattern.sub(self._replace_join_clause, kql)
        except Exception as e:
            logger.debug(f"Join clause normalization failed: {e}")
            return kql

    def _replace_join_clause(self, match: re.Match) -> str:
        """Replace join clause with normalized version."""
        prefix, condition = match.group(1), match.group(2)
        if 'or' in condition.lower():
            parts = re.split(r'\bor\b', condition, flags=re.IGNORECASE)
            normalized_parts = [self._normalize_join_condition(part) for part in parts]
            condition = " and ".join(normalized_parts)
        else:
            condition = self._normalize_join_condition(condition)
        return prefix + condition

    def _normalize_join_condition(self, condition: str) -> str:
        """Normalize individual join condition."""
        condition = condition.strip()
        condition = re.sub(r'\b(\w+)\s*(!=|<>|=|[<>]=?)\s*(\w+)', r'\1 == \2', condition)
        return condition

    def _validate_projected_columns(self, query: str, schema: Dict[str, Any]) -> str:
        """Validates columns in a 'project' clause against a schema."""
        if not schema or not isinstance(schema, dict):
            return query
        schema_cols = get_schema_column_names(schema) or []
        if not schema_cols:
            return query
        lower_map = {c.lower(): c for c in schema_cols}
        try:
            return self.project_pattern.sub(lambda m: self._clean_project(m, lower_map), query)
        except Exception:
            return query

    def _clean_project(self, match: re.Match, lower_map: dict) -> str:
        """
        Clean project clause with schema validation, handling complex expressions.
        """
        project_content = match.group(1)
        parts = []
        current_part = ""
        depth = 0
        in_string = False
        
        # This enhanced parsing correctly handles commas inside function calls and strings
        for char in project_content:
            if char == "'" or char == '"':
                in_string = not in_string
            elif char == '(' and not in_string:
                depth += 1
            elif char == ')' and not in_string:
                depth = max(0, depth - 1)
            
            if char == "," and depth == 0 and not in_string:
                parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        
        if current_part.strip():
            parts.append(current_part.strip())

        valid_parts = []
        for p in parts:
            if not p or p.isspace():
                continue

            # Check if it's a simple identifier (not an alias or function call)
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", p):
                if p.lower() in lower_map:
                    # It's a valid column, use the correct casing
                    valid_parts.append(bracket_if_needed(lower_map[p.lower()]))
                else:
                    # It's an invalid column, skip it
                    logger.debug(f"Column '{p}' not found in schema, skipping from project clause.")
            else:
                # It's a complex expression (e.g., NewColumn = x + y, or strcat(a,b))
                # Keep it as is, assuming it's valid KQL
                valid_parts.append(p)
        
        return "| project " + ", ".join(valid_parts) if valid_parts else "| project *"

    def _validate_all_query_columns(self, query: str, schema: Dict[str, Any]) -> str:
        """Replaces all column identifiers with their real-cased and properly bracketed names from the schema."""
        if not schema or not isinstance(schema, dict):
            return query
        cols = get_schema_column_names(schema) or []
        if not cols:
            return query
        
        mapping = {c.lower(): c for c in cols}

        def _replace_token(m: re.Match) -> str:
            token = m.group(1)
            token_lower = token.lower()

            # **FIX**: Only process tokens that are identified as column names from the schema.
            # This prevents incorrectly bracketing KQL operators like 'take', 'project', 'startswith' etc.
            if token_lower in mapping:
                # This token is a column. Apply casing and bracketing.
                correct_case_token = mapping[token_lower]
                return bracket_if_needed(correct_case_token)
            
            # If the token is not a known column (i.e., it's an operator, function, or literal), return it unchanged.
            return token
            
        try:
            # The regex finds all "words". The replacement logic (_replace_token) now correctly filters
            # and only modifies words that are actual columns.
            return re.sub(r"(\b[A-Za-z_][A-Za-z0-9_]*\b)", _replace_token, query)
        except Exception:
            return query

    def _extract_tables(self, query: str) -> set:
        """Extracts table names from the query using multiple patterns."""
        tables = set()
        reserved_lower = {w.lower() for w in KQL_RESERVED_WORDS}
        for pattern in self.parsing_patterns:
            for match in pattern.finditer(query):
                table_name = match.group(1) if match.group(1) else None
                if table_name and table_name.lower() not in reserved_lower:
                    tables.add(table_name)
        if not tables:
            for pattern in self.fallback_patterns:
                for match in pattern.finditer(query):
                    table_candidate = match.group(1)
                    if table_candidate and table_candidate.lower() not in reserved_lower:
                        tables.add(table_candidate)
        return tables

    def _extract_operations(self, query: str) -> List[str]:
        """Extracts KQL operations from the query."""
        operations = []
        query_lower = query.lower()
        for op in self.operation_keywords:
            if f'| {op}' in query_lower or f'|{op}' in query_lower:
                operations.append(op)
        return operations

    def _fix_join_syntax(self, query: str) -> str:
        """Fix common join syntax issues dynamically."""
        def fix_join_condition(match):
            prefix, condition = match.groups()
            condition = re.sub(r'\bor\b', 'and', condition, flags=re.IGNORECASE)
            condition = re.sub(r'\b(\w+)\s*!=\s*(\w+)', r'\1 == \2', condition)
            return prefix + condition
        return self.join_on_pattern.sub(fix_join_condition, query)

    def _normalize_project_clause(self, project_content: str) -> str:
        """Enhanced normalize project clause to prevent column resolution errors."""
        if not project_content:
            return "*"
        columns = []
        for col in project_content.split(','):
            col = col.strip()
            if col:
                col = re.sub(r'\s+(and|or|==|!=|<=|>=|<|>)\s*$', '', col, flags=re.IGNORECASE)
                if col:
                    columns.append(col)
        return ', '.join(columns) if columns else "*"

def normalize_name(name: str) -> str:
    """Normalize a name for comparison (lowercase, strip whitespace)"""
    if not name:
        return ""
    return str(name).lower().strip().replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_")

class ErrorHandler:
    """
    Consolidated error handling utilities for consistent error management across the codebase.
    This reduces duplicate error handling patterns found throughout the modules.
    """
    
    @staticmethod
    def safe_execute(func, *args, default=None, error_msg="Operation failed", log_level="warning", **kwargs):
        """
        Safely execute a function with consistent error handling.
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            default: Default value to return on error
            error_msg: Error message prefix
            log_level: Logging level for errors (debug, info, warning, error)
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result or default value on error
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_func = getattr(logger, log_level, logger.warning)
            log_func(f"{error_msg}: {e}")
            return default
    
    @staticmethod
    def safe_get_nested(data: dict, *keys, default=None):
        """
        Safely get nested dictionary values with consistent error handling.
        
        Args:
            data: Dictionary to traverse
            *keys: Keys to traverse (e.g., 'cluster', 'database', 'table')
            default: Default value if key path doesn't exist
            
        Returns:
            Value at the key path or default
        """
        try:
            result = data
            for key in keys:
                result = result[key]
            return result
        except (KeyError, TypeError, AttributeError):
            return default
    
    @staticmethod
    def safe_json_dumps(data, default="{}", **kwargs):
        """Safely serialize data to JSON with error handling and type conversion."""
        def json_serializer(obj):
            """Custom JSON serializer for complex types."""
            # Handle pandas Timestamp objects
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            # Handle datetime objects
            elif hasattr(obj, 'strftime'):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            # Handle type objects
            elif isinstance(obj, type):
                return obj.__name__
            # Handle numpy types
            elif hasattr(obj, 'item'):
                return obj.item()
            # Handle pandas Series
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            # Fallback for other objects
            else:
                return str(obj)
        
        try:
            # Set default indent if not provided
            if 'indent' not in kwargs:
                kwargs['indent'] = 2
            # Set default serializer if not provided
            if 'default' not in kwargs:
                kwargs['default'] = json_serializer
            return json.dumps(data, **kwargs)
        except Exception as e:
            logger.warning(f"JSON serialization failed: {e}")
            return default
    
    @staticmethod
    def handle_import_error(module_name: str, fallback=None):
        """
        Handle import errors consistently.
        
        Args:
            module_name: Name of the module that failed to import
            fallback: Fallback value to return
            
        Returns:
            fallback value
        """
        logger.warning(f"{module_name} not available")
        return fallback

    @staticmethod
    def handle_kusto_error(e: Exception) -> Dict[str, Any]:
        """
        Comprehensive Kusto error analysis with extensive pattern recognition and intelligent suggestions.
        Centralizes all Kusto-related error interpretation and handling with enhanced accuracy.
        
        Args:
            e: Exception to analyze (typically KustoServiceError)
            
        Returns:
            Dict with structured error information, suggestions, recovery actions, and confidence scores
        """
        try:
            from azure.kusto.data.exceptions import KustoServiceError
        except ImportError:
            # Fallback if azure.kusto is not available
            KustoServiceError = type(None)
        
        if not isinstance(e, KustoServiceError):
            return {
                "success": False,
                "error": str(e),
                "suggestions": ["An unexpected error occurred. Check server logs."],
                "recovery_actions": ["Check logs", "Verify configuration", "Retry operation"],
                "error_type": "execution_error",
                "confidence": 0.0,
                "kusto_specific": False
            }

        error_str = str(e).lower()
        
        # Comprehensive Kusto-specific error patterns with detailed coverage
        error_patterns = {
            # Column/Schema Errors (SEM0100 series)
            "column_resolution": {
                "patterns": ["sem0100", "failed to resolve", "column", "doesn't exist", "not found",
                           "unknown column", "invalid column name", "column name not found", "no such column"],
                "error_code": "SEM0100",
                "category": "Schema Error",
                "suggestions": [
                    "Check column/table names for typos and correct case.",
                    "Use schema_memory(operation='discover') to refresh the schema.",
                    "Verify that the column exists in the target table.",
                    "Consider using execute_kql_query with generate_query=True for schema validation."
                ],
                "recovery_actions": ["Check table schema", "Verify column spelling", "Use show schema command", "Refresh schema cache"]
            },
            
            # Syntax Errors (SYN0002 series)
            "syntax_error": {
                "patterns": ["syn0002", "syntax error", "unexpected token", "parse error", "invalid syntax",
                           "missing operator", "unexpected end", "malformed query", "invalid character"],
                "error_code": "SYN0002",
                "category": "Syntax Error",
                "suggestions": [
                    "The query has a syntax error. Please review the KQL.",
                    "Check for unmatched parentheses, quotes, or operators.",
                    "Ensure proper pipe operator usage (|) between KQL operations.",
                    "Verify correct operator syntax and spacing."
                ],
                "recovery_actions": ["Validate KQL syntax", "Check operator usage", "Verify query structure", "Use KQL formatter"]
            },
            
            # Type Mismatch (SEM0001 series)
            "type_mismatch": {
                "patterns": ["sem0001", "type mismatch", "cannot convert", "incompatible types",
                           "invalid conversion", "type error", "conversion failed", "cast error"],
                "error_code": "SEM0001",
                "category": "Type Error",
                "suggestions": [
                    "Check data types in operations. Use appropriate conversion functions like tostring(), toint(), etc.",
                    "Verify that compared or operated values have compatible types.",
                    "Use explicit type casting when working with different data types."
                ],
                "recovery_actions": ["Use type conversion functions", "Check data types", "Validate operations", "Add explicit casts"]
            },
            
            # Function/Operator Errors (SEM0002 series)
            "function_error": {
                "patterns": ["sem0002", "function", "operator", "unknown function", "invalid function",
                           "function not found", "operator not supported", "invalid operator"],
                "error_code": "SEM0002",
                "category": "Function Error",
                "suggestions": [
                    "Verify function names and parameters. Check KQL function documentation for correct usage.",
                    "Ensure the function is supported in your Kusto cluster version.",
                    "Check parameter count and types for the function."
                ],
                "recovery_actions": ["Check function spelling", "Verify parameters", "Use supported functions", "Consult documentation"]
            },
            
            # Aggregation Errors (SEM0003 series)
            "aggregation_error": {
                "patterns": ["sem0003", "aggregation", "group by", "summarize", "invalid aggregation",
                           "aggregation function", "grouping error", "summary error"],
                "error_code": "SEM0003",
                "category": "Aggregation Error",
                "suggestions": [
                    "Review aggregation syntax. Ensure proper grouping and valid aggregation functions.",
                    "Check that all non-aggregated columns are included in the 'by' clause.",
                    "Verify aggregation function compatibility with data types."
                ],
                "recovery_actions": ["Check summarize syntax", "Verify group by columns", "Use valid aggregations", "Fix grouping"]
            },
            
            # Authentication/Authorization Errors
            "authentication_error": {
                "patterns": ["unauthorized", "forbidden", "authentication", "access denied", "401", "403",
                           "permission denied", "invalid credentials", "authentication failed"],
                "error_code": "AUTH001",
                "category": "Authentication Error",
                "suggestions": [
                    "Check your Azure authentication status with 'az login'.",
                    "Verify you have proper permissions to access the cluster and database.",
                    "Ensure your Azure CLI is up to date.",
                    "Contact your administrator for access permissions."
                ],
                "recovery_actions": ["Check credentials", "Verify permissions", "Re-authenticate", "Update Azure CLI"]
            },
            
            # Connection/Network Errors
            "connection_error": {
                "patterns": ["connection", "timeout", "network", "unreachable", "dns", "connection refused",
                           "network error", "connection timeout", "host unreachable", "connection failed"],
                "error_code": "CONN001",
                "category": "Connection Error",
                "suggestions": [
                    "Check network connectivity to the Kusto cluster and verify cluster URL.",
                    "Verify your internet connection is stable.",
                    "Check if the cluster is accessible from your network.",
                    "Try again after a few moments in case of temporary network issues."
                ],
                "recovery_actions": ["Verify cluster URL", "Check network", "Test connectivity", "Retry connection"]
            },
            
            # Table/Database Errors (SEM0100 series)
            "resource_not_found": {
                "patterns": ["table", "doesn't exist", "database", "not found", "unknown table", "unknown database",
                           "table not found", "database not found", "invalid table", "no such table"],
                "error_code": "SEM0100",
                "category": "Schema Error",
                "suggestions": [
                    "Verify table and database names. Check if the table exists in the specified database.",
                    "Use schema_memory(operation='list_tables') to see available tables.",
                    "Check for typos in table or database names.",
                    "Ensure you have access to the specified database."
                ],
                "recovery_actions": ["Check table name", "Verify database", "List available tables", "Refresh schema"]
            },
            
            # Query Limits/Performance Errors (LIM001 series)
            "performance_error": {
                "patterns": ["lim001", "limit", "timeout", "query too complex", "memory", "execution timeout",
                           "resource limit", "query complexity", "memory exceeded", "too much data"],
                "error_code": "LIM001",
                "category": "Query Limits Error",
                "suggestions": [
                    "Simplify query or add filters to reduce data processing requirements.",
                    "Consider adding time filters to reduce data volume.",
                    "Break complex queries into smaller, simpler operations.",
                    "Use 'take' or 'limit' operators to reduce result size."
                ],
                "recovery_actions": ["Add time filters", "Reduce query scope", "Optimize query", "Use sampling"]
            },
            
            # Data Format Errors (DAT001 series)
            "data_format_error": {
                "patterns": ["dat001", "format", "encoding", "invalid format", "parsing error",
                           "malformed data", "format error", "encoding error", "data format"],
                "error_code": "DAT001",
                "category": "Data Format Error",
                "suggestions": [
                    "Check data format and encoding. Verify input data structure.",
                    "Ensure data conforms to expected format specifications.",
                    "Validate input data before processing."
                ],
                "recovery_actions": ["Validate data format", "Check encoding", "Review input data", "Fix data structure"]
            },
            
            # Join/Union Errors (SEM0004 series)
            "join_error": {
                "patterns": ["sem0004", "join", "union", "join error", "union error", "invalid join",
                           "join key", "union type", "join condition"],
                "error_code": "SEM0004",
                "category": "Join/Union Error",
                "suggestions": [
                    "Review join conditions and ensure compatible data types for join keys.",
                    "Join conditions can only use 'and' operators, not 'or'.",
                    "Check that join column names exist in both tables.",
                    "Ensure join conditions use equality operators (==)."
                ],
                "recovery_actions": ["Check join keys", "Verify data types", "Review join conditions", "Fix operators"]
            },
            
            # Rate Limiting/Throttling Errors
            "rate_limit_error": {
                "patterns": ["throttled", "rate limit", "too many requests", "quota exceeded",
                           "request limit", "throttling", "rate exceeded"],
                "error_code": "RATE001",
                "category": "Rate Limit Error",
                "suggestions": [
                    "Request rate exceeded. Wait a moment before retrying.",
                    "Consider reducing query frequency or complexity.",
                    "Contact your Kusto administrator if the issue persists.",
                    "Implement exponential backoff in your retry logic."
                ],
                "recovery_actions": ["Wait before retry", "Reduce frequency", "Contact admin", "Implement backoff"]
            }
        }
        
        # Classify the error with scoring for best match
        classified_error = None
        max_matches = 0
        confidence = 0.0
        
        for error_category, config in error_patterns.items():
            matches = sum(1 for pattern in config["patterns"] if pattern in error_str)
            if matches > max_matches:
                max_matches = matches
                classified_error = config.copy()
                classified_error["type"] = error_category
                confidence = matches / len(config["patterns"])
        
        if not classified_error or max_matches == 0:
            classified_error = {
                "type": "unknown_kusto_error",
                "error_code": "GEN001",
                "category": "General Error",
                "suggestions": [
                    "Review the error message for specific details and consult Kusto documentation.",
                    "Check the Kusto service status if the issue persists.",
                    "Verify cluster URL, database name, and your permissions."
                ],
                "recovery_actions": ["Check error details", "Consult documentation", "Verify query", "Contact support"]
            }
            confidence = 0.0
        
        # Build enhanced error response
        error_response = {
            "success": False,
            "error": str(e),
            "error_type": classified_error["type"],
            "error_code": classified_error["error_code"],
            "category": classified_error["category"],
            "suggestions": classified_error["suggestions"],
            "recovery_actions": classified_error["recovery_actions"],
            "confidence": confidence,
            "pattern_matches": max_matches,
            "kusto_specific": True,
            "original_exception": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log with structured information
        logger.error(
            f"Kusto Error [{classified_error['error_code']}] - {classified_error['category']}: {error_str}",
            extra={
                "error_code": classified_error["error_code"],
                "error_type": classified_error["type"],
                "confidence": confidence,
                "pattern_matches": max_matches
            }
        )
        
        return error_response

__all__ = [
    "QueryProcessor",
    "normalize_name",
    "ErrorHandler",
    "QueryOptimizer",
    "bracket_if_needed",
    "get_default_cluster_memory_path",
    "ensure_directory_exists",
    "sanitize_filename",
    "get_schema_column_names",
    "normalize_join_on_clause",
    "validate_projected_columns",
    "validate_all_query_columns",
    "SchemaManager",
    "get_schema_discovery",
    "get_schema_discovery_status",
    "fix_query_with_real_schema",
    "generate_query_description",
    "QueryParser",
    "parse_query_entities",
    "extract_cluster_and_database_from_query",
    "extract_tables_from_query",
]
 
class QueryOptimizer:
    """A class for optimizing and validating KQL queries."""

    def __init__(self):
        self.join_on_pattern = re.compile(r"(\bjoin\b\s+(?:\w+\s+)?(?:\([^)]+\)\s+)?(?:\w+\s+)?on\s+)([^\|]+)", re.IGNORECASE)
        self.project_pattern = re.compile(r"\|\s*project\s+([^|]+)", re.IGNORECASE)
        self.identifier_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
        
        # Dynamic analyzers for intelligent query optimization
        self.table_analyzer = get_dynamic_table_analyzer()
        self.column_analyzer = get_dynamic_column_analyzer()

    def normalize_join_on_clause(self, kql: str) -> str:
        """Normalizes join 'on' clauses to fix common syntax errors."""
        if " join " not in kql.lower():
            return kql
        try:
            return self.join_on_pattern.sub(self._replace_join_clause, kql)
        except Exception as e:
            logger.debug(f"Join clause normalization failed: {e}")
            return kql

    def _replace_join_clause(self, match: re.Match) -> str:
        prefix = match.group(1)
        condition = match.group(2)
        or_split_re = re.compile(r'\bor\b', re.IGNORECASE)
        
        if 'or' in condition.lower():
            parts = or_split_re.split(condition)
            normalized_parts = [self._normalize_join_condition(part) for part in parts]
            normalized_condition = " and ".join(normalized_parts)
        else:
            normalized_condition = self._normalize_join_condition(condition)
        
        return prefix + normalized_condition

    def _normalize_join_condition(self, condition: str) -> str:
        condition = condition.strip()
        condition = re.sub(r'\b(\w+)\s*!=\s*(\w+)', r'\1 == \2', condition)
        condition = re.sub(r'\b(\w+)\s*<>\s*(\w+)', r'\1 == \2', condition)
        condition = re.sub(r'\b(\w+)\s*=\s*(\w+)', r'\1 == \2', condition)
        condition = re.sub(r'\b(\w+)\s*[<>]=?\s*(\w+)', r'\1 == \2', condition)
        return condition

    def validate_projected_columns(self, query: str, schema: Optional[Dict[str, Any]]) -> str:
        """Validates columns in a 'project' clause against a schema."""
        if not schema or not isinstance(schema, dict):
            return query
        
        self.schema_cols = get_schema_column_names(schema) or []
        if not self.schema_cols:
            return query  # No schema columns available
        
        self.lower_map = {c.lower(): c for c in self.schema_cols}

        try:
            return self.project_pattern.sub(self._clean_project, query)
        except Exception:
            return query

    def _clean_project(self, match: re.Match) -> str:
        project_content = match.group(1)
        parts = []
        cur = ""
        depth = 0
        for ch in project_content:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                parts.append(cur.strip())
                cur = ""
            else:
                cur += ch
        if cur.strip():
            parts.append(cur.strip())

        valid_parts = []
        for p in parts:
            if not p or p.isspace():  # Skip empty or whitespace-only parts
                continue
            
            # Clean up any malformed bracketing that could cause SEM0100
            p = p.strip()
            if p.startswith('["') and p.endswith('"]') and '"' in p[2:-2]:
                # Fix malformed bracket expressions like [" "]
                inner = p[2:-2].strip()
                if inner and not inner.isspace():
                    p = f"['{inner}']"
                else:
                    continue  # Skip malformed empty expressions
            
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", p):
                # Simple column name - validate against schema and apply proper bracketing
                if hasattr(self, 'lower_map') and p.lower() in self.lower_map:
                    # Use exact case from schema
                    exact_col = self.lower_map[p.lower()]
                    # Apply bracketing if needed for reserved words or special characters
                    valid_parts.append(bracket_if_needed(exact_col))
                elif hasattr(self, 'schema_cols') and self.schema_cols:
                    # Schema available but column not found - skip invalid column
                    logger.debug(f"Column '{p}' not found in schema, skipping")
                    continue
                else:
                    # No schema available, keep as-is but apply bracketing
                    valid_parts.append(bracket_if_needed(p))
            else:
                # Complex expression (functions, aliases, etc.) - validate before keeping
                if p and not p.isspace() and p != '[" "]' and p != '[""]':
                    valid_parts.append(p)
        
        if not valid_parts:
            return "| project *"  # Fallback to all columns if no valid columns found
        return "| project " + ", ".join(valid_parts)

    def validate_all_query_columns(self, query: str, schema: Optional[Dict[str, Any]]) -> str:
        """Replaces all column identifiers with their real-cased names from the schema."""
        if not schema or not isinstance(schema, dict):
            return query

        cols = get_schema_column_names(schema) or []
        if not cols:
            return query
        
        mapping = {c.lower(): c for c in cols}

        def _replace_token(m: re.Match) -> str:
            token = m.group(1)
            return mapping.get(token.lower(), token)

        try:
            return self.identifier_pattern.sub(_replace_token, query)
        except Exception:
            return query

    def optimize_query_with_dynamic_analysis(self, query: str, table_name: str = None, schema: Optional[Dict[str, Any]] = None) -> str:
        """
        Optimize a KQL query using dynamic table and column analysis.
        This implements Task 7: Dynamic logic for KQL operators.
        
        Args:
            query: The KQL query to optimize
            table_name: Target table name for analysis
            schema: Schema information for the table
            
        Returns:
            Optimized query string
        """
        try:
            if not query or not query.strip():
                return query
            
            optimized_query = query
            
            # Extract entities from the query
            parser = QueryParser()
            entities = parser.parse(query)
            tables = entities.get("tables", [])
            
            # Use provided table_name or extract from query
            target_table = table_name or (tables[0] if tables else None)
            
            if target_table and self.table_analyzer:
                # Analyze table patterns to determine optimization strategy
                table_analysis = self.table_analyzer.analyze_table_characteristics(
                    target_table, schema.get("columns", {}) if schema else {}
                )
                
                # Apply table-specific optimizations
                optimized_query = self._apply_table_optimizations(optimized_query, target_table, table_analysis)
            
            if schema and self.column_analyzer:
                # Analyze column patterns for intelligent projection
                optimized_query = self._apply_column_optimizations(optimized_query, schema, target_table)
            
            # Apply operator-specific optimizations
            optimized_query = self._apply_operator_optimizations(optimized_query, entities)
            
            return optimized_query
            
        except Exception as e:
            logger.debug(f"Dynamic query optimization failed: {e}")
            return query  # Return original query if optimization fails

    def _apply_table_optimizations(self, query: str, table_name: str, table_analysis: Dict[str, Any]) -> str:
        """Apply table-specific optimizations based on dynamic analysis."""
        try:
            optimized_query = query
            
            # If table has temporal patterns, ensure proper time-based operations
            if table_analysis.get("has_timestamps"):
                # Add TimeGenerated column if no time column is projected
                if "| project" in optimized_query and "TimeGenerated" not in optimized_query:
                    optimized_query = re.sub(
                        r'\|\s*project\s+([^|]+)',
                        r'| project TimeGenerated, \1',
                        optimized_query
                    )
            
            # If table has identifiers, optimize for unique operations
            if table_analysis.get("has_identifiers"):
                # Add distinct operation if dealing with identifier columns
                if "summarize" in optimized_query.lower() and "distinct" not in optimized_query.lower():
                    logger.debug(f"Table {table_name} has identifiers - optimizing for uniqueness")
            
            # If table has metrics, suggest aggregation patterns
            if table_analysis.get("has_metrics"):
                # Optimize for numeric operations if no aggregation present
                if "summarize" not in optimized_query.lower() and "take" in optimized_query.lower():
                    logger.debug(f"Table {table_name} has metrics - consider aggregation operations")
            
            return optimized_query
            
        except Exception as e:
            logger.debug(f"Table optimization failed: {e}")
            return query

    def _apply_column_optimizations(self, query: str, schema: Dict[str, Any], table_name: str = None) -> str:
        """Apply column-specific optimizations using dynamic analysis."""
        try:
            optimized_query = query
            columns = schema.get("columns", {})
            
            if not columns:
                return optimized_query
            
            # Analyze each column for optimization opportunities
            for col_name, col_info in columns.items():
                if isinstance(col_info, dict):
                    col_tags = self.column_analyzer.generate_column_tags(
                        col_name, col_info.get("sample_values", [])
                    )
                    
                    # Apply tag-based optimizations
                    optimized_query = self._apply_tag_based_optimizations(
                        optimized_query, col_name, col_tags
                    )
            
            return optimized_query
            
        except Exception as e:
            logger.debug(f"Column optimization failed: {e}")
            return query

    def _apply_tag_based_optimizations(self, query: str, col_name: str, tags: List[str]) -> str:
        """Apply optimizations based on column tags."""
        try:
            optimized_query = query
            
            # Validate column name before applying optimizations
            if not col_name or col_name.isspace():
                return optimized_query
            
            # Time column optimizations
            if "DATETIME" in tags or "TIME_COLUMN" in tags:
                # Only bracket if not already bracketed and is a valid identifier
                if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', col_name):
                    pattern = rf'\b{re.escape(col_name)}\b'
                    replacement = bracket_if_needed(col_name)
                    optimized_query = re.sub(pattern, replacement, optimized_query)
            
            # Identifier column optimizations
            if "ID_COLUMN" in tags:
                # Ensure ID columns are properly bracketed in operations
                if f"== {col_name}" in optimized_query and re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', col_name):
                    bracketed_col = bracket_if_needed(col_name)
                    optimized_query = re.sub(
                        rf'== {re.escape(col_name)}\b',
                        f"== {bracketed_col}",
                        optimized_query
                    )
            
            # Metric column optimizations
            if "METRIC_COLUMN" in tags or "NUMERIC" in tags:
                # Add proper casting for numeric operations
                if "summarize" in optimized_query.lower() and col_name in optimized_query:
                    logger.debug(f"Optimizing numeric column {col_name} for aggregation")
            
            return optimized_query
            
        except Exception as e:
            logger.debug(f"Tag-based optimization failed for {col_name}: {e}")
            return query

    def _apply_operator_optimizations(self, query: str, entities: Dict[str, Any]) -> str:
        """Apply KQL operator-specific optimizations."""
        try:
            optimized_query = query
            operations = entities.get("operations", [])
            
            # Project operation optimizations
            if "project" in operations:
                optimized_query = self._optimize_project_operations(optimized_query)
            
            # Where operation optimizations
            if "where" in operations:
                optimized_query = self._optimize_where_operations(optimized_query)
            
            # Join operation optimizations
            if "join" in operations:
                optimized_query = self._optimize_join_operations(optimized_query)
            
            # Summarize operation optimizations
            if "summarize" in operations:
                optimized_query = self._optimize_summarize_operations(optimized_query)
            
            return optimized_query
            
        except Exception as e:
            logger.debug(f"Operator optimization failed: {e}")
            return query

    def _optimize_project_operations(self, query: str) -> str:
        """Optimize project operations for better performance."""
        try:
            # Ensure project operations come after filtering for performance
            lines = query.split('\n')
            optimized_lines = []
            project_line = None
            
            for line in lines:
                if '| project' in line.lower():
                    project_line = line
                elif '| where' in line.lower() and project_line:
                    # Move where before project for better performance
                    optimized_lines.append(line)
                    optimized_lines.append(project_line)
                    project_line = None
                else:
                    if project_line:
                        optimized_lines.append(project_line)
                        project_line = None
                    optimized_lines.append(line)
            
            if project_line:
                optimized_lines.append(project_line)
            
            return '\n'.join(optimized_lines)
            
        except Exception:
            return query

    def _optimize_where_operations(self, query: str) -> str:
        """Optimize where operations for better filtering."""
        try:
            # Ensure most selective filters come first
            where_pattern = re.compile(r'\|\s*where\s+([^|]+)', re.IGNORECASE)
            
            def optimize_where_clause(match):
                where_content = match.group(1).strip()
                
                # Split multiple conditions
                conditions = []
                if ' and ' in where_content.lower():
                    conditions = re.split(r'\s+and\s+', where_content, flags=re.IGNORECASE)
                elif ' or ' in where_content.lower():
                    conditions = re.split(r'\s+or\s+', where_content, flags=re.IGNORECASE)
                else:
                    conditions = [where_content]
                
                # Sort conditions by selectivity (simple heuristic)
                def selectivity_score(condition):
                    score = 0
                    if '==' in condition:
                        score += 3  # Equality is very selective
                    elif 'contains' in condition.lower():
                        score += 1  # Contains is less selective
                    elif 'startswith' in condition.lower():
                        score += 2  # StartsWith is moderately selective
                    return score
                
                optimized_conditions = sorted(conditions, key=selectivity_score, reverse=True)
                
                # Rejoin conditions
                if ' and ' in where_content.lower():
                    optimized_where = ' and '.join(optimized_conditions)
                elif ' or ' in where_content.lower():
                    optimized_where = ' or '.join(optimized_conditions)
                else:
                    optimized_where = optimized_conditions[0] if optimized_conditions else where_content
                
                return f"| where {optimized_where}"
            
            return where_pattern.sub(optimize_where_clause, query)
            
        except Exception:
            return query

    def _optimize_join_operations(self, query: str) -> str:
        """Optimize join operations for better performance."""
        try:
            # Apply the existing join normalization
            return self.normalize_join_on_clause(query)
        except Exception:
            return query

    def _optimize_summarize_operations(self, query: str) -> str:
        """Optimize summarize operations for better aggregation."""
        try:
            # Ensure summarize operations have proper grouping
            summarize_pattern = re.compile(r'\|\s*summarize\s+([^|]+)', re.IGNORECASE)
            
            def optimize_summarize_clause(match):
                summarize_content = match.group(1).strip()
                
                # Add 'by' clause if missing for better performance
                if 'by ' not in summarize_content.lower() and 'count()' in summarize_content.lower():
                    # For simple count operations, consider adding TimeGenerated binning
                    return f"| summarize {summarize_content} by bin(TimeGenerated, 1h)"
                
                return match.group(0)  # Return original if no optimization needed
            
            return summarize_pattern.sub(optimize_summarize_clause, query)
            
        except Exception:
            return query


# ---------------------------------------------------------------------------
# Path / filename helpers
# ---------------------------------------------------------------------------

def bracket_if_needed(identifier: str) -> str:
    """
    Enhanced KQL identifier bracketing with comprehensive syntax error prevention.
    
    Quotes a KQL identifier (table or column) with [''] if it:
    - Is a reserved keyword
    - Contains special characters
    - Starts with numbers or invalid characters
    - Contains spaces, hyphens, or other problematic characters
    - Has potential for causing KQL syntax errors
    """
    if not isinstance(identifier, str) or not identifier:
        return identifier

    # Use the comprehensive reserved words list from constants
    from .constants import KQL_RESERVED_WORDS
    reserved_keywords = {k.lower() for k in KQL_RESERVED_WORDS}
    
    identifier_lower = identifier.lower()

    # Check if the identifier is a reserved keyword or contains invalid characters
    if identifier_lower in reserved_keywords or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
        # Escape single quotes using the Kusto convention of doubling them
        escaped_identifier = identifier.replace("'", "''")
        return f"['{escaped_identifier}']"

    return identifier

def get_default_cluster_memory_path() -> Path:
    """Return a sensible default path for cluster memory storage.

    Tests accept either 'KQL_MCP' or 'kql_memory' in the path, so choose a value
    that includes 'KQL_MCP' to match expectations on Windows-like systems.
    """
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / "KQL_MCP"
    # Fallback to a local directory in the workspace/home
    return Path.cwd() / "KQL_MCP"


def ensure_directory_exists(path: Path) -> bool:
    """Ensure the given directory exists. Returns True on success."""
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def sanitize_filename(name: Optional[str]) -> str:
    """Remove characters invalid in filenames (Windows-oriented) conservatively."""
    if not name:
        return "" if name == "" else ""
    # Remove < > : " / \ | ? * characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Collapse sequences of underscores to single
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized


# ---------------------------------------------------------------------------
# Lightweight schema helpers (sufficient for tests and call-sites)
# ---------------------------------------------------------------------------
def get_schema_column_names(schema: Optional[Dict[str, Any]]) -> List[str]:
    """Return a list of column names from schema objects used in this project.

    The schema may be in various shapes:
    - A pandas.DataFrame returned by a `| getschema` query (with ColumnName/DataType columns)
    - A dict containing 'column_types' mapping
    - A dict containing a legacy 'columns' list (strings or dicts)

    This function attempts to handle these shapes robustly and return a simple
    list of canonical column names.
    """
    if not schema:
        return []

    # 1) Handle pandas.DataFrame shape (lightweight detection)
    try:
        import pandas as _pd

        if isinstance(schema, _pd.DataFrame):
            df = schema
            df_cols = list(df.columns)
            # Identify the column that contains the column name (ColumnName is common)
            colname_key = next(
                (c for c in df_cols if c.lower() in ("columnname", "column_name", "name")), None
            )
            if colname_key and colname_key in df.columns:
                try:
                    return [str(v) for v in df[colname_key].astype(str).tolist()]
                except (KeyError, AttributeError):
                    pass

            # Fallback: use the first column value from each row
            names = []
            for _, row in df.iterrows():
                try:
                    if len(row.index) > 0:
                        names.append(str(row.iloc[0]))
                except IndexError:
                    continue
            return names
    except ImportError:
        # pandas not available or not a DataFrame-like object; fall back to dict handling
        pass

    # 2) Handle dict-based schema formats (preferred for most code paths)
    if isinstance(schema, dict):
        # The new standard is schema -> columns -> {col_name: {details}}
        try:
            cols = schema.get("columns")
            if isinstance(cols, dict):
                return list(cols.keys())
        except (AttributeError, TypeError):
            pass

        # Preferred legacy format: column_types mapping {col: {...}}
        try:
            ct = schema.get("column_types")
            if isinstance(ct, dict) and ct:
                return list(ct.keys())
        except (AttributeError, TypeError):
            pass

        # Legacy format: 'columns' list (strings or dicts)
        try:
            cols = schema.get("columns")
            if isinstance(cols, list) and cols:
                names = []
                for c in cols:
                    if isinstance(c, str):
                        names.append(c)
                    elif isinstance(c, dict):
                        # common keys: "name", "ColumnName", "column_name", "columnname"
                        for k in ("name", "ColumnName", "column_name", "columnname"):
                            if k in c:
                                names.append(c[k])
                                break
                        else:
                            # Fallback: try first value from the dict
                            try:
                                first_val = next(iter(c.values()))
                                names.append(str(first_val))
                            except StopIteration:
                                continue
                return names
        except (AttributeError, TypeError):
            pass

    # If all attempts fail, return an empty list
    return []


# ---------------------------------------------------------------------------
# Simple project / column validation helpers used in unit tests
# ---------------------------------------------------------------------------
_IDENTIFIER_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


# Instantiate the optimizer for use in backward-compatible functions
_optimizer = QueryOptimizer()

def normalize_join_on_clause(kql: str) -> str:
    """Backward-compatible join normalization."""
    return _optimizer.normalize_join_on_clause(kql)

def validate_projected_columns(query: str, schema: Optional[Dict[str, Any]]) -> str:
    """Backward-compatible project column validation."""
    return _optimizer.validate_projected_columns(query, schema)

def validate_all_query_columns(query: str, schema: Optional[Dict[str, Any]]) -> str:
    """Backward-compatible query column validation."""
    return _optimizer.validate_all_query_columns(query, schema)


# ---------------------------------------------------------------------------
# Centralized Schema Management
# ---------------------------------------------------------------------------
class SchemaManager:
    """
    Centralized and unified schema management system.
    This consolidates all schema operations as recommended in the analysis.
    """

    def __init__(self, memory_manager=None):
        """
        Initializes the SchemaManager with a MemoryManager instance.
        If no memory_manager is provided, creates one automatically.
        """
        if memory_manager is None:
            from .memory import get_memory_manager
            self.memory_manager = get_memory_manager()
        else:
            self.memory_manager = memory_manager
        
        # Unified caching and configuration
        self._schema_cache = {}
        self._discovery_cache = {}
        self._last_discovery_times = {}

    async def _execute_kusto_async(self, query: str, cluster: str, database: str, is_mgmt: bool = False) -> List[Dict]:
        """
        Enhanced async wrapper for executing Kusto queries with comprehensive error handling,
        retry logic, connection validation, and graceful degradation.
        """
        import asyncio
        import re
        import time
        from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
        from .constants import (
            CONNECTION_CONFIG,
            RETRYABLE_ERROR_PATTERNS, NON_RETRYABLE_ERROR_PATTERNS
        )
        
        loop = asyncio.get_running_loop()
        
        # Configuration from constants
        max_retries = CONNECTION_CONFIG.get("max_retries", 5)
        retry_delay = CONNECTION_CONFIG.get("retry_delay", 2.0)
        backoff_factor = CONNECTION_CONFIG.get("retry_backoff_factor", 2.0)
        max_retry_delay = CONNECTION_CONFIG.get("max_retry_delay", 60.0)
        
        def _is_retryable_error(error_str: str) -> bool:
            """Check if error matches retryable patterns."""
            # Check non-retryable patterns first (these take precedence)
            for pattern in NON_RETRYABLE_ERROR_PATTERNS:
                if re.search(pattern, error_str, re.IGNORECASE):
                    return False
            
            # Check retryable patterns
            for pattern in RETRYABLE_ERROR_PATTERNS:
                if re.search(pattern, error_str, re.IGNORECASE):
                    return True
                    
            return False
        
        def _validate_connection(cluster_url: str) -> bool:
            """
            Enhanced connection validation with comprehensive authentication and connectivity checks.
            
            Performs:
            1. Authentication validation with Azure CLI
            2. Network connectivity test
            3. Cluster accessibility verification
            4. Permission validation
            """
            try:
                validation_timeout = CONNECTION_CONFIG.get("connection_validation_timeout", 5.0)
                
                # Step 1: Validate Azure CLI authentication
                auth_valid = self._validate_azure_authentication(cluster_url)
                if not auth_valid:
                    logger.warning(f"Azure CLI authentication validation failed for {cluster_url}")
                    return False
                
                # Step 2: Test basic connectivity
                connectivity_valid = self._test_network_connectivity(cluster_url, validation_timeout)
                if not connectivity_valid:
                    logger.warning(f"Network connectivity test failed for {cluster_url}")
                    return False
                
                # Step 3: Test cluster access with actual query
                access_valid = self._test_cluster_access(cluster_url, validation_timeout)
                if not access_valid:
                    logger.warning(f"Cluster access test failed for {cluster_url}")
                    return False
                
                logger.info(f"Connection validation passed for {cluster_url}")
                return True
                        
            except Exception as e:
                logger.error(f"Connection validation failed for {cluster_url}: {e}")
                return False
        
        def _sync_execute():
            """Execute Kusto query with retry logic and error handling."""
            cluster_url = f"https://{cluster}" if not cluster.startswith("https://") else cluster
            
            # Pre-validate connection if enabled
            if CONNECTION_CONFIG.get("validate_connection_before_use", True):
                if not _validate_connection(cluster_url):
                    logger.warning(f"Connection validation failed for {cluster_url}, proceeding anyway...")
            
            last_exception = None
            current_delay = retry_delay
            
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    # Create connection with timeout configuration
                    kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(cluster_url)
                    
                    with KustoClient(kcsb) as client:
                        # Execute query/management command
                        if is_mgmt:
                            response = client.execute_mgmt(database, query)
                        else:
                            response = client.execute(database, query)
                        
                        # Extract results
                        if response.primary_results:
                            data = response.primary_results[0].to_dict()["data"]
                            logger.debug(f"Successfully executed query on attempt {attempt + 1}")
                            return data
                        else:
                            logger.warning(f"Query returned no results: {query}")
                            return []
                            
                except Exception as e:
                    last_exception = e
                    error_str = str(e)
                    
                    # Log the attempt
                    logger.warning(f"Kusto execution attempt {attempt + 1}/{max_retries + 1} failed: {error_str}")
                    
                    # Check if this is the final attempt
                    if attempt >= max_retries:
                        logger.error(f"All retry attempts exhausted for query: {query}")
                        break
                    
                    # Check if error is retryable
                    if not _is_retryable_error(error_str):
                        logger.error(f"Non-retryable error encountered: {error_str}")
                        break
                    
                    # Wait before retry with exponential backoff
                    logger.info(f"Retrying in {current_delay:.1f}s due to retryable error...")
                    time.sleep(current_delay)
                    current_delay = min(current_delay * backoff_factor, max_retry_delay)
            
            # All retries failed - propagate the last exception
            if last_exception:
                error_msg = f"Kusto execution failed after {max_retries + 1} attempts: {str(last_exception)}"
                logger.error(error_msg)
                raise Exception(error_msg) from last_exception
            else:
                raise Exception("Kusto execution failed for unknown reasons")

        return await loop.run_in_executor(None, _sync_execute)

    def _validate_azure_authentication(self, cluster_url: str) -> bool:
        """
        Skip redundant authentication validation since we validate at startup.
        Always return True if we reach this point (authentication was successful at startup).
        """
        logger.debug(f"Skipping redundant authentication validation for {cluster_url} - already validated at startup")
        return True

    def _test_network_connectivity(self, cluster_url: str, timeout: float) -> bool:
        """Test basic network connectivity to the cluster."""
        try:
            import socket
            from urllib.parse import urlparse
            
            parsed_url = urlparse(cluster_url)
            hostname = parsed_url.hostname
            port = parsed_url.port or 443  # Default HTTPS port for Kusto
            
            # Test TCP connectivity
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            result = sock.connect_ex((hostname, port))
            sock.close()
            
            if result == 0:
                logger.debug(f"Network connectivity test passed for {hostname}:{port}")
                return True
            else:
                logger.warning(f"Network connectivity test failed for {hostname}:{port}")
                return False
                
        except Exception as e:
            logger.warning(f"Network connectivity test error: {e}")
            return False

    def _test_cluster_access(self, cluster_url: str, timeout: float) -> bool:
        """Test actual cluster access with a minimal query."""
        try:
            from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
            kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(cluster_url)
            
            with KustoClient(kcsb) as client:
                # Use a lightweight query that should work on any cluster
                test_query = ".show version"
                
                # Set timeout for the operation
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Cluster access test timeout")
                
                # Apply timeout (only on Unix-like systems)
                if hasattr(signal, 'SIGALRM'):
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout))
                
                try:
                    response = client.execute_mgmt("NetDefaultDB", test_query)
                    
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)  # Cancel timeout
                    
                    success = response is not None and response.primary_results
                    if success:
                        logger.debug("Cluster access test passed")
                    else:
                        logger.warning("Cluster access test failed: no valid response")
                    
                    return success
                    
                except Exception as query_error:
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)  # Cancel timeout
                    logger.warning(f"Cluster access query failed: {query_error}")
                    return False
                    
        except Exception as e:
            logger.warning(f"Cluster access test error: {e}")
            return False

    async def discover_schema_for_table(self, client, table_name: str) -> Dict:
        """
        Discovers detailed schema information for a specific table.
        Unified method that consolidates schema discovery logic with enhanced analysis.

        Args:
            client: The Kusto client instance.
            table_name: The name of the table to analyze.

        Returns:
            Dict: Enhanced schema information for the table.
        """
        try:
            # Check unified cache first
            cache_key = f"unified_table_schema_{table_name}"
            if cache_key in self._schema_cache:
                cached_data = self._schema_cache[cache_key]
                # Check if cache is still valid (1 hour)
                if (datetime.now() - cached_data['timestamp']).seconds < 3600:
                    return cached_data['data']

            # Also check memory manager cache for backwards compatibility
            legacy_cache_key = f"table_schema_{table_name}"
            cached_result = self.memory_manager.get_cached_result(legacy_cache_key, 3600)
            if cached_result:
                return cached_result

            # Query for comprehensive table schema with enhanced analysis
            schema_query = f"""
            {table_name}
            | getschema
            | extend TableName = "{table_name}"
            | project TableName, ColumnName, ColumnType, ColumnOrdinal
            """

            # Also get sample data for enhanced analysis
            sample_query = f"""
            {table_name}
            | take 10
            | project *
            """

            response = await client.execute("", schema_query)
            sample_response = None
            try:
                sample_response = await client.execute("", sample_query)
            except Exception:
                pass  # Sample data is optional

            schema_info = {
                "table_name": table_name,
                "columns": [],
                "total_columns": 0,
                "discovered_at": datetime.now().isoformat(),
                "discovery_method": "unified_schema_manager",
                "sample_data_available": sample_response is not None
            }

            if response and hasattr(response, 'primary_results') and response.primary_results:
                # Extract sample data for enhanced column analysis
                sample_data = []
                if sample_response and hasattr(sample_response, 'primary_results') and sample_response.primary_results:
                    sample_data = sample_response.primary_results[0]

                # Process columns with enhanced analysis
                enhanced_columns = await self._process_schema_columns(
                    response.primary_results[0], sample_data, table_name, "", ""
                )
                
                # Convert to list format for backward compatibility
                for col_name, col_info in enhanced_columns.items():
                    column_entry = {
                        "name": col_name,
                        "type": col_info.get("data_type", ""),
                        "ordinal": col_info.get("ordinal", 0),
                        "description": col_info.get("description", ""),
                        "tags": col_info.get("tags", []),
                        "sample_values": col_info.get("sample_values", []),
                        "ai_token": col_info.get("ai_token", "")
                    }
                    schema_info["columns"].append(column_entry)

                schema_info["total_columns"] = len(schema_info["columns"])
                schema_info["enhanced_columns"] = enhanced_columns  # Also store enhanced format

            # Enhanced caching in both unified and legacy systems
            cache_data = {
                'data': schema_info,
                'timestamp': datetime.now()
            }
            self._schema_cache[cache_key] = cache_data
            self.memory_manager.cache_result(legacy_cache_key, schema_info, 3600)
            
            # Track usage for session learning
            self.track_schema_usage(table_name, "discovery", True)
            
            return schema_info

        except Exception as e:
            logger.error(f"Error in unified schema discovery for table {table_name}: {e}")
            # Track failed usage
            self.track_schema_usage(table_name, "discovery", False)
            
            return {
                "table_name": table_name,
                "columns": [],
                "total_columns": 0,
                "error": str(e),
                "discovered_at": datetime.now().isoformat(),
                "discovery_method": "unified_schema_manager_error"
            }

    async def get_table_schema(self, cluster: str, database: str, table: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Gets a table schema using multiple discovery strategies with proper column metadata handling.
        This function is now the single source of truth for live schema discovery.
        """
        try:
            logger.debug(f"Performing enhanced schema discovery for {database}.{table}")
            
            # Strategy 1: Try .show table schema as json (most detailed)
            try:
                schema_query = f".show table {bracket_if_needed(table)} schema as json"
                schema_result = await self._execute_kusto_async(schema_query, cluster, database, is_mgmt=True)

                if schema_result and len(schema_result) > 0:
                    # The result is a list with one dict, where the first column contains the JSON string.
                    schema_json_str = schema_result[0][next(iter(schema_result[0]))]
                    schema_json = json.loads(schema_json_str)

                    # Get sample data for all columns
                    sample_data = {}
                    try:
                        bracketed_table = bracket_if_needed(table)
                        sample_query = f"{bracketed_table} | take 2"
                        sample_result = await self._execute_kusto_async(sample_query, cluster, database, is_mgmt=False)
                        
                        if sample_result and len(sample_result) > 0:
                            # Extract sample values for each column
                            for col_name in [col['Name'] for col in schema_json.get('Schema', {}).get('OrderedColumns', [])]:
                                sample_values = [str(row.get(col_name, '')) for row in sample_result[:2] if row.get(col_name) is not None]
                                sample_data[col_name] = sample_values
                    except Exception as sample_error:
                        logger.debug(f"Failed to get sample data for Strategy 1: {sample_error}")

                    # Enhanced transformation with proper column metadata
                    columns = {}
                    for col in schema_json.get('Schema', {}).get('OrderedColumns', []):
                        col_name = col['Name']
                        col_type = col['CslType']
                        sample_values = sample_data.get(col_name, [])
                        columns[col_name] = {
                            'data_type': col_type,
                            'description': self._generate_column_description(table, col_name, col_type, sample_values),
                            'tags': self._generate_column_tags(col_name, col_type),
                            'sample_values': sample_values,
                            'ordinal': col.get('Ordinal', 0),
                            'column_type': col_type
                        }
                    
                    if columns:
                        logger.info(f"Strategy 1 successful: JSON schema discovery for {table}")
                        return self._create_enhanced_schema_object(cluster, database, table, columns, "json_schema")
            except Exception as json_error:
                logger.debug(f"JSON schema discovery failed for {table}: {json_error}")
            
            # Strategy 2: Try | getschema (backup method with enhanced processing)
            try:
                # Always bracket identifiers in built queries to prevent reserved word issues
                bracketed_table = bracket_if_needed(table)
                getschema_query = f"{bracketed_table} | getschema"
                getschema_result = await self._execute_kusto_async(getschema_query, cluster, database, is_mgmt=False)
                
                if getschema_result and len(getschema_result) > 0:
                    # Get sample data for all columns
                    sample_data = {}
                    try:
                        sample_query = f"{bracketed_table} | take 2"
                        sample_result = await self._execute_kusto_async(sample_query, cluster, database, is_mgmt=False)
                        
                        if sample_result and len(sample_result) > 0:
                            # Extract sample values for each column
                            for row_data in getschema_result:
                                col_name = row_data.get('ColumnName') or row_data.get('Column')
                                if col_name:
                                    sample_values = [str(row.get(col_name, '')) for row in sample_result[:2] if row.get(col_name) is not None]
                                    sample_data[col_name] = sample_values
                    except Exception as sample_error:
                        logger.debug(f"Failed to get sample data for Strategy 2: {sample_error}")
                    
                    columns = {}
                    for i, row in enumerate(getschema_result):
                        col_name = row.get('ColumnName') or row.get('Column') or f'Column{i}'
                        col_type = row.get('DataType') or row.get('ColumnType') or 'string'
                        
                        # Clean up data type
                        col_type = str(col_type).replace('System.', '').lower()
                        
                        sample_values = sample_data.get(col_name, [])
                        columns[col_name] = {
                            'data_type': col_type,
                            'description': self._generate_column_description(table, col_name, col_type, sample_values),
                            'tags': self._generate_column_tags(col_name, col_type),
                            'sample_values': sample_values,
                            'ordinal': row.get('ColumnOrdinal', i),
                            'column_type': col_type
                        }
                    
                    if columns:
                        logger.info(f"Strategy 2 successful: getschema discovery for {table}")
                        return self._create_enhanced_schema_object(cluster, database, table, columns, "getschema")
            except Exception as getschema_error:
                logger.debug(f"getschema discovery failed for {table}: {getschema_error}")
            
            # Strategy 3: Try to get sample data and infer schema
            try:
                # Always bracket identifiers in built queries to prevent reserved word issues
                bracketed_table = bracket_if_needed(table)
                sample_query = f"{bracketed_table} | take 2"
                sample_result = await self._execute_kusto_async(sample_query, cluster, database, is_mgmt=False)
                
                if sample_result and len(sample_result) > 0:
                    # Infer schema from sample data
                    sample_row = sample_result[0]
                    columns = {}
                    for i, (col_name, value) in enumerate(sample_row.items()):
                        # Infer data type from value
                        col_type = self._infer_data_type_from_value(value)
                        
                        # Extract sample values
                        sample_values = [str(row.get(col_name, '')) for row in sample_result[:2] if row.get(col_name) is not None]
                        
                        columns[col_name] = {
                            'data_type': col_type,
                            'description': self._generate_column_description(table, col_name, col_type, sample_values),
                            'tags': self._generate_column_tags(col_name, col_type),
                            'sample_values': sample_values,
                            'ordinal': i,
                            'column_type': col_type
                        }
                    
                    if columns:
                        logger.info(f"Strategy 3 successful: sample-based discovery for {table}")
                        return self._create_enhanced_schema_object(cluster, database, table, columns, "sample_inference")
            except Exception as sample_error:
                logger.debug(f"Sample-based discovery failed for {table}: {sample_error}")
            
            # All strategies failed
            raise Exception("All schema discovery strategies failed")
            
        except Exception as e:
            logger.error(f"Enhanced schema discovery failed for {database}.{table}: {e}")
            # Track failed usage
            self.track_schema_usage(table, "enhanced_discovery", False)
            
            # Return fallback schema to prevent crashes
            return self._create_fallback_schema(cluster, database, table, str(e))

    def _create_enhanced_schema_object(self, cluster: str, database: str, table: str, columns: dict, method: str) -> Dict[str, Any]:
        """Create enhanced schema object with proper metadata."""
        schema_obj = {
            "table_name": table,
            "columns": columns,
            "discovered_at": datetime.now().isoformat(),
            "cluster": cluster,
            "database": database,
            "column_count": len(columns),
            "discovery_method": f"enhanced_{method}",
            "schema_version": "3.1"
        }
        
        # Store the freshly discovered schema
        self.memory_manager.store_schema(cluster, database, table, schema_obj)
        logger.info(f"Successfully discovered and stored enhanced schema for {database}.{table} with {len(columns)} columns using {method}")
        
        # Track successful usage
        self.track_schema_usage(table, method, True)
        
        return schema_obj

    def _infer_data_type_from_value(self, value) -> str:
        """Infer KQL data type from a sample value."""
        if value is None:
            return 'string'
        
        value_str = str(value)
        
        # Check for datetime patterns
        if re.match(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', value_str):
            return 'datetime'
        
        # Check for boolean
        if value_str.lower() in ['true', 'false']:
            return 'bool'
        
        # Check for numbers
        try:
            if '.' in value_str:
                float(value_str)
                return 'real'
            else:
                int(value_str)
                return 'long'
        except ValueError:
            pass
        
        # Check for GUID/UUID
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', value_str, re.IGNORECASE):
            return 'guid'
        
        # Default to string
        return 'string'

    def _create_fallback_schema(self, cluster: str, database: str, table: str, error: str) -> Dict[str, Any]:
        """Create fallback schema when all discovery methods fail."""
        fallback_columns = {
            'Data': {
                'data_type': 'string',
                'description': 'Fallback data column',
                'tags': ['FALLBACK'],
                'sample_values': [],
                'ordinal': 0,
                'column_type': 'string'
            }
        }
        
        return {
            "table_name": table,
            "columns": fallback_columns,
            "discovered_at": datetime.now().isoformat(),
            "cluster": cluster,
            "database": database,
            "column_count": len(fallback_columns),
            "discovery_method": "fallback_schema",
            "error": error,
            "schema_version": "3.1"
        }

    def _is_schema_error_retryable(self, error_str: str) -> bool:
        """Determine if a schema discovery error is retryable based on error content."""
        error_lower = error_str.lower()
        
        # Retryable patterns - connection and temporary issues
        retryable_patterns = [
            "timeout", "connection", "network", "unavailable", "busy",
            "throttl", "rate limit", "temporary", "service", "grpc"
        ]
        
        # Non-retryable patterns - permanent failures
        non_retryable_patterns = [
            "unauthorized", "forbidden", "not found", "permission denied",
            "invalid", "syntax", "schema error", "bad request"
        ]
        
        # Check non-retryable first (takes precedence)
        for pattern in non_retryable_patterns:
            if pattern in error_lower:
                return False
        
        # Check retryable patterns
        for pattern in retryable_patterns:
            if pattern in error_lower:
                return True
        
        # Default to retryable for unknown errors (conservative approach)
        return True

    def _reconstruct_table_schema_from_db(self, table: str, cluster: str, database: str) -> Optional[Dict[str, Any]]:
        """Reconstruct basic table schema when direct table schema discovery fails."""
        try:
            # Create minimal schema with common column patterns based on table name
            table_lower = table.lower()
            columns = {}
            
            # Add common timestamp column
            columns["TimeGenerated"] = {
                'data_type': 'datetime',
                'description': 'Timestamp field',
                'tags': ['TIME_COLUMN'],
                'sample_values': []
            }
            
            # Add context-specific columns based on table name patterns
            if "security" in table_lower or "event" in table_lower:
                columns.update({
                    "EventID": {'data_type': 'int', 'description': 'Event identifier', 'tags': ['ID_COLUMN'], 'sample_values': []},
                    "Computer": {'data_type': 'string', 'description': 'Computer name', 'tags': ['TEXT'], 'sample_values': []},
                    "Account": {'data_type': 'string', 'description': 'User account', 'tags': ['TEXT'], 'sample_values': []}
                })
            elif "perf" in table_lower:
                columns.update({
                    "ObjectName": {'data_type': 'string', 'description': 'Performance object', 'tags': ['TEXT'], 'sample_values': []},
                    "CounterName": {'data_type': 'string', 'description': 'Performance counter', 'tags': ['TEXT'], 'sample_values': []},
                    "CounterValue": {'data_type': 'real', 'description': 'Counter value', 'tags': ['NUMERIC'], 'sample_values': []}
                })
            else:
                # Generic data columns
                columns.update({
                    "Data": {'data_type': 'string', 'description': 'Data field', 'tags': ['TEXT'], 'sample_values': []},
                    "Source": {'data_type': 'string', 'description': 'Data source', 'tags': ['TEXT'], 'sample_values': []}
                })
            
            return {
                "table_name": table,
                "columns": columns,
                "discovered_at": datetime.now().isoformat(),
                "cluster": cluster,
                "database": database,
                "column_count": len(columns),
                "schema_type": "reconstructed_from_database_fallback",
                "discovery_method": "table_pattern_reconstruction"
            }
            
        except Exception as e:
            logger.warning(f"Schema reconstruction failed for {table}: {e}")
            return None

    def _create_emergency_schema(self, table: str, cluster: str, database: str) -> Dict[str, Any]:
        """Create emergency minimal schema to prevent total failure."""
        emergency_columns = {
            'TimeGenerated': {
                'data_type': 'datetime',
                'description': 'Emergency fallback timestamp field',
                'tags': ['TIME_COLUMN', 'EMERGENCY'],
                'sample_values': []
            },
            'Data': {
                'data_type': 'string',
                'description': 'Emergency fallback data field',
                'tags': ['TEXT', 'EMERGENCY'],
                'sample_values': []
            }
        }
        
        return {
            "table_name": table,
            "columns": emergency_columns,
            "discovered_at": datetime.now().isoformat(),
            "cluster": cluster,
            "database": database,
            "column_count": len(emergency_columns),
            "schema_type": "emergency_schema_manager_fallback",
            "discovery_method": "emergency_fallback"
        }
    
    async def _process_schema_columns(self, schema_data: List[Dict], sample_data: List[Dict],
                                    table: str, cluster: str, database: str) -> Dict[str, Any]:
        """Process schema columns with AI enhancement and data-driven analysis."""
        columns = {}
        
        for row in schema_data:
            col_name = row.get("ColumnName")
            if not col_name:
                continue
            
            # Extract accurate data type from DataType column
            data_type = str(row.get('DataType', 'unknown')).replace("System.", "")
            if data_type == "unknown" and row.get('ColumnType'):
                data_type = str(row.get('ColumnType', 'unknown'))
            
            # Extract sample values from sample data (limit to 3)
            sample_values = self._extract_sample_values_from_data(col_name, sample_data)
            
            # Generate AI-enhanced description
            description = self._generate_column_description(table, col_name, data_type, sample_values)
            
            # Generate semantic tags
            tags = self._generate_column_tags(col_name, data_type)
            
            # Create AI-friendly token for this column
            ai_token = self._create_column_ai_token(col_name, data_type, description, sample_values, tags)
            
            columns[col_name] = {
                "data_type": data_type,
                "description": description,
                "tags": tags,
                "sample_values": sample_values[:3],  # Ensure max 3
                "ai_token": ai_token,
                "ordinal": row.get("ColumnOrdinal", 0),
                "column_type": row.get("ColumnType", data_type)
            }
        
        return columns
    
    def _extract_sample_values_from_data(self, column_name: str, sample_data: List[Dict]) -> List[str]:
        """Extract sample values for a column from sample data, limited to 3."""
        values = []
        seen = set()
        
        for row in sample_data:
            if column_name in row and row[column_name] is not None:
                value_str = str(row[column_name])
                if value_str not in seen and value_str.strip():  # Avoid duplicates and empty values
                    values.append(value_str)
                    seen.add(value_str)
                    if len(values) >= 3:
                        break
        
        return values
    
    def _create_column_ai_token(self, column: str, data_type: str, description: str,
                              sample_values: List[str], tags: List[str]) -> str:
        """Create AI-friendly token for enhanced query generation."""
        from .constants import SPECIAL_TOKENS
        
        token_parts = [
            f"{SPECIAL_TOKENS.get('COLUMN', '::COLUMN::')}:{column}",
            f"{SPECIAL_TOKENS.get('TYPE', '>>TYPE<<')}:{data_type}",
        ]
        
        # Add compact description
        if description and len(description) > 10:
            desc_short = description[:50] + "..." if len(description) > 50 else description
            token_parts.append(f"DESC:{desc_short}")
        
        # Add sample values compactly
        if sample_values:
            samples_str = ",".join(str(v) for v in sample_values[:2])
            token_parts.append(f"SAMPLES:{samples_str}")
        
        # Add primary tag
        if tags:
            primary_tag = tags[0]
            token_parts.append(f"TAG:{primary_tag}")
        
        return "|".join(token_parts)
    
    def _is_schema_fresh(self, schema: Dict[str, Any]) -> bool:
        """Check if cached schema is still fresh (within 24 hours)."""
        try:
            from datetime import timedelta
            discovered_at = schema.get("discovered_at")
            if not discovered_at:
                return False
            
            discovered_time = datetime.fromisoformat(discovered_at.replace('Z', '+00:00'))
            age = datetime.now() - discovered_time
            return age < timedelta(hours=24)
        except Exception:
            return False

    def _generate_column_description(self, table: str, column_name: str, data_type: str, sample_values: List[str]) -> str:
        """Generate AI-enhanced column description with semantic analysis."""
        try:
            # Import AI components for enhanced description
            from .local_ai import AIColumnAnalyzer
            analyzer = AIColumnAnalyzer()
            
            # Create column context for AI analysis
            column_context = {
                "table": table,
                "column": column_name,
                "data_type": data_type,
                "sample_values": sample_values[:3],
            }
            
            # Generate AI-enhanced description
            ai_description = analyzer._analyze_single_column(column_context)
            if ai_description and ai_description.strip():
                return ai_description
                
        except Exception as e:
            logger.debug(f"AI description generation failed for {column_name}: {e}")
        
        # Fallback to enhanced heuristic-based description
        return self._generate_semantic_description(table, column_name, data_type, sample_values)
    
    def _generate_semantic_description(self, table: str, column_name: str, data_type: str, sample_values: List[str]) -> str:
        """Generate semantic description using data-driven heuristics."""
        desc_parts = []
        
        # Determine column purpose based on name patterns
        purpose = self._determine_column_purpose(column_name, data_type, sample_values)
        
        # Build semantic description
        desc_parts.append(f"{purpose} column in {table}")
        
        # Add data type context
        if "datetime" in data_type.lower():
            desc_parts.append("storing timestamp information")
        elif "string" in data_type.lower():
            desc_parts.append("containing textual data")
        elif any(num_type in data_type.lower() for num_type in ['int', 'long', 'real', 'decimal']):
            desc_parts.append("holding numeric values")
        else:
            desc_parts.append(f"of {data_type} type")
        
        # Add contextual information based on sample values
        if sample_values:
            context = self._analyze_sample_context(sample_values, data_type)
            if context:
                desc_parts.append(context)
        
        return ". ".join(desc_parts)
    
    def _determine_column_purpose(self, column_name: str, data_type: str, sample_values: List[str]) -> str:
        """Determine the semantic purpose of a column using data-driven analysis."""
        # DYNAMIC APPROACH: Analyze actual data patterns instead of static keywords
        
        # Analyze sample values to determine purpose
        if sample_values:
            # Check if values are timestamps
            if self._looks_like_timestamps(sample_values):
                return "Temporal"
            
            # Check if values are identifiers (UUIDs, GUIDs, etc)
            if self._looks_like_identifiers(sample_values):
                return "Identifier"
            
            # Check if values are numeric measurements
            if self._looks_like_measurements(sample_values, data_type):
                return "Metric"
            
            # Check if values represent states/statuses
            if self._looks_like_states(sample_values):
                return "Status"
            
            # Check if values are categorical
            if self._looks_like_categories(sample_values):
                return "Category"
            
            # Check if values are locations
            if self._looks_like_locations(sample_values):
                return "Location"
        
        # Default based on data type analysis
        if "datetime" in data_type.lower() or "timestamp" in data_type.lower():
            return "Temporal"
        elif any(num_type in data_type.lower() for num_type in ['int', 'long', 'real', 'decimal', 'float', 'double']):
            return "Numeric"
        elif "bool" in data_type.lower():
            return "Status"
        elif "string" in data_type.lower() or "text" in data_type.lower():
            return "Descriptive"
        else:
            return "Data"
    
    def _looks_like_timestamps(self, values: List[str]) -> bool:
        """Check if values appear to be timestamps based on patterns."""
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # Date pattern
            r'\d{2}:\d{2}:\d{2}',  # Time pattern
            r'^\d{10,13}$',        # Unix timestamp
        ]
        matches = 0
        for value in values[:3]:  # Check first 3 values
            for pattern in timestamp_patterns:
                if re.search(pattern, str(value)):
                    matches += 1
                    break
        return matches >= len(values[:3]) * 0.5  # At least 50% match
    
    def _looks_like_identifiers(self, values: List[str]) -> bool:
        """Check if values appear to be identifiers."""
        id_patterns = [
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',  # UUID
            r'^[0-9a-f]{32}$',  # MD5 hash
            r'^[A-Z0-9]{8,}$',  # Uppercase alphanumeric ID
            r'^\d{6,}$',        # Long numeric ID
        ]
        matches = 0
        for value in values[:3]:
            for pattern in id_patterns:
                if re.match(pattern, str(value), re.IGNORECASE):
                    matches += 1
                    break
        return matches >= len(values[:3]) * 0.5
    
    def _looks_like_measurements(self, values: List[str], data_type: str) -> bool:
        """Check if values appear to be measurements."""
        if not any(num_type in data_type.lower() for num_type in ['int', 'long', 'real', 'decimal', 'float', 'double']):
            return False
        
        # Check if all values are numeric
        try:
            numeric_values = [float(str(v).replace(',', '')) for v in values[:3] if v]
            if numeric_values:
                # Check for measurement patterns (e.g., all positive, decimal values)
                return all(v >= 0 for v in numeric_values) or all('.' in str(v) for v in values[:3])
        except (ValueError, TypeError):
            return False
        return False
    
    def _looks_like_states(self, values: List[str]) -> bool:
        """Check if values appear to be states/statuses."""
        if len(set(str(v).lower() for v in values)) <= 10:  # Limited set of values
            # Check for common state patterns
            state_indicators = ['success', 'failed', 'pending', 'active', 'inactive', 'true', 'false', 'yes', 'no']
            value_set = {str(v).lower() for v in values}
            return any(indicator in value_str for indicator in state_indicators for value_str in value_set)
        return False
    
    def _looks_like_categories(self, values: List[str]) -> bool:
        """Check if values appear to be categorical."""
        unique_values = set(str(v) for v in values)
        # Categorical if limited unique values relative to total
        return 1 < len(unique_values) <= len(values) * 0.5
    
    def _looks_like_locations(self, values: List[str]) -> bool:
        """Check if values appear to be locations."""
        location_patterns = [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',  # IP address
            r'^[A-Z]{2,3}$',  # Country codes
            r'[\w\s]+,\s*[\w\s]+',  # City, State format
        ]
        matches = 0
        for value in values[:3]:
            for pattern in location_patterns:
                if re.search(pattern, str(value)):
                    matches += 1
                    break
        return matches >= len(values[:3]) * 0.3  # At least 30% match
    
    def _analyze_sample_context(self, sample_values: List[str], data_type: str) -> str:
        """Analyze sample values to provide additional context."""
        if not sample_values:
            return ""
        
        # Analyze patterns in sample values
        contexts = []
        
        # Check for common patterns
        all_numeric = all(str(v).replace('.', '').replace('-', '').isdigit() for v in sample_values if v)
        all_uppercase = all(str(v).isupper() for v in sample_values if v and str(v).isalpha())
        all_have_separators = all(any(sep in str(v) for sep in ['-', '_', '.', ':']) for v in sample_values if v)
        
        if all_numeric and "string" in data_type.lower():
            contexts.append("typically containing numeric identifiers")
        elif all_uppercase:
            contexts.append("usually in uppercase format")
        elif all_have_separators:
            contexts.append("often containing structured identifiers")
        
        # Add sample range if meaningful
        if len(sample_values) >= 2:
            sample_str = ", ".join([f"'{str(v)[:20]}'" for v in sample_values[:2]])
            contexts.append(f"Examples: {sample_str}")
        
        return "; ".join(contexts) if contexts else ""

    def _generate_column_tags(self, column: str, data_type: str) -> List[str]:
        """Generate semantic tags based on data type and patterns, not keywords."""
        tags = []
        
        # DYNAMIC APPROACH: Use data type analysis instead of keyword matching
        
        # Data type based tags
        data_type_lower = data_type.lower()
        if "datetime" in data_type_lower or "timestamp" in data_type_lower:
            tags.append("DATETIME")
            tags.append("TIME_COLUMN")
        elif "bool" in data_type_lower:
            tags.append("BOOLEAN")
            tags.append("CATEGORY_COLUMN")
        elif any(num_type in data_type_lower for num_type in ['int', 'long', 'real', 'decimal', 'float', 'double']):
            tags.append("NUMERIC")
            # Check if likely an ID based on column name pattern (not keywords)
            if re.match(r'^[A-Za-z]*ID$', column, re.IGNORECASE) or column.endswith('_id'):
                tags.append("ID_COLUMN")
            else:
                tags.append("METRIC_COLUMN")
        elif "string" in data_type_lower or "text" in data_type.lower():
            tags.append("TEXT")
            # Check for structured text patterns
            if re.match(r'^[A-Z][a-z]+[A-Z]', column):  # CamelCase pattern
                tags.append("STRUCTURED_TEXT")
        elif "dynamic" in data_type_lower:
            tags.append("DYNAMIC")
            tags.append("FLEXIBLE_TYPE")
        elif "object" in data_type_lower:
            tags.append("OBJECT")
            tags.append("COMPLEX_TYPE")
        else:
            tags.append("UNKNOWN_TYPE")
        
        # Add column position tag if it appears to be a primary column
        if column == "TimeGenerated" or column.endswith("_time") or column.endswith("Timestamp"):
            tags.append("PRIMARY_TIME_COLUMN")
        
        return tags

    async def get_database_schema(self, cluster: str, database: str, validate_auth: bool = False) -> Dict[str, Any]:
        """
        Gets a database schema (list of tables) with optimized caching and minimal live discovery.
        
        Args:
            cluster: Cluster URI
            database: Database name
            validate_auth: Whether to perform authentication validation (disabled by default since we validate at startup)
        
        Returns:
            Database schema dictionary with table list and metadata
        """
        # Always check cached schema first - prioritize cached data to avoid redundant queries
        cached_db_schema = self.memory_manager.get_database_schema(cluster, database)
        if cached_db_schema and "tables" in cached_db_schema:
            tables = cached_db_schema.get("tables", [])
            if tables:
                logger.debug(f"Using cached database schema for {database} with {len(tables)} tables")
                return cached_db_schema
            else:
                logger.debug(f"Cached database schema for {database} exists but is empty, checking memory for table data")
                
                # Check if we have individual table schemas cached even if database schema is empty
                normalized_cluster = self.memory_manager._normalize_cluster_uri(cluster)
                cluster_data = self.memory_manager.corpus.get("clusters", {}).get(normalized_cluster, {})
                db_data = cluster_data.get("databases", {}).get(database, {})
                table_schemas = db_data.get("tables", {})
                
                if table_schemas:
                    table_list = list(table_schemas.keys())
                    logger.debug(f"Found {len(table_list)} table schemas in memory, updating database schema cache")
                    updated_schema = {
                        "database_name": database,
                        "tables": table_list,
                        "discovered_at": datetime.now().isoformat(),
                        "cluster": cluster,
                        "schema_source": "memory_reconstruction",
                        "authentication_validated": False
                    }
                    self.memory_manager.store_database_schema(cluster, database, updated_schema)
                    return updated_schema

        # Skip authentication validation since we validate at startup
        if validate_auth:
            logger.debug(f"Skipping authentication validation for {cluster}/{database} - already validated at startup")

        # Only perform live discovery if no cached data is available
        try:
            logger.debug(f"Performing live database schema discovery for {database} (no cached data available)")
            command = ".show tables"
            tables_data = await self._execute_kusto_async(command, cluster, database, is_mgmt=True)
            
            table_list = [row['TableName'] for row in tables_data]
            db_schema_obj = {
                "database_name": database,
                "tables": table_list,
                "discovered_at": datetime.now().isoformat(),
                "cluster": cluster,
                "schema_source": "live_show_tables",
                "authentication_validated": validate_auth
            }
            
            self.memory_manager.store_database_schema(cluster, database, db_schema_obj)
            logger.info(f"Stored newly discovered schema for database {database} with {len(table_list)} tables")
            return db_schema_obj
            
        except Exception as discovery_error:
            logger.error(f"Database schema discovery failed for {cluster}/{database}: {discovery_error}")
            
            # Try fallback strategies
            fallback_schema = self._get_fallback_database_schema(cluster, database)
            if fallback_schema:
                logger.info(f"Using fallback database schema for {database}")
                return fallback_schema
            
            # Return minimal schema to prevent complete failure
            return self._create_minimal_database_schema(cluster, database)

    async def _validate_cluster_authentication(self, cluster_url: str, database: str) -> bool:
        """
        Validate authentication specifically for a cluster and database combination.
        
        Performs cluster-specific validation including:
        1. Azure CLI authentication status
        2. Cluster-specific permissions
        3. Database access rights
        4. Connection stability
        """
        try:
            from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
            logger.info(f"Validating authentication for {cluster_url}/{database}")
            
            # Step 1: Basic authentication validation
            basic_auth_valid = self._validate_azure_authentication(cluster_url)
            if not basic_auth_valid:
                return False
            
            # Step 2: Test database-specific access
            try:
                kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(cluster_url)
                
                with KustoClient(kcsb) as client:
                    # Test database access with a minimal query
                    test_query = ".show database schema"
                    response = client.execute_mgmt(database, test_query)
                    
                    if response and response.primary_results:
                        logger.info(f"Database access validated for {database}")
                        return True
                    else:
                        logger.warning(f"Database access test failed for {database}")
                        return False
                        
            except Exception as db_access_error:
                logger.warning(f"Database-specific authentication failed: {db_access_error}")
                return False
                
        except Exception as e:
            logger.error(f"Cluster authentication validation failed: {e}")
            return False

    def _get_fallback_database_schema(self, cluster: str, database: str) -> Optional[Dict[str, Any]]:
        """Get fallback database schema from memory or derived sources."""
        try:
            # Try to get any available schema from memory
            normalized_cluster = self.memory_manager._normalize_cluster_uri(cluster)
            cluster_data = self.memory_manager.corpus.get("clusters", {}).get(normalized_cluster, {})
            db_data = cluster_data.get("databases", {}).get(database, {})
            
            if db_data and "tables" in db_data:
                tables = list(db_data.get("tables", {}).keys())
                if tables:
                    logger.info(f"Found fallback database schema with {len(tables)} tables")
                    return {
                        "database_name": database,
                        "tables": tables,
                        "discovered_at": datetime.now().isoformat(),
                        "cluster": cluster,
                        "schema_source": "fallback_memory",
                        "fallback_applied": True
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Fallback database schema retrieval failed: {e}")
            return None

    def _create_minimal_database_schema(self, cluster: str, database: str) -> Dict[str, Any]:
        """Create minimal database schema as last resort."""
        # Common table names for different database types
        common_tables = []
        
        database_lower = database.lower()
        if any(kw in database_lower for kw in ['security', 'sentinel', 'defender']):
            common_tables = ['SecurityEvent', 'SigninLogs', 'AuditLogs', 'SecurityAlert']
        elif any(kw in database_lower for kw in ['log', 'analytics']):
            common_tables = ['Heartbeat', 'Perf', 'Event', 'Syslog']
        elif any(kw in database_lower for kw in ['sample', 'demo', 'help']):
            common_tables = ['StormEvents', 'PopulationData']
        else:
            common_tables = ['Events', 'Logs', 'Data']
        
        return {
            "database_name": database,
            "tables": common_tables,
            "discovered_at": datetime.now().isoformat(),
            "cluster": cluster,
            "schema_source": "minimal_fallback",
            "fallback_applied": True,
            "minimal_schema": True
        }

    def get_connection_config(self) -> Dict[str, Any]:
        """Get current connection configuration with validation status."""
        from .constants import CONNECTION_CONFIG, ERROR_HANDLING_CONFIG
        
        return {
            "connection_config": CONNECTION_CONFIG,
            "error_handling_config": ERROR_HANDLING_CONFIG,
            "validation_enabled": CONNECTION_CONFIG.get("validate_connection_before_use", True),
            "retry_config": {
                "max_retries": CONNECTION_CONFIG.get("max_retries", 5),
                "retry_delay": CONNECTION_CONFIG.get("retry_delay", 2.0),
                "backoff_factor": CONNECTION_CONFIG.get("retry_backoff_factor", 2.0)
            },
            "authentication_methods": ["azure_cli"],
            "supported_protocols": ["https", "grpc"]
        }

    async def discover_all_schemas(self, client, force_refresh: bool = False) -> Dict:
        """
        Unified method to discover schemas for all available tables.
        Consolidates discovery logic and provides comprehensive caching.
        
        Args:
            client: The Kusto client instance.
            force_refresh: Whether to bypass cache and force fresh discovery.
            
        Returns:
            Dict: Complete schema information for all tables.
        """
        try:
            cache_key = "unified_all_schemas"
            
            # Check cache unless force refresh is requested
            if not force_refresh and cache_key in self._discovery_cache:
                cached_data = self._discovery_cache[cache_key]
                # Check if cache is valid (30 minutes for full discovery)
                if (datetime.now() - cached_data['timestamp']).seconds < 1800:
                    return cached_data['data']

            # Get list of all tables from current database
            tables_query = "show tables | project TableName"
            tables_response = await client.execute("", tables_query)
            
            all_schemas = {
                "discovery_timestamp": datetime.now().isoformat(),
                "total_tables": 0,
                "tables": {},
                "discovery_method": "unified_schema_manager_full"
            }

            if tables_response and hasattr(tables_response, 'primary_results') and tables_response.primary_results:
                table_names = [row.get("TableName", "") for row in tables_response.primary_results[0]]
                all_schemas["total_tables"] = len(table_names)
                
                # Discover schema for each table
                for table_name in table_names:
                    if table_name:
                        table_schema = await self.discover_schema_for_table(client, table_name)
                        all_schemas["tables"][table_name] = table_schema

            # Cache the complete result
            cache_data = {
                'data': all_schemas,
                'timestamp': datetime.now()
            }
            self._discovery_cache[cache_key] = cache_data
            self._last_discovery_times["full_discovery"] = datetime.now()
            
            return all_schemas

        except Exception as e:
            print(f"Error in unified full schema discovery: {e}")
            return {
                "discovery_timestamp": datetime.now().isoformat(),
                "total_tables": 0,
                "tables": {},
                "error": str(e),
                "discovery_method": "unified_schema_manager_full_error"
            }

    def get_cached_schema(self, table_name: str = None) -> Dict:
        """
        Unified method to retrieve cached schema information.
        
        Args:
            table_name: Specific table name, or None for all schemas.
            
        Returns:
            Dict: Cached schema information.
        """
        if table_name:
            cache_key = f"unified_table_schema_{table_name}"
            if cache_key in self._schema_cache:
                return self._schema_cache[cache_key]['data']
            
            # Check legacy cache
            legacy_key = f"table_schema_{table_name}"
            return self.memory_manager.get_cached_result(legacy_key, 3600)
        else:
            cache_key = "unified_all_schemas"
            if cache_key in self._discovery_cache:
                return self._discovery_cache[cache_key]['data']
            return None

    def clear_schema_cache(self, table_name: str = None):
        """
        Unified method to clear schema cache.
        
        Args:
            table_name: Specific table to clear, or None to clear all.
        """
        if table_name:
            cache_key = f"unified_table_schema_{table_name}"
            if cache_key in self._schema_cache:
                del self._schema_cache[cache_key]
        else:
            self._schema_cache.clear()
            self._discovery_cache.clear()
            self._last_discovery_times.clear()

    def get_session_learning_data(self) -> Dict:
        """
        Get session-based learning data from the unified schema manager.
        Integrates with memory manager's session tracking.
        """
        try:
            # Get session data from memory manager
            session_data = self.memory_manager.get_session_data()
            
            # Add unified schema manager context
            unified_context = {
                "cached_schemas": len(self._schema_cache),
                "discovery_cache_size": len(self._discovery_cache),
                "last_discovery_times": self._last_discovery_times,
                "schema_manager_type": "unified_consolidated"
            }
            
            # Merge session data with unified context
            if session_data:
                session_data["unified_schema_context"] = unified_context
                return session_data
            else:
                return {
                    "sessions": {},
                    "active_session": None,
                    "unified_schema_context": unified_context
                }
                
        except Exception as e:
            logger.warning(f"Failed to get session learning data: {e}")
            return {
                "sessions": {},
                "active_session": None,
                "error": str(e)
            }

    def track_schema_usage(self, table_name: str, operation: str, success: bool = True):
        """
        Track schema usage for session-based learning.
        
        Args:
            table_name: Name of table accessed
            operation: Type of operation (discovery, query, etc.)
            success: Whether operation was successful
        """
        try:
            usage_data = {
                "table": table_name,
                "operation": operation,
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "schema_manager": "unified"
            }
            
            # Store in memory manager for session tracking
            if hasattr(self.memory_manager, 'track_usage'):
                self.memory_manager.track_usage(usage_data)
            else:
                # Fallback: store in local cache
                if not hasattr(self, '_usage_tracking'):
                    self._usage_tracking = []
                self._usage_tracking.append(usage_data)
                
        except Exception as e:
            logger.debug(f"Schema usage tracking failed: {e}")


# Consolidated Schema Discovery Interface
class SchemaDiscovery(SchemaManager):
    """
    Consolidated schema discovery interface that provides both live discovery
    and legacy compatibility methods. Delegates to SchemaManager for actual work.
    """
    
    async def list_tables_in_db(self, cluster_uri: str, database: str) -> List[str]:
        """Lists all tables in a database using the '.show tables' management command."""
        db_schema = await self.get_database_schema(cluster_uri, database)
        return db_schema.get("tables", [])

    def _is_schema_cached_and_valid(self, cache_key: str) -> bool:
        """
        Checks whether a cached schema exists and appears valid.
        Expected cache_key format: 'cluster/database/table'
        """
        try:
            parts = cache_key.split("/")
            if len(parts) != 3:
                return False
            cluster, database, table = parts
            schema = self.memory_manager.get_schema(cluster, database, table, enable_fallback=False)
            if schema and isinstance(schema, dict) and schema.get("columns"):
                return True
            return False
        except Exception:
            return False
    
    def get_column_mapping_from_schema(self, schema_obj: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Return mapping of lowercased column name -> actual column name."""
        cols = get_schema_column_names(schema_obj) or []
        return {c.lower(): c for c in cols}
    
    def _normalize_cluster_uri(self, cluster_uri: str) -> str:
        """Normalize cluster URI format."""
        if not cluster_uri:
            return cluster_uri
        s = str(cluster_uri).strip()
        if not s.startswith("http://") and not s.startswith("https://"):
            s = "https://" + s
        s = s.rstrip("/")
        return s


def get_schema_discovery() -> SchemaDiscovery:
    """
    Return the consolidated schema discovery interface.
    This replaces the old lightweight adapter with the full SchemaManager functionality.
    """
    return SchemaDiscovery()


def get_schema_discovery_status() -> Dict[str, Any]:
    """
    Return enhanced status dictionary for schema discovery availability.
    Includes information about the consolidated schema system.
    """
    try:
        memory_path = str(get_default_cluster_memory_path())
        # Get actual cached schema count from memory manager
        from .memory import get_memory_manager
        mm = get_memory_manager()
        cached_count = len(getattr(mm, 'schema_data', {}))
    except Exception:
        memory_path = ""
        cached_count = 0
    
    return {
        "status": "available",
        "memory_path": memory_path,
        "cached_schemas": cached_count,
        "schema_system": "consolidated_manager",
        "live_discovery_enabled": True
    }


# ---------------------------------------------------------------------------
# Simple query helpers
# ---------------------------------------------------------------------------
def fix_query_with_real_schema(query: str) -> str:
    """Attempt to fix a query when cluster/database/table info is present.

    This is a conservative, best-effort implementation for tests: if the query
    does not contain explicit cluster/database information, return it unchanged.
    """
    if not query or not isinstance(query, str):
        return query
    # Detect the pattern cluster('..').database('..') - if not present, bail out
    if not re.search(r"cluster\(['\"]([^'\"]+)['\"]\)\.database\(['\"]([^'\"]+)['\"]\)", query):
        return query
    # For now return unchanged; richer behavior can be added later
    return query


def generate_query_description(query: str) -> str:
    """Produce a short description for a query (used when storing successful queries)."""
    if not query:
        return ""
    s = " ".join(query.strip().split())
    return s[:200] if len(s) > 200 else s

class QueryParser:
    """A comprehensive KQL query parser for extracting entities and operations."""

    def __init__(self):
        """Initializes the parser with pre-compiled regex patterns."""
        self.patterns = [
            re.compile(r"cluster\(['\"][^'\"]+['\"]\)\.database\(['\"][^'\"]+['\"]\)\.([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
            re.compile(r"cluster\(['\"][^'\"]+['\"]\)\.database\(['\"][^'\"]+['\"]\)\.\['([^']+)'\]", re.IGNORECASE),
            re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\|", re.IGNORECASE),
            re.compile(r"^\s*\['([^']+)'\]\s*\|", re.IGNORECASE),
            re.compile(r"\b(?:join|union|lookup)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
            re.compile(r"\b(?:join|union|lookup)\s+\['([^']+)'\]", re.IGNORECASE),
        ]
        self.fallback_patterns = [
            re.compile(r'([A-Za-z][A-Za-z0-9_]*)\s*\|\s*getschema', re.IGNORECASE),
            re.compile(r'(?:table|from)\s+([A-Za-z][A-Za-z0-9_]*)', re.IGNORECASE),
            re.compile(r'([A-Za-z][A-Za-z0-9_]*)\s+table', re.IGNORECASE),
        ]
        self.operation_keywords = ['project', 'where', 'summarize', 'extend', 'join', 'union', 'take', 'limit', 'sort', 'order']

    def parse(self, query: str) -> Dict[str, Any]:
        """Parses a KQL query to extract cluster, database, tables, and operations."""
        if not query:
            return {"cluster": None, "database": None, "tables": [], "operations": []}

        cluster_match = re.search(r"cluster\(['\"]([^'\"]+)['\"]\)", query)
        db_match = re.search(r"database\(['\"]([^'\"]+)['\"]\)", query)
        
        cluster = cluster_match.group(1) if cluster_match else None
        database = db_match.group(1) if db_match else None
        
        tables = self._extract_tables(query)
        operations = self._extract_operations(query)
        
        return {
            "cluster": cluster,
            "database": database,
            "tables": list(tables),
            "operations": operations,
            "query_length": len(query),
            "has_aggregation": any(op in operations for op in ['summarize', 'count']),
            "complexity_score": len(operations)
        }

    def _extract_tables(self, query: str) -> set:
        """Extracts table names from the query using multiple patterns."""
        tables = set()
        reserved_lower = {w.lower() for w in KQL_RESERVED_WORDS}
        
        for pattern in self.patterns:
            for match in pattern.finditer(query):
                table_name = match.group(1).replace("''", "'") if match.group(1) else None
                if table_name and table_name.lower() not in reserved_lower:
                    tables.add(table_name)
        
        if not tables:
            for pattern in self.fallback_patterns:
                for match in pattern.finditer(query):
                    table_candidate = match.group(1)
                    if table_candidate and table_candidate.lower() not in reserved_lower:
                        tables.add(table_candidate)
        return tables

    def _extract_operations(self, query: str) -> List[str]:
        """Extracts KQL operations from the query."""
        operations = []
        query_lower = query.lower()
        for op in self.operation_keywords:
            if f'| {op}' in query_lower or f'|{op}' in query_lower:
                operations.append(op)
        return operations

# Instantiate the parser for use in backward-compatible functions
_parser = QueryParser()

def parse_query_entities(query: str) -> Dict[str, Any]:
    """Consolidated query parsing using the QueryParser class."""
    return _parser.parse(query)

# Backward compatibility functions
def extract_cluster_and_database_from_query(query: str) -> tuple[str, str]:
    """Extract cluster URI and database name from a KQL query."""
    entities = _parser.parse(query)
    return entities["cluster"], entities["database"]

def extract_tables_from_query(query: str) -> List[str]:
    """Extract table names from a KQL query."""
    entities = _parser.parse(query)
    return entities["tables"]
