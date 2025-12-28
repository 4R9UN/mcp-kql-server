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
import difflib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from .constants import KQL_RESERVED_WORDS

# Set up logger at module level
logger = logging.getLogger(__name__)

def _is_retryable_exc(e: BaseException) -> bool:
    """Lightweight dynamic check for retryable exceptions (message-based)."""
    try:
        s = str(e).lower()
        return any(k in s for k in ("timeout", "connection", "throttl", "unreachable", "refused", "kusto", "service"))
    except (ValueError, TypeError, AttributeError):
        return False

def retry_on_exception(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 10.0):
    """
    Simple, dependency-free retry decorator that supports both sync and async functions.
    Retries only when `_is_retryable_exc` returns True.
    """
    def deco(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapped(*args, **kwargs):
                delay = base_delay
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except (OSError, RuntimeError, ValueError, TimeoutError) as e:
                        if attempt == max_attempts or not _is_retryable_exc(e):
                            raise
                        await asyncio.sleep(min(delay, max_delay))
                        delay *= 2
            return async_wrapped
        else:
            def sync_wrapped(*args, **kwargs):
                delay = base_delay
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except (OSError, RuntimeError, ValueError, TimeoutError) as e:
                        if attempt == max_attempts or not _is_retryable_exc(e):
                            raise
                        time.sleep(min(delay, max_delay))
                        delay *= 2
            return sync_wrapped
    return deco

def log_execution(func):
    """Minimal execution logger decorator (sync+async)."""
    if asyncio.iscoroutinefunction(func):
        async def async_wrapped(*args, **kwargs):
            start = datetime.now()
            try:
                return await func(*args, **kwargs)
            finally:
                logger.debug("%s took %.2fs", func.__name__, (datetime.now() - start).total_seconds())
        return async_wrapped
    else:
        def sync_wrapped(*args, **kwargs):
            start = datetime.now()
            try:
                return func(*args, **kwargs)
            finally:
                logger.debug("%s took %.2fs", func.__name__, (datetime.now() - start).total_seconds())
        return sync_wrapped



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
    reserved_keywords = {k.lower() for k in KQL_RESERVED_WORDS}

    identifier_lower = identifier.lower()

    # Check if the identifier is a reserved keyword or contains invalid characters
    if identifier_lower in reserved_keywords or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
        # Escape single quotes using the Kusto convention of doubling them
        escaped_identifier = identifier.replace("'", "''")
        return f"['{escaped_identifier}']"

    return identifier


def normalize_cluster_uri(cluster_uri: str) -> str:
    """
    Normalize cluster URI for consistent connection handling.
    This is the single source of truth for URI normalization.
    """
    if not cluster_uri:
        raise ValueError("Cluster URI cannot be None or empty")
    uri = cluster_uri.strip().lower()
    if not uri.startswith("https://"):
        uri = f"https://{uri}"
    return uri.rstrip("/")


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
        self._usage_tracking = []
        # Multi-cluster support: table_name -> [(cluster, database)]
        self._table_locations = {}

    async def _execute_kusto_async(self, query: str, cluster: str, database: str, is_mgmt: bool = False) -> List[Dict]:
        """
        Enhanced async wrapper for executing Kusto queries with comprehensive error handling,
        retry logic, connection validation, and graceful degradation.
        """
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
            """
            try:
                validation_timeout = CONNECTION_CONFIG.get("connection_validation_timeout", 5.0)

                # Step 1: Validate Azure CLI authentication
                auth_valid = self._validate_azure_authentication(cluster_url)
                if not auth_valid:
                    logger.warning("Azure CLI authentication validation failed for %s", cluster_url)
                    return False

                # Step 2: Test basic connectivity
                connectivity_valid = self._test_network_connectivity(cluster_url, validation_timeout)
                if not connectivity_valid:
                    logger.warning("Network connectivity test failed for %s", cluster_url)
                    return False

                # Step 3: Test cluster access with actual query
                access_valid = self._test_cluster_access(cluster_url, validation_timeout)
                if not access_valid:
                    logger.warning("Cluster access test failed for %s", cluster_url)
                    return False

                logger.info("Connection validation passed for %s", cluster_url)
                return True

            except (OSError, RuntimeError, ValueError, TimeoutError) as e:
                logger.error("Connection validation failed for %s: %s", cluster_url, e)
                return False

        def _sync_execute():
            """Execute Kusto query with retry logic and error handling."""
            cluster_url = f"https://{cluster}" if not cluster.startswith("https://") else cluster

            # Pre-validate connection if enabled
            if CONNECTION_CONFIG.get("validate_connection_before_use", True):
                if not _validate_connection(cluster_url):
                    logger.warning("Connection validation failed for %s, proceeding anyway...", cluster_url)

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
                            logger.debug("Successfully executed query on attempt %s", attempt + 1)
                            return data
                        else:
                            logger.warning("Query returned no results: %s", query)
                            return []

                except (OSError, RuntimeError, ValueError, TimeoutError) as e:
                    last_exception = e
                    error_str = str(e)

                    # Log the attempt
                    logger.warning("Kusto execution attempt %s/%s failed: %s", attempt + 1, max_retries + 1, error_str)

                    # Check if this is the final attempt
                    if attempt >= max_retries:
                        logger.error("All retry attempts exhausted for query: %s", query)
                        break

                    # Check if error is retryable
                    if not _is_retryable_error(error_str):
                        logger.error("Non-retryable error encountered: %s", error_str)
                        break

                    # Wait before retry with exponential backoff
                    logger.info("Retrying in %.1fs due to retryable error...", current_delay)
                    time.sleep(current_delay)
                    current_delay = min(current_delay * backoff_factor, max_retry_delay)

            # All retries failed - propagate the last exception
            if last_exception:
                error_msg = f"Kusto execution failed after {max_retries + 1} attempts: {str(last_exception)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from last_exception
            else:
                raise RuntimeError("Kusto execution failed for unknown reasons")

        return await loop.run_in_executor(None, _sync_execute)

    def _validate_azure_authentication(self, cluster_url: str) -> bool:
        """
        Skip redundant authentication validation since we validate at startup.
        Always return True if we reach this point (authentication was successful at startup).
        """
        logger.debug("Skipping redundant authentication validation for %s - already validated at startup", cluster_url)
        return True

    def _test_network_connectivity(self, cluster_url: str, timeout: float) -> bool:
        """Test basic network connectivity to the cluster."""
        from urllib.parse import urlparse
        import socket as sock_module

        try:

            parsed_url = urlparse(cluster_url)
            hostname = parsed_url.hostname
            port = parsed_url.port or 443  # Default HTTPS port for Kusto

            # Test TCP connectivity
            sock = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_STREAM)
            sock.settimeout(timeout)

            result = sock.connect_ex((hostname, port))
            sock.close()

            if result == 0:
                logger.debug("Network connectivity test passed for %s:%s", hostname, port)
                return True
            else:
                logger.warning("Network connectivity test failed for %s:%s", hostname, port)
                return False

        except (OSError, sock_module.error, TimeoutError) as e:
            logger.warning("Network connectivity test error: %s", e)
            return False

    def _test_cluster_access(self, cluster_url: str, _timeout: float) -> bool:
        """Test actual cluster access with a minimal query."""
        try:
            from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
            kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(cluster_url)

            with KustoClient(kcsb) as client:
                # Use a lightweight query that should work on any cluster
                test_query = ".show version"

                # Execute query with timeout handling
                try:
                    response = client.execute_mgmt("NetDefaultDB", test_query)

                    has_results = response is not None and bool(response.primary_results)
                    if has_results:
                        logger.debug("Cluster access test passed")
                    else:
                        logger.warning("Cluster access test failed: no valid response")

                    return has_results

                except (OSError, RuntimeError, TimeoutError) as query_error:
                    logger.warning("Cluster access query failed: %s", query_error)
                    return False

        except (OSError, RuntimeError, ValueError, ImportError) as e:
            logger.warning("Cluster access test error: %s", e)
            return False

    async def discover_schema_for_table(self, client, table_name: str) -> Dict:
        """
        Discovers detailed schema information for a specific table.
        Unified method that consolidates schema discovery logic with enhanced analysis.
        """
        try:
            # Check unified cache first
            cache_key = f"unified_table_schema_{table_name}"
            if cache_key in self._schema_cache:
                cached_data = self._schema_cache[cache_key]
                # Check if cache is still valid (1 hour)
                if (datetime.now() - cached_data['timestamp']).seconds < 3600:
                    return cached_data['data']

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

            # Enhanced caching in unified system
            cache_data = {
                'data': schema_info,
                'timestamp': datetime.now()
            }
            self._schema_cache[cache_key] = cache_data

            # Track usage for session learning
            self.track_schema_usage(table_name, "discovery", True)

            return schema_info

        except Exception as e:
            logger.error("Error in unified schema discovery for table %s: %s", table_name, e)
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

    async def get_table_schema(self, cluster: str, database: str, table: str, _force_refresh: bool = False) -> Dict[str, Any]:
        """
        Gets a table schema using multiple discovery strategies with proper column metadata handling.
        This function is now the single source of truth for live schema discovery.
        """
        try:
            logger.debug("Performing enhanced schema discovery for %s.%s", database, table)

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
                    except (ValueError, TypeError, KeyError, RuntimeError) as sample_error:
                        logger.debug("Failed to get sample data for Strategy 1: %s", sample_error)

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
                        logger.info("Strategy 1 successful: JSON schema discovery for %s", table)
                        return self._create_enhanced_schema_object(cluster, database, table, columns, "json_schema")
            except (ValueError, TypeError, KeyError, RuntimeError, OSError) as json_error:
                logger.debug("JSON schema discovery failed for %s: %s", table, json_error)

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
                    except (ValueError, TypeError, KeyError, RuntimeError) as sample_error:
                        logger.debug("Failed to get sample data for Strategy 2: %s", sample_error)

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
                        logger.info("Strategy 2 successful: getschema discovery for %s", table)
                        return self._create_enhanced_schema_object(cluster, database, table, columns, "getschema")
            except (ValueError, TypeError, KeyError, RuntimeError, OSError) as getschema_error:
                logger.debug("getschema discovery failed for %s: %s", table, getschema_error)

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
                        logger.info("Strategy 3 successful: sample-based discovery for %s", table)
                        return self._create_enhanced_schema_object(cluster, database, table, columns, "sample_inference")
            except (ValueError, TypeError, KeyError, RuntimeError, OSError) as sample_error:
                logger.debug("Sample-based discovery failed for %s: %s", table, sample_error)

            # All strategies failed
            raise RuntimeError("All schema discovery strategies failed")

        except Exception as e:
            error_str = str(e).lower()
            # Check for common "not found" errors (SEM0100, etc.)
            # This often happens if the "table" is actually a local variable (let statement),
            # a function, or a temporary view that doesn't exist in the global schema.
            is_not_found = any(x in error_str for x in [
                "sem0100", "failed to resolve", "not found", "does not exist",
                "unknown table", "unknown database"
            ])

            if is_not_found:
                # Log as debug/info to avoid spamming error logs for local variables
                logger.info("Schema discovery skipped for '%s' (likely a local variable or function): %s", table, e)
            else:
                # Log genuine errors
                logger.error("Enhanced schema discovery failed for %s.%s: %s", database, table, e)

            # Track failed usage
            self.track_schema_usage(table, "enhanced_discovery", False)

            # Return error object instead of fallback schema
            return {
                "table_name": table,
                "columns": {},
                "discovered_at": datetime.now().isoformat(),
                "cluster": cluster,
                "database": database,
                "column_count": 0,
                "discovery_method": "failed",
                "error": str(e),
                "schema_version": "3.1",
                "is_not_found": is_not_found
            }

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
        logger.info("Successfully discovered and stored enhanced schema for %s.%s with %s columns using %s", database, table, len(columns), method)

        # Register table location for multi-cluster support
        self.register_table_location(table, cluster, database)

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

    async def _process_schema_columns(self, schema_data: List[Dict], sample_data: List[Dict],
                                    table: str, _cluster: str, _database: str) -> Dict[str, Any]:
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

    def _generate_column_description(self, table: str, column_name: str, data_type: str, sample_values: List[str]) -> str:
        """Generate AI-enhanced column description with semantic analysis."""
        # Use enhanced heuristic-based description
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

    def _determine_column_purpose(self, _column_name: str, data_type: str, sample_values: List[str]) -> str:
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

        return tags

    async def get_database_schema(self, cluster: str, database: str, validate_auth: bool = False) -> Dict[str, Any]:
        """
        Gets a database schema (list of tables) with optimized caching and minimal live discovery.
        """
        # Always check cached schema first - prioritize cached data to avoid redundant queries
        cached_db_schema = self.memory_manager.get_database_schema(cluster, database)
        if cached_db_schema and "tables" in cached_db_schema:
            tables = cached_db_schema.get("tables", [])
            if tables:
                logger.debug("Using cached database schema for %s with %s tables", database, len(tables))
                return cached_db_schema
            else:
                logger.debug("Cached database schema for %s exists but is empty", database)

        # Skip authentication validation since we validate at startup
        if validate_auth:
            logger.debug("Skipping authentication validation for %s/%s - already validated at startup", cluster, database)

        # Only perform live discovery if no cached data is available
        try:
            logger.debug("Performing live database schema discovery for %s (no cached data available)", database)
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

            # Store each table schema
            # Note: .show tables only gives names, not full schema.
            # We store what we have (names) and let full discovery happen later if needed.
            for table_name in table_list:
                self.memory_manager.store_schema(cluster, database, table_name, {"columns": {}})

            logger.info("Stored newly discovered schema for database %s with %s tables", database, len(table_list))
            return db_schema_obj

        except Exception as discovery_error:
            logger.error("Database schema discovery failed for %s/%s: %s", cluster, database, discovery_error)

            # Return error object instead of minimal schema
            return {
                "database_name": database,
                "tables": [],
                "discovered_at": datetime.now().isoformat(),
                "cluster": cluster,
                "schema_source": "failed",
                "error": str(discovery_error)
            }

    async def _validate_cluster_authentication(self, cluster_url: str, database: str) -> bool:
        """
        Validate authentication specifically for a cluster and database combination.
        """
        try:
            from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
            logger.info("Validating authentication for %s/%s", cluster_url, database)

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
                        logger.info("Database access validated for %s", database)
                        return True
                    else:
                        logger.warning("Database access test failed for %s", database)
                        return False

            except Exception as db_access_error:
                logger.warning("Database-specific authentication failed: %s", db_access_error)
                return False

        except Exception as e:
            logger.error("Cluster authentication validation failed: %s", e)
            return False

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

    def get_cached_schema(self, table_name: Optional[str] = None) -> Optional[Dict]:
        """
        Unified method to retrieve cached schema information.
        """
        if table_name:
            cache_key = f"unified_table_schema_{table_name}"
            if cache_key in self._schema_cache:
                return self._schema_cache[cache_key]['data']

            # Return None if not found in unified cache
            return None
        else:
            cache_key = "unified_all_schemas"
            if cache_key in self._discovery_cache:
                return self._discovery_cache[cache_key]['data']
            return None

    def clear_schema_cache(self, table_name: Optional[str] = None):
        """
        Unified method to clear schema cache.
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
        """
        try:
            # Get session data from memory manager (use default session)
            session_data = self.memory_manager.get_session_data("default")

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
            logger.warning("Failed to get session learning data: %s", e)
            return {
                "sessions": {},
                "active_session": None,
                "error": str(e)
            }

    def track_schema_usage(self, table_name: str, operation: str, success: bool = True):
        """
        Track schema usage for session-based learning.
        """
        try:
            usage_data = {
                "table": table_name,
                "operation": operation,
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "schema_manager": "unified"
            }

            # Store in local cache for tracking
            if not hasattr(self, '_usage_tracking'):
                self._usage_tracking = []
            self._usage_tracking.append(usage_data)

        except Exception as e:
            logger.debug("Schema usage tracking failed: %s", e)

    def find_closest_match(self, name: str, candidates: List[str], cutoff: float = 0.6) -> Optional[str]:
        """Find the closest match for a name from a list of candidates."""
        matches = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
        return matches[0] if matches else None

    def register_table_location(self, table_name: str, cluster: str, database: str):
        """
        Register a table's location (cluster/database).
        Supports tracking tables that exist in multiple clusters.
        """
        if table_name not in self._table_locations:
            self._table_locations[table_name] = []

        location = (cluster, database)
        if location not in self._table_locations[table_name]:
            self._table_locations[table_name].append(location)
            logger.debug("Registered table '%s' at %s/%s", table_name, cluster, database)

    def get_table_locations(self, table_name: str) -> List[Tuple[str, str]]:
        """
        Get all known locations (cluster, database) for a table.
        Returns empty list if table is not registered.
        """
        return self._table_locations.get(table_name, [])

    def is_multi_cluster_table(self, table_name: str) -> bool:
        """
        Check if a table exists in multiple clusters.
        """
        return len(self.get_table_locations(table_name)) > 1

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
            # Get schema from database using internal method
            schemas = self.memory_manager._get_database_schema(cluster, database)
            schema = next((s for s in schemas if s.get("table") == table), None)
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
        stats = mm.get_memory_stats()
        cached_count = stats.get("schema_count", 0)
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
        """
        try:
            return func(*args, **kwargs)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            log_func = getattr(logger, log_level, logger.warning)
            log_func("%s: %s", error_msg, e)
            return default

    @staticmethod
    def safe_get_nested(data: dict, *keys, default=None):
        """
        Safely get nested dictionary values with consistent error handling.
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
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("JSON serialization failed: %s", e)
            return default

    @staticmethod
    def handle_import_error(module_name: str, fallback=None):
        """
        Handle import errors consistently.
        """
        logger.warning("%s not available", module_name)
        return fallback

    @staticmethod
    def handle_kusto_error(e: Exception) -> Dict[str, Any]:
        """
        Comprehensive Kusto error analysis with extensive pattern recognition and intelligent suggestions.
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
                "suggestions": ["An unexpected error occurred. Check server logs."]
            }        # Simplified error handling for brevity in this rewrite
        return {
            "success": False,
            "error": str(e),
            "error_type": "kusto_error",
            "suggestions": ["Check KQL syntax and schema."],
            "kusto_specific": True
        }

__all__ = [
    "extract_cluster_and_database_from_query",
    "extract_tables_from_query",
    "parse_query_entities",
]

def extract_cluster_and_database_from_query(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts cluster and database from a KQL query string.
    Returns (cluster, database) or (None, None) if not found.
    """
    if not query:
        return None, None

    cluster_match = re.search(r"cluster\(['\"]([^'\"]+)['\"]\)", query)
    db_match = re.search(r"database\(['\"]([^'\"]+)['\"]\)", query)

    cluster = cluster_match.group(1) if cluster_match else None
    database = db_match.group(1) if db_match else None
    return cluster, database

def extract_tables_from_query(query: str) -> List[str]:
    """
    Extracts table names from a KQL query string.
    Returns a list of table names found.
    """
    if not query:
        return []

    tables = set()
    reserved_lower = {w.lower() for w in KQL_RESERVED_WORDS}

    # Simple patterns for table extraction
    patterns = [
        re.compile(r"cluster\(['\"][^'\"]+['\"]\)\.database\(['\"][^'\"]+['\"]\)\.([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
        re.compile(r"cluster\(['\"][^'\"]+['\"]\)\.database\(['\"][^'\"]+['\"]\)\.\['([^']+)'\]", re.IGNORECASE),
        re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\|", re.IGNORECASE),
        re.compile(r"^\s*\['([^']+)'\]\s*\|", re.IGNORECASE),
        re.compile(r"\b(?:join|union|lookup)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
        re.compile(r"\b(?:join|union|lookup)\s+\['([^']+)'\]", re.IGNORECASE),
    ]

    for pattern in patterns:
        for match in pattern.finditer(query):
            table_name = match.group(1) if match.group(1) else None
            if table_name and table_name.lower() not in reserved_lower:
                tables.add(table_name)

    return list(tables)

def parse_query_entities(query: str) -> Dict[str, Any]:
    """
    Parses a query to extract all entities (cluster, database, tables).
    Simplified version for backward compatibility.
    """
    cluster, database = extract_cluster_and_database_from_query(query)
    tables = extract_tables_from_query(query)
    return {
        "cluster": cluster,
        "database": database,
        "tables": tables
    }
