"""
KQL Query Validator

Pre-execution validation to catch errors before sending queries to Kusto.
Validates column names, table names, and operator syntax against schema.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import re
import logging
from typing import Dict, List, Any

from .constants import KQL_RESERVED_WORDS

logger = logging.getLogger(__name__)


class KQLValidator:
    """Validates KQL queries against schema before execution."""

    def __init__(self, memory_manager, schema_manager):
        """
        Initialize validator with memory and schema managers.

        Args:
            memory_manager: MemoryManager instance for schema lookup
            schema_manager: SchemaManager instance for live schema discovery
        """
        self.memory = memory_manager
        self.schema_manager = schema_manager

    async def validate_query(
        self,
        query: str,
        cluster: str,
        database: str,
        auto_discover: bool = True
    ) -> Dict[str, Any]:
        """
        Validate KQL query against schema.

        Args:
            query: KQL query to validate
            cluster: Cluster URL
            database: Database name
            auto_discover: Whether to auto-discover missing schemas

        Returns:
            Dict with validation results:
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str],
                "suggestions": List[str],
                "tables_used": List[str],
                "columns_validated": int
            }
        """
        errors = []
        warnings = []
        suggestions = []

        try:
            # Extract tables from query
            tables = self._extract_tables(query)
            logger.debug("Extracted tables from query: %s", tables)

            if not tables:
                warnings.append("No tables detected in query")
                return {
                    "valid": True,
                    "errors": [],
                    "warnings": warnings,
                    "suggestions": ["Ensure your query references at least one table"],
                    "tables_used": [],
                    "columns_validated": 0
                }

            # Ensure schemas exist for all tables
            if auto_discover:
                await self._ensure_schemas_exist(cluster, database, tables)

            # Validate each table and its columns
            columns_validated = 0
            for table in tables:
                table_errors, table_warnings, validated_count = await self._validate_table_usage(
                    query, table, cluster, database
                )
                errors.extend(table_errors)
                warnings.extend(table_warnings)
                columns_validated += validated_count

            # Validate operator syntax
            syntax_errors = self._validate_operator_syntax(query)
            errors.extend(syntax_errors)

            # Generate suggestions based on errors
            if errors:
                suggestions = self._generate_suggestions(errors, tables)

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "suggestions": suggestions,
                "tables_used": tables,
                "columns_validated": columns_validated
            }

        except Exception as e:
            logger.error("Validation error: %s", e, exc_info=True)
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "suggestions": ["Check query syntax and try again"],
                "tables_used": [],
                "columns_validated": 0
            }

    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from KQL query."""
        tables = set()

        # Pattern 1: Table at start or after pipe
        # Matches: TableName | ..., | TableName, etc.
        pattern1 = r'(?:^|\|\s*)([A-Za-z_][\w]*)\s*(?:\||$)'

        # Pattern 2: Join/union operations
        # Matches: join Table, union Table, lookup Table
        pattern2 = r'(?:join|union|lookup)\s+(?:kind\s*=\s*\w+\s+)?([A-Za-z_][\w]*)'

        # Pattern 3: Bracketed table names
        # Matches: ['TableName'], ["TableName"]
        pattern3 = r'\[[\'"]([\ w\s-]+)[\'"]\]'

        for pattern in [pattern1, pattern2, pattern3]:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                table_name = match.group(1).strip()
                # Filter out KQL keywords
                if not self._is_kql_keyword(table_name):
                    tables.add(table_name)

        return list(tables)

    def _is_kql_keyword(self, word: str) -> bool:
        """Check if word is a KQL keyword using centralized reserved words."""
        return word.lower() in {w.lower() for w in KQL_RESERVED_WORDS}

    async def _ensure_schemas_exist(
        self,
        cluster: str,
        database: str,
        tables: List[str]
    ) -> None:
        """Ensure schemas exist for all tables, trigger discovery if needed."""
        for table in tables:
            try:
                schemas = self.memory._get_database_schema(cluster, database)
                schema = next((s for s in schemas if s.get("table") == table), None)

                if not schema or not schema.get("columns"):
                    logger.info("Auto-discovering schema for %s", table)
                    await self.schema_manager.get_table_schema(
                        cluster=cluster,
                        database=database,
                        table=table,
                        _force_refresh=False
                    )
            except Exception as e:
                logger.warning("Schema discovery failed for %s: %s", table, e)

    async def _validate_table_usage(
        self,
        query: str,
        table: str,
        cluster: str,
        database: str
    ) -> tuple[List[str], List[str], int]:
        """
        Validate column usage for a specific table.

        Returns:
            (errors, warnings, validated_count)
        """
        errors: List[str] = []
        warnings = []
        validated_count = 0

        try:
            # Get schema for table
            schemas = self.memory._get_database_schema(cluster, database)
            schema = next((s for s in schemas if s.get("table") == table), None)

            if not schema:
                warnings.append(f"Schema not found for table '{table}'")
                return errors, warnings, 0

            columns = schema.get("columns", {})
            if not columns:
                errors.append(f"No columns found in schema for '{table}'. Cannot validate query.")
                return errors, warnings, 0

            # Extract column references for this table
            column_refs = self._extract_column_references(query, table)

            # Validate each column reference
            for col_ref in column_refs:
                if col_ref not in columns:
                    errors.append(
                        f"Column '{col_ref}' not found in table '{table}'. "
                        f"Available columns: {', '.join(sorted(columns.keys()))}"
                    )
                else:
                    validated_count += 1

            return errors, warnings, validated_count

        except Exception as e:
            logger.warning("Table validation failed for %s: %s", table, e)
            return errors, warnings, 0

    def _extract_column_references(self, query: str, _table: str) -> List[str]:
        """
        Extract column references from query that need validation.
        
        Only extracts columns being READ from tables, NOT aliases being CREATED.
        """
        columns = set()
        
        # First, extract all aliases being created (should NOT validate these)
        created_aliases = self._extract_created_aliases(query)
        
        # KQL aggregation functions that create output columns
        aggregation_functions = {
            'count', 'dcount', 'sum', 'avg', 'min', 'max', 'percentile',
            'make_set', 'make_list', 'make_bag', 'countif', 'dcountif',
            'sumif', 'avgif', 'arg_max', 'arg_min', 'any', 'take_any',
            'stdev', 'variance', 'count_distinct', 'binary_all_and',
            'binary_all_or', 'binary_all_xor', 'buildschema', 'hll',
            'hll_merge', 'tdigest', 'tdigest_merge', 'percentiles'
        }

        # Pattern for columns in WHERE clause (reading from table)
        where_pattern = r'\|\s*where\s+(.+?)(?=\||$)'
        for match in re.finditer(where_pattern, query, re.IGNORECASE | re.DOTALL):
            where_clause = match.group(1)
            # Extract column names from conditions (left side of operators)
            col_pattern = r'([A-Za-z_][\w]*)\s*(?:==|!=|<>|<=|>=|<|>|contains|has|startswith|endswith|matches|in\s*\(|!in\s*\(|has_any|has_all|!contains|!has)'
            for col_match in re.finditer(col_pattern, where_clause, re.IGNORECASE):
                col_name = col_match.group(1).strip()
                if not self._is_kql_keyword(col_name) and col_name not in created_aliases:
                    columns.add(col_name)
            
            # Also check isnotempty/isempty/isnull/isnotnull function arguments
            func_pattern = r'(?:isnotempty|isempty|isnull|isnotnull|strlen|toupper|tolower)\s*\(\s*([A-Za-z_][\w]*)\s*\)'
            for func_match in re.finditer(func_pattern, where_clause, re.IGNORECASE):
                col_name = func_match.group(1).strip()
                if not self._is_kql_keyword(col_name) and col_name not in created_aliases:
                    columns.add(col_name)
        
        # Pattern for columns in summarize BY clause (reading from table)
        summarize_by_pattern = r'\|\s*summarize\s+.+?\s+by\s+([^|]+?)(?=\||$)'
        for match in re.finditer(summarize_by_pattern, query, re.IGNORECASE | re.DOTALL):
            by_clause = match.group(1)
            # Extract columns in by clause, skip bin() and other functions
            for part in by_clause.split(','):
                part = part.strip()
                # Skip bin(Column, ...) - extract column inside
                bin_match = re.match(r'bin\s*\(\s*([A-Za-z_][\w]*)', part, re.IGNORECASE)
                if bin_match:
                    col_name = bin_match.group(1).strip()
                    if not self._is_kql_keyword(col_name) and col_name not in created_aliases:
                        columns.add(col_name)
                # Skip alias assignments like Timestamp = bin(...)
                elif '=' in part and not part.strip().startswith('='):
                    continue
                else:
                    # Simple column reference
                    col_match = re.match(r'^([A-Za-z_][\w]*)\s*$', part)
                    if col_match:
                        col_name = col_match.group(1).strip()
                        if not self._is_kql_keyword(col_name) and col_name not in created_aliases:
                            columns.add(col_name)
        
        # Pattern for columns inside aggregation functions (reading from table)
        # e.g., dcount(ApplicationEventId), min(CreatedOn), max(Timestamp)
        agg_func_pattern = r'(?:' + '|'.join(aggregation_functions) + r')\s*\(\s*([A-Za-z_][\w]*)'
        for match in re.finditer(agg_func_pattern, query, re.IGNORECASE):
            col_name = match.group(1).strip()
            if not self._is_kql_keyword(col_name) and col_name not in created_aliases:
                columns.add(col_name)
        
        # Pattern for columns in order by / sort by (only validate if before summarize or references table columns)
        # Skip these as they often reference summarize output columns
        
        # Pattern for columns in join conditions
        join_pattern = r'\|\s*join\s+.+?\s+on\s+([^|]+?)(?=\||$)'
        for match in re.finditer(join_pattern, query, re.IGNORECASE | re.DOTALL):
            on_clause = match.group(1)
            for part in on_clause.split(','):
                part = part.strip()
                # Handle $left.Col == $right.Col syntax
                col_match = re.findall(r'(?:\$left\.|\$right\.)?([A-Za-z_][\w]*)', part)
                for col_name in col_match:
                    if not self._is_kql_keyword(col_name) and col_name not in created_aliases:
                        columns.add(col_name)

        return list(columns)
    
    def _extract_created_aliases(self, query: str) -> set:
        """
        Extract column aliases that are CREATED in the query (not read from tables).
        These should NOT be validated against table schema.
        """
        aliases = set()
        
        # KQL aggregation functions
        aggregation_functions = {
            'count', 'dcount', 'sum', 'avg', 'min', 'max', 'percentile',
            'make_set', 'make_list', 'make_bag', 'countif', 'dcountif',
            'sumif', 'avgif', 'arg_max', 'arg_min', 'any', 'take_any',
            'stdev', 'variance', 'count_distinct', 'percentiles'
        }
        
        # Pattern 1: Explicit aliases in summarize - Alias = function()
        # e.g., TotalAlerts = count(), FirstSeen = min(Timestamp)
        alias_pattern = r'([A-Za-z_][\w]*)\s*=\s*(?:' + '|'.join(aggregation_functions) + r')\s*\('
        for match in re.finditer(alias_pattern, query, re.IGNORECASE):
            aliases.add(match.group(1).strip())
        
        # Pattern 2: Explicit aliases in extend
        # e.g., extend NewCol = expression
        extend_pattern = r'\|\s*extend\s+([A-Za-z_][\w]*)\s*='
        for match in re.finditer(extend_pattern, query, re.IGNORECASE):
            aliases.add(match.group(1).strip())
        
        # Pattern 3: Explicit aliases in project
        # e.g., project NewName = OldName
        project_alias_pattern = r'\|\s*project(?:-reorder|-away|-keep|-rename)?\s+.*?([A-Za-z_][\w]*)\s*='
        for match in re.finditer(project_alias_pattern, query, re.IGNORECASE):
            aliases.add(match.group(1).strip())
        
        # Pattern 4: Default aggregation column names (when no alias given)
        # e.g., count() creates count_, dcount(X) creates count_X
        if re.search(r'\bcount\s*\(\s*\)', query, re.IGNORECASE):
            aliases.add('count_')
        
        # Pattern 5: Aliases in summarize by with assignment
        # e.g., by NewTimestamp = bin(Timestamp, 1h)
        by_alias_pattern = r'\bby\s+(?:[^|]*?,\s*)*([A-Za-z_][\w]*)\s*='
        for match in re.finditer(by_alias_pattern, query, re.IGNORECASE):
            aliases.add(match.group(1).strip())
        
        return aliases

    def _validate_operator_syntax(self, query: str) -> List[str]:
        """Validate KQL operator syntax."""
        errors = []

        # Check for invalid negation operators with spaces
        invalid_patterns = [
            (r'!\s+=', "Invalid negation '! ='. Use '!=' without space"),
            (r'!\s+contains', "Invalid negation '! contains'. Use '!contains' without space"),
            (r'!\s+in\b', "Invalid negation '! in'. Use '!in' without space"),
            (r'!\s+has\b', "Invalid negation '! has'. Use '!has' without space"),
            (r'!has_any', "Invalid operator '!has_any'. Use '!in' instead"),
        ]

        for pattern, error_msg in invalid_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                errors.append(error_msg)

        return errors

    def _generate_suggestions(self, errors: List[str], tables: List[str]) -> List[str]:
        """Generate helpful suggestions based on errors."""
        suggestions = []

        if any("not found in table" in err for err in errors):
            suggestions.append("Check column names against the schema")
            suggestions.append("Use schema discovery to refresh table schemas")

        if any("negation" in err.lower() for err in errors):
            suggestions.append("Remove spaces in negation operators (use !=, !contains, !in, !has)")

        if any("!has_any" in err for err in errors):
            suggestions.append("Replace !has_any with !in operator")

        if not suggestions:
            suggestions.append(f"Review the query syntax for tables: {', '.join(tables)}")

        return suggestions
