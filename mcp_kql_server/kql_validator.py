"""
KQL Query Validator

Pre-execution validation to catch errors before sending queries to Kusto.
Validates column names, table names, and operator syntax against schema.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import re
import logging
import difflib
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
        """
        Extract table names from KQL query.
        
        Handles multiple query patterns including:
        - Simple table references: TableName | where ...
        - Cross-cluster: cluster('...').database('...').TableName
        - Let blocks: let x = ...; TableName | where ...
        - Join/union operations
        - Bracketed table names
        """
        tables = set()

        # Pattern 1: Cross-cluster pattern (HIGHEST PRIORITY)
        # Matches: cluster('...').database('...').TableName
        pattern_cross_cluster = r'\.database\s*\(\s*[\'"][^\'"]+[\'"]\s*\)\s*\.([A-Za-z_][\w]*)'
        
        # Pattern 2: Table after semicolon (for let statements)
        # Matches: ; TableName | ... or ;\nTableName
        pattern_after_let = r';\s*\n?\s*([A-Za-z_][\w]*)\s*(?:\||$)'
        
        # Pattern 3: Table at start of line or after pipe (with MULTILINE support)
        # Matches: TableName | ..., | TableName, etc.
        pattern_line_start = r'(?:^|\|\s*)([A-Za-z_][\w]*)\s*(?:\||$)'

        # Pattern 4: Join/union operations
        # Matches: join Table, union Table, lookup Table
        pattern_join = r'(?:join|union|lookup)\s+(?:kind\s*=\s*\w+\s+)?([A-Za-z_][\w]*)'

        # Pattern 5: Bracketed table names
        # Matches: ['TableName'], ["TableName"]
        pattern_bracket = r'\[[\'"]([A-Za-z_][\w\s-]+)[\'"]\]'
        
        # Pattern 6: Table in datatable or materialize
        # Matches: datatable (...) [...] | ..., materialize(TableName)
        pattern_materialize = r'materialize\s*\(\s*([A-Za-z_][\w]*)\s*\)'

        # Apply patterns with appropriate flags
        patterns_with_flags = [
            (pattern_cross_cluster, re.IGNORECASE),
            (pattern_after_let, re.IGNORECASE | re.MULTILINE),
            (pattern_line_start, re.IGNORECASE | re.MULTILINE),
            (pattern_join, re.IGNORECASE),
            (pattern_bracket, re.IGNORECASE),
            (pattern_materialize, re.IGNORECASE),
        ]

        for pattern, flags in patterns_with_flags:
            try:
                matches = re.finditer(pattern, query, flags)
                for match in matches:
                    table_name = match.group(1).strip()
                    # Filter out KQL keywords and common false positives
                    if table_name and not self._is_kql_keyword(table_name):
                        # Additional filter: skip variable-like names in let statements
                        if not self._is_let_variable(query, table_name):
                            tables.add(table_name)
            except re.error as e:
                logger.warning("Regex error in table extraction: %s", e)

        # If no tables found, try fallback extraction
        if not tables:
            tables = self._fallback_table_extraction(query)
        
        logger.debug("Extracted tables from query: %s", list(tables))
        return list(tables)
    
    def _is_let_variable(self, query: str, name: str) -> bool:
        """
        Check if a name is defined as a let variable in the query.
        This prevents extracting let variable names as table names.
        """
        # Pattern: let VariableName = ...
        let_pattern = rf'\blet\s+{re.escape(name)}\s*='
        return bool(re.search(let_pattern, query, re.IGNORECASE))
    
    def _fallback_table_extraction(self, query: str) -> set:
        """
        Fallback table extraction for complex queries.
        Uses heuristics when standard patterns fail.
        """
        tables = set()
        
        # Remove comments and string literals to avoid false positives
        cleaned = re.sub(r'//.*$', '', query, flags=re.MULTILINE)  # Remove line comments
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)  # Remove block comments
        cleaned = re.sub(r'"[^"]*"', '""', cleaned)  # Replace string literals
        cleaned = re.sub(r"'[^']*'", "''", cleaned)  # Replace string literals
        
        # Split by common KQL delimiters and look for table-like identifiers
        # Tables typically appear: after database().TABLE or at statement boundaries
        
        # Look for identifiers that look like table names (PascalCase or with Events/Log suffix)
        table_like_pattern = r'\b([A-Z][a-zA-Z0-9]*(?:Events?|Log|Table|Data|Records?)?)\b'
        for match in re.finditer(table_like_pattern, cleaned):
            candidate = match.group(1)
            if (len(candidate) > 3 and 
                not self._is_kql_keyword(candidate) and
                not self._is_let_variable(query, candidate)):
                tables.add(candidate)
        
        return tables

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
            column_names = list(columns.keys())

            # Validate each column reference
            for col_ref in column_refs:
                if col_ref not in columns:
                    # Use fuzzy matching to suggest similar column names
                    suggestions = self._find_similar_columns(col_ref, column_names)
                    if suggestions:
                        errors.append(
                            f"Column '{col_ref}' not found in table '{table}'. "
                            f"Did you mean: {', '.join(suggestions)}?"
                        )
                    else:
                        # Show first 10 available columns as reference
                        sample_cols = sorted(column_names)[:10]
                        errors.append(
                            f"Column '{col_ref}' not found in table '{table}'. "
                            f"Sample columns: {', '.join(sample_cols)}"
                            + (f"... ({len(column_names)} total)" if len(column_names) > 10 else "")
                        )
                else:
                    validated_count += 1

            return errors, warnings, validated_count

        except Exception as e:
            logger.warning("Table validation failed for %s: %s", table, e)
            return errors, warnings, 0

    def _find_similar_columns(
        self, 
        invalid_col: str, 
        valid_columns: List[str],
        n: int = 3,
        cutoff: float = 0.5
    ) -> List[str]:
        """
        Find similar column names using fuzzy matching.
        
        Args:
            invalid_col: The invalid column name from the query
            valid_columns: List of valid column names from schema
            n: Maximum number of suggestions to return
            cutoff: Minimum similarity ratio (0-1) for a match
            
        Returns:
            List of similar column names, sorted by similarity
        """
        # Use difflib for fuzzy matching
        matches = difflib.get_close_matches(
            invalid_col.lower(),
            [c.lower() for c in valid_columns],
            n=n,
            cutoff=cutoff
        )
        
        # Map back to original case
        lower_to_original = {c.lower(): c for c in valid_columns}
        return [lower_to_original.get(m, m) for m in matches]

    def _extract_column_references(self, query: str, _table: str) -> List[str]:
        """
        Extract column references from query that need validation.
        
        Only extracts columns being READ from tables, NOT aliases being CREATED.
        Covers: WHERE, PROJECT, EXTEND, SUMMARIZE, SORT/ORDER BY, JOIN, TOP
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
        
        # ============================================================
        # NEW: Pattern for columns in PROJECT clause (critical fix)
        # Handles: | project Col1, Col2, Alias = Col3
        # ============================================================
        project_pattern = r'\|\s*project\s+([^|]+?)(?=\||$)'
        for match in re.finditer(project_pattern, query, re.IGNORECASE | re.DOTALL):
            project_clause = match.group(1).strip()
            # Parse each item in the project list
            for col_item in self._split_by_comma_respecting_parens(project_clause):
                col_item = col_item.strip()
                if not col_item:
                    continue
                
                if '=' in col_item:
                    # Format: Alias = SourceColumn or Alias = expression
                    # Extract columns from the right side (source)
                    _, right_side = col_item.split('=', 1)
                    source_cols = self._extract_columns_from_expression(
                        right_side, created_aliases, aggregation_functions
                    )
                    columns.update(source_cols)
                else:
                    # Simple column reference (no alias)
                    col_name = col_item.strip()
                    # Check it's a valid identifier
                    if re.match(r'^[A-Za-z_][\w]*$', col_name):
                        if not self._is_kql_keyword(col_name) and col_name not in created_aliases:
                            columns.add(col_name)
        
        # ============================================================
        # NEW: Pattern for project-away and project-keep
        # All columns listed are being READ from the table
        # ============================================================
        project_away_pattern = r'\|\s*project-(?:away|keep)\s+([^|]+?)(?=\||$)'
        for match in re.finditer(project_away_pattern, query, re.IGNORECASE | re.DOTALL):
            cols_list = match.group(1).strip()
            for col_item in self._split_by_comma_respecting_parens(cols_list):
                col_name = col_item.strip()
                if re.match(r'^[A-Za-z_][\w]*$', col_name):
                    if not self._is_kql_keyword(col_name) and col_name not in created_aliases:
                        columns.add(col_name)
        
        # ============================================================
        # NEW: Pattern for source columns in EXTEND expressions
        # Handles: | extend NewCol = SourceCol + 1
        # ============================================================
        extend_pattern = r'\|\s*extend\s+([^|]+?)(?=\||$)'
        for match in re.finditer(extend_pattern, query, re.IGNORECASE | re.DOTALL):
            extend_clause = match.group(1).strip()
            # Parse each assignment in the extend
            for assignment in self._split_by_comma_respecting_parens(extend_clause):
                if '=' in assignment:
                    # Extract source columns from the expression (right side)
                    _, expr = assignment.split('=', 1)
                    source_cols = self._extract_columns_from_expression(
                        expr, created_aliases, aggregation_functions
                    )
                    columns.update(source_cols)
        
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
        
        # ============================================================
        # NEW: Pattern for SORT BY / ORDER BY columns (before summarize)
        # Only validate columns that appear before any summarize clause
        # ============================================================
        # Check if query has summarize - if so, sort/order columns after may be aliases
        has_summarize = bool(re.search(r'\|\s*summarize\s+', query, re.IGNORECASE))
        
        if not has_summarize:
            # Safe to validate sort/order by columns
            sort_pattern = r'\|\s*(?:sort|order)\s+by\s+([^|]+?)(?=\||$)'
            for match in re.finditer(sort_pattern, query, re.IGNORECASE | re.DOTALL):
                sort_clause = match.group(1).strip()
                for part in sort_clause.split(','):
                    # Remove asc/desc modifiers
                    part = re.sub(r'\b(asc|desc|nulls\s+first|nulls\s+last)\b', '', part, flags=re.IGNORECASE)
                    col_name = part.strip()
                    if re.match(r'^[A-Za-z_][\w]*$', col_name):
                        if not self._is_kql_keyword(col_name) and col_name not in created_aliases:
                            columns.add(col_name)
        
        # ============================================================
        # NEW: Pattern for TOP N BY column
        # Handles: | top 10 by ColumnName
        # ============================================================
        top_pattern = r'\|\s*top\s+\d+\s+by\s+([A-Za-z_][\w]*)'
        for match in re.finditer(top_pattern, query, re.IGNORECASE):
            col_name = match.group(1).strip()
            if not self._is_kql_keyword(col_name) and col_name not in created_aliases:
                columns.add(col_name)
        
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
    
    def _split_by_comma_respecting_parens(self, text: str) -> List[str]:
        """
        Split text by commas, but respect parentheses nesting.
        E.g., 'a, func(b, c), d' -> ['a', 'func(b, c)', 'd']
        """
        result = []
        current = []
        depth = 0
        
        for char in text:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                result.append(''.join(current))
                current = []
            else:
                current.append(char)
        
        if current:
            result.append(''.join(current))
        
        return result
    
    def _extract_columns_from_expression(
        self, 
        expr: str, 
        created_aliases: set,
        aggregation_functions: set
    ) -> set:
        """
        Extract column references from a KQL expression.
        Filters out KQL functions, keywords, and created aliases.
        """
        columns = set()
        
        # Remove string literals to avoid false positives
        expr = re.sub(r'"[^"]*"', '""', expr)
        expr = re.sub(r"'[^']*'", "''", expr)
        
        # KQL scalar functions to exclude
        kql_functions = {
            'toupper', 'tolower', 'strlen', 'substring', 'strcat', 'split',
            'replace', 'trim', 'ltrim', 'rtrim', 'extract', 'parse_json',
            'toint', 'tolong', 'todouble', 'tostring', 'todatetime', 'totimespan',
            'datetime', 'ago', 'now', 'bin', 'floor', 'ceiling', 'round',
            'abs', 'sign', 'sqrt', 'pow', 'exp', 'log', 'log10',
            'isnotempty', 'isempty', 'isnull', 'isnotnull', 'coalesce',
            'iff', 'iif', 'case', 'pack', 'pack_all', 'dynamic',
            'array_length', 'array_concat', 'array_slice', 'mv_expand',
            'parse_path', 'parse_url', 'parse_urlquery', 'base64_encode_tostring',
            'base64_decode_tostring', 'hash', 'hash_sha256', 'format_datetime',
            'make_datetime', 'datetime_diff', 'datetime_add', 'startofday',
            'startofweek', 'startofmonth', 'startofyear', 'endofday',
            'gettype', 'typeof', 'toreal', 'tobool', 'toguid',
        }
        all_functions = kql_functions | aggregation_functions
        
        # Find all identifiers in the expression
        identifiers = re.findall(r'\b([A-Za-z_][\w]*)\b', expr)
        
        for ident in identifiers:
            ident_lower = ident.lower()
            # Skip if it's a KQL function, keyword, or created alias
            if (ident_lower not in {f.lower() for f in all_functions} and
                not self._is_kql_keyword(ident) and
                ident not in created_aliases):
                columns.add(ident)
        
        return columns
    
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
