"""
Tests for KQL Validator - Column and Table Extraction

Tests the enhanced column extraction that validates columns in:
- WHERE clause
- PROJECT clause (simple and with aliases)
- EXTEND clause (source columns)
- SUMMARIZE BY clause
- SORT/ORDER BY clause
- TOP N BY clause
- JOIN ON clause
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_kql_server.kql_validator import KQLValidator


@pytest.fixture
def validator():
    """Create a KQLValidator with mocked dependencies."""
    mm = MagicMock()
    sm = MagicMock()
    val = KQLValidator(mm, sm)
    val._ensure_schemas_exist = AsyncMock()
    return val


@pytest.fixture
def mock_schema():
    """Mock schema for CreateProcessEvents table."""
    return {
        "table": "CreateProcessEvents",
        "columns": {
            "TimeGenerated": "datetime",
            "DeviceName": "string",
            "ProcessName": "string",
            "ProcessCommandLine": "string",
            "AccountName": "string",
            "InitiatingProcessName": "string",
            "InitiatingProcessCommandLine": "string",
            "ReportTime": "datetime",
            "AlertId": "string",
        }
    }


class TestTableExtraction:
    """Tests for _extract_tables method."""
    
    def test_simple_table(self, validator):
        """Test extracting simple table name."""
        query = "CreateProcessEvents | where TimeGenerated > ago(1h)"
        tables = validator._extract_tables(query)
        assert "CreateProcessEvents" in tables
    
    def test_table_with_let_statement(self, validator):
        """Test extracting table after let statement."""
        query = """
        let lookback = ago(1h);
        CreateProcessEvents
        | where TimeGenerated > lookback
        """
        tables = validator._extract_tables(query)
        assert "CreateProcessEvents" in tables
        assert "lookback" not in tables  # let variable should be excluded
    
    def test_cross_cluster_table(self, validator):
        """Test extracting table from cross-cluster query."""
        query = """
        cluster('prod').database('security').CreateProcessEvents
        | where TimeGenerated > ago(1h)
        """
        tables = validator._extract_tables(query)
        assert "CreateProcessEvents" in tables
    
    def test_join_tables(self, validator):
        """Test extracting tables from join query."""
        query = """
        CreateProcessEvents
        | join kind=inner NetworkEvents on DeviceName
        """
        tables = validator._extract_tables(query)
        assert "CreateProcessEvents" in tables or "NetworkEvents" in tables


class TestColumnExtraction:
    """Tests for _extract_column_references method."""
    
    def test_where_clause_columns(self, validator):
        """Test extracting columns from WHERE clause."""
        query = "CreateProcessEvents | where TimeGenerated > ago(1h) and DeviceName == 'test'"
        columns = validator._extract_column_references(query, "CreateProcessEvents")
        assert "TimeGenerated" in columns
        assert "DeviceName" in columns
    
    def test_project_simple_columns(self, validator):
        """Test extracting simple columns from PROJECT clause."""
        query = "CreateProcessEvents | project TimeGenerated, DeviceName, ProcessName"
        columns = validator._extract_column_references(query, "CreateProcessEvents")
        assert "TimeGenerated" in columns
        assert "DeviceName" in columns
        assert "ProcessName" in columns
    
    def test_project_with_alias(self, validator):
        """Test extracting source columns when PROJECT uses aliases."""
        query = "CreateProcessEvents | project EventTime = TimeGenerated, Name = ProcessName"
        columns = validator._extract_column_references(query, "CreateProcessEvents")
        assert "TimeGenerated" in columns  # Source column should be extracted
        assert "ProcessName" in columns  # Source column should be extracted
        # Aliases should NOT be in the list (they're created, not read)
    
    def test_extend_source_columns(self, validator):
        """Test extracting source columns from EXTEND clause."""
        query = "CreateProcessEvents | extend NameLength = strlen(ProcessName)"
        columns = validator._extract_column_references(query, "CreateProcessEvents")
        assert "ProcessName" in columns  # Source column in function
    
    def test_summarize_by_columns(self, validator):
        """Test extracting columns from SUMMARIZE BY clause."""
        query = "CreateProcessEvents | summarize count() by DeviceName, ProcessName"
        columns = validator._extract_column_references(query, "CreateProcessEvents")
        assert "DeviceName" in columns
        assert "ProcessName" in columns
    
    def test_top_by_column(self, validator):
        """Test extracting column from TOP BY clause."""
        query = "CreateProcessEvents | top 10 by TimeGenerated"
        columns = validator._extract_column_references(query, "CreateProcessEvents")
        assert "TimeGenerated" in columns


class TestColumnValidation:
    """Tests for full column validation against schema."""
    
    @pytest.mark.asyncio
    async def test_valid_columns_pass(self, validator, mock_schema):
        """Test that valid columns pass validation."""
        validator.memory._get_database_schema.return_value = [mock_schema]
        
        query = """
        CreateProcessEvents
        | where TimeGenerated > ago(1h)
        | project DeviceName, ProcessName
        """
        
        result = await validator.validate_query(
            query, "cluster", "db", auto_discover=False
        )
        
        assert result["valid"]
        assert len(result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_invalid_column_in_project_fails(self, validator, mock_schema):
        """Test that invalid column in PROJECT clause fails validation."""
        validator.memory._get_database_schema.return_value = [mock_schema]
        
        # EventTime and Title don't exist in the schema
        query = """
        CreateProcessEvents
        | where TimeGenerated > ago(1h)
        | project EventTime, Title, ProcessName
        """
        
        result = await validator.validate_query(
            query, "cluster", "db", auto_discover=False
        )
        
        assert not result["valid"]
        assert any("EventTime" in err for err in result["errors"])
        assert any("Title" in err for err in result["errors"])
    
    @pytest.mark.asyncio
    async def test_invalid_column_in_where_fails(self, validator, mock_schema):
        """Test that invalid column in WHERE clause fails validation."""
        validator.memory._get_database_schema.return_value = [mock_schema]
        
        query = """
        CreateProcessEvents
        | where EventTime > ago(1h)
        """
        
        result = await validator.validate_query(
            query, "cluster", "db", auto_discover=False
        )
        
        assert not result["valid"]
        assert any("EventTime" in err for err in result["errors"])


class TestFuzzyMatching:
    """Tests for fuzzy column name suggestions."""
    
    def test_similar_column_suggestions(self, validator):
        """Test that similar column names are suggested."""
        valid_columns = ["TimeGenerated", "ProcessName", "DeviceName", "AccountName"]
        
        # Test typo in column name
        suggestions = validator._find_similar_columns("ProcessNam", valid_columns)
        assert "ProcessName" in suggestions
        
        # Test partial match
        suggestions = validator._find_similar_columns("TimGenerated", valid_columns)
        assert "TimeGenerated" in suggestions
    
    def test_no_suggestions_for_completely_different(self, validator):
        """Test that no suggestions for completely different names."""
        valid_columns = ["TimeGenerated", "ProcessName", "DeviceName"]
        
        suggestions = validator._find_similar_columns("XyzAbcDef", valid_columns)
        assert len(suggestions) == 0


class TestProjectAway:
    """Tests for project-away and project-keep extraction."""
    
    def test_project_away_columns(self, validator):
        """Test extracting columns from project-away clause."""
        query = "CreateProcessEvents | project-away TimeGenerated, DeviceName"
        columns = validator._extract_column_references(query, "CreateProcessEvents")
        assert "TimeGenerated" in columns
        assert "DeviceName" in columns
    
    def test_project_keep_columns(self, validator):
        """Test extracting columns from project-keep clause."""
        query = "CreateProcessEvents | project-keep ProcessName, AccountName"
        columns = validator._extract_column_references(query, "CreateProcessEvents")
        assert "ProcessName" in columns
        assert "AccountName" in columns


class TestCreatedAliases:
    """Tests for _extract_created_aliases method."""
    
    def test_summarize_aliases_excluded(self, validator):
        """Test that summarize aliases are not validated as table columns."""
        query = "CreateProcessEvents | summarize TotalEvents = count() by DeviceName"
        aliases = validator._extract_created_aliases(query)
        assert "TotalEvents" in aliases
        
        # The alias should NOT appear in column references to validate
        columns = validator._extract_column_references(query, "CreateProcessEvents")
        assert "TotalEvents" not in columns
        assert "DeviceName" in columns  # This is a real column
    
    def test_extend_aliases_excluded(self, validator):
        """Test that extend aliases are not validated as table columns."""
        query = "CreateProcessEvents | extend ProcessLength = strlen(ProcessName)"
        aliases = validator._extract_created_aliases(query)
        assert "ProcessLength" in aliases
        
        columns = validator._extract_column_references(query, "CreateProcessEvents")
        assert "ProcessLength" not in columns
        assert "ProcessName" in columns  # Source column should be validated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
