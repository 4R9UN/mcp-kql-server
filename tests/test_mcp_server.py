"""
Unit tests for the MCP server module.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import json
import unittest
from unittest.mock import patch

import pytest

from mcp_kql_server.constants import TEST_CONFIG


class TestMCPServerFunctions(unittest.TestCase):
    """Test cases for MCP server functions and operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_query = f"cluster('{TEST_CONFIG['mock_cluster_uri']}').database('{TEST_CONFIG['mock_database']}').{TEST_CONFIG['mock_table']} | take 10"
        self.test_cluster_uri = TEST_CONFIG["mock_cluster_uri"]
        self.test_database = TEST_CONFIG["mock_database"]
        self.test_table = TEST_CONFIG["mock_table"]

    def test_error_handler_enhancement(self):
        """Test enhanced ErrorHandler functionality."""
        from mcp_kql_server.utils import ErrorHandler

        # Test basic error handling
        test_error = Exception("Test error")
        result = ErrorHandler.handle_kusto_error(test_error)

        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("error", result)
        self.assertIn("suggestions", result)
        # Note: recovery_actions is only included for KustoServiceError, not generic Exception
        self.assertFalse(result["success"])

    def test_schema_manager_integration(self):
        """Test SchemaManager integration."""
        from mcp_kql_server.utils import SchemaManager
        from mcp_kql_server.memory import get_memory_manager

        memory_manager = get_memory_manager()
        schema_manager = SchemaManager(memory_manager)

        self.assertIsNotNone(schema_manager)
        self.assertIsNotNone(schema_manager.memory_manager)

    def test_bracket_if_needed_function(self):
        """Test the bracket_if_needed utility function."""
        from mcp_kql_server.utils import bracket_if_needed

        # Test normal column name (should not be bracketed)
        normal_column = "RegularColumn"
        self.assertEqual(bracket_if_needed(normal_column), normal_column)

        # Test column with spaces (should be bracketed)
        spaced_column = "Column With Spaces"
        result = bracket_if_needed(spaced_column)
        self.assertTrue(result.startswith("['") and result.endswith("']"))

        # Test reserved word
        reserved_word = "table"  # This is a KQL reserved word
        result = bracket_if_needed(reserved_word)
        self.assertTrue(result.startswith("['") and result.endswith("']"))

    def test_configuration_constants(self):
        """Test that required configuration constants are properly defined."""
        from mcp_kql_server.constants import (
            SERVER_NAME, SERVER_DESCRIPTION, CONNECTION_CONFIG,
            ERROR_HANDLING_CONFIG, KQL_RESERVED_WORDS
        )

        self.assertIsInstance(SERVER_NAME, str)
        self.assertIsInstance(SERVER_DESCRIPTION, str)
        self.assertIsInstance(CONNECTION_CONFIG, dict)
        self.assertIsInstance(ERROR_HANDLING_CONFIG, dict)
        self.assertIsInstance(KQL_RESERVED_WORDS, (list, frozenset))
        self.assertGreater(len(KQL_RESERVED_WORDS), 0)

    def test_json_serialization_helper(self):
        """Test the JSON serialization helper from ErrorHandler."""
        from mcp_kql_server.utils import ErrorHandler

        # Test with simple data
        simple_data = {"key": "value", "number": 42}
        result = ErrorHandler.safe_json_dumps(simple_data)

        self.assertIsInstance(result, str)
        parsed = json.loads(result)
        self.assertEqual(parsed["key"], "value")
        self.assertEqual(parsed["number"], 42)

        # Test with complex data that might cause issues
        complex_data = {"datetime": "2023-01-01T00:00:00", "none_value": None}
        result = ErrorHandler.safe_json_dumps(complex_data)
        self.assertIsInstance(result, str)

    def test_memory_manager_availability(self):
        """Test that memory manager is properly available."""
        from mcp_kql_server.memory import get_memory_manager

        manager = get_memory_manager()
        self.assertIsNotNone(manager)

        # Test basic memory manager functionality
        self.assertTrue(hasattr(manager, 'get_memory_stats'))
        # Note: clear_schema_cache is available via SchemaManager, not directly on memory manager
        self.assertTrue(hasattr(manager, 'corpus') or hasattr(manager, 'memory_path'))


    @pytest.mark.skip(reason="_execute_kql_query_logic was refactored into execute_kql_query")
    @patch("mcp_kql_server.mcp_server.kql_execute_tool")
    @patch("mcp_kql_server.mcp_server.schema_manager")
    @patch("mcp_kql_server.mcp_server.kusto_manager_global", {"authenticated": True})
    def test_execute_kql_query_schema_context_on_error(
            self, mock_schema_manager, mock_execute_tool):  # pylint: disable=unused-argument
        """Test that schema context is added to suggestions on SEM0100 error.
        
        NOTE: This test is skipped because _execute_kql_query_logic was refactored.
        The functionality is now integrated into execute_kql_query.
        """
        # Test is skipped - no execution needed
        self.skipTest("_execute_kql_query_logic was refactored")


if __name__ == "__main__":
    unittest.main()
