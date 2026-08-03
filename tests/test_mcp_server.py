"""
Unit tests for the MCP server module.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import asyncio
import json
import subprocess
import sys
import unittest
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import LATEST_PROTOCOL_VERSION

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

    def test_stdio_ignores_blank_terminal_input(self):
        """A blank PowerShell line is not an MCP JSON-RPC message."""
        result = subprocess.run(
            [sys.executable, "-m", "mcp_kql_server"],
            input="\n",
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Invalid JSON", result.stderr)
        self.assertNotIn("Internal Server Error", result.stderr)

    def test_stdio_preserves_valid_json_rpc(self):
        """Filtering blank lines must not alter valid MCP messages."""
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "stdio-regression-test", "version": "1.0"},
            },
        }
        result = subprocess.run(
            [sys.executable, "-m", "mcp_kql_server"],
            input=f"\n{json.dumps(initialize)}\n",
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        response = json.loads(result.stdout)
        self.assertEqual(response["id"], 1)
        self.assertIn("result", response)
        self.assertNotIn("Invalid JSON", result.stderr)

    @patch("sys.argv", ["mcp-kql-server"])
    @patch("mcp_kql_server.version_checker.startup_version_check")
    @patch("mcp_kql_server.mcp_server._run_stdio_server", new_callable=AsyncMock)
    @patch("mcp_kql_server.mcp_server._start_kusto_authentication")
    def test_main_authenticates_before_stdio_startup(
            self, mock_start_authentication, mock_run_stdio, mock_version_check):
        """Default startup begins authentication before accepting MCP requests."""
        from mcp_kql_server.mcp_server import main

        main()

        mock_start_authentication.assert_called_once_with(interactive=True)
        mock_version_check.assert_not_called()
        mock_run_stdio.assert_awaited_once_with()

    @patch(
        "sys.argv",
        [
            "mcp-kql-server",
            "--transport", "http",
            "--port", "8123",
            "--http-path", "/kql",
            "--stateless-http",
        ],
    )
    @patch("mcp_kql_server.version_checker.startup_version_check")
    @patch("mcp_kql_server.mcp_server.mcp.run")
    @patch("mcp_kql_server.mcp_server._start_kusto_authentication")
    def test_main_configures_streamable_http_transport(
            self, mock_start_authentication, mock_run, mock_version_check):
        """Test shared HTTP mode configures the official MCP SDK transport."""
        from mcp_kql_server.mcp_server import main, mcp

        main()

        mock_start_authentication.assert_called_once_with(interactive=True)
        mock_version_check.assert_not_called()
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs["transport"], "streamable-http")
        self.assertEqual(mcp.settings.port, 8123)
        self.assertEqual(mcp.settings.streamable_http_path, "/kql")
        self.assertTrue(mcp.settings.stateless_http)

    @patch(
        "sys.argv",
        ["mcp-kql-server", "--transport", "http", "--host", "0.0.0.0"],
    )
    def test_main_rejects_unauthenticated_remote_http_binding(self):
        """HTTP transport cannot expose the operator's Azure identity remotely."""
        from mcp_kql_server.mcp_server import main

        with self.assertRaises(SystemExit):
            main()

    @patch("mcp_kql_server.mcp_server.authenticate_kusto")
    def test_tool_authentication_does_not_trigger_login(self, mock_authenticate):
        """Tool execution only consumes authentication established at startup."""
        import mcp_kql_server.mcp_server as server

        startup_status = {
            "authenticated": True,
            "message": "Azure CLI token ready.",
        }
        with patch.object(server, "kusto_manager_global", startup_status):
            result = server._ensure_kusto_authenticated()

        self.assertTrue(result["authenticated"])
        mock_authenticate.assert_not_called()

    def test_execute_tool_does_not_authenticate_during_request(self):
        """A Kusto tool request must not start Azure CLI authentication."""
        import mcp_kql_server.mcp_server as server

        async def run_test():
            execute_fn = getattr(server.execute_kql_query, "fn", server.execute_kql_query)
            with patch.object(
                server,
                "kusto_manager_global",
                {"authenticated": True, "auth_method": "azure-cli"},
            ), patch.object(server, "_kusto_auth_ready") as mock_auth_ready, \
                 patch.object(server, "authenticate_kusto") as mock_authenticate:
                mock_auth_ready.is_set.return_value = True
                await execute_fn(query="TestTable | take 1")
            return mock_authenticate

        mock_authenticate = asyncio.run(run_test())
        mock_authenticate.assert_not_called()

    def test_local_schema_operation_does_not_authenticate(self):
        """Local memory statistics must remain available without Azure login."""
        import mcp_kql_server.mcp_server as server

        async def run_test():
            schema_fn = getattr(server.schema_memory, "fn", server.schema_memory)
            with patch.object(server.asyncio, "to_thread", new=AsyncMock()) as mock_to_thread, \
                 patch.object(
                     server,
                     "_schema_get_stats_operation",
                     new=AsyncMock(return_value='{"success": true}'),
                 ):
                result = await schema_fn(operation="get_stats")
            return result, mock_to_thread

        result, mock_to_thread = asyncio.run(run_test())
        self.assertEqual(json.loads(result), {"success": True})
        mock_to_thread.assert_not_awaited()


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
