"""Tests for the hardcoded Kusto servertimeout, error classifier, retry policy,
and ADX dry-run helper. These tests pin the regression surface for the
NL2KQL accuracy + 10-minute query-timeout work.

Author: Arjun Trivedi
"""

import unittest
from datetime import timedelta
from unittest.mock import MagicMock, patch

from azure.kusto.data import ClientRequestProperties
from azure.kusto.data.exceptions import KustoServiceError

from mcp_kql_server import execute_kql
from mcp_kql_server.constants import (
    KUSTO_MAX_QUERY_TIMEOUT_SECONDS,
    KUSTO_MIN_QUERY_TIMEOUT_SECONDS,
)
from mcp_kql_server.utils import _is_retryable_exc


# ---------------------------------------------------------------------------
# Hardcoded 10-minute servertimeout via ClientRequestProperties
# ---------------------------------------------------------------------------


class TimeoutClampTests(unittest.TestCase):
    """The hardcoded ceiling and floor must be enforced by _clamp_timeout."""

    def test_ceiling_caps_long_timeouts(self):
        self.assertEqual(execute_kql._clamp_timeout(99999), KUSTO_MAX_QUERY_TIMEOUT_SECONDS)

    def test_floor_protects_against_zero_or_negative(self):
        self.assertEqual(execute_kql._clamp_timeout(0), KUSTO_MAX_QUERY_TIMEOUT_SECONDS)
        self.assertEqual(execute_kql._clamp_timeout(-5), KUSTO_MAX_QUERY_TIMEOUT_SECONDS)

    def test_in_range_value_passes_through(self):
        self.assertEqual(execute_kql._clamp_timeout(120), 120)

    def test_below_floor_is_raised(self):
        self.assertGreaterEqual(
            execute_kql._clamp_timeout(1), KUSTO_MIN_QUERY_TIMEOUT_SECONDS
        )

    def test_constant_is_ten_minutes(self):
        self.assertEqual(KUSTO_MAX_QUERY_TIMEOUT_SECONDS, 600)


class RequestPropertiesTests(unittest.TestCase):
    """Every Kusto call must ship a ClientRequestProperties with servertimeout."""

    def test_servertimeout_option_is_set_to_clamped_value(self):
        crp = execute_kql._build_request_properties(99999, "TableX | take 1")
        opt = crp.get_option(ClientRequestProperties.request_timeout_option_name, None)
        self.assertEqual(opt, timedelta(seconds=KUSTO_MAX_QUERY_TIMEOUT_SECONDS))

    def test_client_request_id_is_unique(self):
        a = execute_kql._build_request_properties(60, "T1")
        b = execute_kql._build_request_properties(60, "T1")
        self.assertNotEqual(a.client_request_id, b.client_request_id)
        self.assertTrue(a.client_request_id.startswith("mcp-kql-"))


class ExecuteForwardsCRPTests(unittest.TestCase):
    """Pin that _execute_kusto_query_sync passes a CRP positional arg to ADX."""

    @patch("mcp_kql_server.execute_kql._get_kusto_client")
    def test_execute_passes_crp_to_client_execute(self, mock_get_client):
        client = MagicMock()
        resp = MagicMock()
        resp.primary_results = [MagicMock()]
        col = MagicMock()
        col.column_name = "c"
        resp.primary_results[0].columns = [col]
        resp.primary_results[0].__iter__ = lambda x: iter([["v"]])
        client.execute.return_value = resp
        mock_get_client.return_value = client

        execute_kql._execute_kusto_query_sync(
            "TableX | take 1", "https://c.kusto.windows.net", "DB", 60
        )

        self.assertTrue(client.execute.called)
        args, _ = client.execute.call_args
        self.assertEqual(len(args), 3, "expected (database, query, crp)")
        self.assertIsInstance(args[2], ClientRequestProperties)

    @patch("mcp_kql_server.execute_kql._get_kusto_client")
    def test_mgmt_passes_crp_to_client_execute_mgmt(self, mock_get_client):
        client = MagicMock()
        resp = MagicMock()
        resp.primary_results = [MagicMock()]
        col = MagicMock()
        col.column_name = "TableName"
        resp.primary_results[0].columns = [col]
        resp.primary_results[0].__iter__ = lambda x: iter([["T"]])
        client.execute_mgmt.return_value = resp
        mock_get_client.return_value = client

        execute_kql._execute_kusto_query_sync(
            ".show tables", "https://c.kusto.windows.net", "DB", 60
        )

        self.assertTrue(client.execute_mgmt.called)
        args, _ = client.execute_mgmt.call_args
        self.assertEqual(len(args), 3, "expected (database, query, crp)")
        self.assertIsInstance(args[2], ClientRequestProperties)


# ---------------------------------------------------------------------------
# Recoverable vs non-recoverable error classifier
# ---------------------------------------------------------------------------


class ClassifyKustoErrorTests(unittest.TestCase):
    def test_server_timeout(self):
        self.assertEqual(
            execute_kql.classify_kusto_error("Request timed out (servertimeout)"),
            execute_kql.ERROR_CLASS_TIMEOUT,
        )

    def test_throttled(self):
        self.assertEqual(
            execute_kql.classify_kusto_error("TooManyRequests: throttled"),
            execute_kql.ERROR_CLASS_THROTTLED,
        )

    def test_schema_drift_sem0100(self):
        self.assertEqual(
            execute_kql.classify_kusto_error("SEM0100: column not found"),
            execute_kql.ERROR_CLASS_SCHEMA_DRIFT,
        )

    def test_schema_drift_failed_to_resolve(self):
        self.assertEqual(
            execute_kql.classify_kusto_error("Failed to resolve scalar expression"),
            execute_kql.ERROR_CLASS_SCHEMA_DRIFT,
        )

    def test_auth(self):
        self.assertEqual(
            execute_kql.classify_kusto_error("Unauthorized: invalid_token"),
            execute_kql.ERROR_CLASS_AUTH,
        )

    def test_transient(self):
        self.assertEqual(
            execute_kql.classify_kusto_error("Connection refused"),
            execute_kql.ERROR_CLASS_TRANSIENT,
        )

    def test_permanent_other_for_unknown(self):
        self.assertEqual(
            execute_kql.classify_kusto_error("Some unrelated banana error"),
            execute_kql.ERROR_CLASS_PERMANENT_OTHER,
        )

    def test_empty_input(self):
        self.assertEqual(
            execute_kql.classify_kusto_error(""),
            execute_kql.ERROR_CLASS_PERMANENT_OTHER,
        )


# ---------------------------------------------------------------------------
# Retry policy: server-side timeout must NOT be auto-retried
# ---------------------------------------------------------------------------


class RetryPolicyTests(unittest.TestCase):
    def test_server_timeout_is_not_retryable(self):
        self.assertFalse(_is_retryable_exc(TimeoutError("Request timed out (servertimeout)")))
        self.assertFalse(_is_retryable_exc(RuntimeError("deadline_exceeded")))

    def test_connection_refused_is_retryable(self):
        self.assertTrue(_is_retryable_exc(OSError("connection refused")))

    def test_throttling_is_retryable(self):
        self.assertTrue(_is_retryable_exc(RuntimeError("TooManyRequests throttled")))

    def test_arbitrary_runtime_error_is_not_retryable(self):
        self.assertFalse(_is_retryable_exc(RuntimeError("syntax error in query")))


# ---------------------------------------------------------------------------
# ADX dry-run wrapping: <query> | take 0
# ---------------------------------------------------------------------------


class DryRunQueryTests(unittest.IsolatedAsyncioTestCase):
    @patch("mcp_kql_server.execute_kql._execute_kusto_query_sync")
    async def test_dry_run_appends_take_zero_and_returns_executable(self, mock_exec):
        mock_exec.return_value = MagicMock()  # returned df ignored
        result = await execute_kql.dry_run_query(
            "TableX | where col == 'a'", "https://c.kusto.windows.net", "DB"
        )
        self.assertTrue(result["executable"])
        self.assertTrue(result["valid"])
        # Inspect probe query (first positional arg)
        probe = mock_exec.call_args[0][0]
        self.assertIn("| take 0", probe)
        self.assertTrue(probe.startswith("TableX"))

    @patch("mcp_kql_server.execute_kql._execute_kusto_query_sync")
    async def test_dry_run_skips_management_commands(self, mock_exec):
        result = await execute_kql.dry_run_query(
            ".show tables", "https://c.kusto.windows.net", "DB"
        )
        self.assertTrue(result.get("skipped"))
        self.assertTrue(result["executable"])
        mock_exec.assert_not_called()

    @patch("mcp_kql_server.execute_kql._execute_kusto_query_sync")
    async def test_dry_run_classifies_kusto_error(self, mock_exec):
        mock_exec.side_effect = KustoServiceError("SEM0100: column missing")
        result = await execute_kql.dry_run_query(
            "TableX | project NotAColumn", "https://c.kusto.windows.net", "DB"
        )
        self.assertFalse(result["executable"])
        self.assertEqual(result["error_class"], execute_kql.ERROR_CLASS_SCHEMA_DRIFT)


if __name__ == "__main__":
    unittest.main()
