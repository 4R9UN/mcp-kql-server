"""
Unit tests for the kql_auth module.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from mcp_kql_server.kql_auth import (
    authenticate,
    build_kusto_connection_string,
    kql_auth,
    trigger_az_cli_auth,
)


class TestKQLAuth(unittest.TestCase):
    """Test cases for KQL authentication functionality."""

    def setUp(self):
        """Clear cache before each test."""
        kql_auth.cache_clear()

    @patch("mcp_kql_server.kql_auth.subprocess.run")
    def test_kql_auth_success(self, mock_run):
        """Test successful authentication check."""
        # Mock successful az command
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")

        result = kql_auth()

        self.assertTrue(result["authenticated"])
        self.assertIn("user is authenticated", result["message"].lower())

    @patch("mcp_kql_server.kql_auth.subprocess.run")
    def test_kql_auth_failure(self, mock_run):
        """Test failed authentication check."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "az", stderr="Authentication failed"
        )

        result = kql_auth()

        self.assertFalse(result["authenticated"])
        self.assertIn("user is not authenticated", result["message"].lower())

    @patch("mcp_kql_server.kql_auth.subprocess.run")
    def test_kql_auth_exception(self, mock_run):
        """Test authentication check with unexpected exception."""
        # Mock exception
        mock_run.side_effect = Exception("Unexpected error")

        result = kql_auth()

        self.assertFalse(result.get("authenticated"))
        self.assertIn("unexpected error", result.get("message", "").lower())

    @patch("mcp_kql_server.kql_auth.subprocess.run")
    def test_trigger_az_cli_auth_success(self, mock_run):
        """Test successful Azure CLI authentication trigger."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stderr=""),
            MagicMock(returncode=0, stdout="{}", stderr=""),
        ]

        result = trigger_az_cli_auth()

        self.assertTrue(result["authenticated"])
        self.assertIn("successful", result["message"].lower())
        login_call = mock_run.call_args_list[0]
        self.assertEqual(login_call.args[0][1:], ["login"])
        self.assertIs(login_call.kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(login_call.kwargs["stdout"], sys.stderr)
        self.assertIs(login_call.kwargs["stderr"], sys.stderr)
        self.assertEqual(mock_run.call_args_list[1].args[0][1:], ["account", "get-access-token"])
        self.assertIs(mock_run.call_args_list[1].kwargs["stdin"], subprocess.DEVNULL)

    @patch("mcp_kql_server.kql_auth.subprocess.run")
    def test_trigger_az_cli_auth_failure(self, mock_run):
        """Test failed Azure CLI authentication trigger."""
        # Mock failed login
        mock_run.return_value = MagicMock(returncode=1, stderr="Login failed")

        result = trigger_az_cli_auth()

        self.assertFalse(result["authenticated"])
        self.assertIn("Login failed", result["message"])

    @patch("mcp_kql_server.kql_auth.subprocess.run")
    def test_trigger_az_cli_auth_requires_token_after_login(self, mock_run):
        """Test login is not accepted until Azure CLI can return a token."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stderr=""),
            subprocess.CalledProcessError(1, "az", stderr="Token unavailable"),
        ]

        result = trigger_az_cli_auth()

        self.assertFalse(result["authenticated"])
        self.assertIn("token acquisition failed", result["message"].lower())

    @patch("mcp_kql_server.kql_auth.subprocess.run")
    def test_trigger_az_cli_auth_timeout(self, mock_run):
        """Test Azure CLI authentication timeout."""
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired("az", 120)

        result = trigger_az_cli_auth()

        self.assertFalse(result.get("authenticated"))
        self.assertIn("timed out", result.get("message", "").lower())

    @patch("mcp_kql_server.kql_auth.kql_auth")
    def test_authenticate_already_authenticated(self, mock_kql_auth):
        """Test authenticate when already authenticated."""
        # Mock already authenticated
        mock_kql_auth.return_value = {
            "authenticated": True,
            "message": "Already authenticated",
        }

        result = authenticate()

        self.assertTrue(result["authenticated"])
        self.assertIn("Already authenticated", result["message"])

    @patch.dict("os.environ", {"MCP_KQL_AUTH_MODE": "managed-identity", "AZURE_CLIENT_ID": "client-id"})
    @patch("mcp_kql_server.kql_auth.subprocess.run")
    def test_managed_identity_authentication_skips_azure_cli(self, mock_run):
        """Managed identity mode never launches Azure CLI."""
        result = authenticate()

        self.assertTrue(result["authenticated"])
        self.assertEqual(result["auth_method"], "managed-identity")
        mock_run.assert_not_called()

    @patch.dict("os.environ", {"MCP_KQL_AUTH_MODE": "managed-identity", "AZURE_CLIENT_ID": "client-id"})
    @patch("azure.kusto.data.KustoConnectionStringBuilder.with_aad_managed_service_identity_authentication")
    def test_managed_identity_connection_builder_uses_client_id(self, mock_builder):
        """User-assigned managed identity is forwarded to the Kusto SDK."""
        build_kusto_connection_string("https://cluster.kusto.windows.net")

        mock_builder.assert_called_once_with(
            "https://cluster.kusto.windows.net",
            client_id="client-id",
        )

    @patch("mcp_kql_server.kql_auth.trigger_az_cli_auth")
    @patch("mcp_kql_server.kql_auth.kql_auth")
    def test_authenticate_needs_login(self, mock_kql_auth, mock_trigger_auth):
        """Test authenticate when login is needed."""
        # Mock not authenticated initially
        mock_kql_auth.return_value = {
            "authenticated": False,
            "message": "Not authenticated",
        }

        # Mock successful login
        mock_trigger_auth.return_value = {
            "authenticated": True,
            "message": "Login successful",
        }

        result = authenticate()

        self.assertTrue(result["authenticated"])
        mock_trigger_auth.assert_called_once()

    @patch("mcp_kql_server.kql_auth.trigger_az_cli_auth")
    @patch("mcp_kql_server.kql_auth.kql_auth")
    def test_authenticate_non_interactive_does_not_login(self, mock_kql_auth, mock_trigger_auth):
        """Test non-interactive auth check never starts Azure CLI login."""
        mock_kql_auth.return_value = {
            "authenticated": False,
            "message": "Not authenticated",
        }

        result = authenticate(interactive=False)

        self.assertFalse(result["authenticated"])
        self.assertTrue(result["interactive_required"])
        mock_trigger_auth.assert_not_called()

    @patch("mcp_kql_server.kql_auth.trigger_az_cli_auth")
    @patch("mcp_kql_server.kql_auth.kql_auth")
    def test_authenticate_login_fails(self, mock_kql_auth, mock_trigger_auth):
        """Test authenticate when login fails."""
        # Mock not authenticated initially
        mock_kql_auth.return_value = {
            "authenticated": False,
            "message": "Not authenticated",
        }

        # Mock failed login
        mock_trigger_auth.return_value = {
            "authenticated": False,
            "message": "Login failed",
        }

        result = authenticate()

        self.assertFalse(result["authenticated"])
        mock_trigger_auth.assert_called_once()

    @patch("mcp_kql_server.kql_auth.platform.system")
    @patch("mcp_kql_server.kql_auth.subprocess.run")
    def test_platform_specific_commands(self, mock_run, mock_platform):
        """Test platform-specific command selection."""
        # Test Windows
        mock_platform.return_value = "Windows"
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")
        kql_auth()

        # Check that az.cmd was used for Windows
        first_call_args = mock_run.call_args_list[0].args[0]
        self.assertIn("az.cmd", first_call_args)

        # Reset mock and clear cache
        mock_run.reset_mock()
        kql_auth.cache_clear()

        # Test Linux/Mac
        mock_platform.return_value = "Linux"
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")
        kql_auth()

        # Check that az was used for Linux
        first_call_args = mock_run.call_args_list[0].args[0]
        self.assertEqual("az", first_call_args[0])


if __name__ == "__main__":
    unittest.main()
