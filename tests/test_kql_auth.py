"""
Unit tests for the kql_auth module.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import subprocess
import unittest
from unittest.mock import MagicMock, patch

from mcp_kql_server.kql_auth import authenticate, kql_auth, trigger_az_cli_auth


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
        # Mock first call success, second call failure
        mock_run.side_effect = [
            MagicMock(returncode=0),  # for 'az config'
            subprocess.CalledProcessError(
                1, "az", stderr="Authentication failed"
            ),  # for 'az account'
        ]

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
        # Mock successful login
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = trigger_az_cli_auth()

        self.assertTrue(result["authenticated"])
        self.assertIn("successful", result["message"].lower())

    @patch("mcp_kql_server.kql_auth.subprocess.run")
    def test_trigger_az_cli_auth_failure(self, mock_run):
        """Test failed Azure CLI authentication trigger."""
        # Mock failed login
        mock_run.return_value = MagicMock(returncode=1, stderr="Login failed")

        result = trigger_az_cli_auth()

        self.assertFalse(result["authenticated"])
        self.assertIn("Login failed", result["message"])

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
        """Test non-interactive auth check never starts device-code login."""
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
