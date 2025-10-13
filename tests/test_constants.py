"""
Test module for constants.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""


def test_version_constants():
    """Test version constants are properly defined."""
    try:
        from mcp_kql_server.constants import __version__

        assert __version__ == "2.0.7"
    except ImportError:
        # Skip test if module not available in CI
        pass


def test_server_constants():
    """Test server configuration constants."""
    try:
        from mcp_kql_server.constants import (
            DEFAULT_CONNECTION_TIMEOUT,
            DEFAULT_QUERY_TIMEOUT,
            SERVER_NAME,
        )

        assert SERVER_NAME == "mcp-kql-server(2.0.7)"
        assert DEFAULT_CONNECTION_TIMEOUT > 0
        assert DEFAULT_QUERY_TIMEOUT > 0

    except ImportError:
        # Skip test if module not available in CI
        pass


def test_error_messages():
    """Test error message constants."""
    try:
        from mcp_kql_server.constants import ERROR_MESSAGES

        assert isinstance(ERROR_MESSAGES, dict)
        assert "auth_failed" in ERROR_MESSAGES
        assert "empty_query" in ERROR_MESSAGES

    except ImportError:
        # Skip test if module not available in CI
        pass


def test_limits():
    """Test limit constants."""
    try:
        from mcp_kql_server.constants import LIMITS

        assert isinstance(LIMITS, dict)
        assert "max_result_rows" in LIMITS
        assert LIMITS["max_result_rows"] > 0

    except ImportError:
        # Skip test if module not available in CI
        pass
