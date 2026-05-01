"""
Test module for constants.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

from mcp_kql_server.constants import (
    DEFAULT_CONNECTION_TIMEOUT,
    DEFAULT_QUERY_TIMEOUT,
    LIMITS,
    SERVER_NAME,
    __version__,
)


def test_version_constants():
    """Test version constants are properly defined."""
    assert __version__ == "2.1.2"


def test_server_constants():
    """Test server configuration constants."""
    assert SERVER_NAME == f"mcp-kql-server({__version__})"
    assert DEFAULT_CONNECTION_TIMEOUT > 0
    assert DEFAULT_QUERY_TIMEOUT > 0


def test_limits():
    """Test limit constants."""
    assert isinstance(LIMITS, dict)
    assert "max_result_rows" in LIMITS
    assert LIMITS["max_result_rows"] > 0
