"""
Test module for constants.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import asyncio
import inspect

from mcp_kql_server.constants import (
    DEFAULT_CONNECTION_TIMEOUT,
    DEFAULT_QUERY_TIMEOUT,
    LIMITS,
    REGISTERED_TOOL_NAMES,
    SERVER_NAME,
    TOOL_KQL_EXECUTE_NAME,
    TOOL_KQL_SCHEMA_NAME,
    __version__,
    format_display_path,
)


def test_version_constants():
    """Test version constants are properly defined."""
    assert __version__ == "2.1.3"


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


def test_format_display_path_redacts_appdata(monkeypatch):
    """Test user-facing paths use environment variable placeholders."""
    monkeypatch.setenv("APPDATA", r"C:\Users\example\AppData\Roaming")

    display_path = format_display_path(r"C:\Users\example\AppData\Roaming\KQL_MCP\kql_memory.db")

    assert display_path == r"%APPDATA%\KQL_MCP\kql_memory.db"
    assert "C:\\Users\\example" not in display_path


def test_registered_tool_name_constants_match_fastmcp_registry():
    """Test documented tool names match the FastMCP registry."""
    from mcp_kql_server.mcp_server import mcp

    if hasattr(mcp, "get_tools"):
        registered_tool_names = set(asyncio.run(mcp.get_tools()))
    elif hasattr(mcp, "get_tool"):
        registered_tool_names = set()
        for tool_name in REGISTERED_TOOL_NAMES:
            tool = mcp.get_tool(tool_name)
            if inspect.isawaitable(tool):
                tool = asyncio.run(tool)
            if tool is not None:
                registered_tool_names.add(tool_name)
    else:
        registry = getattr(mcp, "_tool_manager", None) or getattr(mcp, "tool_manager", None)
        tool_map = getattr(registry, "_tools", {}) if registry else {}
        registered_tool_names = set(tool_map)

    assert TOOL_KQL_EXECUTE_NAME == "execute_kql_query"
    assert TOOL_KQL_SCHEMA_NAME == "kql_schema_memory"
    assert set(REGISTERED_TOOL_NAMES).issubset(registered_tool_names)
