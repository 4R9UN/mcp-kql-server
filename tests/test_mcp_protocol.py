"""Protocol-level tests for the public MCP tool contracts."""

import asyncio
import json

from mcp.types import LATEST_PROTOCOL_VERSION

from mcp_kql_server.constants import REGISTERED_TOOL_NAMES
from mcp_kql_server.mcp_server import _print_info, mcp


def _discovered_tools():
    return {tool.name: tool for tool in asyncio.run(mcp.list_tools())}


def test_public_tool_discovery_contract():
    """The SDK must discover exactly the supported public tools and schemas."""
    tools = _discovered_tools()

    assert set(tools) == set(REGISTERED_TOOL_NAMES)
    assert "query" in tools["execute_kql_query"].inputSchema["required"]
    assert "operation" in tools["kql_schema_memory"].inputSchema["required"]
    assert tools["execute_kql_query"].outputSchema["properties"]["result"]["type"] == "string"


def test_tool_annotations_match_side_effects():
    """Discovery hints must not describe mutating cache operations as read-only."""
    tools = _discovered_tools()
    execute_annotations = tools["execute_kql_query"].annotations
    schema_annotations = tools["kql_schema_memory"].annotations

    assert execute_annotations.readOnlyHint is True
    assert execute_annotations.destructiveHint is False
    assert schema_annotations.readOnlyHint is False
    assert schema_annotations.destructiveHint is True


def test_info_uses_installed_sdk_protocol_version(capsys):
    """Diagnostic metadata follows the installed MCP SDK protocol version."""
    _print_info(as_json=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["mcp_protocol_version"] == LATEST_PROTOCOL_VERSION