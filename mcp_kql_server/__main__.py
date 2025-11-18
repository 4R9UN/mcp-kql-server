"""
Entry point for running the MCP KQL Server as a module.

This file allows the package to be executed directly with:
python -m mcp_kql_server
"""

from .mcp_server import main

if __name__ == "__main__":
    main()
