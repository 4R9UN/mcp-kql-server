"""
REST API endpoints for the KQL MCP Server.

Provides simple HTTP endpoints alongside the MCP protocol so users can
query via curl or any HTTP client without needing the MCP SDK.
"""

import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


async def health(request: Request) -> JSONResponse:
    """Health check endpoint for k8s probes."""
    return JSONResponse({"status": "ok"})


async def query(request: Request) -> JSONResponse:
    """
    Execute a KQL query or natural language query.

    POST /query
    {
        "query": "show me top 10 storm events",   # required
        "cluster_url": "https://...",              # optional, defaults to KUSTO_CLUSTER_URL env
        "database": "mydb",                        # optional, defaults to KUSTO_DEFAULT_DATABASE env
        "generate_query": true                     # optional, default true (NL2KQL)
    }
    """
    from .constants import KUSTO_CLUSTER_URL, KUSTO_DEFAULT_DATABASE

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    query_text = body.get("query")
    if not query_text:
        return JSONResponse({"error": "Missing required field: query"}, status_code=400)

    cluster_url = body.get("cluster_url") or KUSTO_CLUSTER_URL
    database = body.get("database") or KUSTO_DEFAULT_DATABASE
    generate_query = body.get("generate_query", True)

    if not cluster_url:
        return JSONResponse({"error": "No cluster_url provided and KUSTO_CLUSTER_URL env not set"}, status_code=400)
    if not database:
        return JSONResponse({"error": "No database provided and KUSTO_DEFAULT_DATABASE env not set"}, status_code=400)

    # Import the MCP tool and call its underlying function (.fn)
    # since @mcp.tool() wraps it in a FunctionTool object
    from .mcp_server import execute_kql_query

    try:
        result = await execute_kql_query.fn(
            query=query_text,
            cluster_url=cluster_url,
            database=database,
            generate_query=generate_query,
        )
        return JSONResponse(json.loads(result))
    except Exception as e:
        logger.error("REST API query error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


rest_routes = [
    Route("/health", health, methods=["GET"]),
    Route("/query", query, methods=["POST"]),
]
