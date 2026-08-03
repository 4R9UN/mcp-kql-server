"""Public package metadata and lazy exports for MCP KQL Server."""

from importlib import import_module
from typing import Any, TYPE_CHECKING

from .constants import __version__ as VERSION

if TYPE_CHECKING:
    from .execute_kql import execute_kql_query
    from .mcp_server import main
    from .version_checker import check_for_updates, get_current_version

# Version information - single source of truth lives in constants.py
__version__ = VERSION
__author__ = "Arjun Trivedi"
__email__ = "arjuntrivedi42@yahoo.com"

_LAZY_EXPORTS = {
    "main": ("mcp_kql_server.mcp_server", "main"),
    "execute_kql_query": ("mcp_kql_server.execute_kql", "execute_kql_query"),
    "check_for_updates": ("mcp_kql_server.version_checker", "check_for_updates"),
    "get_current_version": ("mcp_kql_server.version_checker", "get_current_version"),
}


def __getattr__(name: str) -> Any:
    """Load runtime-heavy public objects only when callers request them."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in interactive discovery."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "main",
    "execute_kql_query",
    "VERSION",
    "__version__",
    "check_for_updates",
    "get_current_version",
]
