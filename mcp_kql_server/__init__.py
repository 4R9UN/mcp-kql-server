"""
MCP KQL Server - AI-Powered KQL Query Execution with Intelligent Schema Memory

A Model Context Protocol (MCP) server that provides intelligent KQL (Kusto Query Language)
query execution with AI-powered schema caching and context assistance for Azure Data Explorer clusters.

This package automatically sets up the required directories and configuration on import.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import logging
import os
import sys

from .constants import __version__ as VERSION
from .execute_kql import execute_kql_query
from .mcp_server import main
from .version_checker import check_for_updates, get_current_version

# Version information
__version__ = "2.1.0"
__author__ = "Arjun Trivedi"
__email__ = "arjuntrivedi42@yahoo.com"

# Force UTF-8 encoding for stdout/stderr to handle any Unicode gracefully
# Use getattr() to avoid Pylance type checker issues with TextIO vs TextIOWrapper
_stdout_reconfigure = getattr(sys.stdout, 'reconfigure', None)
if _stdout_reconfigure is not None:
    try:
        _stdout_reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass

_stderr_reconfigure = getattr(sys.stderr, 'reconfigure', None)
if _stderr_reconfigure is not None:
    try:
        _stderr_reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass

# Configure basic logging (ASCII-safe format)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)

logger = logging.getLogger(__name__)


def _setup_memory_directories():
    """Initialize unified data directory and corpus file on import."""
    try:
        from .constants import get_data_dir
        from .memory import get_memory_manager

        # Ensure unified data dir exists
        data_dir = get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize unified corpus (creates knowledge_corpus.json if missing)
        mm = get_memory_manager()
        mm.save_corpus()
        # Use canonical attribute name `memory_path` directly; avoid deprecated alias `corpus_path`
        logger.info("Unified memory initialized at: %s", mm.memory_path)
    except (OSError, RuntimeError, ImportError) as e:
        # Narrow exception types to avoid masking unexpected failures
        logger.debug("Unified memory setup skipped: %s", e)


def _suppress_azure_logs():
    """Suppress verbose Azure SDK logs by default."""
    try:
        # Set Azure Core to only show errors
        os.environ.setdefault("AZURE_CORE_ONLY_SHOW_ERRORS", "true")

        # Suppress Azure SDK debug logs
        azure_loggers = [
            "azure.core.pipeline.policies.http_logging_policy",
            "azure.kusto.data",
            "urllib3.connectionpool",
            "azure.identity",
        ]

        for logger_name in azure_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    except (OSError, RuntimeError):
        # Ignore if loggers don't exist yet or environment variable issues
        pass


def _suppress_fastmcp_branding():
    """Suppress FastMCP branding and verbose output."""
    try:
        # Set environment variables to suppress FastMCP branding (both string and numeric values)
        os.environ["FASTMCP_QUIET"] = "1"
        os.environ["FASTMCP_NO_BANNER"] = "1"
        os.environ["FASTMCP_SUPPRESS_BRANDING"] = "1"
        os.environ["FASTMCP_NO_LOGO"] = "1"
        os.environ["FASTMCP_SILENT"] = "1"
        os.environ["NO_COLOR"] = "1"
        os.environ["PYTHONIOENCODING"] = "utf-8"

        # Additional suppression flags
        os.environ["FASTMCP_DISABLE_BANNER"] = "1"
        os.environ["MCP_QUIET"] = "1"

        # Suppress FastMCP and related loggers
        fastmcp_loggers = ["fastmcp", "rich", "rich.console", "rich.progress", "fastmcp.server", "fastmcp.cli"]

        for logger_name in fastmcp_loggers:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    except (OSError, RuntimeError):
        # Ignore if loggers don't exist yet
        pass


# Perform automatic setup on import
_suppress_fastmcp_branding()
_suppress_azure_logs()
_setup_memory_directories()

__all__ = [
    "main",
    "execute_kql_query",
    "VERSION",
    "__version__",
    "check_for_updates",
    "get_current_version",
]
