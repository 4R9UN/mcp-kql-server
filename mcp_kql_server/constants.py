"""
Constants for the MCP KQL Server.

This module contains all the constants used throughout the MCP KQL Server,
including default values, configuration settings, error messages, and other
static values.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import os
import re
import ntpath
import posixpath
from pathlib import Path
from typing import Any, Dict, List


# Version information - single source of truth for the package version.
# Keep this in sync with pyproject.toml and server.json (CI enforces parity).
__version__ = "2.1.5"
VERSION = __version__

# Server configuration
SERVER_NAME = f"mcp-kql-server({__version__})"
SERVER_VERSION = __version__
FEATURES = [
    {
        "title": "Security-focused table & column pattern recognition",
        "summary": (
            "Pattern and semantic-based detection of security-related tables/columns using "
            "configurable regexes/heuristics and optional ML classification. Safe defaults "
            "prevent logging of sensitive values."
        ),
    },
    {
        "title": "Real-time query visualization and result formatting",
        "summary": (
            "Progressive/streaming rendering of large result sets with configurable row/column "
            "limits and pluggable visualization backends to avoid large-memory buffering."
        ),
    },
    {
        "title": "FastMCP: low-latency, high-throughput protocol",
        "summary": (
            "Optimized implementation with connection pooling, optional compression, "
            "backpressure-aware streaming, and efficient serialization to minimize latency "
            "and memory usage."
        ),
    },
]


def _format_features_list(features: List[Dict[str, str]]) -> str:
    """Format features into bullet lines with safe truncation and fallbacks."""
    if not features:
        return (
            "- Security-focused table and column pattern recognition\n"
            "- Real-time query visualization and result formatting\n"
            "- FastMCP implementation for optimal performance"
        )

    lines = []
    for f in features:
        # Defensive conversions & truncation to avoid extremely long lines
        title = str(f.get("title", "<unnamed feature>"))[:120]
        summary = str(f.get("summary", "")).strip()
        if len(summary) > 300:
            summary = summary[:297] + "..."
        lines.append(f"- {title}: {summary}")
    return "\n".join(lines)


# Build description once at import time to avoid repeated formatting at runtime.
_SERVER_FEATURE_BULLETS = _format_features_list(FEATURES)

SERVER_DESCRIPTION = f"""AI-Enhanced KQL Server for Cybersecurity Analytics

An intelligent Model Context Protocol (MCP) server that provides advanced KQL query execution
capabilities with AI-powered schema discovery and intelligent security analytics assistance.

Key Features:
{_SERVER_FEATURE_BULLETS}

Perfect for SOC analysts, threat hunters, and security researchers working with Azure Data Explorer."""

# FastAPI Configuration
FASTAPI_TITLE = "MCP KQL Local AI Agent"
FASTAPI_VERSION = __version__  # Use single source of truth
FASTAPI_DESCRIPTION = "Local AI-powered KQL Server with no external dependencies"
FASTAPI_HOST = "0.0.0.0"
FASTAPI_PORT = 8000

# Tool Configuration
TOOL_KQL_EXECUTE_NAME = "execute_kql_query"
TOOL_KQL_EXECUTE_DESCRIPTION = """Execute KQL (Kusto Query Language) queries for Azure Data Explorer, Application Insights, Log Analytics, and other Kusto databases.

**Use this tool when users ask about:**
- KQL queries, Kusto queries, Azure Data Explorer queries
- Log Analytics queries, Application Insights queries
- Searching logs, events, metrics, telemetry data
- Data analysis, security investigations, performance monitoring
- Questions about tables like SecurityEvent, Heartbeat, Perf, Event, Syslog
- Time-based queries (last hour, today, past week, etc.)
- Aggregations, statistics, trends, patterns in data
- Converting natural language to KQL or executing direct KQL

**Examples of queries to use this tool for:**
- "Show security events from last hour"
- "Find failed login attempts"
- "Get performance data trends"
- "Search application logs for errors"
- "cluster('<cluster>').database('<database>').<table> | take 10"

Supports both direct KQL execution and natural language to KQL conversion with intelligent schema detection."""

TOOL_KQL_SCHEMA_NAME = "kql_schema_memory"
REGISTERED_TOOL_NAMES = (
    TOOL_KQL_EXECUTE_NAME,
    TOOL_KQL_SCHEMA_NAME,
)
TOOL_KQL_SCHEMA_DESCRIPTION = """Discover database schemas, table structures, and generate KQL queries using AI-powered schema intelligence.

**Use this tool when users need:**
- Schema discovery for Kusto/Azure Data Explorer databases
- Table structure information and column details
- Smart KQL query generation from natural language
- Context-aware query suggestions based on available data
- AI-enhanced schema analysis and pattern matching

**Examples:**
- "What tables are available in this database?"
- "Generate a query to find security incidents"
- "Help me understand the schema structure"
- "Convert this natural language to KQL with proper table/column names"

This tool uses local AI to provide intelligent schema discovery and KQL generation without external API dependencies."""

# KQL Validation tool removed - not implemented in current server

# Default configuration values - completely data-driven approach
DEFAULT_CLUSTER = None  # Will be extracted from user's KQL query
DEFAULT_DATABASE = None  # Will be discovered from query execution
DEFAULT_TABLE = None  # Will be discovered from schema analysis

# Default paths and directories
DEFAULT_MEMORY_DIR_NAME = "KQL_MCP"
DEFAULT_CLUSTER_MEMORY_DIR = "cluster_memory"
SCHEMA_FILE_EXTENSION = ".json"

# Azure and KQL configuration
DEFAULT_KUSTO_DOMAIN = "kusto.windows.net"
SYSTEM_DATABASES = {"$systemdb"}
DEFAULT_CONNECTION_TIMEOUT = 60
DEFAULT_QUERY_TIMEOUT = 120
# Hard server-side ceiling for any single Kusto query, in seconds.
# Sent to ADX via ClientRequestProperties.servertimeout.
# 10 minutes is the deliberate upper bound for this MCP server.
KUSTO_MAX_QUERY_TIMEOUT_SECONDS = 600
# Floor that prevents accidental sub-second misconfigurations.
KUSTO_MIN_QUERY_TIMEOUT_SECONDS = 5

# Schema memory configuration
SCHEMA_CACHE_MAX_AGE_DAYS = 7
MAX_SCHEMA_FILE_SIZE_MB = 10
MAX_TABLES_PER_DATABASE = 1000
MAX_COLUMNS_PER_TABLE = 500

# Query validation
MAX_QUERY_LENGTH = 100000
MIN_QUERY_LENGTH = 10

# Enhanced error handling configuration used by discovery/fetch routines
ENHANCED_ERROR_HANDLING = {
    "MAX_RETRIES": 3,
    "BASE_RETRY_DELAY": 1.0,
    "RETRY_BACKOFF_FACTOR": 2.0,
}

# Memory optimization controls used by AI token generation and context management
MEMORY_OPTIMIZATION = {
    "MAX_CONTEXT_SIZE": 4000,
    "AI_TOKEN_COMPRESSION": True,
    "TRIM_AI_TOKENS_TO": 1024,
}

# Performance tuning config (used as default values across modules)
PERFORMANCE_CONFIG = {
    "SCHEMA_CACHE_TTL_HOURS": 24,
}

# Thread safety/limits defaults
THREAD_SAFETY_CONFIG = {
    "USE_THREAD_LOCKS": True,
    "MAX_THREADS": 50,
}

# File and directory permissions
FILE_PERMISSIONS = {
    "schema_file": 0o600,
    "memory_dir": 0o700,
}

# Limits and constraints
LIMITS = {
    "max_concurrent_queries": 5,
    "max_result_rows": 10000,
    "max_result_bytes": 2_000_000,
    "max_visualization_rows": 1000,
    "max_column_description_length": 500,
    "max_table_description_length": 1000,
    "max_retry_attempts": 3,
    "retry_delay_seconds": 2,
}

# KQL operators that definitively indicate a KQL query
KQL_OPERATORS = frozenset({
    "count",
    "datatable",
    "distinct",
    "evaluate",
    "extend",
    "facet",
    "find",
    "getschema",
    "invoke",
    "join",
    "let",
    "limit",
    "lookup",
    "make-series",
    "materialize",
    "mv-expand",
    "order",
    "parse",
    "print",
    "project",
    "project-away",
    "project-rename",
    "range",
    "render",
    "sample",
    "sample-distinct",
    "sort",
    "summarize",
    "take",
    "top",
    "union",
    "where",
})

# ============================================================================
# KQL OPERATOR SYNTAX GUIDANCE FOR AI QUERY GENERATION
# ============================================================================
# These constants provide comprehensive guidance for correct KQL operator syntax
# to prevent common errors in AI-generated queries.

# Valid KQL comparison operators (NO spaces allowed)
KQL_COMPARISON_OPERATORS = frozenset({
    "==",   # Equal
    "!=",   # Not equal (NO SPACE: correct is "!=", not "! =")
    "<",    # Less than
    "<=",   # Less than or equal
    ">",    # Greater than
    ">=",   # Greater than or equal
    "=~",   # Case-insensitive equal
    "!~",   # Case-insensitive not equal (NO SPACE)
})

# Valid KQL string operators (NO spaces in negations)
KQL_STRING_OPERATORS = frozenset({
    "contains",       # Contains substring
    "!contains",      # Does not contain (NO SPACE: correct is "!contains", not "! contains")
    "contains_cs",    # Contains case-sensitive
    "!contains_cs",   # Does not contain case-sensitive (NO SPACE)
    "has",            # Has whole term (word boundary)
    "!has",           # Does not have term (NO SPACE: correct is "!has", not "! has")
    "has_cs",         # Has term case-sensitive
    "!has_cs",        # Does not have term case-sensitive (NO SPACE)
    "hasprefix",      # Has prefix
    "!hasprefix",     # Does not have prefix (NO SPACE)
    "hassuffix",      # Has suffix
    "!hassuffix",     # Does not have suffix (NO SPACE)
    "startswith",     # Starts with
    "!startswith",    # Does not start with (NO SPACE)
    "startswith_cs",  # Starts with case-sensitive
    "!startswith_cs", # Does not start with case-sensitive (NO SPACE)
    "endswith",       # Ends with
    "!endswith",      # Does not end with (NO SPACE)
    "endswith_cs",    # Ends with case-sensitive
    "!endswith_cs",   # Does not end with case-sensitive (NO SPACE)
    "matches regex",  # Matches regular expression
    "in~",            # Case-insensitive in list
    "!in~",           # Case-insensitive not in list (NO SPACE)
})

# Valid KQL list/set operators (NO spaces in negations)
KQL_LIST_OPERATORS = frozenset({
    "in",       # In list (use this for checking if value is in a list)
    "!in",      # Not in list (NO SPACE: correct is "!in", not "! in" or "!has_any")
    "in~",      # Case-insensitive in list
    "!in~",     # Case-insensitive not in list (NO SPACE)
    "has_any",  # Has any of the terms (for arrays/dynamic columns)
    "has_all",  # Has all of the terms
})

# IMPORTANT: Invalid operators that should NEVER be used
KQL_INVALID_OPERATORS = frozenset({
    "! =",        # WRONG: Use "!=" instead (no space)
    "! in",       # WRONG: Use "!in" instead (no space)
    "! has",      # WRONG: Use "!has" instead (no space)
    "! contains", # WRONG: Use "!contains" instead (no space)
    "! startswith", # WRONG: Use "!startswith" instead (no space)
    "! endswith",   # WRONG: Use "!endswith" instead (no space)
    "!has_any",   # WRONG: Use "!in" for negating list membership, or "not (col has_any (list))"
})

# KQL operator usage guidelines for AI query generation
KQL_OPERATOR_GUIDELINES = {
    "negation_syntax": {
        "description": "KQL negation operators must NOT have spaces between ! and the operator",
        "correct_examples": [
            "where Status != 'Active'",
            "where Name !contains 'test'",
            "where Category !in ('A', 'B', 'C')",
            "where Title !has 'error'",
            "where Path !startswith '/temp'",
            "where not (Status == 'Active')",  # Alternative using 'not' keyword
        ],
        "incorrect_examples": [
            "where Status ! = 'Active'",  # WRONG: Space between ! and =
            "where Name ! contains 'test'",  # WRONG: Space between ! and contains
            "where Category ! in ('A', 'B', 'C')",  # WRONG: Space between ! and in
            "where Category !has_any ('A', 'B', 'C')",  # WRONG: !has_any doesn't exist
        ],
    },
    "list_membership": {
        "description": "Use 'in' for list membership, '!in' for negation (not !has_any)",
        "correct_examples": [
            "where RuleName in ('Rule1', 'Rule2', 'Rule3')",  # Check if in list
            "where RuleName !in ('Rule1', 'Rule2', 'Rule3')", # Exclude from list
            "where Category has_any ('cat1', 'cat2')",  # For dynamic/array columns
            "where not (RuleName in ('Rule1', 'Rule2'))",  # Alternative negation
        ],
        "incorrect_examples": [
            "where RuleName !has_any ('Rule1', 'Rule2')",  # WRONG: !has_any doesn't exist
            "where RuleName ! in ('Rule1', 'Rule2')",  # WRONG: Space in operator
        ],
    },
    "alternative_negation": {
        "description": "Alternative: use 'not' keyword for negation",
        "examples": [
            "where not (Status == 'Active')",
            "where not (Name contains 'test')",
            "where not (Category in ('A', 'B', 'C'))",
        ],
    },
    "string_matching": {
        "description": "Choose appropriate string operator based on match requirements",
        "operators": {
            "has": "Whole word/term matching (use for exact terms)",
            "contains": "Substring matching (use for partial matches)",
            "startswith": "Prefix matching",
            "endswith": "Suffix matching",
            "matches regex": "Pattern matching",
        },
    },
}

# Regex pattern to detect invalid operators in queries
KQL_INVALID_OPERATOR_PATTERN = re.compile(
    r'!\s+(=|in|has|contains|startswith|endswith)|!has_any',
    re.IGNORECASE
)

# Natural language indicators
NL_INDICATORS = {
    "show",
    "find",
    "what",
    "how",
    "when",
    "where",
    "who",
    "which",
    "list",
    "get",
    "can",
    "could",
    "would",
    "should",
    "tell",
    "give",
    "display",
    "search",
    "look",
    "help",
    "explain",
    "describe",
}

# Question patterns for natural language detection
QUESTION_PATTERNS = [
    r"\?$",
    r"^(what|how|when|where|who|which|why)\s+",
    r"^(can|could|would|should)\s+you\s+",
    r"^(show|find|get|list)\s+me\s+",
]

# KQL syntax patterns for definitive KQL detection
# Note: Use raw regex strings here (not precompiled) so callers can compile with desired flags.
KQL_SYNTAX_PATTERNS = [
    r"\|\s*(project|extend|where|summarize|join|find|let|evaluate|render|take|sort|order|top|count|distinct|mv-expand|make-series|parse)\b",
    r"ago\s*\(\s*\d+[dhms]\s*\)",
    r"between\s*\(.*?\.\..*?\)",
    r"\s(has|contains|startswith|endswith|==|=~)\s",
    r"^\s*let\s+",
    r"datatable\s*\(",
    r"(cluster|database)\s*\(\s*['\"].*?['\"]\s*\)",
]

# Classification confidence threshold
KQL_CONFIDENCE_THRESHOLD = 0.7

# Score weights for classification
KQL_SYNTAX_SCORE_WEIGHT = 2.0
KQL_OPERATOR_SCORE_WEIGHT = 3.0
NL_QUESTION_SCORE_WEIGHT = 2.0
NL_INDICATOR_SCORE_WEIGHT = 1.5
KQL_TABLE_SCORE_WEIGHT = 1.0
NL_CONVERSATIONAL_SCORE_WEIGHT = 1.0
KQL_CONVERSATIONAL_PENALTY = -0.5

# Memory configuration - default memory path under %APPDATA%/KQL_MCP on Windows; fallback to user's home when not set
DEFAULT_MEMORY_PATH = str(Path(os.environ.get("APPDATA", str(Path.home()))) / "KQL_MCP" / "kql_mcp_memory.json")


def format_display_path(path: Any) -> str:
    """Return a user-safe display path for logs and diagnostics.

    Runtime paths can include a local profile directory. For user-facing output,
    replace well-known profile roots with environment variable placeholders while
    leaving the actual storage path unchanged.
    """
    path_text = str(path)
    if not path_text:
        return path_text

    for env_name, placeholder in (
        ("APPDATA", "%APPDATA%"),
        ("LOCALAPPDATA", "%LOCALAPPDATA%"),
        ("USERPROFILE", "%USERPROFILE%"),
        ("HOME", "~"),
    ):
        env_value = os.environ.get(env_name)
        if not env_value:
            continue
        for path_module, separator in ((os.path, os.sep), (ntpath, "\\"), (posixpath, "/")):
            normalized_path = path_module.normcase(path_module.normpath(path_text))
            normalized_env = path_module.normcase(path_module.normpath(env_value))
            if normalized_path == normalized_env:
                return placeholder
            if normalized_path.startswith(normalized_env + separator):
                suffix = path_module.normpath(path_text)[len(path_module.normpath(env_value)):].lstrip("\\/")
                display_separator = "\\" if "\\" in path_text else "/"
                return f"{placeholder}{display_separator}{suffix}" if suffix else placeholder

    return path_text

# Query execution configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
DEFAULT_PREVIEW_ROWS = 100
DEFAULT_VISUALIZATION_ROWS = 10

# Agent routing configuration
ROUTE_KQL = "KQL"
ROUTE_NL = "NL"

# Status messages
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_LEARNING = "learning"
STATUS_HEALTHY = "healthy"
STATUS_OPERATIONAL = "operational"
STATUS_DEGRADED = "degraded"

# Agent capabilities
CAPABILITY_FAST_PATH = "Direct KQL execution"
CAPABILITY_SMART_PATH = "Natural language to KQL with local AI"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Environment variables
ENV_AZURE_CORE_ONLY_SHOW_ERRORS = "AZURE_CORE_ONLY_SHOW_ERRORS"
ENV_FASTMCP_QUIET = "FASTMCP_QUIET"
ENV_FASTMCP_NO_BANNER = "FASTMCP_NO_BANNER"
ENV_FASTMCP_SUPPRESS_BRANDING = "FASTMCP_SUPPRESS_BRANDING"
ENV_FASTMCP_NO_LOGO = "FASTMCP_NO_LOGO"
ENV_FASTMCP_SILENT = "FASTMCP_SILENT"
ENV_NO_COLOR = "NO_COLOR"
ENV_PYTHONIOENCODING = "PYTHONIOENCODING"

# Note: Monitoring system removed - configurations no longer needed

# Special tokens for knowledge corpus and AI-driven query chaining
SPECIAL_TOKENS = {
    "CLUSTER_START": "<CLUSTER>",
    "CLUSTER_END": "</CLUSTER>",
    "DATABASE_START": "<DATABASE>",
    "DATABASE_END": "</DATABASE>",
    "TABLE_START": "<TABLE>",
    "TABLE_END": "</TABLE>",
    "COLUMN_START": "<COLUMN>",
    "COLUMN_END": "</COLUMN>",
    "PATTERN_START": "<PATTERN>",
    "PATTERN_END": "</PATTERN>",
    "QUERY_START": "<QUERY>",
    "QUERY_END": "</QUERY>",
    "RESULT_START": "<RESULT>",
    "RESULT_END": "</RESULT>",
    "CONTEXT_START": "<CONTEXT>",
    "CONTEXT_END": "</CONTEXT>",
    "MEMORY_START": "<MEMORY>",
    "MEMORY_END": "</MEMORY>",
    "AI_START": "<AI>",
    "AI_END": "</AI>",
    "SEPARATOR": "<SEP>",
    "PAD_TOKEN": "<PAD>",
    "UNK_TOKEN": "<UNK>",
    "MASK_TOKEN": "<MASK>",
    # Metadata extraction and context tuning tokens
    "METADATA_START": "<METADATA>",
    "METADATA_END": "</METADATA>",
    "EXECUTION_RESULT_START": "<EXEC_RESULT>",
    "EXECUTION_RESULT_END": "</EXEC_RESULT>",
    "PLAN_START": "<PLAN>",
    "PLAN_END": "</PLAN>",
    "CHAIN_START": "<CHAIN>",
    "CHAIN_END": "</CHAIN>",
    "FEEDBACK_START": "<FEEDBACK>",
    "FEEDBACK_END": "</FEEDBACK>",
    "LEARNING_START": "<LEARNING>",
    "LEARNING_END": "</LEARNING>",
    "OPTIMIZATION_START": "<OPTIMIZATION>",
    "OPTIMIZATION_END": "</OPTIMIZATION>",
    "SCHEMA_TRIGGER_START": "<SCHEMA_TRIGGER>",
    "SCHEMA_TRIGGER_END": "</SCHEMA_TRIGGER>",
    # Compact AI-friendly tokens (backwards-compatible aliases)
    "CLUSTER": "@@CLUSTER@@",
    "DATABASE": "##DATABASE##",
    "TABLE": "##TABLE##",
    "COLUMN": "::COLUMN::",
    "TYPE": ">>TYPE<<",
}

# Comprehensive set of KQL reserved words and keywords.
KQL_RESERVED_WORDS = frozenset({
    'and', 'as', 'by', 'contains', 'count', 'distinct', 'extend', 'false',
    'from', 'group', 'has', 'in', 'join', 'let', 'limit', 'not', 'on', 'or',
    'order', 'project', 'sort', 'startswith', 'summarize', 'take', 'then',
    'top', 'true', 'union', 'where', 'with', 'avg', 'max', 'min', 'sum',
    'between', 'endswith', 'having', 'kind', 'show', 'type', 'inner', 'outer',
    'anti', 'leftanti', 'getschema', 'string', 'long', 'int', 'datetime',
    'timespan', 'real', 'bool', 'dynamic', 'guid', 'decimal', 'render',
    'evaluate', 'parse', 'mv-expand', 'make-series', 'range', 'datatable',
    'print', 'sample', 'sample-distinct', 'facet', 'find', 'invoke', 'lookup',
    'materialize', 'fork', 'partition', 'serialize', 'toscalar', 'topnested',
    'scan', 'search', 'externaldata', 'cluster', 'database', 'table'
})

# Deprecated: Maintained for backward compatibility, use KQL_RESERVED_WORDS instead.
KQL_RESERVED_KEYWORDS = list(KQL_RESERVED_WORDS)

# KQL functions for AI analysis
KQL_FUNCTIONS = [
    "ago",
    "now",
    "datetime",
    "timespan",
    "bin",
    "floor",
    "ceiling",
    "round",
    "abs",
    "exp",
    "log",
    "log10",
    "pow",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "strlen",
    "substring",
    "split",
    "strcat",
    "replace",
    "trim",
    "tolower",
    "toupper",
    "startswith",
    "endswith",
    "contains",
    "matches",
    "extract",
    "extractall",
    "toint",
    "tolong",
    "todouble",
    "tostring",
    "tobool",
    "todatetime",
    "totimespan",
    "isnull",
    "isnotnull",
    "isempty",
    "isnotempty",
    "iif",
    "case",
    "iff",
    "array_length",
    "array_slice",
    "array_split",
    "pack_array",
    "pack",
    "bag_keys",
    "bag_merge",
    "treepath",
    "zip",
    "repeat",
    "range",
]

# KQL query patterns for AI analysis
KQL_QUERY_PATTERNS = {
    # Matches most common filter operations, including various operators and value types.
    "basic_filter": re.compile(r"\|\s*where\s+\w+\s+(?:==|!=|>=|<=|>|<|contains|has|!contains|!has|startswith)\s+"),

    # Specifically identifies time-based filtering using the ago() function.
    "time_filter": re.compile(r"\|\s*where\s+\w+\s*(?:>|>=|<|<=|==|!=)\s*ago\s*\("),

    # Identifies aggregation, capturing both simple counts and grouped summaries with "by".
    "aggregation": re.compile(r"\|\s*summarize\b.*?\bby\b"),

    # Matches the main data shaping operators: project, project-away, and project-rename.
    "projection": re.compile(r"\|\s*(?:project|project-away|project-rename)\b"),

    # Catches calculated column creation with the "extend" operator.
    "extend_column": re.compile(r"\|\s*extend\b\s+\w+\s*="),

    # Identifies join operations, including the common "on" keyword.
    "join": re.compile(r"\|\s*(?:join|lookup)\b.*?\bon\b"),

    # Finds sorting operations.
    "sorting": re.compile(r"\|\s*(?:order|sort)\s+by\b"),

    # Matches operators that limit the number of results.
    "limit_results": re.compile(r"\|\s*(?:take|limit|top)\b\s+\d+"),

    # Specifically identifies the "top N by column" pattern.
    "top_by_column": re.compile(r"\|\s*top\s+\d+\s+by\b"),
}

# Test configuration for unit tests
TEST_CONFIG = {
    "mock_cluster_uri": "help.kusto.windows.net",
    "mock_database": "Samples",
    "mock_table": "StormEvents"
}

def get_data_dir() -> Path:
    """Get the data directory for storing corpus and other data files."""
    if os.name == "nt":  # Windows
        base_dir = Path(os.environ.get("APPDATA", Path.home()))
        data_dir = base_dir / "KQL_MCP"
    else:  # Linux/macOS
        base_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        data_dir = base_dir / "mcp-kql-server"

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# Network Connection Configuration
CONNECTION_CONFIG = {
    "max_retries": 5,
    "retry_delay": 2.0,
    "retry_backoff_factor": 2.0,
    "max_retry_delay": 60.0,
    "connection_timeout": 30.0,
    "read_timeout": 300.0,
    "total_timeout": 600.0,
    "enable_connection_pooling": True,
    "pool_max_size": 10,
    "pool_block": False,
    "validate_connection_before_use": True,
    "connection_validation_timeout": 5.0,
}

# Error Handling Configuration
ERROR_HANDLING_CONFIG = {
    "enable_graceful_degradation": True,
    "isolate_thread_errors": True,
    "fallback_strategies": ["cached_schema", "query_derived_schema"],
    "error_recovery_timeout": 30.0,
    "max_consecutive_failures": 3,
    "failure_recovery_delay": 60.0,
    "enable_circuit_breaker": True,
    "circuit_breaker_threshold": 5,
    "circuit_breaker_recovery_time": 120.0,
}

# Retryable Error Patterns - Network and connection errors
RETRYABLE_ERROR_PATTERNS = [
    # Socket and connection errors
    r"SocketException.*?10060",  # Connection timed out
    r"ConnectionError",
    r"TimeoutError",
    r"timeout",
    r"Connection refused",
    r"Connection reset",
    r"Connection aborted",
    r"Network is unreachable",
    r"Host is unreachable",

    # gRPC specific errors
    r"grpc.*?unavailable",
    r"grpc.*?deadline_exceeded",
    r"grpc.*?resource_exhausted",
    r"subchannel.*?connection.*?failed",
    r"DNS resolution failed",

    # Kusto/Azure specific transient errors
    r"ServiceNotAvailable",
    r"InternalServerError",
    r"TooManyRequests",
    r"ThrottlingError",
    r"TemporaryFailure",
    r"ClusterNotFound.*?temporarily",

    # Authentication token refresh errors
    r"TokenExpired",
    r"AuthenticationFailed.*?retry",
    r"Invalid.*?token.*?refresh",
]

# Non-Retryable Error Patterns - Permanent failures
NON_RETRYABLE_ERROR_PATTERNS = [
    r"Unauthorized",
    r"Forbidden",
    r"NotFound.*?permanent",
    r"InvalidQuery",
    r"SyntaxError",
    r"ValidationError",
    r"SchemaError",
    r"PermissionDenied",
    r"QuotaExceeded.*?permanent",
]

# Precompiled regex sets for reuse across modules (reduce repeated compilation)
KQL_SYNTAX_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in KQL_SYNTAX_PATTERNS]
QUESTION_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in QUESTION_PATTERNS]

# Default await timeout for async wrappers that run sync work in background threads
DEFAULT_AWAIT_TIMEOUT = 30
