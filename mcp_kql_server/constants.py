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
from pathlib import Path
from typing import Any, Dict, List


# Version information
__version__ = "2.0.7"
VERSION = __version__
MCP_PROTOCOL_VERSION = "2024-11-05"

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
FASTAPI_VERSION = "2.0.7"
FASTAPI_DESCRIPTION = "Local AI-powered KQL Server with no external dependencies"
FASTAPI_HOST = "0.0.0.0"
FASTAPI_PORT = 8000

# Tool Configuration
TOOL_KQL_EXECUTE_NAME = "kql_execute"
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
DEFAULT_QUERY_TIMEOUT = 600

# Schema memory configuration
SCHEMA_CACHE_MAX_AGE_DAYS = 7
MAX_SCHEMA_FILE_SIZE_MB = 10
MAX_TABLES_PER_DATABASE = 1000
MAX_COLUMNS_PER_TABLE = 500

# Query validation
MAX_QUERY_LENGTH = 100000
MIN_QUERY_LENGTH = 10

# Default configuration (no environment variables required)
DEFAULT_CONFIG = {
    "DEBUG_MODE": False,
    "AZURE_ERRORS_ONLY": True,
    "CONNECTION_TIMEOUT": DEFAULT_CONNECTION_TIMEOUT,
    "QUERY_TIMEOUT": DEFAULT_QUERY_TIMEOUT,
    "AUTO_CREATE_MEMORY_PATH": True,
    "ENABLE_LOGGING": True,
}

# Caching and strategy defaults (sensible fallbacks for missing configuration)
CACHE_STRATEGIES = {
    # Maximum number of schema entries to retain in memory (LRU eviction)
    "SCHEMA_CACHE_SIZE": 1000,
    # Maximum number of pattern analysis entries to retain
    "PATTERN_CACHE_SIZE": 500,
    # Column mapping cache size
    "COLUMN_MAPPING_CACHE_SIZE": 500,
}

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


# Dynamic Schema Analysis Configuration
class DynamicSchemaAnalyzer:
    """
    Replace static patterns with dynamic analysis.
    This class provides methods for dynamic table and column analysis.
    """
    
    @staticmethod
    def analyze_table_semantics(table_name: str, sample_data: List[Dict] = None) -> Dict[str, Any]:
        """
        Dynamically analyze table purpose from actual data patterns.
        Replaces static keyword-based table categorization.
        """
        if not table_name:
            return {"purpose": "unknown", "confidence": 0.0}
        
        # Dynamic analysis based on data patterns
        analysis = {
            "purpose": "data_table",
            "confidence": 0.5,
            "temporal_columns": [],
            "key_columns": [],
            "relationships": []
        }
        
        # Analyze sample data if available
        if sample_data and len(sample_data) > 0:
            analysis.update(DynamicSchemaAnalyzer._analyze_data_patterns(sample_data))
        
        # Table name pattern analysis (dynamic, not hardcoded keywords)
        analysis.update(DynamicSchemaAnalyzer._analyze_table_name_patterns(table_name))
        
        return analysis
    
    @staticmethod
    def _analyze_data_patterns(sample_data: List[Dict]) -> Dict[str, Any]:
        """Analyze actual data patterns to determine table characteristics."""
        if not sample_data:
            return {}
        
        patterns = {
            "has_timestamps": False,
            "has_identifiers": False,
            "has_metrics": False,
            "data_quality": "unknown"
        }
        
        # Check for common column patterns in data
        first_row = sample_data[0] if sample_data else {}
        
        for column, value in first_row.items():
            if value is None:
                continue
                
            value_str = str(value)
            
            # Detect temporal patterns
            if DynamicSchemaAnalyzer._looks_like_timestamp(value_str):
                patterns["has_timestamps"] = True
            
            # Detect identifier patterns
            if DynamicSchemaAnalyzer._looks_like_identifier(value_str):
                patterns["has_identifiers"] = True
            
            # Detect metric patterns
            if DynamicSchemaAnalyzer._looks_like_metric(value_str):
                patterns["has_metrics"] = True
        
        return patterns
    
    @staticmethod
    def _analyze_table_name_patterns(table_name: str) -> Dict[str, Any]:
        """Analyze table name for semantic patterns without hardcoded keywords."""
        name_lower = table_name.lower()
        
        # Dynamic pattern recognition
        patterns = {
            "table_type": "general",
            "likely_temporal": False,
            "likely_transactional": False
        }
        
        # Use pattern length and structure analysis instead of keywords
        if len(name_lower) > 10 and any(sep in name_lower for sep in ['_', '-']):
            patterns["table_type"] = "structured"
        
        # Check for timestamp-like naming patterns
        if any(time_indicator in name_lower for time_indicator in ['time', 'date', 'log', 'event']):
            patterns["likely_temporal"] = True
        
        # Check for transaction-like patterns
        if name_lower.endswith(('s', 'es', 'ies')):  # Plural forms often indicate collections
            patterns["likely_transactional"] = True
        
        return patterns
    
    @staticmethod
    def _looks_like_timestamp(value: str) -> bool:
        """Dynamic timestamp detection."""
        import re
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # Date pattern
            r'\d{2}:\d{2}:\d{2}',  # Time pattern
            r'^\d{10,13}$',        # Unix timestamp
        ]
        return any(re.search(pattern, value) for pattern in timestamp_patterns)
    
    @staticmethod
    def _looks_like_identifier(value: str) -> bool:
        """Dynamic identifier detection."""
        import re
        id_patterns = [
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',  # UUID
            r'^[0-9a-f]{32}$',  # MD5 hash
            r'^[A-Z0-9]{8,}$',  # Uppercase alphanumeric ID
        ]
        return any(re.match(pattern, value, re.IGNORECASE) for pattern in id_patterns)
    
    @staticmethod
    def _looks_like_metric(value: str) -> bool:
        """Dynamic metric detection."""
        try:
            float(value.replace(',', ''))
            return True
        except ValueError:
            return False


# Dynamic Column Analysis Configuration
class DynamicColumnAnalyzer:
    """
    Replace static column patterns with dynamic analysis.
    """
    
    @staticmethod
    def generate_column_tags(column_name: str, data_type: str, sample_values: List[str] = None) -> List[str]:
        """
        Generate smart column tags based on data type and patterns, not static keywords.
        Replaces the static keyword-based approach in memory.py and utils.py.
        """
        tags = []
        
        # Dynamic data type analysis
        type_lower = data_type.lower() if data_type else ""
        
        # Type-based tags (primary categorization)
        if any(t in type_lower for t in ['datetime', 'timestamp', 'date', 'time']):
            tags.append("TEMPORAL")
        elif any(t in type_lower for t in ['int', 'long', 'short', 'byte']):
            tags.append("INTEGER")
        elif any(t in type_lower for t in ['real', 'double', 'float', 'decimal']):
            tags.append("DECIMAL")
        elif any(t in type_lower for t in ['bool', 'boolean']):
            tags.append("BOOLEAN")
        elif any(t in type_lower for t in ['guid', 'uuid']):
            tags.append("IDENTIFIER")
        elif any(t in type_lower for t in ['dynamic', 'json', 'object']):
            tags.append("STRUCTURED")
        elif any(t in type_lower for t in ['string', 'text', 'varchar', 'char']):
            tags.append("TEXT")
        
        # Dynamic sample value analysis
        if sample_values:
            value_tags = DynamicColumnAnalyzer._analyze_sample_values(sample_values)
            tags.extend(value_tags)
        
        # Dynamic column name analysis (pattern-based, not keyword matching)
        name_tags = DynamicColumnAnalyzer._analyze_column_name_patterns(column_name)
        tags.extend(name_tags)
        
        return list(dict.fromkeys(tags))[:3]  # Remove duplicates and limit to 3 tags
    
    @staticmethod
    def _analyze_sample_values(sample_values: List[str]) -> List[str]:
        """Analyze sample values to determine column characteristics."""
        tags = []
        
        if not sample_values:
            return tags
        
        # Check if all values are unique (likely identifier)
        if len(set(sample_values)) == len(sample_values):
            tags.append("UNIQUE_VALUES")
        
        # Check if values are categorizable (limited set)
        if len(set(sample_values)) <= len(sample_values) * 0.5:
            tags.append("CATEGORICAL")
        
        # Check for numerical patterns
        try:
            numeric_values = [float(str(v).replace(',', '')) for v in sample_values if v]
            if numeric_values:
                if all(v.is_integer() for v in numeric_values):
                    tags.append("COUNT_LIKE")
                else:
                    tags.append("MEASUREMENT_LIKE")
        except (ValueError, AttributeError):
            pass
        
        return tags
    
    @staticmethod
    def _analyze_column_name_patterns(column_name: str) -> List[str]:
        """Analyze column name patterns dynamically."""
        tags = []
        name_lower = column_name.lower()
        
        # Dynamic pattern analysis
        if name_lower.endswith('id') or name_lower.endswith('_id'):
            tags.append("ID_COLUMN")
        
        if 'time' in name_lower or 'date' in name_lower:
            tags.append("TIME_COLUMN")
        
        if name_lower.startswith('is_') or name_lower.startswith('has_'):
            tags.append("BOOLEAN_FLAG")
        
        if any(sep in name_lower for sep in ['_count', '_total', '_sum']):
            tags.append("AGGREGATION_COLUMN")
        
        return tags


# Replace static patterns with dynamic factory functions
def get_dynamic_table_analyzer() -> DynamicSchemaAnalyzer:
    """Get dynamic table analyzer instance."""
    return DynamicSchemaAnalyzer()


def get_dynamic_column_analyzer() -> DynamicColumnAnalyzer:
    """Get dynamic column analyzer instance."""
    return DynamicColumnAnalyzer()

# Monitoring system removed - no placeholder functions needed

# Query Chaining Configuration
QUERY_CHAINING_CONFIG = {
    "enable_query_chaining": True,
    "enable_background_schema_discovery": True,
    "enable_context_aware_planning": True,
    "enable_execution_result_feedback": True,
    "max_chain_length": 5,
    "chain_timeout_seconds": 300,
    "context_window_size": 1000,
    "enable_adaptive_throttling": True,
}

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
    "fallback_strategies": ["cached_schema", "query_derived_schema", "minimal_schema"],
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

BACKGROUND_SCHEMA_CONFIG = {
    "enable_trigger_based_discovery": True,
    "discovery_trigger_conditions": {
        "new_table_reference": True,
        "unknown_column_usage": True,
        "query_execution_success": True,
        "schema_validation_failure": True,
        "context_gap_detected": True,
    },
}

# Types of operations that can be orchestrated via query chaining
CHAINING_OPERATION_TYPES = [
    "discover_schema",
    "update_schema",
    "generate_kql",
    "execute_kql",
    "persist_memory",
]

def get_query_chaining_config() -> dict:
    """Return the active chaining configuration (fallback to defaults)."""
    return QUERY_CHAINING_CONFIG

def get_chaining_threshold(default: float = 0.5) -> float:
    """Return the chaining threshold used to decide whether to trigger chaining behavior."""
    try:
        return float(QUERY_CHAINING_CONFIG.get("chaining_threshold", default))
    except Exception:
        return default

def is_chaining_feature_enabled(feature_name: str, config_section: str = "chaining") -> bool:
    config = QUERY_CHAINING_CONFIG if config_section == "chaining" else {}
    return bool(config.get(feature_name, False))

def should_trigger_background_schema_discovery(trigger_type: str) -> bool:
    if not is_chaining_feature_enabled("enable_background_schema_discovery"):
        return False
    trigger_conditions = BACKGROUND_SCHEMA_CONFIG.get("discovery_trigger_conditions", {})
    return trigger_conditions.get(trigger_type, False)

# Precompiled regex sets for reuse across modules (reduce repeated compilation)
KQL_SYNTAX_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in KQL_SYNTAX_PATTERNS]
QUESTION_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in QUESTION_PATTERNS]

def is_performance_feature_enabled(feature_name: str) -> bool:
    """Return whether a performance-related feature is enabled (case-insensitive)."""
    try:
        return bool(PERFORMANCE_CONFIG.get(feature_name.upper(), False))
    except Exception:
        return False

# Default await timeout for async wrappers that run sync work in background threads
DEFAULT_AWAIT_TIMEOUT = 30
