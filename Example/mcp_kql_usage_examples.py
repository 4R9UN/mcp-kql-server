#!/usr/bin/env python3
"""
MCP KQL Server Usage Examples

This file demonstrates how to use the MCP KQL Server tools
for various KQL query scenarios and schema management.
"""

import json
from typing import Any, Dict

# ============================================================================
# EXAMPLE 1: Basic KQL Query Execution
# ============================================================================


def example_basic_query() -> Dict[str, Any]:
    """Execute a basic KQL query with visualization."""

    request = {
        "tool": "kql_execute",
        "input": {
            "query": """
                cluster('help.kusto.windows.net')
                .database('Samples')
                .StormEvents
                | where State == 'TEXAS'
                | where StartTime >= datetime(2007-01-01)
                | summarize EventCount = count() by EventType
                | top 5 by EventCount desc
            """,
            "visualize": True,
            "use_schema_context": True,
        },
    }

    # Expected response structure:
    expected_response = {
        "status": "success",
        "result": {
            "columns": ["EventType", "EventCount"],
            "rows": [
                ["Hail", 742],
                ["Thunderstorm Wind", 568],
                ["Flash Flood", 289],
                ["Tornado", 187],
                ["Heavy Rain", 156],
            ],
            "row_count": 5,
            "visualization": "| EventType | EventCount |\n|---|---|\n| Hail | 742 |\n...",
            "schema_context": [
                "Table: StormEvents - Contains storm and weather event data...",
                "Key columns: StartTime, EndTime, State, EventType, DamageProperty",
            ],
        },
    }

    return {
        "description": "Basic query with aggregation and filtering",
        "request": request,
        "expected_response": expected_response,
    }


# ============================================================================
# EXAMPLE 2: Complex JSON Processing Query
# ============================================================================


def example_json_processing() -> Dict[str, Any]:
    """Execute a complex query with JSON parsing and extraction."""

    request = {
        "tool": "kql_execute",
        "input": {
            "query": """
                cluster('mycluster.kusto.windows.net')
                .database('ApplicationLogs')
                .Events
                | where Timestamp >= ago(24h)
                | extend EventProps = parse_json(Properties)
                | extend UserId = tostring(EventProps.userId)
                | extend SessionId = tostring(EventProps.sessionId)
                | extend ActionType = tostring(EventProps.actionType)
                | where isnotempty(UserId)
                | summarize
                    UniqueActions = dcount(ActionType),
                    TotalEvents = count(),
                    LastActivity = max(Timestamp)
                  by UserId
                | where UniqueActions >= 5
                | order by TotalEvents desc
                | limit 10
            """,
            "visualize": True,
            "use_schema_context": True,
        },
    }

    return {
        "description": "Complex query with JSON processing and user analytics",
        "request": request,
        "use_case": "Analyze user activity patterns from application logs",
    }


# ============================================================================
# EXAMPLE 3: Security Analysis Query
# ============================================================================


def example_security_analysis() -> Dict[str, Any]:
    """Execute a security-focused query with threat detection."""

    request = {
        "tool": "kql_execute",
        "input": {
            "query": """
                cluster('security.kusto.windows.net')
                .database('SecurityEvents')
                .SigninLogs
                | where TimeGenerated >= ago(7d)
                | where ResultType != "0"  // Failed sign-ins only
                | extend GeoInfo = parse_json(LocationDetails)
                | extend Country = tostring(GeoInfo.countryOrRegion)
                | extend City = tostring(GeoInfo.city)
                | summarize
                    FailedAttempts = count(),
                    UniqueIPs = dcount(IPAddress),
                    Countries = make_set(Country, 10)
                  by UserPrincipalName
                | where FailedAttempts >= 10 or UniqueIPs >= 5
                | order by FailedAttempts desc
            """,
            "visualize": True,
            "use_schema_context": True,
        },
    }

    return {
        "description": "Security analysis for suspicious login patterns",
        "request": request,
        "use_case": "Detect potential brute force attacks or compromised accounts",
    }


# ============================================================================
# EXAMPLE 5: Performance Optimized Query
# ============================================================================


def example_performance_optimized() -> Dict[str, Any]:
    """Execute a query optimized for performance."""

    request = {
        "tool": "kql_execute",
        "input": {
            "query": """
                cluster('logs.kusto.windows.net')
                .database('Telemetry')
                .PerformanceCounters
                | where TimeGenerated >= ago(1h)
                | where CounterName in ('% Processor Time', 'Available MBytes')
                | summarize
                    AvgValue = avg(CounterValue),
                    MaxValue = max(CounterValue),
                    MinValue = min(CounterValue)
                  by bin(TimeGenerated, 5m), CounterName, Computer
                | order by TimeGenerated desc
            """,
            "visualize": False,  # Disable for faster response
            "use_schema_context": False,  # Skip context loading for speed
        },
    }

    return {
        "description": "Performance-optimized query execution",
        "request": request,
        "optimization_techniques": [
            "Disabled visualization for faster response",
            "Disabled schema context loading",
            "Used specific time ranges",
            "Efficient aggregation with bin()",
        ],
    }


# ============================================================================
# EXAMPLE 6: Custom Memory Path
# ============================================================================


def example_custom_memory_path() -> Dict[str, Any]:
    """Use custom memory path for project-specific schema caching."""

    request = {
        "tool": "kql_execute",
        "input": {
            "query": """
                cluster('project-cluster.kusto.windows.net')
                .database('ProjectData')
                .UserEvents
                | take 100
            """,
            "cluster_memory_path": "/projects/myproject/kql_memory",
            "visualize": True,
            "use_schema_context": True,
        },
    }

    return {
        "description": "Query with custom memory path for project isolation",
        "request": request,
        "use_case": "Separate schema caches for different projects",
    }


# ============================================================================
# EXAMPLE 7: Error Handling and Debugging
# ============================================================================


def example_error_scenarios() -> Dict[str, Any]:
    """Examples of common errors and how the system handles them."""

    # Example of query with syntax error
    error_request = {
        "tool": "kql_execute",
        "input": {
            "query": """
                cluster('help.kusto.windows.net')
                .database('Samples')
                .NonExistentTable  // This table doesn't exist
                | take 10
            """,
            "visualize": True,
        },
    }

    expected_error_response = {
        "status": "error",
        "error": "KQL execution: KustoServiceError - Table 'NonExistentTable' not found. Did you mean 'StormEvents' or 'PopulationData'?",
    }

    return {
        "description": "Error handling with AI-powered suggestions",
        "error_request": error_request,
        "expected_error_response": expected_error_response,
        "ai_features": [
            "Suggests similar table names",
            "Provides context about available tables",
            "Enhanced error messages with solutions",
        ],
    }


# ============================================================================
# USAGE PATTERNS AND BEST PRACTICES
# ============================================================================


def usage_patterns():
    """Document common usage patterns and best practices."""

    patterns = {
        "workflow_2_development": [
            "1. Use visualize=True for data exploration",
            "2. Use visualize=False for production queries",
            "3. Enable debug mode for troubleshooting",
        ],
        "workflow_3_performance": [
            "1. Run schema discovery once per cluster",
            "2. Use custom memory paths for project isolation",
            "3. Disable context loading for high-frequency queries",
        ],
        "best_practices": [
            "Always authenticate with Azure CLI first (az login)",
            "Use specific time ranges to limit query scope",
            "Enable schema context for development, disable for production",
            "Use force_refresh when cluster schema changes",
            "Monitor memory file size for large clusters",
        ],
    }

    return patterns


# ============================================================================
# MAIN EXAMPLE RUNNER
# ============================================================================


def main():
    """Run all examples and display their structure."""

    examples = [
        example_basic_query(),
        example_json_processing(),
        example_security_analysis(),
        example_performance_optimized(),
        example_custom_memory_path(),
        example_error_scenarios(),
    ]

    print("=== MCP KQL Server Usage Examples ===\n")

    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example['description']}")
        print(f"Request: {json.dumps(example['request'], indent=2)}")
        if "expected_response" in example:
            print(f"Response: {json.dumps(example['expected_response'], indent=2)}")
        print("-" * 60)

    print("\nUsage Patterns:")
    patterns = usage_patterns()
    for pattern_name, steps in patterns.items():
        print(f"\n{pattern_name.replace('_', ' ').title()}:")
        for step in steps:
            print(f"  {step}")


if __name__ == "__main__":
    main()
