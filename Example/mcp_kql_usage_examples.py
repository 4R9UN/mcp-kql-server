#!/usr/bin/env python3
"""Current MCP KQL Server usage examples for the v2.1.1 tool contract."""

import json
from typing import Any, Dict

def example_basic_query() -> Dict[str, Any]:
    """Execute a basic KQL query with the current tool shape."""

    request = {
        "tool": "execute_kql_query",
        "input": {
            "query": "StormEvents | where State == 'TEXAS' | where StartTime >= datetime(2007-01-01) | summarize EventCount = count() by EventType | top 5 by EventCount desc",
            "cluster_url": "https://help.kusto.windows.net",
            "database": "Samples",
            "output_format": "table",
        },
    }

    # Expected response structure:
    expected_response = {
        "success": True,
        "query": "StormEvents | where State == 'TEXAS' | where StartTime >= datetime(2007-01-01) | summarize EventCount = count() by EventType | top 5 by EventCount desc",
        "row_count": 5,
        "columns": ["EventType", "EventCount"],
        "data": [
            ["Hail", 742],
            ["Thunderstorm Wind", 568],
            ["Flash Flood", 289],
            ["Tornado", 187],
            ["Heavy Rain", 156],
        ],
    }

    return {
        "description": "Basic query with aggregation and filtering",
        "request": request,
        "expected_response": expected_response,
    }


def example_json_processing() -> Dict[str, Any]:
    """Execute a complex query against a specific cluster and database."""

    request = {
        "tool": "execute_kql_query",
        "input": {
            "query": "Events | where Timestamp >= ago(24h) | extend EventProps = parse_json(Properties) | extend UserId = tostring(EventProps.userId) | extend SessionId = tostring(EventProps.sessionId) | extend ActionType = tostring(EventProps.actionType) | where isnotempty(UserId) | summarize UniqueActions = dcount(ActionType), TotalEvents = count(), LastActivity = max(Timestamp) by UserId | where UniqueActions >= 5 | order by TotalEvents desc | limit 10",
            "cluster_url": "https://mycluster.kusto.windows.net",
            "database": "ApplicationLogs",
            "output_format": "json",
        },
    }

    return {
        "description": "Complex query with JSON processing and user analytics",
        "request": request,
        "use_case": "Analyze user activity patterns from application logs",
    }


def example_security_analysis() -> Dict[str, Any]:
    """Execute a security-focused query with threat detection."""

    request = {
        "tool": "execute_kql_query",
        "input": {
            "query": "SigninLogs | where TimeGenerated >= ago(7d) | where ResultType != '0' | extend GeoInfo = parse_json(LocationDetails) | extend Country = tostring(GeoInfo.countryOrRegion) | extend City = tostring(GeoInfo.city) | summarize FailedAttempts = count(), UniqueIPs = dcount(IPAddress), Countries = make_set(Country, 10) by UserPrincipalName | where FailedAttempts >= 10 or UniqueIPs >= 5 | order by FailedAttempts desc",
            "cluster_url": "https://security.kusto.windows.net",
            "database": "SecurityEvents",
            "output_format": "table",
        },
    }

    return {
        "description": "Security analysis for suspicious login patterns",
        "request": request,
        "use_case": "Detect potential brute force attacks or compromised accounts",
    }


def example_performance_optimized() -> Dict[str, Any]:
    """Execute a query optimized for performance."""

    request = {
        "tool": "execute_kql_query",
        "input": {
            "query": "PerformanceCounters | where TimeGenerated >= ago(1h) | where CounterName in ('% Processor Time', 'Available MBytes') | summarize AvgValue = avg(CounterValue), MaxValue = max(CounterValue), MinValue = min(CounterValue) by bin(TimeGenerated, 5m), CounterName, Computer | order by TimeGenerated desc",
            "cluster_url": "https://logs.kusto.windows.net",
            "database": "Telemetry",
            "output_format": "json",
        },
    }

    return {
        "description": "Performance-optimized query execution",
        "request": request,
        "optimization_techniques": [
            "Pinned the cluster and database explicitly",
            "Requested JSON output for cheaper transport",
            "Used a narrow time range",
            "Kept aggregation work on the cluster side",
        ],
    }


def example_schema_memory() -> Dict[str, Any]:
    """Discover schema using the current schema_memory tool."""

    request = {
        "tool": "schema_memory",
        "input": {
            "operation": "discover",
            "cluster_url": "https://project-cluster.kusto.windows.net",
            "database": "ProjectData",
            "table_name": "UserEvents",
        },
    }

    return {
        "description": "Schema discovery for a target table",
        "request": request,
        "use_case": "Warm the schema cache before NL2KQL or troubleshooting",
    }


def example_error_scenarios() -> Dict[str, Any]:
    """Examples of common errors and how the system handles them."""

    # Example of query with syntax error
    error_request = {
        "tool": "execute_kql_query",
        "input": {
            "query": "NonExistentTable | take 10",
            "cluster_url": "https://help.kusto.windows.net",
            "database": "Samples",
            "output_format": "json",
        },
    }

    expected_error_response = {
        "success": False,
        "error": "KQL execution failed because the table could not be resolved.",
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
            "1. Run az login before starting the server locally",
            "2. Use schema_memory(discover/list_tables/get_context) before NL2KQL-heavy sessions",
            "3. Prefer table or JSON output depending on how your MCP client renders results",
        ],
        "workflow_3_performance": [
            "1. Keep time windows tight on large telemetry tables",
            "2. Use summarize/bin() to reduce result volume before transport",
            "3. Reuse cached schema context instead of rediscovering tables unnecessarily",
        ],
        "best_practices": [
            "Always authenticate with Azure CLI first (az login)",
            "Use specific time ranges to limit query scope",
            "Use schema_memory(refresh_schema) when table metadata changes",
            "Use schema_memory(get_context) to ground NL2KQL with the right tables",
            "Keep local Azure CLI auth and Azure-hosted managed identity flows documented separately",
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
        example_schema_memory(),
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
