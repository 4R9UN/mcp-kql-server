"""Offline quality gates for schema-based table ranking."""

import pytest

from mcp_kql_server.memory import MemoryManager


@pytest.fixture
def ranking_memory(tmp_path):
    manager = MemoryManager(tmp_path / "ranking-quality.db")
    manager.semantic_search.encode = lambda _text: None
    schemas = {
        "SigninLogs": {
            "TimeGenerated": {"data_type": "datetime"},
            "UserPrincipalName": {"data_type": "string"},
            "ResultType": {"data_type": "string"},
            "IPAddress": {"data_type": "string"},
        },
        "DeviceProcessEvents": {
            "Timestamp": {"data_type": "datetime"},
            "DeviceName": {"data_type": "string"},
            "ProcessCommandLine": {"data_type": "string"},
            "AccountName": {"data_type": "string"},
        },
        "EmailEvents": {
            "Timestamp": {"data_type": "datetime"},
            "SenderFromAddress": {"data_type": "string"},
            "RecipientEmailAddress": {"data_type": "string"},
            "ThreatTypes": {"data_type": "string"},
        },
        "SecurityAlert": {
            "TimeGenerated": {"data_type": "datetime"},
            "AlertId": {"data_type": "string"},
            "Severity": {"data_type": "string"},
            "Description": {"data_type": "string"},
        },
        "CloudAppEvents": {
            "Timestamp": {"data_type": "datetime"},
            "Application": {"data_type": "string"},
            "ActionType": {"data_type": "string"},
            "AccountId": {"data_type": "string"},
        },
        "DeviceNetworkEvents": {
            "Timestamp": {"data_type": "datetime"},
            "DeviceName": {"data_type": "string"},
            "RemoteIP": {"data_type": "string"},
            "RemotePort": {"data_type": "int"},
        },
    }
    for table, columns in schemas.items():
        manager.store_schema("cluster.kusto.windows.net", "db", table, {"columns": columns})
    return manager


def test_lexical_ranking_quality(ranking_memory):
    """Schema retrieval maintains perfect top-one accuracy on core intents."""
    cases = [
        ("show failed signins by user over the last day", "SigninLogs"),
        ("find suspicious process command lines by device", "DeviceProcessEvents"),
        ("list phishing email events by sender and recipient", "EmailEvents"),
        ("count security alerts grouped by severity", "SecurityAlert"),
        ("show cloud application actions by account", "CloudAppEvents"),
        ("list remote network connections and ports by device", "DeviceNetworkEvents"),
    ]
    reciprocal_ranks = []
    top_one_hits = 0

    for query, expected_table in cases:
        ranked = ranking_memory.rank_tables_for_query(
            "cluster.kusto.windows.net", "db", query, limit=6
        )
        tables = [item["table"] for item in ranked]
        rank = tables.index(expected_table) + 1
        reciprocal_ranks.append(1.0 / rank)
        top_one_hits += rank == 1

    assert top_one_hits / len(cases) == 1.0
    assert sum(reciprocal_ranks) / len(reciprocal_ranks) == 1.0