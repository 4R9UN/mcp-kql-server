"""Focused tests for the CAG ranking and NL-to-KQL generation pipeline."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from mcp_kql_server.memory import MemoryManager
from mcp_kql_server import mcp_server


@pytest.fixture
def memory_manager(tmp_path):
    """Create an isolated memory manager for ranking tests."""
    return MemoryManager(tmp_path / "ranking.db")


def test_rank_tables_for_query_prefers_column_overlap(memory_manager):
    """Hybrid ranking should prefer tables whose schema matches the request."""
    memory_manager.store_schema(
        "cluster",
        "db",
        "SigninLogs",
        {
            "TimeGenerated": {"data_type": "datetime"},
            "UserPrincipalName": {"data_type": "string"},
            "ResultType": {"data_type": "string"},
            "IPAddress": {"data_type": "string"},
        },
    )
    memory_manager.store_schema(
        "cluster",
        "db",
        "DeviceInventory",
        {
            "DeviceId": {"data_type": "string"},
            "DeviceName": {"data_type": "string"},
            "OSPlatform": {"data_type": "string"},
        },
    )

    ranked = memory_manager.rank_tables_for_query(
        "cluster",
        "db",
        "show failed signins by user in the last 24 hours",
        limit=2,
    )

    assert ranked
    assert ranked[0]["table"] == "SigninLogs"
    assert "UserPrincipalName" in ranked[0]["matched_columns"]


def test_build_cag_bundle_fingerprints_columns(memory_manager):
    """Compact CAG should keep only the highest-value columns."""
    memory_manager.store_schema(
        "cluster",
        "db",
        "SecurityAlert",
        {
            "TimeGenerated": {"data_type": "datetime"},
            "AlertId": {"data_type": "string"},
            "Severity": {"data_type": "string"},
            "ProviderName": {"data_type": "string"},
            "Description": {"data_type": "string"},
            "ExtendedProperties": {"data_type": "dynamic"},
        },
    )

    bundle = memory_manager.build_cag_bundle(
        "cluster",
        "db",
        "count alerts by severity in the last 7 days",
        max_tables=1,
        max_columns=3,
    )

    assert bundle["tables"]
    top_table = bundle["tables"][0]
    assert len(top_table["selected_columns"]) <= 3
    assert "Severity" in top_table["selected_columns"]
    assert bundle["context"].startswith("<CAG_CONTEXT>")


@pytest.mark.asyncio
async def test_generate_kql_from_nl_uses_ranked_candidates():
    """Generation should choose an aggregation candidate when the intent asks for counts."""
    bundle = {
        "tables": [
            {
                "table": "SecurityAlert",
                "score": 9.0,
                "matched_columns": ["Severity"],
                "columns": {
                    "TimeGenerated": {"data_type": "datetime"},
                    "Severity": {"data_type": "string"},
                    "AlertId": {"data_type": "string"},
                },
                "fingerprinted_columns": [
                    {"name": "Severity", "data_type": "string", "score": 4.0},
                    {"name": "TimeGenerated", "data_type": "datetime", "score": 3.0},
                    {"name": "AlertId", "data_type": "string", "score": 1.0},
                ],
                "selected_columns": {
                    "Severity": {"data_type": "string"},
                    "TimeGenerated": {"data_type": "datetime"},
                    "AlertId": {"data_type": "string"},
                },
            }
        ],
        "similar_queries": [],
        "join_hints": [],
        "context": "<CAG_CONTEXT>SecurityAlert(Severity:s, TimeGenerated:dt, AlertId:s)</CAG_CONTEXT>",
    }

    validator = AsyncMock(return_value={"valid": True, "errors": [], "columns_validated": 2})

    with patch.object(mcp_server, "memory_manager", MagicMock()) as mocked_memory, \
         patch.object(mcp_server, "kql_validator", MagicMock()) as mocked_validator:
        mocked_memory.build_cag_bundle.return_value = bundle
        mocked_validator.validate_query = validator

        result = await mcp_server._generate_kql_from_natural_language(
            "count alerts by severity in the last 24 hours",
            "cluster",
            "db",
        )

    assert result["success"] is True
    assert result["generation_method"] == "cag_hybrid_ranking"
    assert result["target_table"] == "SecurityAlert"
    assert "summarize count_ = count() by Severity" in result["query_plain"]
    assert result["query"] == result["query_plain"]


@pytest.mark.asyncio
async def test_generate_kql_from_nl_can_use_join_hints():
    """Join-oriented requests should be able to use stored join hints."""
    bundle = {
        "tables": [
            {
                "table": "SecurityAlert",
                "score": 9.0,
                "matched_columns": ["DeviceId"],
                "columns": {
                    "TimeGenerated": {"data_type": "datetime"},
                    "DeviceId": {"data_type": "string"},
                    "AlertId": {"data_type": "string"},
                },
                "fingerprinted_columns": [
                    {"name": "DeviceId", "data_type": "string", "score": 4.0},
                    {"name": "TimeGenerated", "data_type": "datetime", "score": 3.0},
                    {"name": "AlertId", "data_type": "string", "score": 1.0},
                ],
                "selected_columns": {
                    "DeviceId": {"data_type": "string"},
                    "TimeGenerated": {"data_type": "datetime"},
                    "AlertId": {"data_type": "string"},
                },
            },
            {
                "table": "DeviceInventory",
                "score": 8.0,
                "matched_columns": ["DeviceId"],
                "columns": {
                    "DeviceId": {"data_type": "string"},
                    "DeviceName": {"data_type": "string"},
                    "OSPlatform": {"data_type": "string"},
                },
                "fingerprinted_columns": [
                    {"name": "DeviceId", "data_type": "string", "score": 4.0},
                    {"name": "DeviceName", "data_type": "string", "score": 3.0},
                    {"name": "OSPlatform", "data_type": "string", "score": 2.0},
                ],
                "selected_columns": {
                    "DeviceId": {"data_type": "string"},
                    "DeviceName": {"data_type": "string"},
                    "OSPlatform": {"data_type": "string"},
                },
            },
        ],
        "similar_queries": [],
        "join_hints": ["SecurityAlert joins with DeviceInventory on DeviceId"],
        "context": "<CAG_CONTEXT>join context</CAG_CONTEXT>",
    }

    validator = AsyncMock(return_value={"valid": True, "errors": [], "columns_validated": 3})

    with patch.object(mcp_server, "memory_manager", MagicMock()) as mocked_memory, \
         patch.object(mcp_server, "kql_validator", MagicMock()) as mocked_validator:
        mocked_memory.build_cag_bundle.return_value = bundle
        mocked_validator.validate_query = validator

        result = await mcp_server._generate_kql_from_natural_language(
            "join alerts with device inventory across device id",
            "cluster",
            "db",
        )

    assert result["success"] is True
    assert "join kind=inner DeviceInventory on DeviceId" in result["query_plain"]


@pytest.mark.asyncio
async def test_execute_kql_query_repairs_invalid_time_column():
    """Direct KQL with a wrong time column should be repaired from schema context before execution."""
    validation_results = [
        {
            "valid": False,
            "errors": [
                "Column 'Timestamp' not found in table 'MtpAlertEvidence'. Sample columns: AlertId, ReportTime, DeviceId"
            ],
            "warnings": [],
            "suggestions": ["Check column names against the schema"],
            "tables_used": ["MtpAlertEvidence"],
            "columns_validated": 1,
        },
        {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "tables_used": ["MtpAlertEvidence"],
            "columns_validated": 2,
        },
        {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "tables_used": ["MtpAlertEvidence"],
            "columns_validated": 2,
        },
    ]

    with patch.object(mcp_server, "kusto_manager_global", {"authenticated": True}), \
         patch.object(mcp_server, "kql_execute_tool", return_value=pd.DataFrame([{"AlertId": "a1", "ReportTime": "2026-01-01"}])), \
         patch.object(mcp_server, "schema_manager", MagicMock()) as mocked_schema_manager, \
         patch.object(mcp_server, "kql_validator", MagicMock()) as mocked_validator:
        mocked_schema_manager.get_table_schema = AsyncMock(return_value={
            "table_name": "MtpAlertEvidence",
            "columns": {
                "AlertId": {"data_type": "string"},
                "ReportTime": {"data_type": "datetime"},
                "DeviceId": {"data_type": "string"},
            },
        })
        mocked_validator.validate_query = AsyncMock(side_effect=validation_results)

        result = await mcp_server.execute_kql_query.fn(
            query="MtpAlertEvidence | where AlertId == \"abc\" | order by Timestamp asc",
            cluster_url="cluster",
            database="db",
            generate_query=False,
        )

    payload = json.loads(result)
    assert payload["success"] is True
    assert payload["repair_applied"] is True
    assert payload["replacements"]["Timestamp"] == "ReportTime"


@pytest.mark.asyncio
async def test_schema_get_context_can_be_scoped_to_table():
    """Context retrieval should support a strict, single-table schema contract."""
    with patch.object(mcp_server, "schema_manager", MagicMock()) as mocked_schema_manager, \
         patch.object(mcp_server, "memory_manager", MagicMock()) as mocked_memory:
        mocked_schema_manager.get_table_schema = AsyncMock(return_value={
            "columns": {
                "AlertId": {"data_type": "string"},
                "ReportTime": {"data_type": "datetime"},
                "DeviceId": {"data_type": "string"},
            }
        })
        mocked_memory.find_similar_queries.return_value = []
        mocked_memory.get_join_hints.return_value = []
        mocked_memory.fingerprint_columns.return_value = [
            {"name": "AlertId", "data_type": "string", "score": 2.0},
            {"name": "ReportTime", "data_type": "datetime", "score": 1.0},
        ]
        mocked_memory._to_toon.return_value = "<CAG_CONTEXT>MtpAlertEvidence(AlertId:s, ReportTime:dt)</CAG_CONTEXT>"

        result = await mcp_server._schema_get_context_operation(
            "cluster",
            "db",
            "find alert evidence by alert id",
            table_name="MtpAlertEvidence",
        )

    payload = json.loads(result)
    assert payload["success"] is True
    assert payload["tables"] == ["MtpAlertEvidence"]
    assert "AlertId" in payload["strict_schema"]["MtpAlertEvidence"]["allowed_columns"]
    assert payload["strict_schema"]["MtpAlertEvidence"]["preferred_time_column"] == "ReportTime"
