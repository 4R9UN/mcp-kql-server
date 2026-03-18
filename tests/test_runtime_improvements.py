"""Regression tests for cache scoping, table tracking, and runtime safety."""

import json
from unittest.mock import AsyncMock

import pytest

from mcp_kql_server.memory import MemoryManager
from mcp_kql_server.performance import ConnectionInfo, KustoConnectionPool
from mcp_kql_server.utils import SchemaManager


@pytest.fixture
def memory_manager(tmp_path):
    """Create an isolated memory manager."""
    return MemoryManager(tmp_path / "runtime.db")


def test_cache_isolated_by_cluster_and_namespace(memory_manager):
    """The same query text should not collide across clusters or cache namespaces."""
    query = "SecurityAlert | take 1"
    memory_manager.cache_query_result(
        query,
        json.dumps({"source": "cluster-a-json"}),
        1,
        cluster="https://cluster-a.kusto.windows.net",
        database="db",
        cache_namespace="execute:json",
    )
    memory_manager.cache_query_result(
        query,
        json.dumps({"source": "cluster-b-table"}),
        1,
        cluster="https://cluster-b.kusto.windows.net",
        database="db",
        cache_namespace="execute:table",
    )

    cached_a = memory_manager.get_cached_result(
        query,
        cluster="https://cluster-a.kusto.windows.net",
        database="db",
        cache_namespace="execute:json",
    )
    cached_b = memory_manager.get_cached_result(
        query,
        cluster="https://cluster-b.kusto.windows.net",
        database="db",
        cache_namespace="execute:table",
    )

    assert json.loads(cached_a)["source"] == "cluster-a-json"
    assert json.loads(cached_b)["source"] == "cluster-b-table"


def test_store_schema_registers_persistent_table_locations(memory_manager):
    """Schema storage should also populate the persistent table location registry."""
    schema = {"columns": {"TimeGenerated": {"data_type": "datetime"}}}
    memory_manager.store_schema("https://cluster-a.kusto.windows.net", "db", "SecurityAlert", schema)
    memory_manager.store_schema("https://cluster-b.kusto.windows.net", "db", "SecurityAlert", schema)

    locations = memory_manager.get_table_locations("SecurityAlert")

    assert len(locations) == 2
    assert {item["cluster"] for item in locations} == {
        "https://cluster-a.kusto.windows.net",
        "https://cluster-b.kusto.windows.net",
    }


@pytest.mark.asyncio
async def test_cached_schema_is_reused_without_reindex(memory_manager):
    """If a real schema already exists, schema retrieval should use it without live re-discovery."""
    memory_manager.store_schema(
        "https://cluster-a.kusto.windows.net",
        "db",
        "SecurityAlert",
        {"columns": {"AlertId": {"data_type": "string"}, "ReportTime": {"data_type": "datetime"}}},
    )
    manager = SchemaManager(memory_manager)
    manager._execute_kusto_async = AsyncMock(side_effect=AssertionError("live discovery should not run"))

    schema = await manager.get_table_schema("https://cluster-a.kusto.windows.net", "db", "SecurityAlert")

    assert schema["discovery_method"] == "cached_schema_memory"
    assert "AlertId" in schema["columns"]


def test_clear_query_cache_can_be_scoped(memory_manager):
    """Scoped cache clear should only remove matching entries."""
    query = "SecurityAlert | take 1"
    memory_manager.cache_query_result(query, "a", 1, cluster="https://cluster-a.kusto.windows.net", database="db")
    memory_manager.cache_query_result(query, "b", 1, cluster="https://cluster-b.kusto.windows.net", database="db")

    removed = memory_manager.clear_query_cache(cluster="https://cluster-a.kusto.windows.net", database="db")

    assert removed == 1
    assert memory_manager.get_cached_result(
        query,
        cluster="https://cluster-a.kusto.windows.net",
        database="db",
    ) is None
    assert memory_manager.get_cached_result(
        query,
        cluster="https://cluster-b.kusto.windows.net",
        database="db",
    ) == "b"


def test_memory_stats_include_generation_telemetry(memory_manager):
    """Generation events should be visible in memory stats."""
    memory_manager.store_learning_result(
        "count alerts by severity",
        {"query": "SecurityAlert | summarize count() by Severity"},
        execution_type="generation",
    )

    stats = memory_manager.get_memory_stats()

    assert stats["learning_by_type"]["generation"] == 1


def test_health_checker_uses_registered_database():
    """Health checks should probe a registered database, not the cluster URL."""
    pool = KustoConnectionPool()
    pool._health_check_databases.clear()  # pylint: disable=protected-access

    class FakeClient:
        def __init__(self):
            self.calls = []

        def execute(self, database, query):
            self.calls.append((database, query))
            return {"ok": True}

    client = FakeClient()
    conn_info = ConnectionInfo(client=client, cluster_url="https://cluster.kusto.windows.net")
    pool.register_health_check_database("https://cluster.kusto.windows.net", "SecurityDB")

    try:
        assert pool._check_connection_health(conn_info) is True  # pylint: disable=protected-access
        assert client.calls == [("SecurityDB", "print 1")]
    finally:
        pool.stop_health_checker()
