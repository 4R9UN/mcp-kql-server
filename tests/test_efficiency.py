import pytest
from unittest.mock import MagicMock, patch
import json
from mcp_kql_server.memory import MemoryManager
# _schema_find_tables_operation was removed - test uses MemoryManager.find_relevant_tables directly

class TestEfficiencyFeatures:
    @pytest.fixture
    def memory_manager(self, tmp_path):
        # Use temp file DB for testing - :memory: creates new DB per connection
        db_path = str(tmp_path / "test.db")
        mm = MemoryManager(db_path)
        # Mock semantic search to avoid loading model
        mm.semantic_search = MagicMock()
        mm.semantic_search.encode.return_value = b'\x00' * 4  # Dummy float32 bytes
        return mm

    def test_semantic_table_search(self, memory_manager):
        """Test storing schema with embedding and finding relevant tables."""
        # Store schema
        memory_manager.store_schema("cluster1", "db1", "Users", {"columns": {"Name": "string", "Email": "string"}})
        memory_manager.store_schema("cluster1", "db1", "Logs", {"columns": {"Timestamp": "datetime", "Level": "string"}})
        
        # Mock find_relevant_tables to return results (since we mocked encode)
        with patch.object(memory_manager, 'find_relevant_tables') as mock_find:
            mock_find.return_value = [{"table": "Users", "score": 0.9}]
            
            results = memory_manager.find_relevant_tables("cluster1", "db1", "user data")
            assert len(results) == 1
            assert results[0]["table"] == "Users"

    def test_query_caching(self, memory_manager):
        """Test caching query results."""
        query = "Table | take 10"
        result = json.dumps([{"col": "val"}])
        
        # Cache result
        memory_manager.cache_query_result(query, result, 1)
        
        # Retrieve result
        cached = memory_manager.get_cached_result(query)
        assert cached == result
        
        # Retrieve non-existent
        assert memory_manager.get_cached_result("other query") is None

    def test_join_hints(self, memory_manager):
        """Test storing and retrieving join hints."""
        memory_manager.store_join_hint("TableA", "TableB", "A.id == B.id")
        
        hints = memory_manager.get_join_hints(["TableA", "TableC"])
        assert len(hints) == 1
        assert "TableA joins with TableB on A.id == B.id" in hints[0]

    @pytest.mark.asyncio
    async def test_find_tables_via_memory_manager(self):
        """Test find_relevant_tables via MemoryManager directly."""
        mm = MemoryManager(":memory:")
        mm.semantic_search = MagicMock()
        mm.semantic_search.encode.return_value = b'\x00' * 4
        
        with patch.object(mm, 'find_relevant_tables') as mock_find:
            mock_find.return_value = [{"table": "T1", "score": 0.8}]
            
            result = mm.find_relevant_tables("c", "d", "q")
            assert len(result) == 1
            assert result[0]["table"] == "T1"
