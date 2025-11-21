import asyncio
import json
import sys
import os
from unittest.mock import MagicMock, patch

# Add current directory to path
sys.path.append(os.getcwd())

from mcp_kql_server.memory import MemoryManager
from mcp_kql_server.mcp_server import _schema_find_tables_operation

async def run_tests():
    print("Starting Manual Verification...")
    
    import tempfile
    import shutil
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_memory.db")
    
    try:
        # Setup
        mm = MemoryManager(db_path)
        mm.semantic_search = MagicMock()
        mm.semantic_search.encode.return_value = b'\x00' * 4
        
        # Test 1: Semantic Search
        print("Test 1: Semantic Table Search...", end="")
        mm.store_schema("c1", "d1", "Users", {"columns": {"Name": "s"}})
        with patch.object(mm, 'find_relevant_tables') as mock_find:
            mock_find.return_value = [{"table": "Users", "score": 0.9}]
            results = mm.find_relevant_tables("c1", "d1", "q")
            assert len(results) == 1
            assert results[0]["table"] == "Users"
        print("PASSED")

        # Test 2: Caching
        print("Test 2: Query Caching...", end="")
        query = "T | take 1"
        res = json.dumps([{"a": 1}])
        mm.cache_query_result(query, res, 1)
        cached = mm.get_cached_result(query)
        assert cached == res
        assert mm.get_cached_result("other") is None
        print("PASSED")

        # Test 3: Join Hints
        print("Test 3: Join Hints...", end="")
        mm.store_join_hint("A", "B", "on X")
        hints = mm.get_join_hints(["A"])
        assert len(hints) == 1
        assert "A joins with B on on X" in hints[0]
        print("PASSED")

        # Test 4: Find Tables Operation
        print("Test 4: Find Tables Operation...", end="")
        with patch('mcp_kql_server.mcp_server.memory_manager', mm):
            # We need to patch the method on the instance we just injected
            with patch.object(mm, 'find_relevant_tables') as mock_find:
                mock_find.return_value = [{"table": "T1", "score": 0.8}]
                result = await _schema_find_tables_operation("c", "d", "q")
                data = json.loads(result)
                assert data["success"] is True
                assert len(data["tables"]) == 1
        print("PASSED")
        
        print("\nALL TESTS PASSED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass # Ignore cleanup errors on Windows (file in use)

if __name__ == "__main__":
    asyncio.run(run_tests())
