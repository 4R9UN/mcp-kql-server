import asyncio
# import json - removed unused import
import logging
import os
import sys
from unittest.mock import MagicMock, AsyncMock
from fastmcp import Context

# Add current directory to path
sys.path.insert(0, os.getcwd())

from mcp_kql_server.memory import MemoryManager # noqa: E402
import mcp_kql_server.memory # noqa: E402
print(f"DEBUG: Loaded memory from {mcp_kql_server.memory.__file__}")
print(f"DEBUG: MemoryManager methods: {[m for m in dir(MemoryManager) if 'schema' in m]}")
from mcp_kql_server.mcp_server import ( # noqa: E402
    _schema_refresh_operation,
    _schema_discover_operation,
    # logging_middleware, - removed unused import
    schema_manager
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_tests():
    print("Starting GoFastMCP Features Verification...")
    
    import tempfile
    import shutil
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_memory.db")
    
    try:
        # Setup MemoryManager
        mm = MemoryManager(db_path)
        mm.semantic_search = MagicMock()
        mm.semantic_search.encode.return_value = b'\x00' * 4
        
        # Mock SchemaManager methods
        schema_manager.memory_manager = mm
        schema_manager.get_table_schema = AsyncMock(return_value={
            "columns": {"col1": "string", "col2": "int"},
            "last_updated": "2023-01-01"
        })
        
        # --- Test 2: Progress Reporting ---
        print("\nTest 2: Progress Reporting...")
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.report_progress = AsyncMock()
        mock_ctx.info = MagicMock()

        # Mock SchemaDiscovery
        from unittest.mock import patch
        
        with patch("mcp_kql_server.utils.SchemaDiscovery") as MockSchemaDiscovery:
            instance = MockSchemaDiscovery.return_value
            instance.list_tables_in_db = AsyncMock(return_value=["Table1", "Table2"])
            
            await _schema_refresh_operation("https://cluster.kusto.windows.net", "db", ctx=mock_ctx)
        
        # Verify progress calls
        assert mock_ctx.report_progress.call_count >= 2
        print("Progress Reporting...PASSED")
        
        # --- Test 3: LLM Sampling ---
        print("\nTest 3: LLM Sampling...")
        mock_ctx.sample_llm = AsyncMock(return_value=MagicMock(content="This is a test description."))
        
        await _schema_discover_operation("https://cluster.kusto.windows.net", "db", "Table1", ctx=mock_ctx)
        
        # Verify sampling call
        mock_ctx.sample_llm.assert_called_once()
        
        # Verify description stored in DB
        with mm._lock, mm._get_connection() as conn:
            cursor = conn.execute("SELECT description FROM schemas WHERE table_name='Table1'")
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "This is a test description."
            
        print("LLM Sampling...PASSED")
        
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
            pass

if __name__ == "__main__":
    asyncio.run(run_tests())
