import asyncio
# import json - removed unused import
import os
import sys
# from datetime import datetime - removed unused import

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_kql_server.memory import MemoryManager

async def test_memory_updates():
    print("Testing MemoryManager updates...")
    db_path = "test_memory_opt.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    mm = MemoryManager(db_path=db_path)
    
    # Test schema update (execution_time_ms)
    try:
        mm.add_successful_query("cluster", "db", "query", "desc", execution_time_ms=123.45)
        print("✅ add_successful_query with execution_time_ms passed")
    except Exception as e:
        print(f"❌ add_successful_query failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        mm.store_learning_result("query", {"data": 1}, "test", execution_time_ms=50.0)
        print("✅ store_learning_result with execution_time_ms passed")
    except Exception as e:
        print(f"❌ store_learning_result failed: {e}")
        import traceback
        traceback.print_exc()
        
    metrics = mm.get_performance_metrics()
    print(f"Metrics: {metrics}")
    if "average_execution_time_ms" in metrics:
        print("✅ get_performance_metrics passed")
    else:
        print("❌ get_performance_metrics missing keys")

    # Clean up
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass

async def main():
    await test_memory_updates()

if __name__ == "__main__":
    asyncio.run(main())
