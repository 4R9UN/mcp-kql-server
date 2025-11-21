import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_kql_server.kql_validator import KQLValidator

async def test_validator_hardening():
    print("Testing validator hardening...")
    
    # Mock MemoryManager and SchemaManager
    mm = MagicMock()
    sm = MagicMock()
    
    validator = KQLValidator(mm, sm)
    
    # Mock schema with NO columns
    mm._get_database_schema.return_value = [{"table": "TestTable", "columns": {}}]
    
    # Mock ensure_schemas_exist to do nothing
    validator._ensure_schemas_exist = AsyncMock()
    
    # Test query against table with no columns
    result = await validator.validate_query("TestTable | take 10", "cluster", "db", auto_discover=False)
    
    if not result["valid"] and any("No columns found" in e for e in result["errors"]):
        print("✅ Validator correctly failed for table with no columns")
    else:
        print(f"❌ Validator failed to catch missing columns. Result: {result}")

async def main():
    await test_validator_hardening()

if __name__ == "__main__":
    asyncio.run(main())
