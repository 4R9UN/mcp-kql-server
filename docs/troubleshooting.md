# Troubleshooting Guide - MCP KQL Server

**Version**: 2.2.0
**Last Updated**: December 2025

---

## Table of Contents

1. [Quick Diagnostics](#1-quick-diagnostics)
2. [Authentication Issues](#2-authentication-issues)
3. [Connection Problems](#3-connection-problems)
4. [Query Errors](#4-query-errors)
5. [Schema & Memory Issues](#5-schema--memory-issues)
6. [Performance Problems](#6-performance-problems)
7. [Installation Issues](#7-installation-issues)
8. [MCP Integration Issues](#8-mcp-integration-issues)
9. [Logging & Debugging](#9-logging--debugging)
10. [Common Error Codes](#10-common-error-codes)

---

## 1. Quick Diagnostics

### Health Check Command

```bash
# Check if server is working
python -m mcp_kql_server --version

# Test basic connectivity
python -c "from mcp_kql_server import check_for_updates; print(check_for_updates())"
```

### Diagnostic Script

```python
import sys
print(f"Python Version: {sys.version}")

# Check imports
try:
    from mcp_kql_server import __version__
    print(f"MCP KQL Server Version: {__version__}")
except ImportError as e:
    print(f"Import Error: {e}")

# Check Azure CLI
import subprocess
result = subprocess.run(["az", "account", "show"], capture_output=True, text=True)
if result.returncode == 0:
    print("Azure CLI: Logged in")
else:
    print("Azure CLI: Not logged in")

# Check dependencies
deps = ["fastmcp", "azure.kusto.data", "azure.identity", "sentence_transformers"]
for dep in deps:
    try:
        __import__(dep.split(".")[0])
        print(f"{dep}: OK")
    except ImportError:
        print(f"{dep}: MISSING")
```

---

## 2. Authentication Issues

### Problem: "Not logged in to Azure CLI"

**Symptoms:**
- Error message: "Please log in using 'az login'"
- Authentication dialog doesn't appear

**Solutions:**

1. **Log in to Azure CLI:**
   ```bash
   az login
   ```

2. **For specific tenant:**
   ```bash
   az login --tenant YOUR_TENANT_ID
   ```

3. **Verify login:**
   ```bash
   az account show
   ```

4. **Check subscription access:**
   ```bash
   az account list --output table
   ```

---

### Problem: "Insufficient permissions"

**Symptoms:**
- Error: "User does not have access to database"
- 403 Forbidden errors

**Solutions:**

1. **Verify Kusto permissions:**
   ```kql
   .show database YourDatabase principals
   ```

2. **Request access from admin:**
   - Need at least "Database Viewer" role
   - Contact your Azure Data Explorer administrator

3. **Check if using correct account:**
   ```bash
   az account show --query user.name
   ```

---

### Problem: "Token expired"

**Symptoms:**
- Error after long idle time
- "Token has expired" message

**Solutions:**

1. **Re-authenticate:**
   ```bash
   az login --scope https://kusto.kusto.windows.net/.default
   ```

2. **Clear token cache:**
   ```bash
   az account clear
   az login
   ```

---

## 3. Connection Problems

### Problem: "Cannot connect to cluster"

**Symptoms:**
- Timeout errors
- "Name resolution failed"

**Solutions:**

1. **Verify cluster URL:**
   ```python
   # Correct formats:
   "https://cluster.region.kusto.windows.net"
   "cluster.region.kusto.windows.net"  # Auto-prefixed
   
   # Incorrect:
   "cluster.kusto.windows.net"  # Missing region
   ```

2. **Test network connectivity:**
   ```bash
   # Windows
   Test-NetConnection cluster.region.kusto.windows.net -Port 443
   
   # Linux/Mac
   nc -zv cluster.region.kusto.windows.net 443
   ```

3. **Check firewall/proxy:**
   - Ensure port 443 is open
   - Add cluster URL to proxy allowlist

---

### Problem: "Connection pool exhausted"

**Symptoms:**
- Slow responses after many queries
- "Unable to get connection" errors

**Solutions:**

1. **Increase pool size:**
   ```python
   from mcp_kql_server.performance import KustoConnectionPool
   pool = KustoConnectionPool(max_size=30)
   ```

2. **Clean up idle connections:**
   ```python
   from mcp_kql_server.performance import get_connection_pool
   pool = get_connection_pool()
   pool.cleanup_idle_connections()
   ```

3. **Check pool statistics:**
   ```python
   stats = pool.get_statistics()
   print(f"Connections: {stats['total_connections']}")
   print(f"Hit rate: {stats['hit_rate_percent']}%")
   ```

---

## 4. Query Errors

### Problem: SEM0100 - Column not found

**Symptoms:**
- Error: "SEM0100: Failed to resolve scalar expression named 'ColumnName'"

**Solutions:**

1. **Check available columns:**
   ```python
   from mcp_kql_server.memory import get_memory_manager
   memory = get_memory_manager()
   schema = memory.get_table_schema(cluster, database, table)
   print(schema['columns'])
   ```

2. **Refresh schema cache:**
   ```python
   # Clear and rediscover
   memory.clear_cache(cluster, database)
   # Schema will be rediscovered on next query
   ```

3. **Use schema memory tool:**
   ```
   schema_memory(operation="discover", cluster_url="...", database="...", table_name="...")
   ```

---

### Problem: SEM0102 - Table not found

**Symptoms:**
- Error: "SEM0102: Table 'TableName' not found"

**Solutions:**

1. **List available tables:**
   ```python
   # Via MCP tool
   schema_memory(operation="list_tables", cluster_url="...", database="...")
   ```

2. **Check table name case sensitivity:**
   - Kusto table names are case-sensitive
   - "StormEvents" ≠ "stormevents"

3. **Verify database context:**
   - Ensure you're querying the correct database

---

### Problem: Query timeout

**Symptoms:**
- Error: "Query execution timeout"
- Very long execution times

**Solutions:**

1. **Increase timeout:**
   ```python
   # Set environment variable
   import os
   os.environ["KQL_QUERY_TIMEOUT"] = "600"  # 10 minutes
   ```

2. **Optimize query:**
   ```kql
   // Add time filters
   | where Timestamp > ago(1h)
   
   // Limit rows early
   | take 10000
   
   // Use summarize instead of returning all rows
   | summarize count() by Category
   ```

3. **Check cluster load:**
   - Query during off-peak hours
   - Contact cluster admin if consistently slow

---

### Problem: "Too many rows"

**Symptoms:**
- Error: "Query result exceeds maximum row count"

**Solutions:**

1. **Add limit to query:**
   ```kql
   | take 10000
   | limit 10000
   ```

2. **Use aggregation:**
   ```kql
   | summarize count() by Category
   ```

3. **Adjust max rows setting:**
   ```python
   import os
   os.environ["KQL_MAX_ROWS"] = "50000"
   ```

---

## 5. Schema & Memory Issues

### Problem: Schema cache is stale

**Symptoms:**
- Query fails but works in Kusto Explorer
- Columns exist but not recognized

**Solutions:**

1. **Force schema refresh:**
   ```python
   from mcp_kql_server.memory import get_memory_manager
   memory = get_memory_manager()
   memory.clear_cache(cluster_url, database)
   ```

2. **Use refresh_schema operation:**
   ```
   schema_memory(operation="refresh_schema", cluster_url="...", database="...")
   ```

---

### Problem: Memory database corrupted

**Symptoms:**
- SQLite errors
- "database disk image is malformed"

**Solutions:**

1. **Delete and recreate:**
   ```bash
   # Windows
   del %APPDATA%\KQL_MCP\kql_memory.db
   
   # Linux/Mac
   rm ~/.local/share/KQL_MCP/kql_memory.db
   ```

2. **Use custom path:**
   ```python
   from mcp_kql_server.memory import MemoryManager
   memory = MemoryManager(db_path="/new/path/memory.db")
   ```

---

### Problem: Semantic search not working

**Symptoms:**
- `find_relevant_tables` returns empty
- NL2KQL not finding correct tables

**Solutions:**

1. **Check if embeddings exist:**
   ```python
   stats = memory.get_memory_stats()
   print(f"Embeddings: {stats.get('total_embeddings', 0)}")
   ```

2. **Rediscover schemas to generate embeddings:**
   ```python
   schema_memory(operation="discover", cluster_url="...", database="...", table_name="TableName")
   ```

3. **Check sentence-transformers installation:**
   ```bash
   pip install -U sentence-transformers
   ```

---

## 6. Performance Problems

### Problem: Slow first query

**Symptoms:**
- First query takes 10+ seconds
- Subsequent queries are fast

**Causes & Solutions:**

1. **Model loading (expected):**
   - Sentence-transformers model loads on first use
   - This is normal, cached after first load

2. **Connection establishment:**
   - First connection requires authentication
   - Enable connection pooling to keep connections warm

3. **Schema discovery:**
   - Use schema preloading at startup:
   ```python
   from mcp_kql_server.performance import SchemaPreloader
   preloader = SchemaPreloader()
   preloader.preload_schemas(cluster, database)
   ```

---

### Problem: High memory usage

**Symptoms:**
- Server using > 1GB RAM
- Memory keeps growing

**Solutions:**

1. **Clear old embeddings:**
   ```python
   memory.clear_cache()
   ```

2. **Reduce embedding model size:**
   - Use smaller model variant
   - Contact maintainer for model options

3. **Limit connection pool:**
   ```python
   pool = KustoConnectionPool(max_size=5)
   ```

---

## 7. Installation Issues

### Problem: "Module not found"

**Symptoms:**
- `ModuleNotFoundError: No module named 'mcp_kql_server'`

**Solutions:**

1. **Install package:**
   ```bash
   pip install mcp-kql-server
   ```

2. **Verify installation:**
   ```bash
   pip show mcp-kql-server
   ```

3. **Check Python environment:**
   ```bash
   which python  # Linux/Mac
   where python  # Windows
   pip list | grep mcp
   ```

---

### Problem: Dependency conflicts

**Symptoms:**
- `ERROR: pip's dependency resolver`
- Version incompatibility warnings

**Solutions:**

1. **Create fresh environment:**
   ```bash
   python -m venv mcp_env
   source mcp_env/bin/activate  # Linux/Mac
   mcp_env\Scripts\activate     # Windows
   pip install mcp-kql-server
   ```

2. **Update dependencies:**
   ```bash
   pip install --upgrade mcp-kql-server
   ```

3. **Install specific version:**
   ```bash
   pip install mcp-kql-server==2.2.0
   ```

---

## 8. MCP Integration Issues

### Problem: Server not detected by Claude

**Symptoms:**
- Claude doesn't show KQL tools
- MCP connection fails

**Solutions:**

1. **Check mcp.json configuration:**
   ```json
   {
       "mcpServers": {
           "kql-server": {
               "command": "python",
               "args": ["-m", "mcp_kql_server"]
           }
       }
   }
   ```

2. **Verify Python path:**
   ```json
   {
       "mcpServers": {
           "kql-server": {
               "command": "C:\\Python311\\python.exe",
               "args": ["-m", "mcp_kql_server"]
           }
       }
   }
   ```

3. **Test manually:**
   ```bash
   python -m mcp_kql_server
   # Should output MCP protocol messages
   ```

---

### Problem: Tool calls fail silently

**Symptoms:**
- No error message
- Empty response

**Solutions:**

1. **Enable verbose logging:**
   ```json
   {
       "mcpServers": {
           "kql-server": {
               "command": "python",
               "args": ["-m", "mcp_kql_server"],
               "env": {
                   "KQL_LOG_LEVEL": "DEBUG"
               }
           }
       }
   }
   ```

2. **Check MCP client logs:**
   - Claude Desktop: Check `~/Library/Logs/Claude/` (Mac)
   - VS Code: Check Output panel -> MCP

---

## 9. Logging & Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("mcp_kql_server").setLevel(logging.DEBUG)
```

### Log File Location

| OS | Default Log Path |
|----|------------------|
| Windows | `%APPDATA%\KQL_MCP\logs\` |
| Linux | `~/.local/share/KQL_MCP/logs/` |
| macOS | `~/Library/Application Support/KQL_MCP/logs/` |

### Environment Variables for Debugging

```bash
# Maximum verbosity
export KQL_LOG_LEVEL=DEBUG
export AZURE_LOG_LEVEL=DEBUG

# Log to file
export KQL_LOG_FILE=/path/to/debug.log
```

---

## 10. Common Error Codes

### Kusto Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| SEM0100 | Column not found | Check column name spelling |
| SEM0102 | Table not found | Verify table exists |
| SEM0103 | Function not found | Check function name |
| SYN0001 | Syntax error | Fix KQL syntax |
| GEN0001 | General error | Check error details |
| AUTH001 | Auth failure | Re-authenticate with `az login` |
| RATE001 | Rate limited | Wait and retry |

### MCP KQL Server Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| MCP001 | Server startup failed | Check dependencies |
| MCP002 | Tool registration failed | Restart server |
| MCP003 | Memory initialization failed | Check disk space |
| CONN001 | Connection pool exhausted | Increase pool size |
| CACHE001 | Schema cache miss | Run schema discovery |

---

## Getting Help

If you've tried the above solutions and still have issues:

1. **Check GitHub Issues:**
   https://github.com/4R9UN/mcp-kql-server/issues

2. **Create New Issue:**
   Include:
   - Python version
   - MCP KQL Server version
   - Full error message
   - Steps to reproduce

3. **Contact:**
   Email: arjuntrivedi42@yahoo.com

---

## Related Documentation

- [API Reference](api-reference.md)
- [Architecture](architecture.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Security Policy](../SECURITY.md)
