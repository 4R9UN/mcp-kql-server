# MCP KQL Server

mcp-name: io.github.4R9UN/mcp-kql-server

[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/4r9un-mcp-kql-server-badge.png)](https://mseep.ai/app/4r9un-mcp-kql-server)

> **AI-Powered KQL Query Execution with Natural Language to KQL (NL2KQL) Conversion and Execution**

A Model Context Protocol (MCP) server that transforms natural language questions into optimized KQL queries with intelligent schema discovery, AI-powered caching, and seamless Azure Data Explorer integration. Simply ask questions in plain English and get instant, accurate KQL queries with context-aware results.

<!-- Badges Section -->

[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/18772353-3d90-4b12-a253-cf752efaadd2)
[![PyPI version](https://img.shields.io/pypi/v/mcp-kql-server.svg)](https://pypi.org/project/mcp-kql-server/)
[![Python](https://img.shields.io/pypi/pyversions/mcp-kql-server.svg)](https://pypi.org/project/mcp-kql-server/)

[![CI/CD Pipeline](https://github.com/4R9UN/mcp-kql-server/workflows/MCP%20KQL%20Server%20CI%2FCD%20Pipeline/badge.svg)](https://github.com/4R9UN/mcp-kql-server/actions)
[![codecov](https://codecov.io/gh/4R9UN/mcp-kql-server/branch/main/graph/badge.svg)](https://codecov.io/gh/4R9UN/mcp-kql-server)
[![Security Rating](https://img.shields.io/badge/security-A+-brightgreen.svg)](https://github.com/4R9UN/mcp-kql-server/security)
[![Code Quality](https://img.shields.io/badge/code%20quality-A-brightgreen.svg)](https://github.com/4R9UN/mcp-kql-server)

[![FastMCP](https://img.shields.io/badge/framework-FastMCP-green.svg)](https://github.com/jlowin/fastmcp)
[![Azure Data Explorer](https://img.shields.io/badge/Azure-Data%20Explorer-orange.svg)](https://azure.microsoft.com/en-us/services/data-explorer/)
[![MCP Protocol](https://img.shields.io/badge/MCP-2024--11--05-blue.svg)](https://github.com/anthropics/mcp)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/4R9UN/mcp-kql-server/graphs/commit-activity)
[![MCP Badge](https://lobehub.com/badge/mcp/4r9un-mcp-kql-server?style=for-the-badge)](https://lobehub.com/mcp/4r9un-mcp-kql-server)


## 🎬 Demo

Watch a quick demo of the MCP KQL Server in action:

[![MCP KQL Server Demo](https://img.youtube.com/vi/Ca-yuThJ3Vc/0.jpg)](https://www.youtube.com/watch?v=Ca-yuThJ3Vc)

## 🚀 Features

- **`execute_kql_query`**:
    - **Natural Language to KQL**: Generate KQL queries from natural language descriptions.
    - **Direct KQL Execution**: Execute raw KQL queries.
    - **Multiple Output Formats**: Supports JSON, CSV, and table formats.
    - **Live Schema Validation**: Ensures query accuracy by using live schema discovery.

- **`schema_memory`**:
    - **Schema Discovery**: Discover and cache schemas for tables.
    - **Database Exploration**: List all tables within a database.
    - **AI Context**: Get AI-driven context for tables.
    - **Analysis Reports**: Generate reports with visualizations.
    - **Cache Management**: Clear or refresh the schema cache.
    - **Memory Statistics**: Get statistics about the memory usage.


## 📊 MCP Tools Execution Flow

```mermaid
graph TD
    A[👤 User Submits KQL Query] --> B{🔍 Query Validation}
    B -->|❌ Invalid| C[📝 Syntax Error Response]
    B -->|✅ Valid| D[🧠 Load Schema Context]
    
    D --> E{💾 Schema Cache Available?}
    E -->|✅ Yes| F[⚡ Load from Memory]
    E -->|❌ No| G[🔍 Discover Schema]
    
    F --> H[🎯 Execute Query]
    G --> I[💾 Cache Schema + AI Context]
    I --> H
    
    H --> J{🎯 Query Success?}
    J -->|❌ Error| K[🚨 Enhanced Error Message]
    J -->|✅ Success| L[📊 Process Results]
    
    L --> M[🎨 Generate Visualization]
    M --> N[📤 Return Results + Context]
    
    K --> O[💡 AI Suggestions]
    O --> N
    
    style A fill:#4a90e2,stroke:#2c5282,stroke-width:2px,color:#ffffff
    style B fill:#7c7c7c,stroke:#4a4a4a,stroke-width:2px,color:#ffffff
    style C fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#ffffff
    style D fill:#8e44ad,stroke:#6a1b99,stroke-width:2px,color:#ffffff
    style E fill:#7c7c7c,stroke:#4a4a4a,stroke-width:2px,color:#ffffff
    style F fill:#27ae60,stroke:#1e8449,stroke-width:2px,color:#ffffff
    style G fill:#f39c12,stroke:#d68910,stroke-width:2px,color:#ffffff
    style H fill:#2980b9,stroke:#1f618d,stroke-width:2px,color:#ffffff
    style I fill:#f39c12,stroke:#d68910,stroke-width:2px,color:#ffffff
    style J fill:#7c7c7c,stroke:#4a4a4a,stroke-width:2px,color:#ffffff
    style K fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#ffffff
    style L fill:#27ae60,stroke:#1e8449,stroke-width:2px,color:#ffffff
    style M fill:#8e44ad,stroke:#6a1b99,stroke-width:2px,color:#ffffff
    style N fill:#27ae60,stroke:#1e8449,stroke-width:2px,color:#ffffff
    style O fill:#f39c12,stroke:#d68910,stroke-width:2px,color:#ffffff
```

### Schema Memory Discovery Flow

The `kql_schema_memory` functionality is now seamlessly integrated into the `kql_execute` tool. When you run a query, the server automatically discovers and caches the schema for any tables it hasn't seen before. This on-demand process ensures you always have the context you need without any manual steps.

```mermaid
graph TD
    A[👤 User Requests Schema Discovery] --> B[🔗 Connect to Cluster]
    B --> C[📂 Enumerate Databases]
    C --> D[📋 Discover Tables]
    
    D --> E[🔍 Get Table Schemas]
    E --> F[🤖 AI Analysis]
    F --> G[📝 Generate Descriptions]
    
    G --> H[💾 Store in Memory]
    H --> I[📊 Update Statistics]
    I --> J[✅ Return Summary]
    
    style A fill:#4a90e2,stroke:#2c5282,stroke-width:2px,color:#ffffff
    style B fill:#8e44ad,stroke:#6a1b99,stroke-width:2px,color:#ffffff
    style C fill:#f39c12,stroke:#d68910,stroke-width:2px,color:#ffffff
    style D fill:#2980b9,stroke:#1f618d,stroke-width:2px,color:#ffffff
    style E fill:#7c7c7c,stroke:#4a4a4a,stroke-width:2px,color:#ffffff
    style F fill:#e67e22,stroke:#bf6516,stroke-width:2px,color:#ffffff
    style G fill:#8e44ad,stroke:#6a1b99,stroke-width:2px,color:#ffffff
    style H fill:#f39c12,stroke:#d68910,stroke-width:2px,color:#ffffff
    style I fill:#2980b9,stroke:#1f618d,stroke-width:2px,color:#ffffff
    style J fill:#27ae60,stroke:#1e8449,stroke-width:2px,color:#ffffff
```


## 📋 Prerequisites

- Python 3.10 or higher
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows?view=azure-cli-latest&pivots=msi) installed and authenticated (`az login`)
- Access to Azure Data Explorer cluster(s)

## 🚀 One-Command Installation

### Quick Install (Recommended)

#### From Source

```bash
git clone https://github.com/4R9UN/mcp-kql-server.git && cd mcp-kql-server && pip install -e .
```
### Alternative Installation Methods

```bash
pip install mcp-kql-server
```

**That's it!** The server automatically:
- ✅ Sets up memory directories in `%APPDATA%\KQL_MCP` (Windows) or `~/.local/share/KQL_MCP` (Linux/Mac)
- ✅ Configures optimal defaults for production use
- ✅ Suppresses verbose Azure SDK logs
- ✅ No environment variables required


## 📱 MCP Client Configuration

### Claude Desktop

Add to your Claude Desktop MCP settings file (`mcp_settings.json`):

**Location:**
- **Windows**: `%APPDATA%\Claude\mcp_settings.json`
- **macOS**: `~/Library/Application Support/Claude/mcp_settings.json`
- **Linux**: `~/.config/Claude/mcp_settings.json`

```json
{
  "mcpServers": {
    "mcp-kql-server": {
      "command": "python",
      "args": ["-m", "mcp_kql_server"],
      "env": {}
    }
  }
}
```
### VSCode (with MCP Extension)

Add to your VSCode MCP configuration:

**Settings.json location:**
- **Windows**: `%APPDATA%\Code\User\mcp.json`
- **macOS**: `~/Library/Application Support/Code/User/mcp.json`
- **Linux**: `~/.config/Code/User/mcp.json`

```json
{
 "MCP-kql-server": {
			"command": "python",
			"args": [
				"-m",
				"mcp_kql_server"
			],
			"type": "stdio"
		}
}
```

### Roo-code Or Cline (VS-code Extentions)

Ask or Add to your Roo-code Or Cline MCP settings:

**MCP Settings location:**
- **All platforms**: Through Roo-code extension settings or `mcp_settings.json`

```json
{
   "MCP-kql-server": {
      "command": "python",
      "args": [
        "-m",
        "mcp_kql_server"
      ],
      "type": "stdio",
      "alwaysAllow": [
      ]
    },
}
```

### Generic MCP Client

For any MCP-compatible application:

```bash
# Command to run the server
python -m mcp_kql_server

# Server provides these tools:
# - kql_execute: Execute KQL queries with AI context
# - kql_schema_memory: Discover and cache cluster schemas
```
## 🔧 Quick Start

### 1. Authenticate with Azure (One-time setup)

```bash
az login
```

### 2. Start the MCP Server (Zero configuration)

```bash
python -m mcp_kql_server
```

The server starts immediately with:
- 📁 **Auto-created memory path**: `%APPDATA%\KQL_MCP\cluster_memory`
- 🔧 **Optimized defaults**: No configuration files needed
- 🔐 **Secure setup**: Uses your existing Azure CLI credentials

### 3. Use via MCP Client

The server provides two main tools:

> #### `kql_execute` - Execute KQL Queries with AI Context
> #### `kql_schema_memory` - Discover and Cache Cluster Schemas


## 💡 Usage Examples

### Basic Query Execution

Ask your MCP client (like Claude):
> "Execute this KQL query against the help cluster: `cluster('help.kusto.windows.net').database('Samples').StormEvents | take 10` and summarize the result and give me high level insights "

### Complex Analytics Query

Ask your MCP client:
> "Query the Samples database in the help cluster to show me the top 10 states by storm event count, include visualization"

### Schema Discovery

Ask your MCP client:
> "Discover and cache the schema for the help.kusto.windows.net cluster, then tell me what databases and tables are available"

### Data Exploration with Context

Ask your MCP client:
> "Using the StormEvents table in the Samples database on help cluster, show me all tornado events from 2007 with damage estimates over $1M"

### Time-based Analysis

Ask your MCP client:
> "Analyze storm events by month for the year 2007 in the StormEvents table, group by event type and show as a visualization"



## 🎯 Key Benefits

### For Data Analysts
- **⚡ Faster Query Development**: AI-powered autocomplete and suggestions
- **🎨 Rich Visualizations**: Instant markdown tables for data exploration
- **🧠 Context Awareness**: Understand your data structure without documentation

### For DevOps Teams
- **🔄 Automated Schema Discovery**: Keep schema information up-to-date
- **💾 Smart Caching**: Reduce API calls and improve performance
- **🔐 Secure Authentication**: Leverage existing Azure CLI credentials

### For AI Applications
- **🤖 Intelligent Query Assistance**: AI-generated table descriptions and suggestions
- **📊 Structured Data Access**: Clean, typed responses for downstream processing
- **🎯 Context-Aware Responses**: Rich metadata for better AI decision making

## 🏗️ Architecture

```mermaid
graph TD
    A[MCP Client<br/>Claude/AI/Custom] <--> B[MCP KQL Server<br/>FastMCP Framework]
    B <--> C[Azure Data Explorer<br/>Kusto Clusters]
    B <--> D[Schema Memory<br/>Local AI Cache]
    
    style A fill:#4a90e2,stroke:#2c5282,stroke-width:3px,color:#ffffff
    style B fill:#8e44ad,stroke:#6a1b99,stroke-width:3px,color:#ffffff
    style C fill:#e67e22,stroke:#bf6516,stroke-width:3px,color:#ffffff
    style D fill:#27ae60,stroke:#1e8449,stroke-width:3px,color:#ffffff
```

## 📁 Project Structure

```
mcp-kql-server/
├── mcp_kql_server/
│   ├── __init__.py          # Package initialization
│   ├── mcp_server.py        # Main MCP server implementation
│   ├── execute_kql.py       # KQL query execution logic
│   ├── memory.py            # Advanced memory management
│   ├── kql_auth.py          # Azure authentication
│   ├── utils.py             # Utility functions
│   └── constants.py         # Configuration constants
├── docs/                    # Documentation
├── Example/                 # Usage examples
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## 🔒 Security

- **Azure CLI Authentication**: Leverages your existing Azure device login
- **No Credential Storage**: Server doesn't store authentication tokens
- **Local Memory**: Schema cache stored locally, not transmitted

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   # Re-authenticate with Azure CLI
   az login --tenant your-tenant-id
   ```

2. **Memory Issues**
   ```bash
   # The memory cache is now managed automatically. If you suspect issues,
   # you can clear the cache directory, and it will be rebuilt on the next query.
   # Windows:
   rmdir /s /q "%APPDATA%\KQL_MCP\unified_memory.json"
   
   # macOS/Linux:
   rm -rf ~/.local/share/KQL_MCP/cluster_memory
   ```

3. **Connection Timeouts**
   - Check cluster URI format
   - Verify network connectivity
   - Confirm Azure permissions

## 🤝 Contributing

We welcome contributions! Please do. 

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/4R9UN/mcp-kql-server/issues)
- **PyPI Package**: [PyPI Project Page](https://pypi.org/project/mcp-kql-server/)
- **Author**: [Arjun Trivedi](mailto:arjuntrivedi42@yahoo.com)
- **Certified** : [MCPHub](https://mcphub.com/mcp-servers/4R9UN/mcp-kql-server)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=4R9UN/mcp-kql-server&type=Date)](https://star-history.com/#4R9UN/mcp-kql-server&Date)

---
mcp-name: io.github.4R9UN/mcp-kql-server

**Happy Querying! 🎉**
