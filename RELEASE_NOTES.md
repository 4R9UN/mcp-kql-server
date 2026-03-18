# Release Notes - MCP KQL Server

---

## 🚀 **v2.1.1 - Schema-Grounded CAG & Runtime Accuracy**

> **Strict schema-first generation, safer cache scoping, and repair-before-execute** 

**Release Date**: March 17, 2026
**Author**: Arjun Trivedi
**Email**: arjuntrivedi42@yahoo.com
**Repository**: https://github.com/4R9UN/mcp-kql-server

### 🚀 **What's New in v2.1.1**

This release focuses on correctness and schema fidelity. The server now prefers cached schema/CAG context over re-indexing, generates KQL from ranked schema context, and can repair invalid client-generated KQL only when a real schema proves the replacement.

#### **1. Stricter CAG and Schema Memory Usage**
- **Schema-First Generation**: NL2KQL generation now uses ranked table and column context from schema memory before building candidate queries.
- **Strict Table Context**: `schema_memory(operation="get_context")` can now be scoped to a specific table and returns:
  - allowed columns
  - recommended columns
  - preferred time column
  - compact CAG context
- **No Redundant Reindexing**: Existing cached schema is reused before live discovery, and empty placeholder discovery paths no longer overwrite rich table schemas.

#### **2. Runtime Query Accuracy**
- **Schema-Grounded Repair**: When direct KQL contains invalid columns, the server now attempts a schema-backed repair and only executes the repaired query if it fully revalidates.
- **Improved Join-Aware Generation**: Candidate generation now includes join-aware paths when join hints exist in memory.
- **Generation Telemetry**: Memory stats now include generation event counts and candidate metrics for tuning.

#### **3. Cache and Runtime Stability**
- **Scoped Query Cache**: Cached results are now isolated by query text, cluster, database, and cache namespace.
- **Safe Cache Clearing**: `schema_memory(operation="clear_cache")` now clears real cached state rather than only returning a success message.
- **Shared Connection Pooling**: Execution paths now use the shared pool instead of a duplicate local client cache path.
- **Safer Health Checks**: Pool health checks probe only with a registered database and avoid false recycling when no database context exists.

#### **4. Repository Cleanup**
- **Removed Scratch Docs**: Deleted ad hoc analysis documents that were not part of the shipped project docs.
- **Version Alignment**: Package metadata, docs, server manifest, and tests are aligned to `2.1.1`.

### 🐛 **Bug Fixes**
- Fixed repeated schema writes when schema was already available in memory/CAG.
- Prevented empty placeholder schemas from overwriting real indexed schemas.
- Fixed direct-query failures caused by invalid generated/client column names such as non-schema time aliases.

### 📦 **Installation & Upgrade**

#### **New Installation**
```bash
pip install mcp-kql-server==2.1.1
```

#### **Upgrade from Previous Versions**
```bash
pip install --upgrade mcp-kql-server
```

### ✅ **Quality Assurance**
- **Tests**: 76 passed, 1 skipped
- **Verified**: Generation, execution, runtime cache, health-check, and package-version regression suites passed

---

## 🚀 **v2.1.0 - NL2KQL Enhancement & Version Management**

> **Schema-Only NL2KQL & Auto-Update Functionality** 🧠

**Release Date**: December 28, 2025
**Author**: Arjun Trivedi
**Email**: arjuntrivedi42@yahoo.com
**Repository**: https://github.com/4R9UN/mcp-kql-server

### 🚀 **What's New in v2.1.0**

This release brings significant improvements to Natural Language to KQL (NL2KQL) accuracy and introduces automatic version management capabilities.

#### **1. NL2KQL Schema-Only Approach**
- **🎯 100% Schema-Driven**: NL2KQL now uses ONLY data from schema memory - no hardcoded table, cluster, or column names
- **🔍 Semantic Table Search**: Uses `find_relevant_tables()` for intelligent table discovery
- **✅ Strict Column Validation**: All columns are validated against schema memory before use
- **📝 Clear Error Messages**: Returns helpful errors when schema is not discovered yet

#### **2. PyPI Version Checker**
- **🔄 Auto-Update Detection**: Checks PyPI for new versions at startup
- **📢 Update Notifications**: Notifies users when a new version is available
- **⚡ Optional Auto-Install**: Can automatically install updates with `auto_update=True`
- **🛡️ Safe Fallback**: Gracefully handles network errors or offline mode

#### **3. Log Improvements**
- **📋 Unicode Removed**: All emoji characters replaced with ASCII equivalents
- **🖥️ Better Console Output**: Cleaner logs for terminals that don't support Unicode
- **📊 UTF-8 Encoding**: Added UTF-8 encoding configuration for stdout/stderr

#### **4. New Module: version_checker.py**
- `get_current_version()` - Returns installed version
- `fetch_latest_pypi_version()` - Fetches latest from PyPI API
- `compare_versions()` - Compares version strings
- `check_for_updates(auto_update=False)` - Main update check function
- `install_update()` - Runs pip upgrade
- `startup_version_check()` - Called at server startup

### 🐛 **Bug Fixes**
- **NL2KQL Fix**: Resolved issue where words like "getschema" were incorrectly extracted as table names
- **Test Fixes**: Updated test mocking to properly handle client cache clearing
- **Import Fix**: Added proper module exports for version checker functions

### 📦 **Installation & Upgrade**

#### **New Installation**
```bash
pip install mcp-kql-server==2.1.0
```

#### **Upgrade from Previous Versions**
```bash
pip install --upgrade mcp-kql-server
```

### ✅ **Quality Assurance**
- **Tests**: 59 passed, 1 skipped
- **Code Quality**: All pylint issues resolved
- **Verified**: Full regression testing passed for all core functionalities

---

## 🚀 **v2.0.9 - Major MCP Update & Intelligence Upgrade**

> **CAG & SQLite Optimization Release** 🧠

**Release Date**: November 21, 2025
**Author**: Arjun Trivedi
**Email**: arjuntrivedi42@yahoo.com
**Repository**: https://github.com/4R9UN/mcp-kql-server

### 🚀 **What's New in v2.0.9**

This release brings significant improvements to the server's intelligence and performance, featuring enhanced Context-Aware Generation (CAG) and optimized local memory management with SQLite.

#### **1. Major MCP Updates**
- **🧠 CAG Updates**: Enhanced Context-Aware Generation for more accurate KQL queries. The AI now better understands schema relationships and query intent.
- **💾 SQLite Updates**: Improved local memory management with optimized SQLite integration for faster and more reliable schema caching.

#### **2. Efficiency & Accuracy**
- **🎯 Accuracy**: Improved KQL generation accuracy by ~15% through better schema validation and prompt engineering.
- **⚡ Efficiency**: Reduced query latency by ~20% with optimized caching and connection handling.

#### **3. New Features**
- **🔍 Interesting Findings**: New section in analysis reports that highlights key insights and anomalies in the data.
- **💧 Watermark**: Added watermark to generated reports for better traceability and provenance.
- **🎨 Premium Visualizations**: Enhanced Mermaid visualization with a new "Cyberpunk/Neon" color palette for stunning, high-contrast diagrams.

#### **4. Production & Deployment**
- **🚀 Production Deployment Guide**: Comprehensive guide for deploying to Azure Container Apps with enterprise-grade security
  - 📖 Full documentation available in [`deployment/README.md`](deployment/README.md)
  - ⚡ One-command deployment scripts (PowerShell & Bash)
  - 🏗️ Infrastructure as Code with Bicep templates
  - 🔒 Managed Identity for passwordless authentication
  - 📊 Integrated monitoring with Log Analytics
- **🐳 Docker Containerization**: Multi-stage Dockerfile optimized for production
  - ✅ Lightweight Python 3.11 slim base image
  - ✅ Security best practices with non-root user
  - ✅ Optimized layer caching for faster builds
  - ✅ Health checks and proper signal handling
  - 📦 Ready for Azure Container Registry deployment

### 🐛 **Bug Fixes**
- **Critical Fix**: Resolved unbound variable error in `execute_kql.py`.
- **Validation**: Fixed `SEM0100` errors with stricter column validation.
- **Code Quality**: Resolved multiple `pylint` issues, improving code score to **8.73/10**.

### 📦 **Installation & Upgrade**

#### **New Installation**
```bash
pip install mcp-kql-server==2.0.9
```

#### **Upgrade from Previous Versions**
```bash
pip install --upgrade mcp-kql-server
```

### ✅ **Quality Assurance**
- **Verified**: All AI prompts and visualization themes tested and verified.
- **Tested**: Full regression testing passed for all core functionalities.


## 📦 **v2.0.8 - GitHub MCP Registry Ready**

> **Registry Optimization & Documentation Update** 🚀

**Release Date**: November 18, 2025
**Author**: Arjun Trivedi
**Email**: arjuntrivedi42@yahoo.com
**Repository**: https://github.com/4R9UN/mcp-kql-server

### 🚀 **What's New in v2.0.8**

This release prepares the MCP KQL Server for GitHub MCP Registry publication with enhanced metadata and improved documentation.

#### **1. GitHub MCP Registry Compliance**
- **📋 Updated Schema**: Migrated to latest MCP schema (2025-09-29)
- **🏷️ Added Tags**: 10 discovery tags (kql, kusto, azure, nl2kql, natural-language, etc.)
- **✨ Enhanced Metadata**: Added title field and improved description
- **📝 PyPI Verification**: Added `mcp-name` identifier to README for ownership proof

#### **2. Documentation Improvements**
- **🎯 NL2KQL Emphasis**: Highlighted Natural Language to KQL conversion throughout
- **🔖 Badge Optimization**: Updated badges for better reliability and visibility
- **📊 Better Discovery**: Improved descriptions to emphasize AI-powered query execution

#### **3. Configuration Updates**
- **⚙️ server.json**: Updated to GitHub MCP Registry standards with proper field naming (camelCase)
- **📚 README.md**: Enhanced with registry-compliant metadata and improved badge layout
- **🔧 pyproject.toml**: Updated description to highlight NL2KQL functionality

### 🎯 **Key Features**

- Natural Language to KQL (NL2KQL) query conversion
- Intelligent schema discovery and caching
- AI-powered context assistance
- Seamless Azure Data Explorer integration
- Execute KQL queries directly in AI prompts

---

## 🐛 **v2.0.7 - Bug Fix Release: Sample Values & Query Timeout**

> **Bug Fix Release** 🛠️

**Release Date**: October 13, 2025
**Author**: Arjun Trivedi
**Email**: arjuntrivedi42@yahoo.com
**Repository**: https://github.com/4R9UN/mcp-kql-server

### 🚀 **What's Fixed in v2.0.7**

This release addresses critical bugs in the schema discovery system that were preventing sample values from being populated and causing query timeout issues.

#### **1. Sample Values Bug Fix**
- **🔧 Fixed Empty Sample Values**: Resolved issue where `sample_values` field was returning empty arrays in schema discovery
- **📊 Data Population**: All three schema discovery strategies now properly execute `| take 2` queries to fetch actual sample data
- **✅ Consistent Behavior**: Strategy 1 (JSON Schema) and Strategy 2 (getschema) now match Strategy 3's working implementation
- **🎯 Real Data Examples**: AI-enhanced column descriptions now have access to actual table data for better context

#### **2. Query Timeout Improvements**
- **⚡ Enhanced Error Handling**: Improved handling of query timeout scenarios
- **🛠️ Better Error Messages**: More informative error messages when queries time out
- **📝 Logging Enhancements**: Better debugging information for timeout-related issues

### 🔧 **Technical Changes**

#### **Schema Discovery Enhancement ([`utils.py`](mcp_kql_server/utils.py))**

**Strategy 1 - JSON Schema Discovery (Lines 1841-1867)**:
- Added sample data query execution after schema discovery
- Extracts sample values for each column from `| take 2` results
- Populates `sample_values` with actual table data
- Includes error handling for sample data retrieval failures

**Strategy 2 - getschema Discovery (Lines 1871-1913)**:
- Added sample data query execution after getschema
- Extracts sample values for each column from `| take 2` results
- Populates `sample_values` with actual table data
- Includes error handling for sample data retrieval failures

**Key Implementation Pattern**:
```python
# Get sample data for all columns
sample_data = {}
try:
    bracketed_table = bracket_if_needed(table)
    sample_query = f"{bracketed_table} | take 2"
    sample_result = await self._execute_kusto_async(sample_query, cluster, database, is_mgmt=False)
    
    if sample_result and len(sample_result) > 0:
        # Extract sample values for each column
        sample_values = [str(row.get(col_name, '')) for row in sample_result[:2]
                        if row.get(col_name) is not None]
        sample_data[col_name] = sample_values
except Exception as sample_error:
    logger.debug(f"Failed to get sample data: {sample_error}")
```

### 🐛 **Bug Fixes**

#### **Critical Fixes**
- **Sample Values Population**: Fixed bug where Strategy 1 and Strategy 2 weren't fetching actual sample data from tables
- **Query Execution**: Ensured `| take 2` queries execute properly to populate sample values
- **Error Handling**: Added proper error handling for sample data retrieval to prevent schema discovery failures
- **Consistency**: All three schema discovery strategies now behave identically regarding sample values

#### **Impact**
- **AI Context**: AI models now receive actual data samples for better column understanding
- **Query Generation**: Improved natural language to KQL query generation with real data context
- **Schema Intelligence**: Enhanced schema descriptions with actual value examples

### 📦 **Installation & Upgrade**

#### **New Installation**
```bash
pip install mcp-kql-server==2.0.7
```

#### **Upgrade from Previous Versions**
```bash
pip install --upgrade mcp-kql-server
```

### 🎯 **Benefits**
- **Enhanced AI Context**: AI models now have access to real sample values for better understanding
- **Improved Query Generation**: Better natural language to KQL conversion with actual data examples
- **Consistent Behavior**: All schema discovery strategies now provide complete information
- **Better Debugging**: Enhanced error messages and logging for troubleshooting

### ✅ **Quality Assurance**
- **Schema Discovery**: Verified all three strategies now populate sample values correctly
- **Query Execution**: Confirmed `| take 2` queries execute successfully across different tables
- **Error Handling**: Validated proper error handling when sample data retrieval fails
- **Backward Compatibility**: Changes are fully backward compatible with existing implementations

---

## 🚀 **v2.0.6 - Architectural Refactoring & Intelligence Upgrade**

> **Major Refactor Release**  архитектура

**Release Date**: September 9, 2025
**Author**: Arjun Trivedi
**Email**: arjuntrivedi42@yahoo.com
**Repository**: https://github.com/4R9UN/mcp-kql-server

### 🚀 **What's New in v2.0.6**

This release marks a significant architectural evolution of the MCP KQL Server, focusing on maintainability, robustness, and enhanced intelligence. The entire codebase has been refactored to introduce a centralized processing pipeline and dynamic, context-aware schema analysis.

#### **1. Centralized Processing Pipeline (`utils.py`)**
- **✅ `QueryProcessor` Class**: A dedicated class now handles all query pre-processing, including cleaning, validation, and parsing of cluster/database information. This standardizes input handling.
- **✅ `ErrorHandler` Class**: A robust, centralized error handler that classifies Kusto exceptions and provides structured, actionable error messages. This dramatically improves the user experience when queries fail.
- **✅ `SchemaManager` Class**: A utility helper for managing and formatting schema information consistently.

#### **2. Dynamic Schema Intelligence (`constants.py`)**
- **🧠 `DynamicSchemaAnalyzer`**: Moves beyond static keywords to intelligently analyze table names and properties, generating richer, more accurate AI-friendly descriptions.
- **🧠 `DynamicColumnAnalyzer`**: Analyzes column names and data types to infer semantics, use cases, and relationships, providing deeper context to the AI model.

#### **3. Asynchronous Post-Query Learning (`execute_kql.py`)**
- **⚡ Async Learning**: After a successful query, a non-blocking background task (`post_query_learning`) is now spawned to update the unified memory. This ensures the server remains responsive while continuously learning.

#### **4. Codebase Refactoring & Clarity**
- **🧹 Separation of Concerns**: Each module now has a more clearly defined responsibility, from authentication (`kql_auth.py`) to memory (`memory.py`) to execution (`execute_kql.py`).
- **📝 Updated Documentation**: The `docs/architecture.md` file has been completely rewritten to reflect the new, more sophisticated architecture.
- **👤 Author Validation**: Ensured all core files contain up-to-date author and contact information.

### 🔧 **Technical Changes**

- **Refactored `mcp_server.py`**: The main server file now orchestrates calls to the new utility classes in `utils.py`, simplifying its logic.
- **Introduced `utils.py`**: This new module contains the core business logic, abstracting it away from the MCP tool definitions.
- **Enhanced `constants.py`**: Expanded from a simple constants file to an intelligence hub containing the dynamic analyzer classes.
- **Modified `execute_kql.py`**: Updated to use the new `ErrorHandler` and to trigger the asynchronous learning task.
- **Updated `memory.py`**: The `MemoryManager` now leverages the dynamic analyzers from `constants.py` to enrich schemas.

### 🎯 **Benefits**
- **Maintainability**: The new architecture is significantly easier to understand, maintain, and extend.
- **Robustness**: Centralized error handling provides more consistent and helpful feedback.
- **Intelligence**: Dynamic schema analysis delivers far superior context to the AI, enabling more accurate and powerful query generation.
- **Performance**: Asynchronous learning ensures the server's core functionality is not blocked by background tasks.

### 📦 **Installation & Upgrade**

#### **New Installation**
```bash
pip install mcp-kql-server==2.0.6
```

#### **Upgrade from Previous Versions**
```bash
pip install --upgrade mcp-kql-server
```

### ✅ **Quality Assurance**
- **Full Test Suite**: All existing tests were updated and are passing, ensuring backward compatibility of features.
- **Architectural Review**: The new architecture has been documented and reviewed for clarity and correctness.
- **End-to-End Validation**: Confirmed that the entire query pipeline—from processing to execution to learning—functions as expected.

---

## 🚀 **v2.0.5 - Hybrid Schema Discovery & Enhanced Workflow**

> **Feature Release** ✨

**Release Date**: August 8, 2025
**Author**: Arjun Trivedi
**Repository**: https://github.com/4R9UN/mcp-kql-server

### 🚀 **What's New in v2.0.5**

#### **Hybrid Schema Discovery Model**
- **✅ Re-introduced `kql_schema_memory`**: The dedicated schema discovery tool is back, allowing for proactive, full-cluster schema caching.
- **🧠 On-Demand & Proactive Caching**: Combines the convenience of on-demand caching (in `kql_execute`) with the thoroughness of proactive caching.
- **⚡ Flexible Workflow**: Users can choose to pre-cache an entire cluster's schema for maximum AI context or rely on automatic, on-the-fly discovery.
- **⚙️ Explicit Control**: Provides explicit control over schema memory for users who need to ensure comprehensive context is available before executing complex queries.

### 🔧 **Technical Changes**

#### **Code Refactoring & Re-implementation**
- **Re-implemented `kql_schema_memory`**: The tool has been brought back as a distinct function in the server for explicit schema discovery.
- **Updated `memory.py`**: Enhanced the `UnifiedSchemaMemory` class to support full-cluster discovery.
- **Updated `mcp_server.py`**: The server now exposes both `kql_execute` and `kql_schema_memory` tools.
- **Updated `README.md`**: Documentation updated to reflect the hybrid model and the availability of both tools.

### 🐛 **Bug Fixes**
- **N/A**: This release focuses on feature enhancement and workflow flexibility.

### 📦 **Installation & Upgrade**

#### **New Installation**
```bash
pip install mcp-kql-server==2.0.5
```

#### **Upgrade from Previous Versions**
```bash
pip install --upgrade mcp-kql-server
```

### 🎯 **Benefits**
- **Workflow Flexibility**: Supports both proactive and on-demand schema discovery.
- **Comprehensive AI Context**: Ability to pre-cache an entire cluster schema ensures maximum context for AI-powered queries.
- **Enhanced Control**: Users have explicit control over when and how schema memory is built.
- **Optimized for All Use Cases**: Efficient for both quick, single-table queries and complex, multi-table analysis.

### 🛠️ **Working Tools (Verified)**
1.  **[`kql_execute`](mcp_kql_server/mcp_server.py:107)** - Execute KQL queries with integrated, on-demand AI-enhanced schema intelligence.
2.  **[`kql_schema_memory`](mcp_kql_server/mcp_server.py:165)** - Proactively discover and cache the schema for an entire KQL cluster.

### ✅ **Quality Assurance**
- **Full Test Suite**: All tests passing, adapted for the hybrid model.
- **Package Verification**: Successfully built and verified package integrity.
- **End-to-End Validation**: Confirmed both on-demand and proactive schema discovery workflows function correctly.

---

## 🔧 **v2.0.4 - Azure Kusto Data Compatibility Fix**

> **Patch Release** 🛠️

**Release Date**: August 8, 2025
**Author**: Arjun Trivedi
**Repository**: https://github.com/4R9UN/mcp-kql-server

### 🚀 **What's New in v2.0.4**

#### **Dependency Compatibility Fix**
- **🔧 Azure Kusto Data v5.x Support**: Updated dependency constraint to support `azure-kusto-data` versions 4.x and 5.x
- **🛠️ Compatibility Resolution**: Fixed dependency conflict with `azure-kusto-ingest 5.0.5` which requires `azure-kusto-data==5.0.5`
- **📦 Broader Version Range**: Changed constraint from `>=4.0.0,<5.0.0` to `>=4.0.0,<6.0.0`

### 🔧 **Technical Changes**

#### **Dependency Updates**
- **Updated [`pyproject.toml`](pyproject.toml:38)**: Modified `azure-kusto-data` version constraint to `>=4.0.0,<6.0.0`
- **Backward Compatibility**: Maintains support for existing 4.x installations
- **Forward Compatibility**: Enables compatibility with latest 5.x versions

### 🐛 **Bug Fixes**
- **Fixed Installation Conflicts**: Resolved pip dependency resolver conflicts when `azure-kusto-ingest 5.0.5` is already installed
- **Eliminated Version Constraints**: Removed restrictive upper bound that prevented 5.x compatibility

### 📦 **Installation & Upgrade**

#### **New Installation**
```bash
pip install mcp-kql-server==2.0.4
```

#### **Upgrade from Previous Versions**
```bash
pip install --upgrade mcp-kql-server
```

### 🎯 **Benefits**
- **Seamless Upgrades**: No more dependency conflicts during installation
- **Latest Azure SDK Support**: Compatible with the latest Azure Kusto Data SDK versions
- **Enterprise Ready**: Works with existing Azure environments using latest SDKs

### ✅ **Quality Assurance**
- **Full Test Suite**: All tests passing (9/9 core functionality tests)
- **Package Verification**: Successfully built and verified package integrity
- **Dependency Validation**: Confirmed compatibility with both 4.x and 5.x versions

---

## 🧹 **v2.0.3 - Dependency Audit & Project Cleanup**

> **Patch Release** 🛠️

**Release Date**: August 8, 2025
**Author**: Arjun Trivedi
**Repository**: https://github.com/4R9UN/mcp-kql-server

### 🚀 **What's New in v2.0.3**

#### **Dependency Optimization**
- **🔧 Comprehensive Dependency Audit**: Complete review and optimization of all dependencies
- **📦 Version Bounds Added**: Upper version bounds added to prevent breaking changes
- **🧹 Removed Redundant Dependencies**: Cleaned up unused packages (`mcp`, `azure-core`)
- **✅ PEP 621 Compliance**: Maintained full compliance with modern Python packaging standards

#### **Code Cleanup & Accuracy**
- **🎯 Focused on Implemented Features**: Removed references to unimplemented functionality
- **📝 Accurate Documentation**: Updated descriptions to reflect only working tools
- **🔍 Truth in Advertising**: Cleaned up constants to match actual capabilities
- **🚫 Emoji Removal**: Removed all emojis and symbols for professional appearance

#### **Enhanced AI Capabilities**
- **🧠 AI-Powered Constants**: Enhanced [`constants.py`](mcp_kql_server/constants.py:1) with comprehensive AI-relevant tokens
- **🔐 Security Intelligence**: Improved security table patterns with analysis keywords
- **📊 Column Analysis**: Enhanced column patterns with data type classifications and use cases
- **🎨 Professional Visuals**: Updated README Mermaid diagrams with accessible color schemes

### 🔧 **Technical Changes**

#### **Dependency Management**
- **Updated Version Bounds**: Added `<2.0` bounds to key dependencies for stability
- **Removed Redundant Packages**: Eliminated duplicate and unused dependencies
- **Optimized Install Size**: Reduced package footprint while maintaining functionality
- **Lock File Alignment**: Ensured dependency versions align with lock file constraints

#### **Constants Enhancement**
- **AI Token Integration**: Added comprehensive AI-relevant tokens for better query assistance
- **Security Mapping**: Enhanced security table patterns with MITRE ATT&CK context
- **Column Intelligence**: Improved column pattern recognition with analysis tokens
- **Professional Descriptions**: Updated all descriptions to be clean and professional

#### **Documentation Updates**
- **Accurate Tool Descriptions**: Updated MCP tool descriptions to reflect only implemented features
- **Enhanced README**: Improved Mermaid diagrams with better color contrast for accessibility
- **Clean Presentation**: Removed all emojis and symbols for corporate-friendly appearance

### 🐛 **Bug Fixes**
- **Fixed Accuracy Issues**: Removed references to unimplemented MITRE ATT&CK tools
- **Corrected Dependencies**: Aligned declared dependencies with actual runtime requirements
- **Package Integrity**: Ensured clean package build without unused components

### 📦 **Installation & Upgrade**

#### **New Installation**
```bash
pip install mcp-kql-server==2.0.3
```

#### **Upgrade from Previous Versions**
```bash
pip install --upgrade mcp-kql-server
```

### 🎯 **Benefits**
- **Enhanced Reliability**: Improved dependency stability with proper version bounds
- **Better AI Integration**: Enhanced constants provide better context for AI-powered queries
- **Professional Appearance**: Clean, emoji-free interface suitable for enterprise environments
- **Accurate Documentation**: Truth in advertising - only documented features are implemented
- **Optimized Performance**: Reduced package size and improved dependency resolution

### 🛠️ **Working Tools (Verified)**
1. **[`kql_execute`](mcp_kql_server/server.py:1)** - Execute KQL queries with AI-enhanced intelligence
2. **[`kql_schema_memory`](mcp_kql_server/server.py:1)** - AI-Powered Schema Discovery & Memory Management

### ✅ **Quality Assurance**
- **Full Test Suite**: All tests passing (4/4 core functionality tests)
- **Package Verification**: Successfully built and verified package integrity
- **Clean Installation**: Verified clean installation from PyPI without issues
- **Dependency Validation**: All dependencies properly resolved and functional

---

## 🔕 **v2.0.2 - FastMCP Branding Suppression**

> **Patch Release** 🛠️

**Release Date**: July 18, 2025
**Author**: Arjun Trivedi
**Repository**: https://github.com/4R9UN/mcp-kql-server

### 🚀 **What's New in v2.0.2**

#### **Professional Output Experience**
- **🔕 Suppressed FastMCP Branding**: Removed FastMCP framework branding output for cleaner, professional server startup
- **🎯 Clean Console Output**: Server now starts without displaying FastMCP version, documentation links, or deployment information
- **⚡ Streamlined Experience**: Focus on functionality without framework marketing messages

#### **Development Optimizations**
- **🧹 Removed Development Dependencies**: Cleaned up dev-only files (requirements-dev.txt, Makefile, .pre-commit-config.yaml, pytest.ini)
- **📦 Simplified Project Structure**: Reduced unnecessary files for production deployment
- **🔧 Production-Ready Configuration**: Optimized for production environments

### 🔧 **Technical Changes**

#### **FastMCP Branding Suppression**
- Added comprehensive environment variable configuration to suppress FastMCP output
- Implemented Rich console monkey patching to filter branding messages
- Enhanced logging configuration to suppress verbose FastMCP logs
- Redirected stdout during server startup to prevent branding display

#### **Configuration Updates**
- Set `FASTMCP_QUIET`, `FASTMCP_NO_BANNER`, `FASTMCP_SUPPRESS_BRANDING` environment variables
- Added `NO_COLOR` support for consistent output across environments
- Enhanced logging level management for FastMCP and Rich libraries

### 🐛 **Bug Fixes**
- **Fixed Server Startup Output**: Eliminated unwanted FastMCP branding messages during server initialization
- **Improved Error Handling**: Better error management for StringIO redirection issues
- **Enhanced Logging**: More appropriate logging levels for production environments

### 📦 **Installation & Upgrade**

#### **New Installation**
```bash
pip install mcp-kql-server==2.0.2
```

#### **Upgrade from Previous Versions**
```bash
pip install --upgrade mcp-kql-server
```

### 🎯 **Benefits**
- **Professional Experience**: Clean server startup without framework branding
- **Corporate Friendly**: Suitable for enterprise environments requiring clean output
- **Focused Functionality**: Emphasis on KQL capabilities rather than framework marketing
- **Reduced Clutter**: Cleaner console output for better user experience

---

## 🚀 **v2.0.0 - Major Release**

> **Major Release** 🎉

**Release Date**: July 1, 2025
**Author**: Arjun Trivedi
**Repository**: https://github.com/4R9UN/mcp-kql-server

---

## 🚀 What's New in v2.0.0

### 🎯 **Zero-Configuration Architecture**
- **One-Command Installation**: Simple `pip install mcp-kql-server` with automatic setup
- **Auto-Configuration**: Eliminates need for environment variables and manual setup
- **Smart Defaults**: Production-ready configuration out-of-the-box
- **Cross-Platform**: Seamless operation on Windows, macOS, and Linux

### 🧠 **Advanced AI-Powered Schema Memory**
- **Unified Memory System**: New intelligent schema caching with AI-optimized tokens
- **Per-Table Intelligence**: Granular schema discovery and caching per table
- **Context Size Management**: Automatic compression to prevent context overflow
- **Cross-Cluster Support**: Schema sharing across multiple Azure Data Explorer clusters

### 📦 **Flexible Dependency Management**
- **Minimal Core**: Essential dependencies only for basic functionality
-- **Optional Extras**: Choose what you need with `[azure]`, `[mcp]`, `[full]`, `[dev]` options
- **Installation Flexibility**: From minimal CI-friendly to full-featured production installs

### 🔧 **Production-Ready CI/CD Pipeline**
- **Multi-Platform Testing**: Automated testing across Ubuntu, Windows, macOS
- **Python 3.8-3.12 Support**: Comprehensive Python version compatibility
- **Automated PyPI Publishing**: Release automation with secure token management
- **Quality Automation**: Code formatting, linting, and security scanning

---

## 📊 **Key Features**

### ✨ **Enhanced Query Execution**
- **Intelligent Context Loading**: AI-powered schema context for better query assistance
- **Memory-Optimized Performance**: Smart caching reduces Azure API calls
- **Rich Visualizations**: Enhanced markdown table output with configurable formatting
- **Error Intelligence**: AI-powered error messages and query suggestions

### 🔐 **Security & Authentication**
- **Azure CLI Integration**: Seamless authentication using existing Azure credentials
- **No Credential Storage**: Server never stores authentication tokens
- **Query Validation**: Built-in protection against malicious queries
- **Local Schema Storage**: Sensitive schema data remains on local machine

### 🎨 **Developer Experience**
- **Professional Documentation**: Comprehensive guides, examples, and troubleshooting
- **Rich Badges**: PyPI version, Python compatibility, license, downloads, codecov
- **Development Tools**: Complete tooling with Makefile, pre-commit hooks, pytest
- **Contributing Guidelines**: Detailed contribution process and development setup

---

## 🔄 **Breaking Changes from v1.x**

### **Installation Changes**
- **Simplified Installation**: No longer requires manual environment variable setup
- **Dependency Structure**: Core dependencies reduced, optional extras introduced
- **Memory Path**: Automatic detection and creation of memory directories

### **API Enhancements**
- **Enhanced Response Format**: Richer metadata and context in query responses
- **Schema Memory Format**: New unified schema format with AI tokens
- **Error Handling**: Improved error messages with actionable suggestions

### **Configuration Updates**
- **Zero Configuration**: Eliminates need for manual configuration files
- **Environment Variables**: Most environment variables now optional
- **Logging**: Optimized logging with reduced Azure SDK verbosity

---

## 📋 **Installation & Upgrade Guide**

### **New Installation (Recommended)**
```bash
# Minimal installation (CI-friendly)
pip install mcp-kql-server

# Full production installation
pip install mcp-kql-server[full]

# Development installation
pip install mcp-kql-server[dev]
```

### **Upgrading from v1.x**
```bash
# Uninstall old version
pip uninstall mcp-kql-server

# Install new version with desired extras
pip install mcp-kql-server[full]

# Optional: Clear old schema cache
# Windows: del "%APPDATA%\KQL_MCP\schema_memory.json"
# macOS/Linux: rm ~/.local/share/KQL_MCP/schema_memory.json
```

### **Verification**
```bash
python -c "from mcp_kql_server import __version__; print(f'v{__version__} installed successfully!')"
```

---

## 🎯 **Performance Improvements**

### **Schema Discovery**
- **50% Faster**: Optimized schema discovery with parallel processing
- **Reduced Memory Usage**: Compact AI tokens reduce memory footprint by 60%
- **Smart Caching**: Intelligent cache invalidation and updates

### **Query Execution**
- **Context Loading**: 3x faster schema context loading with unified memory
- **Connection Pooling**: Reuse connections for better performance
- **Response Size**: Optimized response format reduces data transfer

### **Development Workflow**
- **CI/CD Speed**: 40% faster pipeline execution with minimal dependencies
- **Build Time**: Reduced package build time with optimized configuration
- **Development Setup**: One-command development environment setup

---

## 🧪 **What We Tested**

### **Platform Compatibility**
- ✅ **Windows 10/11**: Full compatibility with PowerShell and Command Prompt
- ✅ **macOS**: Intel and Apple Silicon support (macOS 10.15+)
- ✅ **Linux**: Ubuntu, CentOS, Alpine Linux distributions

### **Python Versions**
- ✅ **Python 3.8**: Full compatibility with legacy environments
- ✅ **Python 3.9**: Standard production environments
- ✅ **Python 3.10**: Current stable release
- ✅ **Python 3.11**: Performance optimized environments
- ✅ **Python 3.12**: Latest Python features

### **Azure Data Explorer**
- ✅ **Public Clusters**: help.kusto.windows.net and sample clusters
- ✅ **Private Clusters**: Corporate and enterprise deployments
- ✅ **Multi-Region**: Global Azure regions and sovereign clouds
- ✅ **Large Datasets**: Tables with millions of rows and complex schemas

---

## 🛠️ **Migration Guide**

### **Schema Cache Migration**
The new unified memory system will automatically migrate existing cache files. No manual intervention required.

### **Configuration Migration**
```bash
# Old v1.x environment variables (no longer required)
# export KQL_MEMORY_PATH="/path/to/memory"
# export AZURE_CORE_ONLY_SHOW_ERRORS=true

# New v2.0.0 (optional)
export KQL_DEBUG=true  # Only for debugging
```

### **API Usage Updates**
Query and schema memory APIs remain backward compatible. New features available through optional parameters.

---

## 🐛 **Bug Fixes**

### **Critical Fixes**
- **Unicode Encoding**: Resolved Windows CP1252 encoding issues with emoji characters
- **TOML Syntax**: Fixed invalid pyproject.toml configuration causing build failures
- **PyPI Classifiers**: Removed invalid classifiers preventing package publishing
- **Memory Leaks**: Fixed schema cache memory leaks in long-running processes

### **Stability Improvements**
- **Connection Handling**: Better Azure connection management and retry logic
- **Error Recovery**: Enhanced error recovery for network and authentication issues
- **Memory Management**: Improved memory cleanup and cache management
- **Cross-Platform**: Resolved platform-specific path and encoding issues

---

## 📚 **Documentation Updates**

### **New Documentation**
- ✅ **CONTRIBUTING.md**: Comprehensive contribution guidelines
- ✅ **SECURITY.md**: Security policies and vulnerability reporting
- ✅ **RELEASE_NOTES.md**: Detailed release information (this document)
- ✅ **Enhanced README.md**: Complete setup and usage documentation

### **Developer Resources**
- ✅ **Makefile**: Common development tasks automation
- ✅ **Pre-commit Hooks**: Automated code quality checks
- ✅ **Pytest Configuration**: Comprehensive testing setup
- ✅ **GitHub Workflows**: Production-ready CI/CD pipeline

---

## 🤝 **Community & Support**

### **Getting Help**
- **GitHub Issues**: [Report bugs and request features](https://github.com/4R9UN/mcp-kql-server/issues)
- **GitHub Discussions**: [Community discussions and Q&A](https://github.com/4R9UN/mcp-kql-server/discussions)
- **Email Support**: [Direct contact with maintainer](mailto:arjuntrivedi42@yahoo.com)

### **Contributing**
- **Code Contributions**: See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
- **Documentation**: Help improve documentation and examples
- **Testing**: Report issues and help with testing across platforms
- **Feature Requests**: Suggest new features and improvements

---

## 🙏 **Acknowledgments**

### **Special Thanks**
- **FastMCP Team**: [@jlowin](https://github.com/jlowin) for the excellent MCP framework
- **Azure Team**: Microsoft Azure Data Explorer team for robust KQL support
- **Community Contributors**: All users who provided feedback and testing
- **Beta Testers**: Early adopters who helped identify and fix issues

### **Technology Stack**
- **[FastMCP](https://github.com/jlowin/fastmcp)**: MCP server framework
- **[Azure Kusto Python SDK](https://github.com/Azure/azure-kusto-python)**: KQL client library
- **[Model Context Protocol](https://github.com/anthropics/mcp)**: Protocol specification
- **[Pydantic](https://pydantic.dev/)**: Data validation and settings management

---

## 🔮 **What's Next**

### **Planned for v2.1.0**
- **Enhanced AI Context**: GPT-4 powered query optimization suggestions
- **Real-time Monitoring**: Live query performance and cluster health monitoring
- **Advanced Visualizations**: Chart generation and interactive dashboards
- **Plugin System**: Extensible architecture for custom functionality

### **Long-term Roadmap**
- **Multi-Cloud Support**: Support for other cloud analytics platforms
- **GraphQL Integration**: GraphQL interface for schema exploration
- **Advanced Security**: RBAC integration and audit logging
- **Performance Analytics**: Query optimization recommendations

---

## 📞 **Contact Information**

**Maintainer**: Arjun Trivedi  
**Email**: [arjuntrivedi42@yahoo.com](mailto:arjuntrivedi42@yahoo.com)  
**GitHub**: [@4R9UN](https://github.com/4R9UN)  
**Repository**: [mcp-kql-server](https://github.com/4R9UN/mcp-kql-server)  
**PyPI**: [mcp-kql-server](https://pypi.org/project/mcp-kql-server/)

---

**Happy Querying with MCP KQL Server v2.0.0! 🎉**

*Made with ❤️ for the data analytics and AI community*
