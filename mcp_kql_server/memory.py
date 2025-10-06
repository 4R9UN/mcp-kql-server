"""
Unified Schema Memory System for MCP KQL Server

This module provides a unified schema memory system that:
- Uses AI-friendly special tokens (e.g., @@CLUSTER@@, ##DATABASE##)
- Supports a two-step flow: execute KQL first, then discover/update schema context
- Prevents context size bloat via intelligent compression
- Provides schema memory used for AI-driven query generation

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import json
import logging
import os
import re
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from dataclasses import dataclass

# FastMCP imports removed - using programmatic description generation instead

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of schema validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    validated_query: str
    tables_used: Set[str]
    columns_used: Dict[str, Set[str]]  # table -> columns

# Global lock for thread-safe memory operations
_memory_lock = threading.RLock()

# FastMCP initialization removed - using programmatic description generation

# Enhanced AI-Friendly Special Tokens with XML-style structure
SPECIAL_TOKENS = {
    "CLUSTER_START": "<CLUSTER>",
    "CLUSTER_END": "</CLUSTER>",
    "DATABASE_START": "<DB>",
    "DATABASE_END": "</DB>",
    "TABLE_START": "<TABLE>",
    "TABLE_END": "</TABLE>",
    "COLUMN_START": "<COL>",
    "COLUMN_END": "</COL>",
    "TYPE_START": "<TYPE>",
    "TYPE_END": "</TYPE>",
    "DESCRIPTION_START": "<DESC>",
    "DESCRIPTION_END": "</DESC>",
    "SUMMARY_START": "<SUMMARY>",
    "SUMMARY_END": "</SUMMARY>",
    "TAGS_START": "<TAGS>",
    "TAGS_END": "</TAGS>",
    "SAMPLES_START": "<SAMPLES>",
    "SAMPLES_END": "</SAMPLES>",
    "QUERY_START": "<QUERY>",
    "QUERY_END": "</QUERY>",
}


class ContextSelector:
    """
    Intelligent context selection for query generation.
    Implements the enhanced schema context management recommended in the analysis.
    """
    
    def select_relevant_context(self, query: str, all_schemas: Dict) -> List[str]:
        """Select only relevant schema context using intelligent scoring."""
        
        # Parse query intent
        intent = self._parse_query_intent(query)
        
        # Score each schema for relevance
        scored_schemas = []
        for table, schema in all_schemas.items():
            score = self._calculate_relevance_score(intent, schema, table)
            scored_schemas.append((score, table, schema))
        
        # Select top relevant schemas within token limit
        selected = []
        token_count = 0
        max_tokens = 4000
        
        for score, table, schema in sorted(scored_schemas, reverse=True):
            schema_token = self._create_compact_token(schema, table)
            if token_count + len(schema_token) <= max_tokens:
                selected.append(schema_token)
                token_count += len(schema_token)
            else:
                break
        
        return selected
    
    def _parse_query_intent(self, query: str) -> Dict[str, Any]:
        """Parse query to understand user intent."""
        intent = {
            "operation_types": [],
            "mentioned_tables": [],
            "mentioned_columns": [],
            "temporal_focus": False,
            "aggregation_focus": False,
            "filtering_focus": False
        }
        
        if not query:
            return intent
        
        query_lower = query.lower()
        
        # Detect operation types
        if any(op in query_lower for op in ['summarize', 'count', 'sum', 'avg']):
            intent["aggregation_focus"] = True
            intent["operation_types"].append("aggregation")
        
        if any(op in query_lower for op in ['where', 'filter']):
            intent["filtering_focus"] = True
            intent["operation_types"].append("filtering")
        
        if any(op in query_lower for op in ['ago(', 'between', 'timespan']):
            intent["temporal_focus"] = True
            intent["operation_types"].append("temporal")
        
        # Extract mentioned entities (simplified)
        words = query.split()
        for word in words:
            if word.isalpha() and len(word) > 3:
                intent["mentioned_columns"].append(word)
        
        return intent
    
    def _calculate_relevance_score(self, intent: Dict[str, Any], schema: Dict[str, Any], table: str) -> float:
        """Calculate relevance score for a schema based on query intent."""
        score = 0.0
        
        # Base score for having columns
        columns = schema.get("columns", {})
        if columns:
            score += 1.0
        
        # Score based on column types matching intent
        if intent["temporal_focus"]:
            temporal_columns = [
                col for col, info in columns.items()
                if "datetime" in info.get("data_type", "").lower()
            ]
            score += len(temporal_columns) * 0.5
        
        if intent["aggregation_focus"]:
            numeric_columns = [
                col for col, info in columns.items()
                if any(t in info.get("data_type", "").lower() for t in ['int', 'real', 'decimal'])
            ]
            score += len(numeric_columns) * 0.3
        
        # Score based on table name relevance
        table_lower = table.lower()
        for mentioned in intent["mentioned_columns"]:
            if mentioned.lower() in table_lower:
                score += 2.0
                break
        
        # Score based on recent usage (if schema has successful queries)
        if schema.get("successful_queries"):
            score += 1.0
        
        return score
    
    def _create_compact_token(self, schema: Dict[str, Any], table: str) -> str:
        """Create compact schema token for context."""
        # Use existing ai_token if available, otherwise create minimal one
        if schema.get("ai_token"):
            return schema["ai_token"]
        
        # Create minimal token
        columns = schema.get("columns", {})
        column_count = len(columns)
        
        # Create compact representation
        compact_columns = []
        for col_name, col_info in list(columns.items())[:5]:  # Max 5 columns
            col_type = col_info.get("data_type", "unknown")
            compact_columns.append(f"{col_name}({col_type})")
        
        token = (
            f"{SPECIAL_TOKENS['TABLE_START']}{table}{SPECIAL_TOKENS['TABLE_END']}"
            f"{SPECIAL_TOKENS['SUMMARY_START']}{column_count}_cols{SPECIAL_TOKENS['SUMMARY_END']}"
            f"[{','.join(compact_columns)}]"
        )
        
        return token


class MemoryManager:
    """
    Simplified Memory Manager implementing 2-step flow:
    1. Execute KQL query first and show data
    2. Background schema discovery and AI context preparation
    """

    def __init__(self, custom_memory_path: Optional[str] = None):
        """Initialize memory manager with AI-friendly token system and thread safety."""
        self.memory_path = self._get_memory_path(custom_memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.corpus = self._load_or_create_corpus()
        self._save_scheduled = False
        self._memory_size_limit = 500 * 1024  # 500KB limit per cluster
        self._compression_enabled = True

    def _get_memory_path(self, custom_path: Optional[str] = None) -> Path:
        """Get the path for unified schema memory."""
        if custom_path:
            base_dir = Path(custom_path)
        elif os.name == "nt":  # Windows
            appdata = os.environ.get("APPDATA", "")
            if not appdata:
                # Fallback to user profile if APPDATA not available
                base_dir = Path.home() / "AppData" / "Roaming" / "KQL_MCP"
            else:
                base_dir = Path(appdata) / "KQL_MCP"
        else:  # macOS/Linux
            base_dir = Path.home() / ".local" / "share" / "KQL_MCP"
        
        # Log the memory path for debugging
        memory_path = base_dir / "unified_memory.json"
        logger.info(f"Schema memory path: {memory_path}")
        return memory_path

    def _load_or_create_corpus(self) -> Dict[str, Any]:
        """Load existing corpus or create a new one if loading fails or file doesn't exist."""
        if self.memory_path.exists():
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    corpus = json.load(f)
                    logger.info(f"Loaded memory from {self.memory_path}")
                    return self._ensure_corpus_structure(corpus)
            except Exception as e:
                logger.error(f"Failed to load memory from {self.memory_path}: {e}. A new corpus will be created.")
        
        # This block runs if the file doesn't exist or if loading failed.
        logger.info("Creating new schema memory corpus.")
        corpus = self._create_empty_corpus()
        # Temporarily assign to self.corpus so save_corpus() can access it.
        # The final assignment happens in __init__ after this function returns.
        self.corpus = corpus
        try:
            self.save_corpus()
            logger.info(f"Successfully created and saved new corpus at {self.memory_path}")
        except Exception as e:
            # Log error but proceed, as the corpus is in memory.
            logger.error(f"Failed to perform initial save of new corpus: {e}")
        return corpus

    def _create_empty_corpus(self) -> Dict[str, Any]:
        """Create empty corpus structure with enhanced AI-friendly organization."""
        return {
            "version": "3.0",
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "clusters": {},  # cluster_uri -> meta + databases -> tables -> schema + successful_queries
        }

    def _ensure_corpus_structure(self, corpus: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure corpus has required structure and migrate if needed."""
        required_sections = {
            "version": "3.0",
            "clusters": {},
        }

        for section, default in required_sections.items():
            if section not in corpus:
                corpus[section] = default

        # Handle version migration
        current_version = corpus.get("version", "3.0")
        if current_version in ["3.0", "2.0", "1.0"]:
            corpus = self._migrate_to_v31(corpus)

        # Update version and timestamp
        corpus["version"] = "3.0"
        corpus["last_updated"] = datetime.now().isoformat()

        return corpus
    
    def _migrate_to_v31(self, corpus: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate corpus from older versions to v3.1 with meta sections."""
        try:
            migrated_clusters = {}
            
            for cluster_uri, cluster_data in corpus.get("clusters", {}).items():
                # Ensure cluster has meta section
                if isinstance(cluster_data, dict):
                    if "meta" not in cluster_data:
                        cluster_data["meta"] = {
                            "token": f"{SPECIAL_TOKENS['CLUSTER_START']}{self._extract_cluster_name(cluster_uri)}{SPECIAL_TOKENS['CLUSTER_END']}",
                            "description": f"Cluster {cluster_uri}",
                            "last_accessed": datetime.now().isoformat()
                        }
                    
                    # Migrate databases
                    databases = cluster_data.get("databases", {})
                    migrated_databases = {}
                    
                    for db_name, db_data in databases.items():
                        if isinstance(db_data, dict):
                            # Ensure database has meta section
                            if "meta" not in db_data:
                                tables_count = len(db_data.get("tables", {}))
                                db_data["meta"] = {
                                    "token": f"{SPECIAL_TOKENS['DATABASE_START']}{db_name}{SPECIAL_TOKENS['DATABASE_END']}",
                                    "description": f"Database {db_name}",
                                    "table_count": tables_count
                                }
                            
                            # Migrate tables to new structure
                            tables = db_data.get("tables", {})
                            migrated_tables = {}
                            
                            for table_name, table_data in tables.items():
                                if isinstance(table_data, dict):
                                    # Convert old schema format to new structure
                                    if "meta" not in table_data:
                                        table_data["meta"] = {
                                            "token": f"{SPECIAL_TOKENS['TABLE_START']}{table_name}{SPECIAL_TOKENS['TABLE_END']}",
                                            "summary": f"{SPECIAL_TOKENS['SUMMARY_START']}{self._generate_table_summary(table_name, {})}{SPECIAL_TOKENS['SUMMARY_END']}",
                                            "discovered_at": table_data.get("discovered_at", datetime.now().isoformat()),
                                            "last_updated": table_data.get("last_updated", datetime.now().isoformat())
                                        }
                                    
                                    # Ensure successful_queries exists
                                    table_data.setdefault("successful_queries", [])
                                    
                                    migrated_tables[table_name] = table_data
                            
                            db_data["tables"] = migrated_tables
                            migrated_databases[db_name] = db_data
                    
                    cluster_data["databases"] = migrated_databases
                    migrated_clusters[cluster_uri] = cluster_data
            
            corpus["clusters"] = migrated_clusters
            logger.info("Successfully migrated corpus to v3.1 structure")
            
        except Exception as e:
            logger.warning(f"Migration to v3.1 failed: {e}, using current structure")
        
        return corpus

    @lru_cache(maxsize=50)
    def get_schema(self, cluster_uri: str, database: str, table: str, enable_fallback: bool = True) -> Dict[str, Any]:
        """
        Get schema for a specific table using the new structure with fallback strategies.
        
        Args:
            cluster_uri: The cluster URI
            database: Database name
            table: Table name
            enable_fallback: Whether to enable fallback strategies if primary schema fails
        
        Returns:
            Schema dictionary with fallback strategies applied if needed
        """
        try:
            normalized_cluster = self._normalize_cluster_uri(cluster_uri)

            # Navigate the new structure
            cluster_data = self.corpus.get("clusters", {}).get(normalized_cluster, {})
            db_data = cluster_data.get("databases", {}).get(database, {})
            table_data = db_data.get("tables", {}).get(table, {})

            # Extract schema from the new structure
            schema_data = table_data.get("schema", {})
            
            # Use unified schema.columns format only - no redundant column_types storage
            # This eliminates duplicate schema storage and ensures single source of truth

            # If we have a valid schema, return it
            if schema_data and "columns" in schema_data:
                return schema_data

            # Apply fallback strategies if enabled and no valid schema found
            if enable_fallback:
                return self._apply_schema_fallback_strategies(normalized_cluster, database, table)

            return schema_data

        except Exception as e:
            logger.warning(
                f"Failed to get schema for {cluster_uri}/{database}/{table}: {e}"
            )
            
            # Apply fallback strategies on exception
            if enable_fallback:
                try:
                    return self._apply_schema_fallback_strategies(cluster_uri, database, table)
                except Exception as fallback_error:
                    logger.error(f"All schema fallback strategies failed: {fallback_error}")
            
            return {}

    async def get_schema_async(self, cluster_uri: str, database: str, table: str, enable_fallback: bool = True) -> Dict[str, Any]:
        """
        Async wrapper around the sync `get_schema` to allow non-blocking calls from async code.
        Uses the existing lru_cache-backed get_schema for fast lookups.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_schema, cluster_uri, database, table, enable_fallback)
    
    def store_schema(
            self, cluster_uri: str, database: str, table: str, schema_data: Dict[str, Any], samples: Optional[Dict[str, Any]] = None
        ):
        """Store schema with the new AI-optimized structure and thread safety."""
        self._lock = _memory_lock
        self._lock.acquire()
        try:
                # DEBUG: Log incoming schema shape to help diagnose missing schema storage
                try:
                    schema_type = type(schema_data).__name__
                    schema_keys = list(schema_data.keys()) if isinstance(schema_data, dict) else None
                except Exception:
                    schema_type = str(type(schema_data))
                    schema_keys = None
                logger.debug(
                    "store_schema called for %s/%s/%s - schema_type=%s, schema_keys=%s, samples_present=%s",
                    cluster_uri,
                    database,
                    table,
                    schema_type,
                    schema_keys,
                    bool(samples),
                )
                normalized_cluster = self._normalize_cluster_uri(cluster_uri)
                
                # Apply dynamic compression before storing
                if self._compression_enabled:
                    schema_data = self._compress_schema_data(schema_data)
                
                # Minimal pre-filter: only skip obviously invalid data, avoid rejecting valid schemas
                try:
                    # Only skip if ALL columns are literally "Error" (case-insensitive) - very conservative check
                    cols_check: List[str] = []
                    if isinstance(schema_data, dict):
                        if isinstance(schema_data.get("columns"), list) and schema_data.get("columns"):
                            cols_check = [
                                c if isinstance(c, str) else (c.get("name") or c.get("column") or "")
                                for c in schema_data.get("columns", [])
                            ]
                        elif isinstance(schema_data.get("column_types"), dict) and schema_data.get("column_types"):
                            cols_check = list(schema_data.get("column_types").keys())
                    
                    # Only skip if EXACTLY one column named exactly "Error" with no other data
                    if (cols_check and len(cols_check) == 1 and
                        str(cols_check[0]).strip().lower() == "error" and
                        not any(schema_data.get(k) for k in ["table_name", "discovered_at", "cluster", "database"])):
                        logger.debug(f"Skipping schema store for {cluster_uri}/{database}/{table}: detected single 'Error' column with no metadata")
                        return

                    # Very conservative sample checking - only skip obvious error messages in samples
                    if isinstance(samples, dict):
                        error_sample_count = 0
                        total_samples = 0
                        for v in samples.values():
                            if v is not None:
                                total_samples += 1
                                try:
                                    if isinstance(v, str) and re.search(
                                        r"^(kusto service error|semantic error|failed to execute kql):",
                                        str(v).strip(),
                                        flags=re.IGNORECASE,
                                    ):
                                        error_sample_count += 1
                                except Exception:
                                    continue
                        # Only skip if ALL samples are error messages (and we have samples)
                        if total_samples > 0 and error_sample_count == total_samples:
                            logger.debug(f"Skipping schema store for {cluster_uri}/{database}/{table}: all samples are error messages")
                            return
                except Exception as _preerr:
                    logger.debug(f"Schema store pre-filter check failed: {_preerr}")
                
                # Ensure cluster structure
                if normalized_cluster not in self.corpus["clusters"]:
                    self.corpus["clusters"][normalized_cluster] = {
                        "meta": {
                            "token": f"{SPECIAL_TOKENS['CLUSTER_START']}{self._extract_cluster_name(normalized_cluster)}{SPECIAL_TOKENS['CLUSTER_END']}",
                            "description": f"Cluster {normalized_cluster}",
                            "last_accessed": datetime.now().isoformat()
                        },
                        "databases": {}
                    }
                
                cluster_data = self.corpus["clusters"][normalized_cluster]
                
                # Ensure database structure
                if database not in cluster_data["databases"]:
                    cluster_data["databases"][database] = {
                        "meta": {
                            "token": f"{SPECIAL_TOKENS['DATABASE_START']}{database}{SPECIAL_TOKENS['DATABASE_END']}",
                            "description": f"Database {database}",
                            "table_count": 0
                        },
                        "tables": {}
                    }
                
                db_data = cluster_data["databases"][database]
                
                # Process columns from schema_data
                columns = {}
                column_tokens = []
                
                # Handle both legacy columns list, dict-based columns mapping, and new column_types format
                incoming_columns = []
                cols_obj = schema_data.get("columns")
                # Case A: legacy list of columns (strings or dicts)
                if isinstance(cols_obj, list):
                    for col in cols_obj:
                        if isinstance(col, str):
                            incoming_columns.append({"name": col, "type": "unknown", "description": "", "tags": [], "sample_values": []})
                        elif isinstance(col, dict):
                            col_name = col.get("name") or col.get("column") or ""
                            if col_name:
                                incoming_columns.append({
                                    "name": col_name,
                                    "type": col.get("type") or col.get("datatype") or "unknown",
                                    "description": col.get("description") or col.get("desc") or "",
                                    "tags": col.get("tags") or [],
                                    "sample_values": col.get("sample_values") or col.get("examples") or []
                                })
                # Case B: dict mapping of column_name -> metadata (common new shape)
                elif isinstance(cols_obj, dict):
                    for col_name, info in cols_obj.items():
                        if isinstance(info, dict):
                            incoming_columns.append({
                                "name": col_name,
                                "type": info.get("data_type") or info.get("type") or info.get("ColumnType") or "unknown",
                                "description": info.get("description") or info.get("desc") or "",
                                "tags": info.get("tags") or info.get("column_tags") or [],
                                "sample_values": list(info.get("sample_values") or info.get("examples") or [])
                            })
                        else:
                            # simple value mapping - treat as unknown type with provided value ignored
                            incoming_columns.append({
                                "name": col_name,
                                "type": "unknown",
                                "description": "",
                                "tags": [],
                                "sample_values": []
                            })
                # Case C: older 'column_types' mapping
                elif isinstance(schema_data.get("column_types"), dict):
                    for col_name, info in schema_data.get("column_types", {}).items():
                        incoming_columns.append({
                            "name": col_name,
                            "type": info.get("data_type") or info.get("type") or "unknown",
                            "description": info.get("description", "") or "",
                            "tags": info.get("tags") or [],
                            "sample_values": list(info.get("sample_values") or [])
                        })
                
                # Process each column and create enhanced tokens
                for col_data in incoming_columns:
                    col_name = col_data["name"]
                    col_type = col_data["type"]
                    col_description = col_data["description"] or self._generate_ai_description(col_name, col_type, table)
                    col_tags = col_data["tags"]
                    col_samples = col_data["sample_values"][:3]  # Limit to 3 samples
                    
                    # Merge samples if provided
                    if samples and isinstance(samples, dict) and col_name in samples:
                        val = samples.get(col_name)
                        if val is not None:
                            sv = str(val)
                            if not re.search(r"Kusto service error:|Semantic error:|Failed to execute KQL", sv, flags=re.IGNORECASE):
                                if sv not in col_samples:
                                    col_samples.insert(0, sv)
                    
                                        # Generate column token (ensure all samples are strings)
                    sample_strs = [str(s) for s in col_samples[:2]]
                    col_token = (
                        f"{SPECIAL_TOKENS['COLUMN_START']}{col_name}"
                        f"{SPECIAL_TOKENS['TYPE_START']}{col_type}{SPECIAL_TOKENS['TYPE_END']}"
                        f"{SPECIAL_TOKENS['DESCRIPTION_START']}{col_description}{SPECIAL_TOKENS['DESCRIPTION_END']}"
                        f"{SPECIAL_TOKENS['TAGS_START']}{','.join(col_tags)}{SPECIAL_TOKENS['TAGS_END']}"
                        f"{SPECIAL_TOKENS['SAMPLES_START']}{','.join(sample_strs)}{SPECIAL_TOKENS['SAMPLES_END']}"
                        f"{SPECIAL_TOKENS['COLUMN_END']}"
                    )
                    
                    columns[col_name] = {
                        "token": col_token,
                        "data_type": col_type,
                        "description": col_description,
                        "tags": col_tags,
                        "sample_values": col_samples
                    }
                    column_tokens.append(col_token)
                
                # Create table AI token
                table_token = (
                    f"{SPECIAL_TOKENS['CLUSTER_START']}{self._extract_cluster_name(normalized_cluster)}{SPECIAL_TOKENS['CLUSTER_END']}"
                    f"{SPECIAL_TOKENS['DATABASE_START']}{database}{SPECIAL_TOKENS['DATABASE_END']}"
                    f"{SPECIAL_TOKENS['TABLE_START']}{table}{SPECIAL_TOKENS['TABLE_END']}"
                    f"{SPECIAL_TOKENS['SUMMARY_START']}{self._generate_table_summary(table, columns)}{SPECIAL_TOKENS['SUMMARY_END']}"
                    f"{''.join(column_tokens[:10])}"  # Limit to 10 columns
                )
                
                # Check if table already exists to preserve successful_queries
                existing_table = db_data["tables"].get(table, {})
                existing_queries = existing_table.get("successful_queries", [])
                
                # Overwrite schema to ensure it's always up-to-date, but preserve successful queries.
                db_data["tables"][table] = {
                    "meta": {
                        "token": f"{SPECIAL_TOKENS['TABLE_START']}{table}{SPECIAL_TOKENS['TABLE_END']}",
                        "summary": f"{SPECIAL_TOKENS['SUMMARY_START']}{self._generate_table_summary(table, columns)}{SPECIAL_TOKENS['SUMMARY_END']}",
                        "discovered_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat()
                    },
                    "schema": {
                        "columns": columns,
                        "ai_token": table_token
                    },
                    "successful_queries": existing_queries  # Preserve existing queries
                }
                
                # Update table count and table list for database-level queries (merge with existing)
                try:
                    existing_list = db_data["meta"].get("table_list", []) or []
                except Exception:
                    existing_list = []
                merged_list = list(dict.fromkeys(existing_list + list(db_data["tables"].keys())))
                db_data["meta"]["table_count"] = len(db_data["tables"])
                try:
                    db_data["meta"]["table_list"] = merged_list
                except Exception:
                    # Fallback if tables structure is unexpected
                    db_data["meta"].setdefault("table_list", merged_list)
                
                # Check memory limits and apply compression if needed
                if self._should_compress_cluster_data(normalized_cluster):
                    self._compress_cluster_data(normalized_cluster)
                
                # Schedule save (background) and perform an immediate save to ensure persistence.
                # The background save batches multiple updates, but an immediate save here prevents
                # a race where the process exits or another instance overwrites the file before
                # the background thread runs.
                self._schedule_save()
                try:
                    self.save_corpus()
                except Exception as e:
                    logger.debug(f"Immediate save failed (will rely on scheduled save): {e}")
                
                # Clear cached get_schema results
                try:
                    MemoryManager.get_schema.cache_clear()
                except Exception:
                    pass
                
                logger.debug(f"Stored enhanced schema for {normalized_cluster}/{database}/{table}")
                
        except Exception as e:
            logger.error(f"Failed to store schema for {cluster_uri}/{database}/{table}: {e}")
            raise
        finally:
            self._lock.release()

    def get_database_schema(self, cluster: str, database: str) -> Dict[str, Any]:
        """Gets a database schema (list of tables) from the corpus using new structure."""
        try:
            normalized_cluster = self._normalize_cluster_uri(cluster)
            db_data = self.corpus.get("clusters", {}).get(normalized_cluster, {}).get("databases", {}).get(database, {})
            
            # Extract table list from meta section; fall back to actual stored tables if needed
            meta = db_data.get("meta", {})
            tables = meta.get("table_list", [])
            
            # Fallback: if meta.table_list is empty, derive from stored table entries
            if not tables:
                try:
                    tables = list(db_data.get("tables", {}).keys())
                except Exception:
                    tables = []
            
            # Return in expected format
            return {
                "tables": tables,
                "table_count": meta.get("table_count", len(tables)),
                "last_discovered": meta.get("last_discovered"),
                "discovery_method": meta.get("discovery_method", "unknown"),
                "schema_version": meta.get("schema_version", "3.0")
            }
        except Exception as e:
            logger.warning(f"Failed to get database schema for {cluster}/{database}: {e}")
            return {}

    def store_database_schema(self, cluster: str, database: str, db_schema_data: Dict[str, Any]):
        """Stores a database schema object with enhanced metadata using new structure."""
        try:
            # DEBUG: Log incoming database schema details for diagnostics
            try:
                table_count = len(db_schema_data.get("tables", [])) if isinstance(db_schema_data, dict) else None
            except Exception:
                table_count = None
            logger.debug(
                "store_database_schema called for %s/%s - tables_count=%s, keys=%s",
                cluster,
                database,
                table_count,
                list(db_schema_data.keys()) if isinstance(db_schema_data, dict) else None,
            )
            normalized_cluster = self._normalize_cluster_uri(cluster)
            
            # Ensure cluster structure with meta section
            if normalized_cluster not in self.corpus["clusters"]:
                self.corpus["clusters"][normalized_cluster] = {
                    "meta": {
                        "token": f"{SPECIAL_TOKENS['CLUSTER_START']}{self._extract_cluster_name(normalized_cluster)}{SPECIAL_TOKENS['CLUSTER_END']}",
                        "description": f"Cluster {normalized_cluster}",
                        "last_accessed": datetime.now().isoformat()
                    },
                    "databases": {}
                }
            
            cluster_data = self.corpus["clusters"][normalized_cluster]
            
            # Ensure database structure with meta section
            if database not in cluster_data["databases"]:
                cluster_data["databases"][database] = {
                    "meta": {
                        "token": f"{SPECIAL_TOKENS['DATABASE_START']}{database}{SPECIAL_TOKENS['DATABASE_END']}",
                        "description": f"Database {database}",
                        "table_count": 0
                    },
                    "tables": {}
                }
            
            # Update database meta with table count (merge with existing to avoid overwriting previously discovered tables)
            table_list = db_schema_data.get("tables", [])
            existing_meta = cluster_data["databases"][database].setdefault("meta", {})
            existing_table_list = existing_meta.get("table_list", []) or []
            # Merge table lists without removing previously discovered tables
            merged_tables = list(dict.fromkeys(existing_table_list + list(table_list)))
            cluster_data["databases"][database]["meta"]["table_count"] = len(merged_tables)
            
            # Store table list information (for database-level queries like ".show tables")
            cluster_data["databases"][database]["meta"]["table_list"] = merged_tables
            cluster_data["databases"][database]["meta"]["last_discovered"] = datetime.now().isoformat()
            cluster_data["databases"][database]["meta"]["discovery_method"] = "live_schema_discovery"
            cluster_data["databases"][database]["meta"]["schema_version"] = "3.0"
            
            # Schedule background save and perform immediate save to persist DB-level schema changes.
            self._schedule_save()
            try:
                self.save_corpus()
            except Exception as e:
                logger.debug(f"Immediate DB schema save failed (will rely on scheduled save): {e}")
            logger.info(f"Stored enhanced database schema for {cluster}/{database}: {len(table_list)} tables")
            
        except Exception as e:
            logger.error(f"Failed to store database schema: {e}")
 
    def add_successful_query(self, cluster_uri: str, database: str, table: str, kql: str, description: str):
        """Add a successful KQL query to the specific table in memory."""
        try:
            normalized = self._normalize_cluster_uri(cluster_uri)
            
            # Ensure cluster exists
            if normalized not in self.corpus["clusters"]:
                self.corpus["clusters"][normalized] = {
                    "meta": {
                        "token": f"{SPECIAL_TOKENS['CLUSTER_START']}{self._extract_cluster_name(normalized)}{SPECIAL_TOKENS['CLUSTER_END']}",
                        "description": f"Cluster {normalized}",
                        "last_accessed": datetime.now().isoformat()
                    },
                    "databases": {}
                }
            
            cluster_data = self.corpus["clusters"][normalized]
            
            # Ensure database exists
            if database not in cluster_data["databases"]:
                cluster_data["databases"][database] = {
                    "meta": {
                        "token": f"{SPECIAL_TOKENS['DATABASE_START']}{database}{SPECIAL_TOKENS['DATABASE_END']}",
                        "description": f"Database {database}",
                        "table_count": 0
                    },
                    "tables": {}
                }
            
            db_data = cluster_data["databases"][database]
            
            # Ensure table exists
            if table not in db_data["tables"]:
                db_data["tables"][table] = {
                    "meta": {
                        "token": f"{SPECIAL_TOKENS['TABLE_START']}{table}{SPECIAL_TOKENS['TABLE_END']}",
                        "summary": f"{SPECIAL_TOKENS['SUMMARY_START']}{self._generate_table_summary(table, {})}{SPECIAL_TOKENS['SUMMARY_END']}",
                        "discovered_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat()
                    },
                    "schema": {
                        "columns": {},
                        "ai_token": ""
                    },
                    "successful_queries": []
                }
            
            table_data = db_data["tables"][table]
            
            # Add the successful query
            query_entry = {
                "query": kql,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "token": f"{SPECIAL_TOKENS['QUERY_START']}{self._generate_query_token(kql)}{SPECIAL_TOKENS['QUERY_END']}"
            }
            
            # Add to successful_queries list
            table_data.setdefault("successful_queries", []).append(query_entry)
            
            # Limit to last 10 queries to prevent memory bloat
            if len(table_data["successful_queries"]) > 10:
                table_data["successful_queries"] = table_data["successful_queries"][-10:]
            
            # Update table's last_updated timestamp
            table_data["meta"]["last_updated"] = datetime.now().isoformat()
            
            # Schedule save
            self._schedule_save()
            
        except Exception as e:
            logger.warning(f"Failed to add successful query: {e}")

    def add_global_successful_query(self, cluster_uri: str, database: str, kql: str, description: str):
        """Add a successful KQL query to global storage when table association is not available."""
        try:
            normalized = self._normalize_cluster_uri(cluster_uri)
            
            # Ensure cluster exists
            if normalized not in self.corpus["clusters"]:
                self.corpus["clusters"][normalized] = {
                    "meta": {
                        "token": f"{SPECIAL_TOKENS['CLUSTER_START']}{self._extract_cluster_name(normalized)}{SPECIAL_TOKENS['CLUSTER_END']}",
                        "description": f"Cluster {normalized}",
                        "last_accessed": datetime.now().isoformat()
                    },
                    "databases": {}
                }
            
            cluster_data = self.corpus["clusters"][normalized]
            
            # Ensure database exists
            if database not in cluster_data["databases"]:
                cluster_data["databases"][database] = {
                    "meta": {
                        "token": f"{SPECIAL_TOKENS['DATABASE_START']}{database}{SPECIAL_TOKENS['DATABASE_END']}",
                        "description": f"Database {database}",
                        "table_count": 0
                    },
                    "tables": {}
                }
            
            # Add to cluster-level successful queries (global storage)
            if "successful_queries" not in cluster_data:
                cluster_data["successful_queries"] = []
            
            query_entry = {
                "query": kql,
                "description": description,
                "database": database,
                "timestamp": datetime.now().isoformat(),
                "token": f"{SPECIAL_TOKENS['QUERY_START']}{self._generate_query_token(kql)}{SPECIAL_TOKENS['QUERY_END']}"
            }
            
            cluster_data["successful_queries"].append(query_entry)
            
            # Limit to last 20 global queries to prevent memory bloat
            if len(cluster_data["successful_queries"]) > 20:
                cluster_data["successful_queries"] = cluster_data["successful_queries"][-20:]
            
            # Schedule save
            self._schedule_save()
            
            logger.debug(f"Added global successful query for {database}: {description}")
            
        except Exception as e:
            logger.warning(f"Failed to add global successful query: {e}")

    async def validate_query(
        self,
        query: str,
        cluster_uri: str,
        database: str
    ) -> ValidationResult:
        """
        Validate a KQL query against cached schema
        
        Args:
            query: KQL query to validate
            cluster_uri: Cluster URI
            database: Database name
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        suggestions = []
        tables_used = set()
        columns_used = {}
        
        # Normalize query for analysis
        normalized_query = query.strip()
        
        # Extract table references
        table_refs = self._extract_table_references(normalized_query)
        
        # Get cached schema - FIX: Use proper memory structure access
        schema = await self._get_schema_for_validation(cluster_uri, database)
        
        if not schema:
            warnings.append(f"No cached schema found for {database} on {cluster_uri}. Validation limited.")
            return ValidationResult(
                is_valid=True,  # Don't block if no schema
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                validated_query=query,
                tables_used=tables_used,
                columns_used=columns_used
            )
        
        # Validate table references
        for table_ref in table_refs:
            table_name = table_ref.get('table', '')
            if table_name:
                validation = self._validate_table_reference(table_name, schema, database)
                if validation['error']:
                    errors.append(validation['error'])
                if validation['suggestion']:
                    suggestions.append(validation['suggestion'])
                if validation['valid_table']:
                    tables_used.add(validation['valid_table'])
        
        # Validate column references
        column_validations = self._validate_column_references(
            normalized_query,
            schema,
            tables_used
        )
        
        errors.extend(column_validations['errors'])
        warnings.extend(column_validations['warnings'])
        suggestions.extend(column_validations['suggestions'])
        columns_used = column_validations['columns_used']
        
        # Validate data types in operations
        type_validations = self._validate_data_types(
            normalized_query,
            schema,
            tables_used,
            columns_used
        )
        
        errors.extend(type_validations['errors'])
        warnings.extend(type_validations['warnings'])
        suggestions.extend(type_validations['suggestions'])
        
        # Apply query corrections if possible
        corrected_query = self._apply_corrections(
            normalized_query,
            column_validations.get('corrections', {}),
            type_validations.get('corrections', {})
        )
        
        # Validate KQL syntax patterns
        syntax_validations = self._validate_syntax_patterns(corrected_query)
        errors.extend(syntax_validations['errors'])
        warnings.extend(syntax_validations['warnings'])
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            validated_query=corrected_query,
            tables_used=tables_used,
            columns_used=columns_used
        )

    def _extract_table_references(self, query: str) -> List[Dict[str, str]]:
        """
        Extract table references from query using the existing parse_query_entities function
        
        Args:
            query: KQL query
            
        Returns:
            List of table reference dictionaries
        """
        try:
            from .utils import parse_query_entities
            entities = parse_query_entities(query)
            tables = entities.get('tables', [])
            return [{'table': table} for table in tables]
        except Exception as e:
            logger.warning(f"Error extracting table references: {e}")
            return []

    async def _get_schema_for_validation(
        self,
        cluster_uri: str,
        database: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get schema from memory for validation - FIXED to use new memory structure
        
        Args:
            cluster_uri: Cluster URI
            database: Database name
            
        Returns:
            Schema dictionary or None
        """
        try:
            # Use the new memory structure to get schema
            normalized_cluster = self._normalize_cluster_uri(cluster_uri)
            cluster_data = self.corpus.get("clusters", {}).get(normalized_cluster, {})
            db_data = cluster_data.get("databases", {}).get(database, {})
            
            if not db_data:
                logger.debug(f"No database data found for {database} in {normalized_cluster}")
                return None
            
            # Build schema structure from tables
            tables_data = db_data.get("tables", {})
            if not tables_data:
                logger.debug(f"No tables found in {database}")
                return None
            
            # Convert to validation-compatible format
            schema = {
                "tables": {}
            }
            
            for table_name, table_data in tables_data.items():
                if isinstance(table_data, dict):
                    # Check if table has schema data
                    if "schema" in table_data:
                        table_schema = table_data["schema"]
                        columns = table_schema.get("columns", {})
                    else:
                        # Fallback: check if columns are directly in table_data
                        columns = table_data.get("columns", {})
                    
                    if columns:
                        # Convert columns to validation format
                        schema["tables"][table_name] = {
                            "columns": columns
                        }
                        logger.debug(f"Found {len(columns)} columns for table {table_name}")
            
            if schema["tables"]:
                logger.debug(f"Retrieved schema for validation: {len(schema['tables'])} tables")
                return schema
            else:
                logger.debug(f"No valid schemas found for validation in {database}")
                return None
            
        except Exception as e:
            logger.warning(f"Error accessing schema for validation: {e}")
            return None

    def _validate_table_reference(
        self,
        table_name: str,
        schema: Dict[str, Any],
        database: str
    ) -> Dict[str, Any]:
        """
        Validate a table reference
        
        Args:
            table_name: Table name to validate
            schema: Database schema
            database: Database name
            
        Returns:
            Validation result dictionary
        """
        result = {
            'error': None,
            'suggestion': None,
            'valid_table': None
        }
        
        tables = schema.get('tables', {})
        
        # Exact match
        if table_name in tables:
            result['valid_table'] = table_name
            return result
        
        # Case-insensitive match
        from .utils import normalize_name
        normalized_table = normalize_name(table_name)
        for actual_table in tables:
            if normalize_name(actual_table) == normalized_table:
                result['valid_table'] = actual_table
                result['suggestion'] = f"Table '{table_name}' should be '{actual_table}' (case-sensitive)"
                return result
        
        # Find similar tables
        similar = self._find_similar_names(table_name, list(tables.keys()))
        if similar:
            result['error'] = f"Table '{table_name}' not found in {database}"
            result['suggestion'] = f"Did you mean: {', '.join(similar[:3])}?"
        else:
            result['error'] = f"Table '{table_name}' not found in {database}"
        
        return result

    def _validate_column_references(
        self,
        query: str,
        schema: Dict[str, Any],
        tables_used: Set[str]
    ) -> Dict[str, Any]:
        """
        Validate column references in the query
        
        Args:
            query: KQL query
            schema: Database schema
            tables_used: Set of tables used in query
            
        Returns:
            Validation results with errors, warnings, suggestions
        """
        errors = []
        warnings = []
        suggestions = []
        columns_used = {}
        corrections = {}
        
        # Extract column references using enhanced logic
        found_columns = self._extract_columns_from_query(query, list(tables_used))
        
        # Validate each column
        tables = schema.get('tables', {})
        for col in found_columns:
            validated = False
            
            # Check in each used table
            for table in tables_used:
                if table in tables:
                    table_schema = tables[table]
                    columns = table_schema.get('columns', {})
                    
                    # Exact match
                    if col in columns:
                        if table not in columns_used:
                            columns_used[table] = set()
                        columns_used[table].add(col)
                        validated = True
                        break
                    
                    # Case-insensitive match
                    from .utils import normalize_name
                    for actual_col in columns:
                        if normalize_name(actual_col) == normalize_name(col):
                            if table not in columns_used:
                                columns_used[table] = set()
                            columns_used[table].add(actual_col)
                            corrections[col] = actual_col
                            suggestions.append(f"Column '{col}' should be '{actual_col}' (case-sensitive)")
                            validated = True
                            break
            
            if not validated and tables_used:
                # Find similar columns
                all_columns = []
                for table in tables_used:
                    if table in tables:
                        all_columns.extend(tables[table].get('columns', {}).keys())
                
                similar = self._find_similar_names(col, all_columns)
                if similar:
                    errors.append(f"Column '{col}' not found")
                    suggestions.append(f"Did you mean: {', '.join(similar[:3])}?")
                else:
                    errors.append(f"Column '{col}' not found in tables: {', '.join(tables_used)}")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions,
            'columns_used': columns_used,
            'corrections': corrections
        }

    def _validate_data_types(
        self,
        query: str,
        schema: Dict[str, Any],
        tables_used: Set[str],
        columns_used: Dict[str, Set[str]]
    ) -> Dict[str, Any]:
        """
        Validate data type compatibility in operations
        
        Args:
            query: KQL query
            schema: Database schema
            tables_used: Tables used in query
            columns_used: Columns used per table
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        suggestions = []
        corrections = {}
        
        # Check type compatibility in common operations
        type_patterns = [
            (r'(\w+)\s*==\s*"([^"]+)"', 'string_comparison'),
            (r'(\w+)\s*==\s*(\d+)', 'numeric_comparison'),
            (r'(\w+)\s*>\s*(\d+)', 'numeric_operation'),
            (r'(\w+)\s*<\s*(\d+)', 'numeric_operation'),
            (r'sum\s*\(\s*(\w+)\s*\)', 'aggregation'),
            (r'avg\s*\(\s*(\w+)\s*\)', 'aggregation'),
        ]
        
        tables = schema.get('tables', {})
        
        for pattern, op_type in type_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                column = match.group(1)
                
                # Find column type
                column_type = None
                from .utils import normalize_name
                for table, cols in columns_used.items():
                    if column in cols or normalize_name(column) in [normalize_name(c) for c in cols]:
                        if table in tables:
                            table_cols = tables[table].get('columns', {})
                            for col_name, col_info in table_cols.items():
                                if normalize_name(col_name) == normalize_name(column):
                                    column_type = col_info.get('data_type', col_info.get('type', 'unknown'))
                                    break
                
                if column_type:
                    # Validate based on operation type
                    if op_type == 'string_comparison' and column_type not in ['string', 'dynamic']:
                        warnings.append(
                            f"Column '{column}' is type '{column_type}' but used in string comparison"
                        )
                        suggestions.append(f"Consider using tostring({column}) for explicit conversion")
                    
                    elif op_type == 'numeric_operation' and column_type not in ['int', 'long', 'real', 'decimal']:
                        errors.append(
                            f"Column '{column}' is type '{column_type}' but used in numeric operation"
                        )
                        if column_type == 'string':
                            suggestions.append(f"Consider using toint({column}) or toreal({column})")
                    
                    elif op_type == 'aggregation' and column_type not in ['int', 'long', 'real', 'decimal']:
                        errors.append(
                            f"Cannot aggregate column '{column}' of type '{column_type}'"
                        )
        
        return {
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions,
            'corrections': corrections
        }

    def _validate_syntax_patterns(self, query: str) -> Dict[str, List[str]]:
        """
        Validate common KQL syntax patterns
        
        Args:
            query: KQL query
            
        Returns:
            Dictionary with errors and warnings
        """
        errors = []
        warnings = []
        
        # Check for common syntax issues
        syntax_checks = [
            (r'\|\s*\|', "Double pipe operator detected"),
            (r'where\s+and\s+', "Invalid WHERE AND syntax"),
            (r'summarize\s+by\s*$', "Summarize by clause is empty"),
            (r'project\s*$', "Project clause is empty"),
            (r'join\s+kind\s*=', "Join kind syntax should be 'join kind=inner' not 'join kind ='"),
        ]
        
        for pattern, message in syntax_checks:
            if re.search(pattern, query, re.IGNORECASE):
                errors.append(message)
        
        # Check for unbalanced parentheses
        open_parens = query.count('(')
        close_parens = query.count(')')
        if open_parens != close_parens:
            errors.append(f"Unbalanced parentheses: {open_parens} opening, {close_parens} closing")
        
        # Check for unbalanced quotes
        single_quotes = query.count("'")
        double_quotes = query.count('"')
        if single_quotes % 2 != 0:
            errors.append("Unbalanced single quotes")
        if double_quotes % 2 != 0:
            errors.append("Unbalanced double quotes")
        
        # Warn about deprecated syntax
        deprecated_patterns = [
            (r'sort\s+by', "Use 'order by' instead of 'sort by' (deprecated)"),
            (r'limit\s+\d+', "Use 'take' instead of 'limit' (deprecated)"),
        ]
        
        for pattern, message in deprecated_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                warnings.append(message)
        
        return {
            'errors': errors,
            'warnings': warnings
        }

    def _apply_corrections(
        self,
        query: str,
        column_corrections: Dict[str, str],
        type_corrections: Dict[str, str]
    ) -> str:
        """
        Apply automatic corrections to the query
        
        Args:
            query: Original query
            column_corrections: Column name corrections
            type_corrections: Type conversion corrections
            
        Returns:
            Corrected query
        """
        corrected = query
        
        # Apply column name corrections
        for wrong_name, correct_name in column_corrections.items():
            # Use word boundary to avoid partial replacements
            pattern = rf'\b{re.escape(wrong_name)}\b'
            corrected = re.sub(pattern, correct_name, corrected, flags=re.IGNORECASE)
        
        # Apply type corrections
        for col, conversion in type_corrections.items():
            corrected = corrected.replace(col, conversion)
        
        return corrected

    def _find_similar_names(self, target: str, candidates: List[str], max_results: int = 3) -> List[str]:
        """
        Find similar names using simple string similarity.
        
        Args:
            target: Target name to find similarities for
            candidates: List of candidate names
            max_results: Maximum number of results to return
            
        Returns:
            List of similar names sorted by similarity
        """
        if not target or not candidates:
            return []
        
        target_lower = target.lower()
        similarities = []
        
        for candidate in candidates:
            if not candidate:
                continue
                
            candidate_lower = candidate.lower()
            
            # Skip exact matches
            if candidate_lower == target_lower:
                continue
            
            # Calculate similarity score
            score = 0.0
            
            # Check if target is substring of candidate or vice versa
            if target_lower in candidate_lower or candidate_lower in target_lower:
                score += 0.7
            
            # Check for common prefix
            prefix_len = 0
            for i in range(min(len(target_lower), len(candidate_lower))):
                if target_lower[i] == candidate_lower[i]:
                    prefix_len += 1
                else:
                    break
            
            if prefix_len > 0:
                score += (prefix_len / max(len(target_lower), len(candidate_lower))) * 0.5
            
            # Check for common suffix
            suffix_len = 0
            target_rev = target_lower[::-1]
            candidate_rev = candidate_lower[::-1]
            for i in range(min(len(target_rev), len(candidate_rev))):
                if target_rev[i] == candidate_rev[i]:
                    suffix_len += 1
                else:
                    break
            
            if suffix_len > 0:
                score += (suffix_len / max(len(target_lower), len(candidate_lower))) * 0.3
            
            # Simple Levenshtein-like score
            if len(target_lower) > 2 and len(candidate_lower) > 2:
                common_chars = set(target_lower) & set(candidate_lower)
                if common_chars:
                    score += len(common_chars) / max(len(set(target_lower)), len(set(candidate_lower))) * 0.2
            
            if score > 0.1:  # Only include if reasonably similar
                similarities.append((score, candidate))
        
        # Sort by similarity score (descending) and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [name for _, name in similarities[:max_results]]

    def _extract_columns_from_query(self, query: str, tables: List[str]) -> Set[str]:
        """
        Extract column names referenced in the query with enhanced logic.
        This includes columns in project, where, extend, summarize clauses and more.
        
        Args:
            query: KQL query string
            tables: List of table names used in query
            
        Returns:
            Set of column names found in the query
        """
        columns = set()
        
        # Import KQL constants
        try:
            from .constants import KQL_RESERVED_WORDS, KQL_FUNCTIONS
        except ImportError:
            KQL_RESERVED_WORDS = ['where', 'project', 'summarize', 'extend', 'order', 'by', 'take', 'limit']
            KQL_FUNCTIONS = ['count', 'sum', 'avg', 'min', 'max']
        
        # Extract bracketed columns [ColumnName] or ['ColumnName']
        bracketed_columns = re.findall(r'\[\'?([a-zA-Z0-9_]+)\'?\]', query)
        columns.update(bracketed_columns)
        
        # Extract columns from project clauses - enhanced pattern
        project_matches = re.finditer(r'\|\s*project\s+([^|]+)', query, re.IGNORECASE)
        for match in project_matches:
            project_content = match.group(1).strip()
            # Split by comma and clean each column
            for col in project_content.split(','):
                col = col.strip()
                # Remove any alias (column = alias or alias = expression)
                if '=' in col:
                    # Check if it's column = alias or alias = column expression
                    parts = col.split('=')
                    left_part = parts[0].strip()
                    # Take the left part as it's usually the column name
                    col = left_part
                
                # Extract clean column name (handle functions and brackets)
                col = re.sub(r'\(.*?\)', '', col)  # Remove function calls
                col = re.sub(r'\[|\]', '', col)    # Remove brackets
                clean_col = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', col.strip())
                if clean_col and clean_col.group(0).lower() not in [w.lower() for w in KQL_RESERVED_WORDS]:
                    columns.add(clean_col.group(0))
        
        # Extract columns from where clauses - enhanced pattern
        where_matches = re.finditer(r'\|\s*where\s+([^|]+)', query, re.IGNORECASE)
        for match in where_matches:
            where_content = match.group(1).strip()
            # Find column names in various conditions
            col_patterns = [
                r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:==|!=|<=|>=|<|>|contains|startswith|endswith|has|!has)',
                r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:in|!in)\s*\(',
                r'isnotnull\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)',
                r'isnull\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)',
                r'isempty\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)',
                r'isnotempty\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)'
            ]
            
            for pattern in col_patterns:
                col_matches = re.findall(pattern, where_content, re.IGNORECASE)
                for col in col_matches:
                    if col.lower() not in [w.lower() for w in KQL_RESERVED_WORDS]:
                        columns.add(col)
        
        # Extract columns from summarize clauses
        summarize_matches = re.finditer(r'\|\s*summarize\s+([^|]+)', query, re.IGNORECASE)
        for match in summarize_matches:
            summarize_content = match.group(1).strip()
            
            # Extract from aggregation functions
            agg_patterns = [
                r'(?:count|sum|avg|min|max|stdev|variance)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)',
                r'dcount\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)',
                r'countif\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[^)]*\)'
            ]
            
            for pattern in agg_patterns:
                agg_matches = re.findall(pattern, summarize_content, re.IGNORECASE)
                for col in agg_matches:
                    if col.lower() not in [w.lower() for w in KQL_RESERVED_WORDS]:
                        columns.add(col)
            
            # Extract from 'by' clause
            by_match = re.search(r'\bby\s+([^,\|]+)', summarize_content, re.IGNORECASE)
            if by_match:
                by_content = by_match.group(1).strip()
                for col in by_content.split(','):
                    col = col.strip()
                    # Handle functions in by clause
                    col = re.sub(r'\(.*?\)', '', col)
                    clean_col = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', col)
                    if clean_col and clean_col.group(0).lower() not in [w.lower() for w in KQL_RESERVED_WORDS]:
                        columns.add(clean_col.group(0))
        
        # Extract columns from extend clauses
        extend_matches = re.finditer(r'\|\s*extend\s+([^|]+)', query, re.IGNORECASE)
        for match in extend_matches:
            extend_content = match.group(1).strip()
            # Look for column references in expressions
            col_matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', extend_content)
            for col in col_matches:
                if (col.lower() not in [w.lower() for w in KQL_RESERVED_WORDS] and
                    col.lower() not in [w.lower() for w in KQL_FUNCTIONS]):
                    columns.add(col)
        
        # Extract columns from order by clauses
        order_matches = re.finditer(r'\|\s*order\s+by\s+([^|]+)', query, re.IGNORECASE)
        for match in order_matches:
            order_content = match.group(1).strip()
            for col in order_content.split(','):
                col = col.strip()
                # Remove asc/desc
                col = re.sub(r'\s+(asc|desc)$', '', col, flags=re.IGNORECASE)
                clean_col = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', col)
                if clean_col and clean_col.group(0).lower() not in [w.lower() for w in KQL_RESERVED_WORDS]:
                    columns.add(clean_col.group(0))
        
        # Extract columns from join clauses
        join_matches = re.finditer(r'\|\s*join\s+(?:kind\s*=\s*\w+\s+)?([^|]+)', query, re.IGNORECASE)
        for match in join_matches:
            join_content = match.group(1).strip()
            # Look for $left.column and $right.column patterns
            join_col_matches = re.findall(r'\$(?:left|right)\.([a-zA-Z_][a-zA-Z0-9_]*)', join_content)
            for col in join_col_matches:
                if col.lower() not in [w.lower() for w in KQL_RESERVED_WORDS]:
                    columns.add(col)
        
        # Filter out KQL reserved words and operators
        reserved_words_lower = {word.lower() for word in KQL_RESERVED_WORDS}
        columns = {col for col in columns if col.lower() not in reserved_words_lower}
        
        return columns

    def _apply_schema_fallback_strategies(self, cluster_uri: str, database: str, table: str) -> Dict[str, Any]:
        """
        Apply comprehensive schema discovery fallback strategies when live discovery fails.
        
        Fallback Strategy Order:
        1. cached_schema: Use any available cached schema data
        2. query_derived_schema: Derive schema from successful query history
        3. minimal_schema: Generate minimal schema with common patterns
        
        Args:
            cluster_uri: Cluster URI
            database: Database name
            table: Table name
            
        Returns:
            Schema dictionary from the first successful fallback strategy
        """
        from .constants import ERROR_HANDLING_CONFIG
        
        fallback_strategies = ERROR_HANDLING_CONFIG.get("fallback_strategies", [
            "cached_schema", "query_derived_schema", "minimal_schema"
        ])
        
        logger.info(f"Applying schema fallback strategies for {cluster_uri}/{database}/{table}")
        
        for strategy in fallback_strategies:
            try:
                if strategy == "cached_schema":
                    schema = self._fallback_cached_schema(cluster_uri, database, table)
                elif strategy == "query_derived_schema":
                    schema = self._fallback_query_derived_schema(cluster_uri, database, table)
                elif strategy == "minimal_schema":
                    schema = self._fallback_minimal_schema(cluster_uri, database, table)
                else:
                    logger.warning(f"Unknown fallback strategy: {strategy}")
                    continue
                
                if schema and isinstance(schema, dict) and schema.get("columns"):
                    logger.info(f"Fallback strategy '{strategy}' succeeded for {table}")
                    schema["fallback_strategy"] = strategy
                    schema["fallback_applied_at"] = datetime.now().isoformat()
                    return schema
                    
            except Exception as e:
                logger.warning(f"Fallback strategy '{strategy}' failed for {table}: {e}")
                continue
        
        # If all fallback strategies fail, return minimal emergency schema
        logger.error(f"All fallback strategies failed for {table}, using emergency minimal schema")
        return self._emergency_minimal_schema(table)

    def _fallback_cached_schema(self, cluster_uri: str, database: str, table: str) -> Optional[Dict[str, Any]]:
        """
        Fallback Strategy 1: Attempt to use any available cached schema data.
        
        Searches for:
        - Partial schema data in the corpus
        - Similar table schemas in the same database
        - Historical schema data that might still be valid
        """
        try:
            normalized_cluster = self._normalize_cluster_uri(cluster_uri)
            
            # Check for any partial schema data in the corpus
            cluster_data = self.corpus.get("clusters", {}).get(normalized_cluster, {})
            db_data = cluster_data.get("databases", {}).get(database, {})
            table_data = db_data.get("tables", {}).get(table, {})
            
            # Look for any existing schema fragments
            if table_data and isinstance(table_data, dict):
                schema_data = table_data.get("schema", {})
                if schema_data and "columns" in schema_data:
                    logger.info(f"Found partial cached schema for {table}")
                    return schema_data
                
                # Check if we have successful queries that might indicate column structure
                successful_queries = table_data.get("successful_queries", [])
                if successful_queries:
                    derived_schema = self._derive_schema_from_queries(successful_queries, table)
                    if derived_schema:
                        logger.info(f"Derived schema from successful queries for {table}")
                        return derived_schema
            
            # Look for similar tables in the same database (pattern-based matching)
            similar_schema = self._find_similar_table_schema(db_data, table)
            if similar_schema:
                logger.info(f"Found similar table schema for {table}")
                return similar_schema
                
            return None
            
        except Exception as e:
            logger.warning(f"Cached schema fallback failed: {e}")
            return None

    def _fallback_query_derived_schema(self, cluster_uri: str, database: str, table: str) -> Optional[Dict[str, Any]]:
        """
        Fallback Strategy 2: Derive schema from historical successful query patterns.
        
        Analyzes successful queries to infer:
        - Column names from project clauses
        - Data types from filter operations
        - Common column patterns from aggregations
        """
        try:
            normalized_cluster = self._normalize_cluster_uri(cluster_uri)
            
            # Collect all successful queries for this table from the corpus
            cluster_data = self.corpus.get("clusters", {}).get(normalized_cluster, {})
            all_queries = []
            
            # Get queries from table-specific successful_queries
            db_data = cluster_data.get("databases", {}).get(database, {})
            table_data = db_data.get("tables", {}).get(table, {})
            if table_data:
                all_queries.extend(table_data.get("successful_queries", []))
            
            # Get queries from cluster-level successful_queries (legacy)
            cluster_queries = cluster_data.get("successful_queries", [])
            table_specific_queries = [
                q for q in cluster_queries
                if isinstance(q, dict) and table.lower() in str(q.get("query", "")).lower()
            ]
            all_queries.extend(table_specific_queries)
            
            if not all_queries:
                logger.info(f"No query history found for schema derivation of {table}")
                return None
            
            # Analyze queries to derive schema
            derived_columns = self._analyze_queries_for_schema(all_queries, table)
            if not derived_columns:
                return None
            
            # Create schema object
            schema = {
                "table_name": table,
                "columns": derived_columns,
                "discovered_at": datetime.now().isoformat(),
                "cluster": cluster_uri,
                "database": database,
                "derived_from_queries": len(all_queries),
                "derivation_method": "query_analysis"
            }
            
            logger.info(f"Derived schema for {table} from {len(all_queries)} queries")
            return schema
            
        except Exception as e:
            logger.warning(f"Query-derived schema fallback failed: {e}")
            return None

    def _fallback_minimal_schema(self, cluster_uri: str, database: str, table: str) -> Dict[str, Any]:
        """Direct KQL-based schema discovery using getschema and take commands."""
        try:
            # Use direct KQL execution to get real schema - no static keywords
            from .execute_kql import kql_execute_tool
            
            # Get schema using | getschema
            schema_query = f"{table} | getschema"
            schema_df = kql_execute_tool(schema_query, cluster_uri, database)
            
            if schema_df is not None and not schema_df.empty:
                columns = {}
                for _, row in schema_df.iterrows():
                    col_name = row.get("ColumnName", "")
                    col_type = row.get("DataType", row.get("ColumnType", "string"))
                    
                    columns[col_name] = {
                        'data_type': col_type,
                        'description': self._generate_ai_description(col_name, col_type, table),
                        'tags': self._generate_column_tags(col_name, col_type),
                        'sample_values': []
                    }
                
                # Get sample data using | take 2
                sample_query = f"{table} | take 2"
                sample_df = kql_execute_tool(sample_query, cluster_uri, database)
                
                if sample_df is not None and not sample_df.empty:
                    for col_name in columns.keys():
                        if col_name in sample_df.columns:
                            sample_values = sample_df[col_name].dropna().astype(str).tolist()[:2]
                            columns[col_name]['sample_values'] = sample_values
                
                return {
                    "table_name": table,
                    "columns": columns,
                    "discovered_at": datetime.now().isoformat(),
                    "cluster": cluster_uri,
                    "database": database,
                    "column_count": len(columns),
                    "schema_type": "direct_kql_discovery"
                }
            
            return self._emergency_minimal_schema(table)
            
        except Exception as e:
            logger.warning(f"Direct KQL schema discovery failed: {e}")
            return self._emergency_minimal_schema(table)

    def _emergency_minimal_schema(self, table: str) -> Dict[str, Any]:
        """
        Emergency fallback: Generate absolute minimal schema to prevent total failure.
        
        Creates the most basic schema possible to allow query execution.
        """
        emergency_columns = {
            'TimeGenerated': {
                'data_type': 'datetime',
                'description': 'Timestamp field',
                'tags': ['TIME_COLUMN'],
                'sample_values': []
            },
            'Data': {
                'data_type': 'string',
                'description': 'Data field',
                'tags': ['TEXT'],
                'sample_values': []
            }
        }
        
        return {
            "table_name": table,
            "columns": emergency_columns,
            "discovered_at": datetime.now().isoformat(),
            "column_count": len(emergency_columns),
            "schema_type": "emergency_fallback"
        }

    def _derive_schema_from_queries(self, queries: List[Dict[str, Any]], table: str) -> Optional[Dict[str, Any]]:
        """Derive schema information from successful query patterns."""
        column_info = {}
        
        for query_entry in queries:
            query = query_entry.get("query", "") if isinstance(query_entry, dict) else str(query_entry)
            
            # Extract columns from project clauses
            project_matches = re.findall(r'\|\s*project\s+([^|]+)', query, re.IGNORECASE)
            for match in project_matches:
                columns = [col.strip() for col in match.split(',')]
                for col in columns:
                    # Clean column name (remove brackets, aliases, etc.)
                    clean_col = re.sub(r'[\[\]\'"`]', '', col.split(' as ')[0].strip())
                    if clean_col and not any(op in clean_col.lower() for op in ['(', ')', '+', '-', '*', '/']):
                        column_info[clean_col] = {
                            'data_type': 'string',  # Default type
                            'description': 'Column derived from query analysis',
                            'tags': ['DERIVED'],
                            'sample_values': []
                        }
        
        if column_info:
            return {
                "table_name": table,
                "columns": column_info,
                "discovered_at": datetime.now().isoformat(),
                "derivation_method": "query_project_analysis"
            }
        
        return None

    def _find_similar_table_schema(self, db_data: Dict[str, Any], target_table: str) -> Optional[Dict[str, Any]]:
        """Find schema from similar tables in the same database."""
        try:
            target_lower = target_table.lower()
            tables = db_data.get("tables", {})
            
            # Look for tables with similar naming patterns
            for table_name, table_data in tables.items():
                if table_name == target_table:
                    continue
                
                table_lower = table_name.lower()
                
                # Check for similar prefixes, suffixes, or contained patterns
                similarity_score = 0
                
                # Same prefix (first 3+ characters)
                if len(table_lower) > 3 and len(target_lower) > 3:
                    if table_lower[:3] == target_lower[:3]:
                        similarity_score += 1
                
                # Same suffix (last 3+ characters)
                if len(table_lower) > 3 and len(target_lower) > 3:
                    if table_lower[-3:] == target_lower[-3:]:
                        similarity_score += 1
                
                # Contains common substrings
                common_patterns = ['event', 'log', 'security', 'audit', 'network', 'process']
                for pattern in common_patterns:
                    if pattern in table_lower and pattern in target_lower:
                        similarity_score += 2
                
                # If similarity found, use this schema as a template
                if similarity_score >= 1:
                    schema_data = table_data.get("schema", {})
                    if schema_data and "columns" in schema_data:
                        # Copy schema but update metadata
                        similar_schema = schema_data.copy()
                        similar_schema["table_name"] = target_table
                        similar_schema["similarity_source"] = table_name
                        similar_schema["similarity_score"] = similarity_score
                        similar_schema["discovered_at"] = datetime.now().isoformat()
                        
                        logger.info(f"Using schema from similar table {table_name} for {target_table}")
                        return similar_schema
            
            return None
            
        except Exception as e:
            logger.warning(f"Similar table schema lookup failed: {e}")
            return None

    def _analyze_queries_for_schema(self, queries: List[Dict[str, Any]], table: str) -> Dict[str, Any]:
        """Analyze query patterns to derive comprehensive schema information."""
        columns = {}
        
        for query_entry in queries:
            query = query_entry.get("query", "") if isinstance(query_entry, dict) else str(query_entry)
            
            # Extract columns from various KQL operations
            self._extract_columns_from_project(query, columns)
            self._extract_columns_from_where(query, columns)
            self._extract_columns_from_summarize(query, columns)
            self._extract_columns_from_extend(query, columns)
        
        return columns

    def _extract_columns_from_project(self, query: str, columns: Dict[str, Any]):
        """Extract column information from project clauses."""
        project_patterns = re.findall(r'\|\s*project\s+([^|]+)', query, re.IGNORECASE)
        for match in project_patterns:
            column_list = [col.strip() for col in match.split(',')]
            for col in column_list:
                # Handle aliases and clean column names
                if ' as ' in col.lower():
                    original_col = col.split(' as ')[0].strip()
                else:
                    original_col = col
                
                clean_col = re.sub(r'[\[\]\'"`]', '', original_col)
                if clean_col and clean_col not in columns:
                    columns[clean_col] = {
                        'data_type': 'string',
                        'description': 'Column from project analysis',
                        'tags': ['PROJECTED'],
                        'sample_values': []
                    }

    def _extract_columns_from_where(self, query: str, columns: Dict[str, Any]):
        """Extract column information from where clauses with data type hints."""
        where_patterns = re.findall(r'\|\s*where\s+([^|]+)', query, re.IGNORECASE)
        for match in where_patterns:
            # Look for column comparisons
            comparisons = re.findall(r'(\w+)\s*(?:==|!=|>=|<=|>|<|contains|has)\s*([^\\s]+)', match)
            for col_name, value in comparisons:
                if col_name not in columns:
                    # Infer data type from comparison value
                    data_type = 'string'
                    if value.strip().startswith('datetime('):
                        data_type = 'datetime'
                    elif value.isdigit():
                        data_type = 'int'
                    elif re.match(r'^\d+\.\d+$', value):
                        data_type = 'real'
                    
                    columns[col_name] = {
                        'data_type': data_type,
                        'description': 'Column from where clause analysis',
                        'tags': ['FILTERED'],
                        'sample_values': [value] if value not in ['true', 'false', 'null'] else []
                    }

    def _extract_columns_from_summarize(self, query: str, columns: Dict[str, Any]):
        """Extract column information from summarize clauses."""
        summarize_patterns = re.findall(r'\|\s*summarize\s+([^|]+)', query, re.IGNORECASE)
        for match in summarize_patterns:
            # Extract aggregation columns and group by columns
            if ' by ' in match.lower():
                agg_part, by_part = match.lower().split(' by ', 1)
                
                # Process group by columns
                by_columns = [col.strip() for col in by_part.split(',')]
                for col in by_columns:
                    clean_col = re.sub(r'[\[\]\'"`]', '', col)
                    if clean_col and clean_col not in columns:
                        columns[clean_col] = {
                            'data_type': 'string',
                            'description': 'Group by column',
                            'tags': ['GROUPING'],
                            'sample_values': []
                        }

    def _extract_columns_from_extend(self, query: str, columns: Dict[str, Any]):
        """Extract column information from extend clauses."""
        extend_patterns = re.findall(r'\|\s*extend\s+([^|]+)', query, re.IGNORECASE)
        for match in extend_patterns:
            # Extract new column definitions
            definitions = [defn.strip() for defn in match.split(',')]
            for defn in definitions:
                if '=' in defn:
                    new_col = defn.split('=')[0].strip()
                    clean_col = re.sub(r'[\[\]\'"`]', '', new_col)
                    if clean_col and clean_col not in columns:
                        columns[clean_col] = {
                            'data_type': 'string',  # Extended columns often calculated
                            'description': 'Extended/calculated column',
                            'tags': ['CALCULATED'],
                            'sample_values': []
                        }

    def _generate_query_token(self, query: str) -> str:
        """Generate a simplified token for a query."""
        # Remove special characters and normalize
        normalized = re.sub(r'[^\w\s]', '', query.lower())
        # Replace spaces with underscores and limit length
        token = re.sub(r'\s+', '_', normalized)[:30]
        return token

    def get_successful_queries(self, cluster_uri: str, database: str, table: str) -> List[Dict[str, Any]]:
        """Get successful queries for a specific table."""
        try:
            normalized = self._normalize_cluster_uri(cluster_uri)
            return (self.corpus
                    .get("clusters", {})
                    .get(normalized, {})
                    .get("databases", {})
                    .get(database, {})
                    .get("tables", {})
                    .get(table, {})
                    .get("successful_queries", []))
        except Exception as e:
            logger.warning(f"Failed to get successful queries: {e}")
            return []

    def store_learning_result(self, query: str, result_data: Dict[str, Any], execution_type: str):
        """
        Store query execution results for learning and future context building.
        Enhanced with session-based tracking.
        """
        try:
            # Extract cluster, database, and table information from the query
            from .execute_kql import extract_cluster_and_database_from_query, extract_tables_from_query
            
            cluster_uri, database = extract_cluster_and_database_from_query(query)
            tables = extract_tables_from_query(query)
            
            # No hardcoded defaults - require explicit cluster and database
            if not cluster_uri:
                logger.warning("No cluster_uri found in query - cannot store learning result without cluster information")
                return
                    
            if not database:
                logger.warning("No database found in query - cannot store learning result without database information")
                return
            
            # Get session ID from result data or generate one
            session_id = result_data.get("session_id") or self._generate_session_id()
            
            # Use first table or create a generic entry
            primary_table = tables[0] if tables else "UnknownTable"
            
            # Create learning entry with enhanced session tracking
            learning_entry = {
                "query": query,
                "execution_type": execution_type,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "result_metadata": {
                    "row_count": result_data.get("row_count", 0),
                    "column_count": len(result_data.get("columns", [])),
                    "columns": result_data.get("columns", []),
                    "success": result_data.get("success", True),
                    "execution_time_ms": result_data.get("execution_time_ms", 0)
                },
                "learning_insights": {
                    "query_complexity": len(query.split("|")),
                    "has_filters": "where" in query.lower(),
                    "has_aggregation": any(op in query.lower() for op in ["summarize", "count", "sum", "avg"]),
                    "has_time_reference": "ago(" in query.lower(),
                    "data_found": result_data.get("row_count", 0) > 0,
                    "tables_involved": tables,
                    "cluster": cluster_uri,
                    "database": database
                }
            }
            
            # Store the learning result as a successful query for the primary table
            description = f"Learning result from {execution_type} - {result_data.get('row_count', 0)} rows - {execution_type}"
            self.add_successful_query(cluster_uri, database, primary_table, query, description)
            
            # Store in session-based learning section
            self._store_session_learning(session_id, learning_entry)
            
            # Also store in cluster-level learning results for backward compatibility
            self._store_cluster_learning(cluster_uri, learning_entry)
            
            logger.info(f"Stored learning result for session {session_id}: {len(query)} chars, {result_data.get('row_count', 0)} rows")
            
        except Exception as e:
            logger.error(f"Failed to store learning result: {e}")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    def _store_session_learning(self, session_id: str, learning_entry: Dict[str, Any]):
        """Store learning entry in session-based structure."""
        try:
            # Ensure sessions section exists
            if "sessions" not in self.corpus:
                self.corpus["sessions"] = {}
            
            # Ensure session exists
            if session_id not in self.corpus["sessions"]:
                self.corpus["sessions"][session_id] = {
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "query_count": 0,
                    "learning_entries": [],
                    "session_insights": {
                        "total_rows_processed": 0,
                        "unique_tables": set(),
                        "unique_clusters": set(),
                        "query_types": set()
                    }
                }
            
            session_data = self.corpus["sessions"][session_id]
            
            # Add learning entry
            session_data["learning_entries"].append(learning_entry)
            session_data["query_count"] += 1
            session_data["last_updated"] = datetime.now().isoformat()
            
            # Update session insights
            insights = session_data["session_insights"]
            insights["total_rows_processed"] += learning_entry.get("result_metadata", {}).get("row_count", 0)
            insights["unique_tables"].update(learning_entry.get("learning_insights", {}).get("tables_involved", []))
            insights["unique_clusters"].add(learning_entry.get("learning_insights", {}).get("cluster", ""))
            insights["query_types"].add(learning_entry.get("execution_type", ""))
            
            # Convert sets to lists for JSON serialization
            insights["unique_tables"] = list(insights["unique_tables"])
            insights["unique_clusters"] = list(insights["unique_clusters"])
            insights["query_types"] = list(insights["query_types"])
            
            # Limit session entries to prevent memory bloat
            if len(session_data["learning_entries"]) > 100:
                session_data["learning_entries"] = session_data["learning_entries"][-100:]
            
            # Schedule save
            self._schedule_save()
            
        except Exception as e:
            logger.error(f"Failed to store session learning: {e}")

    def _store_cluster_learning(self, cluster_uri: str, learning_entry: Dict[str, Any]):
        """Store learning entry in cluster-level structure for backward compatibility."""
        try:
            normalized_cluster = self._normalize_cluster_uri(cluster_uri)
            
            # Ensure cluster structure exists
            if normalized_cluster not in self.corpus["clusters"]:
                self.corpus["clusters"][normalized_cluster] = {
                    "meta": {
                        "token": f"{SPECIAL_TOKENS['CLUSTER_START']}{self._extract_cluster_name(normalized_cluster)}{SPECIAL_TOKENS['CLUSTER_END']}",
                        "description": f"Cluster {normalized_cluster}",
                        "last_accessed": datetime.now().isoformat()
                    },
                    "databases": {}
                }
            
            cluster_data = self.corpus["clusters"][normalized_cluster]
            
            # Ensure learning_results section exists
            if "learning_results" not in cluster_data:
                cluster_data["learning_results"] = []
            
            # Add to learning results only if the query returned rows
            row_count = learning_entry.get("result_metadata", {}).get("row_count", 0)
            if row_count and row_count > 0:
                cluster_data["learning_results"].append(learning_entry)
                # Keep only the most recent 50 learning results to prevent memory bloat
                if len(cluster_data["learning_results"]) > 50:
                    cluster_data["learning_results"] = cluster_data["learning_results"][-50:]
            
        except Exception as e:
            logger.error(f"Failed to store cluster learning: {e}")

    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """Get session data for analysis and reporting."""
        try:
            return self.corpus.get("sessions", {}).get(session_id, {})
        except Exception as e:
            logger.error(f"Failed to get session data: {e}")
            return {}

    def get_session_queries(self, session_id: str) -> List[Dict[str, Any]]:
        """Get queries for a specific session."""
        try:
            session_data = self.get_session_data(session_id)
            return session_data.get("learning_entries", [])
        except Exception as e:
            logger.error(f"Failed to get session queries: {e}")
            return []

    def list_active_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List active sessions with basic metadata."""
        try:
            sessions = self.corpus.get("sessions", {})
            session_list = []
            
            for session_id, session_data in list(sessions.items())[-limit:]:
                session_summary = {
                    "session_id": session_id,
                    "created_at": session_data.get("created_at"),
                    "last_updated": session_data.get("last_updated"),
                    "query_count": session_data.get("query_count", 0),
                    "total_rows": session_data.get("session_insights", {}).get("total_rows_processed", 0)
                }
                session_list.append(session_summary)
            
            return sorted(session_list, key=lambda x: x["last_updated"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list active sessions: {e}")
            return []
 
    def _create_ai_friendly_token(
        self,
        table: str,
        cluster_uri: str,
        database: str,
        columns: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    ) -> str:
        """Create AI-friendly token with XML-style markers for efficient parsing."""
        
        # Start with cluster, database, and table tokens
        token_parts = [
            f"{SPECIAL_TOKENS['CLUSTER_START']}{self._extract_cluster_name(cluster_uri)}{SPECIAL_TOKENS['CLUSTER_END']}",
            f"{SPECIAL_TOKENS['DATABASE_START']}{database}{SPECIAL_TOKENS['DATABASE_END']}",
            f"{SPECIAL_TOKENS['TABLE_START']}{table}{SPECIAL_TOKENS['TABLE_END']}",
        ]
        
        # Derive column items from input
        column_items: List[Tuple[str, Dict[str, Any]]] = []
        if isinstance(columns, dict):
            # columns is a mapping: name -> metadata
            column_items = list(columns.items())
        elif isinstance(columns, list):
            # columns is a list of dicts
            for col in columns:
                if isinstance(col, dict):
                    name = col.get("name", "unknown")
                    meta = {
                        "data_type": col.get("type") or col.get("data_type", "unknown"),
                        "description": col.get("description", ""),
                        "tags": col.get("tags", []),
                        "sample_values": col.get("sample_values", []),
                    }
                    column_items.append((name, meta))
                elif isinstance(col, str):
                    column_items.append((col, {"data_type": "unknown", "description": "", "tags": [], "sample_values": []}))
        
        # Add table summary
        column_names = [c[0] for c in column_items] if column_items else []
        table_summary = self._generate_table_summary(table, column_names)
        token_parts.append(f"{SPECIAL_TOKENS['SUMMARY_START']}{table_summary}{SPECIAL_TOKENS['SUMMARY_END']}")
        
        # Add columns with enhanced XML-style tokens (limit to prevent bloat)
        for name, meta in column_items[:10]:  # Limit to 10 columns for performance
            col_type = meta.get("data_type", "unknown")
            ai_desc = meta.get("description", "") or self._generate_ai_description(name, col_type, table)
            tags = meta.get("tags", [])
            samples = meta.get("sample_values", [])
            
            # Create column token with XML-style markers
            col_token = (
                f"{SPECIAL_TOKENS['COLUMN_START']}{name}"
                f"{SPECIAL_TOKENS['TYPE_START']}{col_type}{SPECIAL_TOKENS['TYPE_END']}"
                f"{SPECIAL_TOKENS['DESCRIPTION_START']}{ai_desc}{SPECIAL_TOKENS['DESCRIPTION_END']}"
                f"{SPECIAL_TOKENS['TAGS_START']}{','.join(str(t) for t in tags)}{SPECIAL_TOKENS['TAGS_END']}"
                f"{SPECIAL_TOKENS['SAMPLES_START']}{','.join(str(s) for s in samples[:2])}{SPECIAL_TOKENS['SAMPLES_END']}"
                f"{SPECIAL_TOKENS['COLUMN_END']}"
            )
            
            token_parts.append(col_token)
        
        # Add truncation indicator if more columns exist
        if len(column_items) > 10:
            truncated_count = len(column_items) - 10
            token_parts.append(f"[+{truncated_count}_more_columns]")
        
        full_token = "".join(token_parts)
        
        # Log token size for monitoring
        logger.debug(
            "Generated enhanced AI token for %s: %d chars, %d columns",
            table,
            len(full_token),
            len(column_items),
        )
        return full_token

    def _generate_ai_description(self, col_name: str, col_type: str, table: str) -> str:
        """Generate intelligent AI description for column based on type and context."""
        # Dynamic description based on data type
        type_lower = col_type.lower() if col_type else ""
        
        # Type-specific descriptions
        if 'datetime' in type_lower or 'timestamp' in type_lower:
            return self._describe_datetime_column(col_name)
        elif 'int' in type_lower or 'long' in type_lower:
            return self._describe_numeric_column(col_name, 'integer')
        elif 'real' in type_lower or 'double' in type_lower or 'float' in type_lower:
            return self._describe_numeric_column(col_name, 'decimal')
        elif 'bool' in type_lower:
            return f"Boolean flag indicating {self._humanize_column_name(col_name)}"
        elif 'guid' in type_lower or 'uuid' in type_lower:
            return f"Unique identifier for {self._humanize_column_name(col_name)}"
        elif 'dynamic' in type_lower or 'json' in type_lower:
            return f"Dynamic/JSON data for {self._humanize_column_name(col_name)}"
        elif 'string' in type_lower or 'text' in type_lower:
            return self._describe_text_column(col_name)
        else:
            # Generic description
            return f"{self._humanize_column_name(col_name)} field"
    
    def _describe_datetime_column(self, col_name: str) -> str:
        """Generate description for datetime columns based on patterns."""
        name_lower = col_name.lower()
        
        # Look for common datetime patterns
        if any(pattern in name_lower for pattern in ['created', 'creation', 'createdat']):
            return "Timestamp when the record was created"
        elif any(pattern in name_lower for pattern in ['updated', 'modified', 'updatedat']):
            return "Timestamp when the record was last updated"
        elif any(pattern in name_lower for pattern in ['start', 'begin', 'from']):
            return "Start time of the event or period"
        elif any(pattern in name_lower for pattern in ['end', 'finish', 'to', 'until']):
            return "End time of the event or period"
        elif any(pattern in name_lower for pattern in ['generated', 'logged', 'recorded']):
            return "Timestamp when the data was generated or logged"
        elif 'time' in name_lower:
            return f"Timestamp for {self._humanize_column_name(col_name.replace('time', '').replace('Time', ''))}"
        else:
            return f"Timestamp field: {self._humanize_column_name(col_name)}"
    
    def _describe_numeric_column(self, col_name: str, num_type: str) -> str:
        """Generate description for numeric columns."""
        name_lower = col_name.lower()
        
        if any(pattern in name_lower for pattern in ['count', 'total', 'number']):
            return f"Count of {self._humanize_column_name(col_name.replace('count', '').replace('Count', ''))}"
        elif any(pattern in name_lower for pattern in ['size', 'length', 'bytes']):
            return f"Size/length measurement in {self._humanize_column_name(col_name)}"
        elif any(pattern in name_lower for pattern in ['duration', 'elapsed', 'latency']):
            return f"Time duration for {self._humanize_column_name(col_name)}"
        elif any(pattern in name_lower for pattern in ['score', 'rating', 'rank']):
            return f"Score/rating value for {self._humanize_column_name(col_name)}"
        elif 'id' in name_lower and num_type == 'integer':
            return f"Numeric identifier for {self._humanize_column_name(col_name.replace('id', '').replace('Id', ''))}"
        elif 'port' in name_lower:
            return "Network port number"
        elif 'code' in name_lower:
            return f"Numeric code for {self._humanize_column_name(col_name.replace('code', '').replace('Code', ''))}"
        else:
            return f"{num_type.capitalize()} value: {self._humanize_column_name(col_name)}"
    
    def _describe_text_column(self, col_name: str) -> str:
        """Generate description for text columns."""
        name_lower = col_name.lower()
        
        if any(pattern in name_lower for pattern in ['name', 'title', 'label']):
            context = col_name.replace('name', '').replace('Name', '').replace('title', '').replace('Title', '')
            return f"Name/identifier for {self._humanize_column_name(context)}" if context else "Name identifier"
        elif any(pattern in name_lower for pattern in ['message', 'description', 'text', 'content']):
            return f"Descriptive text: {self._humanize_column_name(col_name)}"
        elif any(pattern in name_lower for pattern in ['url', 'uri', 'link', 'href']):
            return "URL/URI reference"
        elif any(pattern in name_lower for pattern in ['path', 'directory', 'folder']):
            return "File system path"
        elif any(pattern in name_lower for pattern in ['email', 'mail']):
            return "Email address"
        elif any(pattern in name_lower for pattern in ['ip', 'address']):
            return "Network address"
        elif any(pattern in name_lower for pattern in ['user', 'username', 'account']):
            return "User account identifier"
        elif any(pattern in name_lower for pattern in ['type', 'category', 'class']):
            return f"Type/category classification for {self._humanize_column_name(col_name)}"
        elif any(pattern in name_lower for pattern in ['version', 'release']):
            return "Version identifier"
        else:
            return f"Text field: {self._humanize_column_name(col_name)}"
    
    def _humanize_column_name(self, col_name: str) -> str:
        """Convert column name to human-readable format."""
        if not col_name:
            return "data"
        
        # Handle camelCase and PascalCase
        import re
        # Insert space before uppercase letters that follow lowercase letters
        humanized = re.sub(r'([a-z])([A-Z])', r'\1 \2', col_name)
        # Insert space before uppercase letters that are followed by lowercase letters
        humanized = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', humanized)
        # Replace underscores and hyphens with spaces
        humanized = humanized.replace('_', ' ').replace('-', ' ')
        # Remove extra spaces and convert to lowercase
        humanized = ' '.join(humanized.split()).lower()
        
        # Remove common suffixes that don't add meaning
        for suffix in ['field', 'column', 'value', 'data']:
            if humanized.endswith(' ' + suffix):
                humanized = humanized[:-len(suffix)-1]
        
        return humanized.strip() or "data"

    def _generate_table_summary(self, table_name: str, columns) -> str:
        """Generate compact table summary."""
        col_count = len(columns) if isinstance(columns, (list, dict)) else 0
        return f"{table_name.lower()}_table_{col_count}cols"
        
    def _generate_column_tags(self, col_name: str, col_type: str) -> List[str]:
        """Generate smart column tags based on data type and patterns."""
        tags = []
        type_lower = col_type.lower() if col_type else ""
        
        # Type-based tags (primary categorization)
        if any(t in type_lower for t in ['datetime', 'timestamp', 'date', 'time']):
            tags.append("TEMPORAL")
        elif any(t in type_lower for t in ['int', 'long', 'short', 'byte']):
            tags.append("INTEGER")
        elif any(t in type_lower for t in ['real', 'double', 'float', 'decimal']):
            tags.append("DECIMAL")
        elif any(t in type_lower for t in ['bool', 'boolean']):
            tags.append("BOOLEAN")
        elif any(t in type_lower for t in ['guid', 'uuid']):
            tags.append("IDENTIFIER")
        elif any(t in type_lower for t in ['dynamic', 'json', 'object']):
            tags.append("STRUCTURED")
        elif any(t in type_lower for t in ['string', 'text', 'varchar', 'char']):
            tags.append("TEXT")
        
        # Add functional tags based on data characteristics
        if any(t in type_lower for t in ['datetime', 'timestamp']) and tags:
            tags.append("SORTABLE")
        if any(t in type_lower for t in ['int', 'long', 'real', 'decimal']) and tags:
            tags.append("AGGREGATABLE")
        
        # Pattern-based functional tags (conservative, based on common patterns)
        name_lower = col_name.lower()
        if 'id' in name_lower and any(t in tags for t in ["INTEGER", "TEXT", "IDENTIFIER"]):
            tags.append("KEY")
        
        return list(dict.fromkeys(tags))[:3]  # Remove duplicates and limit to 3 tags

    def _update_ai_context_cache(
        self, cache_key: str, schema_data: Dict[str, Any], ai_token: str = None
    ):
        """ai_context_cache removed; maintained for backward compatibility as no-op."""
        return


    def get_ai_context_for_tables(
        self, cluster_uri: str, database: str, tables: List[str]
    ) -> List[str]:
        """Get enhanced AI context tokens for tables with intelligent relevance scoring."""
        try:
            # Ensure schemas are discovered before getting context
            from .utils import SchemaManager
            schema_manager = SchemaManager(self)
            
            for table in tables:
                schema = self.get_schema(cluster_uri, database, table, enable_fallback=False)
                if not schema or not schema.get("columns"):
                    # Force schema discovery if missing
                    try:
                        import asyncio
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Create task if loop is running
                            asyncio.create_task(schema_manager.get_table_schema(cluster_uri, database, table, force_refresh=True))
                        else:
                            # Run synchronously if no loop
                            asyncio.run(schema_manager.get_table_schema(cluster_uri, database, table, force_refresh=True))
                        logger.debug(f"Auto-discovered schema for AI context: {table}")
                    except Exception as discovery_error:
                        logger.debug(f"Schema auto-discovery failed for {table}: {discovery_error}")
            
            # Use the enhanced context selector for intelligent filtering
            context_selector = ContextSelector()
            all_schemas = self._get_all_schemas_for_tables(cluster_uri, database, tables)
            
            # Select relevant context using intelligent scoring
            selected_tokens = context_selector.select_relevant_context("", all_schemas)
            
            logger.debug(
                "Generated %d context tokens using intelligent relevance scoring",
                len(selected_tokens)
            )
            return selected_tokens

        except Exception as e:
            logger.warning(f"Failed to get AI context for tables: {e}")
            return [
                f"{SPECIAL_TOKENS['TABLE_START']}{table}{SPECIAL_TOKENS['TABLE_END']}"
                f"{SPECIAL_TOKENS['SUMMARY_START']}context_error{SPECIAL_TOKENS['SUMMARY_END']}"
                for table in tables
            ]

    def _get_all_schemas_for_tables(self, cluster_uri: str, database: str, tables: List[str]) -> Dict[str, Dict]:
        """Get all schemas for specified tables."""
        all_schemas = {}
        normalized_cluster = self._normalize_cluster_uri(cluster_uri)
        
        for table in tables:
            # Get table data from new structure
            cluster_data = self.corpus.get("clusters", {}).get(normalized_cluster, {})
            db_data = cluster_data.get("databases", {}).get(database, {})
            table_data = db_data.get("tables", {}).get(table, {})
            schema_data = table_data.get("schema", {})
            
            if schema_data and schema_data.get("columns"):
                all_schemas[table] = schema_data
            else:
                # Create minimal schema entry for missing tables
                all_schemas[table] = {
                    "columns": {},
                    "ai_token": f"{SPECIAL_TOKENS['TABLE_START']}{table}{SPECIAL_TOKENS['TABLE_END']}"
                }
        
        return all_schemas

    @lru_cache(maxsize=128)
    def get_ai_context_for_query(
        self, cluster_uri: str, database: str, query: str, max_tokens: int = 3000
    ) -> str:
        """Get AI context for a query by extracting tables and building enhanced context."""
        try:
            from .utils import parse_query_entities
            
            entities = parse_query_entities(query)
            extracted_tables = entities.get("tables", [])
            logger.debug(f"Extracted tables for AI context: {extracted_tables}")
            
            if not extracted_tables:
                return ""
            
            context_tokens = self.get_ai_context_for_tables(
                cluster_uri, database, extracted_tables
            )
            
            full_context = " ".join(context_tokens)
            if len(full_context) > max_tokens:
                truncated_tokens = []
                current_length = 0
                for token in context_tokens:
                    if current_length + len(token) + 1 <= max_tokens:
                        truncated_tokens.append(token)
                        current_length += len(token) + 1
                    else:
                        break
                full_context = " ".join(truncated_tokens)
                logger.debug(f"Truncated AI context to {len(full_context)} characters")
            
            return full_context
            
        except Exception as e:
            logger.warning(f"Failed to get AI context for query: {e}")
            return ""

    def _compress_token(self, token: str, max_size: int) -> Optional[str]:
        """Compress token to fit within size limit."""
        if len(token) <= max_size:
            return token

        if max_size < 100:
            return None

        # Split token into parts
        parts = token.split("|")
        if len(parts) < 4:  # Need at least cluster, database, table, summary
            return None

        # Keep essential parts
        essential_parts = parts[:4]  # cluster, database, table, summary
        essential_size = sum(
            len(part) + 1 for part in essential_parts
        )  # +1 for separators

        if essential_size >= max_size:
            return None

        # Add columns until size limit
        remaining_size = max_size - essential_size
        column_parts = []

        for part in parts[4:]:  # Column parts
            if len(part) + 1 <= remaining_size:
                column_parts.append(part)
                remaining_size -= len(part) + 1
            else:
                break

        compressed_parts = essential_parts + column_parts

        # Add truncation indicator if needed
        if len(column_parts) < len(parts) - 4:
            truncated_count = len(parts) - 4 - len(column_parts)
            truncated_part = (
                f"{SPECIAL_TOKENS['COLUMN']}+{truncated_count}_more"
                f"{SPECIAL_TOKENS['DESCRIPTION']}truncated"
            )
            compressed_parts.append(truncated_part)

        return "|".join(compressed_parts)

    def _normalize_cluster_uri(self, cluster_input: str) -> str:
        """Normalize cluster URI to standard format."""
        if not cluster_input or not cluster_input.strip():
            raise ValueError("Cluster input cannot be empty")

        cluster_input = cluster_input.strip()

        # If already a full HTTPS URI
        if cluster_input.startswith("https://"):
            return cluster_input

        # If it's a full domain name
        if "." in cluster_input and not cluster_input.startswith("http"):
            return f"https://{cluster_input}"

        # If it's just a cluster name
        if re.match(r"^[a-zA-Z0-9\-_]+$", cluster_input):
            return f"https://{cluster_input}.kusto.windows.net"

        raise ValueError(f"Invalid cluster format: {cluster_input}")

    def _extract_cluster_name(self, cluster_uri: str) -> str:
        """Extract short cluster name from URI for tokens."""
        if cluster_uri.startswith("https://"):
            hostname = cluster_uri[8:].split(".")[0]
            return hostname
        return cluster_uri

    def _schedule_save(self):
        """Schedule background save."""
        if not self._save_scheduled:
            self._save_scheduled = True
            import threading

            threading.Thread(target=self._background_save, daemon=True).start()

    def _background_save(self):
        """Perform background save."""
        try:
            import time

            time.sleep(2.0)  # Wait to batch changes

            self._save_scheduled = False
            self.save_corpus()

        except Exception as e:
            logger.error(f"Background save failed: {e}")
            self._save_scheduled = False

    def _compress_schema_data(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply dynamic compression to schema data to reduce memory usage."""
        if not isinstance(schema_data, dict):
            return schema_data
        
        compressed = schema_data.copy()
        
        # Compress sample values - keep only unique, non-empty values (max 2)
        if "columns" in compressed and isinstance(compressed["columns"], dict):
            for col_name, col_data in compressed["columns"].items():
                if isinstance(col_data, dict) and "sample_values" in col_data:
                    samples = col_data["sample_values"]
                    if isinstance(samples, list):
                        # Remove duplicates while preserving order, filter empty values
                        unique_samples = []
                        seen = set()
                        for sample in samples:
                            if sample and str(sample).strip() and str(sample) not in seen:
                                unique_samples.append(sample)
                                seen.add(str(sample))
                                if len(unique_samples) >= 2:
                                    break
                        col_data["sample_values"] = unique_samples
        
        return compressed

    def _should_compress_cluster_data(self, cluster_uri: str) -> bool:
        """Check if cluster data exceeds memory limits and needs compression."""
        try:
            cluster_data = self.corpus.get("clusters", {}).get(cluster_uri, {})
            cluster_json = json.dumps(cluster_data)
            cluster_size = len(cluster_json.encode('utf-8'))
            
            return cluster_size > self._memory_size_limit
        except Exception:
            return False

    def _compress_cluster_data(self, cluster_uri: str):
        """Apply compression to cluster data to reduce memory usage."""
        try:
            cluster_data = self.corpus.get("clusters", {}).get(cluster_uri, {})
            if not cluster_data:
                return
            
            # Remove oldest successful queries to maintain memory limits
            for db_name, db_data in cluster_data.get("databases", {}).items():
                for table_name, table_data in db_data.get("tables", {}).items():
                    if isinstance(table_data, dict) and "successful_queries" in table_data:
                        queries = table_data["successful_queries"]
                        if isinstance(queries, list) and len(queries) > 5:
                            # Keep only 5 most recent queries
                            table_data["successful_queries"] = queries[-5:]
            
            # Remove old learning results
            if "learning_results" in cluster_data:
                learning_results = cluster_data["learning_results"]
                if isinstance(learning_results, list) and len(learning_results) > 25:
                    cluster_data["learning_results"] = learning_results[-25:]
            
            logger.debug(f"Applied compression to cluster {cluster_uri}")
            
        except Exception as e:
            logger.warning(f"Failed to compress cluster data for {cluster_uri}: {e}")

    def save_corpus(self):
        """Save corpus to disk with thread safety."""
        with _memory_lock:
            try:
                self.corpus["last_updated"] = datetime.now().isoformat()
                self.memory_path.parent.mkdir(parents=True, exist_ok=True)

                # Atomic save
                temp_path = self.memory_path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(self.corpus, f, indent=2, ensure_ascii=False, default=str)

                temp_path.replace(self.memory_path)
                logger.debug(f"Saved unified memory to {self.memory_path}")

            except Exception as e:
                logger.error(f"Failed to save memory: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        try:
            corpus = self.corpus or {}
            clusters = corpus.get("clusters", {})
            clusters_count = len(clusters)

            total_schemas = 0
            total_queries = 0
            total_tables = 0
            
            for cluster_data in clusters.values():
                if not isinstance(cluster_data, dict):
                    continue
                    
                # Count successful queries at cluster level
                total_queries += len(cluster_data.get("successful_queries", []))
                
                # Count schemas and tables
                dbs = cluster_data.get("databases", {})
                for db_data in dbs.values():
                    if not isinstance(db_data, dict):
                        continue
                        
                    tables = db_data.get("tables", {})
                    total_tables += len(tables)
                    
                    for table_data in tables.values():
                        if isinstance(table_data, dict) and table_data.get("schema", {}).get("columns"):
                            total_schemas += 1
                        # Count table-level successful queries
                        total_queries += len(table_data.get("successful_queries", []))

            # Calculate memory size
            import json
            corpus_json = json.dumps(corpus, default=str)
            memory_size_kb = len(corpus_json.encode('utf-8')) / 1024

            return {
                "clusters_count": clusters_count,
                "total_schemas": total_schemas,
                "total_queries": total_queries,
                "total_tables": total_tables,
                "memory_size_kb": round(memory_size_kb, 2),
                "last_updated": corpus.get("last_updated"),
                "version": corpus.get("version", "3.0")
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {
                "error": str(e),
                "clusters_count": 0,
                "total_schemas": 0,
                "total_queries": 0
            }

    def clear_memory(self) -> bool:
        """Clear all memory."""
        try:
            self.corpus = self._create_empty_corpus()
            self.save_corpus()
            logger.info("Memory cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            return False



# Global instance
_memory_manager = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def get_knowledge_corpus():
    """Compatibility adapter expected by legacy tests: return an object with memory_manager."""
    class KnowledgeCorpus:
        def __init__(self):
            self.memory_manager = get_memory_manager()
    return KnowledgeCorpus()


# Convenience functions for backward compatibility and easy integration
def get_context_for_tables(
    cluster_uri: str,
    database: str,
    tables: List[str],
    memory_path: Optional[str] = None,
) -> List[str]:
    """Get AI context tokens for tables."""
    memory = get_memory_manager()
    return memory.get_ai_context_for_tables(cluster_uri, database, tables)


def ensure_table_in_memory(cluster_uri: str, database: str, table: str) -> bool:
    """Ensure table schema exists in memory (simplified check)."""
    try:
        memory = get_memory_manager()
        schema = memory.get_schema(cluster_uri, database, table)
        return bool(schema)
    except Exception as e:
        logger.warning(f"Failed to check table in memory: {e}")
        return False


def get_table_ai_token(cluster_uri: str, database: str, table: str) -> Optional[str]:
    """Get AI token for a specific table."""
    try:
        memory = get_memory_manager()
        normalized_cluster = memory._normalize_cluster_uri(cluster_uri)

        # Prefer token from stored table entry
        cluster_data = memory.corpus.get("clusters", {}).get(normalized_cluster, {})
        try:
            return cluster_data.get("databases", {}).get(database, {}).get("tables", {}).get(table, {}).get("ai_token")
        except Exception:
            return None
    except Exception as e:
        logger.warning(f"Failed to get AI token: {e}")
        return None


def store_pattern_analysis(
    cluster_uri: str, database: str, table: str, pattern_data: Dict[str, Any]
):
    """Store pattern analysis data (simplified for compatibility)."""
    try:
        memory = get_memory_manager()
        # Store as part of schema data
        schema_data = memory.get_schema(cluster_uri, database, table)
        if schema_data:
            schema_data["pattern_analysis"] = pattern_data
            memory.store_schema(cluster_uri, database, table, schema_data)
    except Exception as e:
        logger.warning(f"Failed to store pattern analysis: {e}")


def update_memory_after_query(
    cluster_uri: str,
    database: str,
    tables: List[str],
    cluster_memory_path: Optional[str] = None,
):
    """Update memory after query execution (placeholder for compatibility)."""
    logger.debug(f"Memory update triggered for {len(tables)} tables")
    # This function is now handled by the 2-step flow in execute_kql.py




def clear_memory_cache() -> bool:
    """Clear memory cache."""
    try:
        memory = get_memory_manager()
        return memory.clear_memory()
    except Exception as e:
        logger.error(f"Failed to clear memory cache: {e}")
        return False


# Simplified utility functions for backward compatibility
def get_dynamic_query_builder():
    """Minimal query builder fallback."""
    class SimpleQueryBuilder:
        def build_from_natural_language(self, query: str, context: dict, session_context=None):
            from .utils import parse_query_entities
            entities = parse_query_entities(query)
            
            class SimpleQuery:
                def __init__(self, kql_query: str):
                    self.kql_query = kql_query
                def to_kql(self):
                    return self.kql_query
                def get_telemetry(self):
                    return {"confidence_score": 0.8, "method": "simple_builder"}
            
            # Use first table or default
            table = entities["tables"][0] if entities["tables"] else "StormEvents"
            return SimpleQuery(f"{table} | take 10")
    
    return SimpleQueryBuilder()

def get_telemetry_collector():
    """Minimal telemetry collector."""
    class SimpleCollector:
        def track_query(self, query_id: str, query_type: str):
            class Tracker:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return Tracker()
    return SimpleCollector()

async def kql_schema_memory_tool(natural_language_query: str = None, session_id: str = None):
    """Enhanced schema memory tool with forced discovery when missing."""
    from .utils import parse_query_entities, SchemaManager
    
    entities = parse_query_entities(natural_language_query or "")
    cluster, database = entities["cluster"], entities["database"]
    tables = entities["tables"]
    
    if not cluster or not database:
        return {"error": "Missing cluster or database specification"}
    
    mm = get_memory_manager()
    
    # List tables request
    if any(keyword in (natural_language_query or "").lower()
           for keyword in ["list tables", "show tables", "what tables"]):
        try:
            schema_manager = SchemaManager()
            db_schema = await schema_manager.get_database_schema(cluster, database)
            
            # Force discovery even without tables - use db schema
            if "list tables" in (natural_language_query or "").lower():
                mm.store_database_schema(cluster, database, db_schema)
            
            return {"schemas": [{"cluster": cluster, "database": database,
                               "tables_available": db_schema.get("tables", [])}]}
        except Exception as e:
            return {"error": f"Failed to list tables: {e}"}
    
    # Table schema request
    table = tables[0] if tables else None
    if not table:
        return {"error": "No table specified"}
    
    schema = mm.get_schema(cluster, database, table)
    
    # Force schema discovery if missing, even on tool call
    if not schema or not schema.get("columns"):
        try:
            schema_manager = SchemaManager()
            schema = await schema_manager.get_table_schema(cluster, database, table, force_refresh=True)
            if schema:
                # Always store post-discovery to ensure persistence
                mm.store_schema(cluster, database, table, schema)
                logger.info(f"Forced schema discovery and storage for {database}.{table}")
        except Exception as e:
            logger.warning(f"Forced schema discovery failed for {table}: {e}")
            return {"error": f"Schema discovery failed: {e}"}
    
    # Extract columns efficiently
    columns = []
    if isinstance(schema, dict):
        cols = schema.get("columns") or schema.get("column_types", {})
        if isinstance(cols, dict):
            columns = list(cols.keys())
        elif isinstance(cols, list):
            columns = [c if isinstance(c, str) else c.get("name", "") for c in cols]
    
    return {"schemas": [{"cluster": cluster, "database": database, "table": table, "columns": columns}]}

async def get_session_context(session_id: str):
    """Minimal session context."""
    return {"session_id": session_id, "conversation_state": "active"}


def get_memory_stats() -> Dict[str, int]:
    """
    Return basic memory statistics for tests and diagnostics.

    Keys provided:
    - clusters_count: number of clusters recorded in the corpus
    - total_schemas: total number of stored table schemas across all clusters/databases
    - total_queries: total number of stored query execution history entries
    """
    try:
        mm = get_memory_manager()
        corpus = getattr(mm, "corpus", {}) or {}
        clusters = corpus.get("clusters", {}) if isinstance(corpus, dict) else {}
        clusters_count = len(clusters)

        total_schemas = 0
        for cluster_data in clusters.values():
            if not isinstance(cluster_data, dict):
                continue
            dbs = cluster_data.get("databases", {}) or {}
            for db_data in dbs.values():
                if not isinstance(db_data, dict):
                    continue
                tables = db_data.get("tables", {}) or {}
                total_schemas += len(tables)

        # Determine total stored successful queries across clusters.
        # The legacy 'query_execution_history' was removed. Derive the total
        # from the per-cluster 'successful_queries' lists (if populated).
        total_queries = 0
        try:
            for cluster_data in clusters.values():
                if not isinstance(cluster_data, dict):
                    continue
                total_queries += len(cluster_data.get("successful_queries", []) or [])
        except Exception:
            total_queries = 0

        return {
            "clusters_count": clusters_count,
            "total_schemas": total_schemas,
            "total_queries": total_queries,
        }
    except Exception as e:
        logger.warning(f"Failed to compute memory stats: {e}")
        return {"clusters_count": 0, "total_schemas": 0, "total_queries": 0}
