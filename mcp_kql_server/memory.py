"""
Unified Schema Memory System for MCP KQL Server (SQLite + CAG + TOON)

This module provides a high-performance memory system that:
- Uses SQLite for robust, zero-config storage of schemas and queries.
- Implements Context Augmented Generation (CAG) to load full schemas into LLM context.
- Uses TOON (Token-Oriented Object Notation) for compact schema representation.
- Supports Semantic Search (using sentence-transformers) for Few-Shot prompting.

Author: Arjun Trivedi
"""

import sqlite3
import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)

# TOON Type Mapping for compression
TOON_TYPE_MAP = {
    'string': 's',
    'int': 'i',
    'long': 'l',
    'real': 'r',
    'double': 'd',
    'decimal': 'd',
    'datetime': 'dt',
    'timespan': 'ts',
    'bool': 'b',
    'boolean': 'b',
    'dynamic': 'dyn',
    'guid': 'g'
}

class SemanticSearch:
    """Handles embedding generation and similarity search with optimized loading."""
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, model_name='all-MiniLM-L6-v2'):
        """Singleton pattern to share model across instances."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False  # pylint: disable=protected-access
            return cls._instance

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if getattr(self, '_initialized', False):
            return
        self.model_name = model_name
        self.model = None
        self._loading = False
        self._load_lock = threading.Lock()
        self._initialized = True

    def preload(self):
        """Preload model in background thread for faster first query."""
        if self.model is None and not self._loading and HAS_SENTENCE_TRANSFORMERS:
            threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        """Thread-safe lazy load of the model."""
        with self._load_lock:
            if self.model is None and HAS_SENTENCE_TRANSFORMERS:
                self._loading = True
                try:
                    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
                    if SentenceTransformer:
                        self.model = SentenceTransformer(self.model_name)
                    logger.info("Loaded Semantic Search model: %s", self.model_name)
                except Exception as e:
                    logger.warning("Failed to load SentenceTransformer: %s", e)
                finally:
                    self._loading = False

    def encode(self, text: str) -> Optional[bytes]:
        """Generate embedding for text and return as bytes."""
        # Lazy load model if needed
        self._load_model()

        if self.model is None:
            return None
        try:
            embedding = self.model.encode(text)
            # Ensure we have a numpy array (handle Tensor or list output)
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            return embedding.astype(np.float32).tobytes()
        except Exception as e:
            logger.error("Encoding failed: %s", e)
            return None

@dataclass
class ValidationResult:
    """Result of query validation against schema."""
    is_valid: bool
    validated_query: str
    errors: List[str]

class MemoryManager:
    """
    SQLite-backed Memory Manager for KQL Schemas and Queries.
    Implements CAG (Context Augmented Generation) with TOON formatting.
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = self._get_db_path(db_path)
        self.semantic_search = SemanticSearch()
        self._lock = threading.RLock()
        self._schema_cache: Dict[str, Any] = {}  # Initialize schema cache in __init__
        self._init_db()

    @property
    def memory_path(self) -> Path:
        """Expose db_path as memory_path for compatibility."""
        return self.db_path

    def _get_db_path(self, custom_path: Optional[str] = None) -> Path:
        """Determine the SQLite database path."""
        if custom_path:
            return Path(custom_path)

        if os.name == "nt":
            base_dir = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "KQL_MCP"
        else:
            base_dir = Path.home() / ".local" / "share" / "KQL_MCP"

        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / "kql_memory.db"

    def _init_db(self):
        """Initialize SQLite database schema."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")

            # Schema table: Stores table definitions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schemas (
                    cluster TEXT,
                    database TEXT,
                    table_name TEXT,
                    columns_json TEXT,
                    embedding BLOB,
                    description TEXT,
                    last_updated TIMESTAMP,
                    PRIMARY KEY (cluster, database, table_name)
                )
            """)

            # Queries table: Stores successful queries with embeddings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster TEXT,
                    database TEXT,
                    query TEXT,
                    description TEXT,
                    embedding BLOB,
                    timestamp TIMESTAMP,
                    execution_time_ms REAL
                )
            """)

            # Join Hints table: Stores discovered relationships
            conn.execute("""
                CREATE TABLE IF NOT EXISTS join_hints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table1 TEXT,
                    table2 TEXT,
                    join_condition TEXT,
                    confidence REAL,
                    last_used TIMESTAMP,
                    UNIQUE(table1, table2, join_condition)
                )
            """)

            # Query Cache table: Stores result hashes
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash TEXT PRIMARY KEY,
                    result_json TEXT,
                    timestamp TIMESTAMP,
                    row_count INTEGER
                )
            """)

            # Indexes for faster lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_schemas_db ON schemas(cluster, database)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_db ON queries(cluster, database)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_join_hints ON join_hints(table1, table2)")

            # Learning Events table: Stores execution learning data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    execution_type TEXT,
                    result_json TEXT,
                    timestamp TIMESTAMP,
                    execution_time_ms REAL
                )
            """)

    def store_schema(self, cluster: str, database: str, table: str,
                     schema: Dict[str, Any], description: Optional[str] = None):
        """Store or update a table schema with embedding and description."""
        columns = schema.get("columns", {})

        # Normalize columns to dict format if it's a list
        if isinstance(columns, list):
            normalized_cols = {}
            for col in columns:
                if isinstance(col, dict):
                    name = col.get("name") or col.get("column")
                    if name:
                        normalized_cols[name] = col
                elif isinstance(col, str):
                    normalized_cols[col] = {"data_type": "string"}
            columns = normalized_cols

        # Generate embedding for table (name + column names)
        col_names = " ".join(columns.keys())
        embedding = self.semantic_search.encode(f"Table {table} contains columns: {col_names}")

        with self._lock, sqlite3.connect(self.db_path) as conn:
            # Check if columns exist (migration for existing DBs)
            try:
                conn.execute("ALTER TABLE schemas ADD COLUMN embedding BLOB")
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute("ALTER TABLE schemas ADD COLUMN description TEXT")
            except sqlite3.OperationalError:
                pass

            # If description is not provided, try to preserve existing one
            if description is None:
                cursor = conn.execute(
                    "SELECT description FROM schemas WHERE cluster=? AND database=? AND table_name=?",
                    (cluster, database, table)
                )
                row = cursor.fetchone()
                if row:
                    description = row[0]

            conn.execute("""
                INSERT OR REPLACE INTO schemas
                (cluster, database, table_name, columns_json, embedding, description, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (cluster, database, table, json.dumps(columns),
                  embedding, description, datetime.now().isoformat()))

        logger.debug("Stored schema for %s in %s", table, database)

    def add_successful_query(self, cluster: str, database: str, query: str,
                             description: str, execution_time_ms: float = 0.0):
        """Store a successful query with its description and embedding."""
        embedding = self.semantic_search.encode(f"{description} {query}")

        with self._lock, sqlite3.connect(self.db_path) as conn:
            # Check for new column (migration)
            try:
                conn.execute("ALTER TABLE queries ADD COLUMN execution_time_ms REAL")
            except sqlite3.OperationalError:
                pass

            conn.execute("""
                INSERT INTO queries
                (cluster, database, query, description, embedding, timestamp, execution_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (cluster, database, query, description, embedding,
                  datetime.now().isoformat(), execution_time_ms))

    def add_global_successful_query(self, cluster: str, database: str, query: str,
                                    description: str, execution_time_ms: float = 0.0):
        """Store a successful query globally (alias for add_successful_query for now)."""
        self.add_successful_query(cluster, database, query, description, execution_time_ms)

    def store_learning_result(self, query: str, result_data: Dict[str, Any],
                              execution_type: str, execution_time_ms: float = 0.0):
        """Store learning result from query execution."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            # Check for new column (migration)
            try:
                conn.execute("ALTER TABLE learning_events ADD COLUMN execution_time_ms REAL")
            except sqlite3.OperationalError:
                pass

            conn.execute("""
                INSERT INTO learning_events
                (query, execution_type, result_json, timestamp, execution_time_ms)
                VALUES (?, ?, ?, ?, ?)
            """, (query, execution_type, json.dumps(result_data),
                  datetime.now().isoformat(), execution_time_ms))

    def find_relevant_tables(self, cluster: str, database: str,
                             query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find tables semantically related to the query."""
        query_embedding = self.semantic_search.encode(query)
        if query_embedding is None:
            return []

        query_vector = np.frombuffer(query_embedding, dtype=np.float32)
        results = []

        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT table_name, columns_json, embedding FROM schemas WHERE cluster = ? AND database = ?",
                (cluster, database)
            )

            for row in cursor:
                if row[2]: # If embedding exists
                    tbl_vec = np.frombuffer(row[2], dtype=np.float32)
                    norm_q = np.linalg.norm(query_vector)
                    norm_t = np.linalg.norm(tbl_vec)
                    if norm_q > 0 and norm_t > 0:
                        similarity = np.dot(query_vector, tbl_vec) / (norm_q * norm_t)
                        results.append({
                            "table": row[0],
                            "columns": json.loads(row[1]),
                            "score": float(similarity)
                        })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def find_similar_queries(self, cluster: str, database: str,
                               query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find similar past queries using vector search."""
        query_embedding = self.semantic_search.encode(query)
        if query_embedding is None:
            return []

        # Rename to query_vector to avoid potential linter confusion with previous name
        query_vector = np.frombuffer(query_embedding, dtype=np.float32)
        results = []

        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT query, description, embedding FROM queries WHERE cluster = ? AND database = ?",
                (cluster, database)
            )

            for row in cursor:
                if row[2]:
                    row_vector = np.frombuffer(row[2], dtype=np.float32)
                    norm_q1 = np.linalg.norm(query_vector)
                    norm_q2 = np.linalg.norm(row_vector)
                    if norm_q1 > 0 and norm_q2 > 0:
                        similarity = np.dot(query_vector, row_vector) / (norm_q1 * norm_q2)
                        results.append({
                            "query": row[0],
                            "description": row[1],
                            "score": float(similarity)
                        })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def cache_query_result(self, query: str, result_json: str, row_count: int):
        """Cache query result."""
        import hashlib
        query_hash = hashlib.sha256(query.encode()).hexdigest()

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO query_cache (query_hash, result_json, timestamp, row_count)
                VALUES (?, ?, ?, ?)
            """, (query_hash, result_json, datetime.now().isoformat(), row_count))

    def get_cached_result(self, query: str, ttl_seconds: int = 300) -> Optional[str]:
        """Get cached result if valid."""
        import hashlib
        query_hash = hashlib.sha256(query.encode()).hexdigest()

        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT result_json, timestamp FROM query_cache WHERE query_hash = ?",
                (query_hash,)
            )
            row = cursor.fetchone()
            if row:
                cached_time = datetime.fromisoformat(row[1])
                if (datetime.now() - cached_time).total_seconds() < ttl_seconds:
                    return row[0]
        return None

    def store_join_hint(self, table1: str, table2: str, condition: str):
        """Store a discovered join relationship."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO join_hints (table1, table2, join_condition, confidence, last_used)
                VALUES (?, ?, ?, 1.0, ?)
            """, (table1, table2, condition, datetime.now().isoformat()))

    def get_join_hints(self, tables: List[str]) -> List[str]:
        """Get join hints relevant to the provided tables."""
        if not tables:
            return []

        placeholders = ','.join(['?'] * len(tables))
        hints = []

        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                SELECT table1, table2, join_condition FROM join_hints
                WHERE table1 IN ({placeholders}) OR table2 IN ({placeholders})
            """, tables + tables)

            for row in cursor:
                hints.append(f"{row[0]} joins with {row[1]} on {row[2]}")

        return list(set(hints))

    def _get_database_schema(self, cluster: str, database: str) -> List[Dict[str, Any]]:
        """Get schema from SQLite with caching."""
        cache_key = f"db_schema_{cluster}_{database}"
        # Simple in-memory cache check
        if hasattr(self, '_schema_cache') and cache_key in self._schema_cache:
            cached = self._schema_cache[cache_key]
            if (datetime.now() - cached['ts']).seconds < 300:  # 5 min TTL
                return cached['data']

        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT table_name, columns_json FROM schemas WHERE cluster = ? AND database = ?",
                (cluster, database)
            )
            schemas = [{"table": row[0], "columns": json.loads(row[1])} for row in cursor]

        # Cache result
        if not hasattr(self, '_schema_cache'):
            self._schema_cache = {}
        self._schema_cache[cache_key] = {'data': schemas, 'ts': datetime.now()}
        return schemas

    def get_relevant_context(self, cluster: str, database: str, user_query: str, max_tables: int = 20) -> str:
        """
        Optimized CAG: Get schema + similar queries + join hints in TOON format.
        Limited to max_tables to prevent token overflow.
        """
        # 1. Get schemas (limited)
        schemas = self._get_database_schema(cluster, database)[:max_tables]
        table_names = [s["table"] for s in schemas]

        # 2. Get similar queries (parallel-safe)
        similar_queries = self.find_similar_queries(cluster, database, user_query, limit=3)

        # 3. Get join hints
        join_hints = self.get_join_hints(table_names) if table_names else []

        # 4. Format as compact TOON
        return self._to_toon(schemas, similar_queries, join_hints)

    def _to_toon(self, schemas: List[Dict], similar_queries: List[Dict],
                 join_hints: Optional[List[str]] = None) -> str:
        """Optimized TOON formatting with size limits."""
        lines = ["<CAG_CONTEXT>"]

        # Compact syntax guidance
        lines.append("# KQL Rules: Use != (not ! =), !contains, !in, !has. No spaces in negation.")

        # Schema Section (compact)
        if schemas:
            lines.append("# Schema (TOON)")
            for schema in schemas:
                table = schema["table"]
                cols = []
                for col_name, col_def in schema["columns"].items():
                    # Handle different column definition formats
                    col_type = "string"
                    if isinstance(col_def, dict):
                        col_type = col_def.get("data_type") or col_def.get("type") or "string"
                    elif isinstance(col_def, str): # simple key-value
                        col_type = col_def

                    # Map to short type
                    short_type = TOON_TYPE_MAP.get(col_type.lower(), 's')
                    cols.append(f"{col_name}:{short_type}")

                lines.append(f"{table}({', '.join(cols)})")
        else:
            lines.append("# No Schema Found (Run queries to discover)")

        # Join Hints Section
        if join_hints:
            lines.append("\n# Join Hints")
            for hint in join_hints:
                lines.append(f"// {hint}")

        # Few-Shot Section
        if similar_queries:
            lines.append("\n# Similar Queries")
            for q in similar_queries:
                lines.append(f"// {q['description']}")
                lines.append(q['query'])

        lines.append("</CAG_CONTEXT>")
        return "\n".join(lines)

    def clear_memory(self) -> bool:
        """Clear all data from the database."""
        try:
            with self._lock, sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM schemas")
                conn.execute("DELETE FROM queries")
                conn.execute("DELETE FROM learning_events")
            return True
        except Exception as e:
            logger.error("Failed to clear memory: %s", e)
            return False

    # Use centralized normalize_cluster_uri from utils.py
    # Import at method level to avoid circular imports
    def normalize_cluster_uri(self, uri: str) -> str:
        """Normalize cluster URI - delegates to utils."""
        from .utils import normalize_cluster_uri as _normalize
        return _normalize(uri) if uri else ""

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory database."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            schema_count = conn.execute("SELECT COUNT(*) FROM schemas").fetchone()[0]
            query_count = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
            learning_count = 0
            try:
                learning_count = conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0]
            except sqlite3.OperationalError:
                pass
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "schema_count": schema_count,
            "query_count": query_count,
            "learning_count": learning_count,
            "db_size_bytes": db_size,
            "db_path": str(self.db_path)
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics (execution time, success rate)."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            # Average execution time
            try:
                avg_time = conn.execute(
                    "SELECT AVG(execution_time_ms) FROM queries WHERE execution_time_ms > 0"
                ).fetchone()[0]
                avg_time = avg_time if avg_time is not None else 0.0
            except sqlite3.OperationalError:
                avg_time = 0.0

            # Total queries
            query_count = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]

        return {
            "average_execution_time_ms": round(avg_time, 2),
            "total_successful_queries": query_count
        }

    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent successful queries."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT query, description, cluster, database, timestamp FROM queries ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            return [
                {
                    "query": row[0],
                    "description": row[1],
                    "cluster": row[2],
                    "database": row[3],
                    "timestamp": row[4],
                    "result_metadata": {"success": True} # Mock for compatibility
                }
                for row in cursor.fetchall()
            ]

    def get_ai_context_for_tables(self, cluster: str, database: str, tables: List[str]) -> str:
        """Wrapper for get_relevant_context to support list of tables."""
        # In CAG, we load the full database schema anyway, but we can filter if needed.
        # For now, we'll just use the first table name as a hint or just pass a generic query
        # if no specific query is provided.
        # Actually, get_relevant_context expects a user_query to find similar queries.
        # If we just want context for tables, we can construct a dummy query or just return schema.

        # If tables is a list, join them
        table_str = ", ".join(tables)
        dummy_query = f"Querying tables: {table_str}"
        return self.get_relevant_context(cluster, database, dummy_query)

    def validate_query(self, query: str, cluster: str, database: str) -> ValidationResult:  # pylint: disable=unused-argument
        """
        Validate query against schema.
        Returns an object with is_valid, validated_query, errors.

        Args:
            query: The KQL query to validate
            cluster: Cluster URL (reserved for future use)
            database: Database name (reserved for future use)
        """
        # Simple validation stub
        return ValidationResult(
            is_valid=True,
            validated_query=query,
            errors=[]
        )

    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """Get session data (stub for compatibility)."""
        return {
            "sessions": {},
            "active_session": session_id
        }

    def get_database_schema(self, cluster: str, database: str) -> Dict[str, Any]:
        """Get database schema in the format expected by utils.py."""
        schemas = self._get_database_schema(cluster, database)
        table_names = [s["table"] for s in schemas]
        return {
            "database_name": database,
            "tables": table_names,
            "cluster": cluster
        }

    @property
    def corpus(self) -> Dict[str, Any]:
        """Compatibility property for legacy corpus access."""
        # Return a dummy dict structure to prevent crashes in legacy code
        # that hasn't been fully migrated yet.
        return {"clusters": {}}

    def save_corpus(self):
        """Compatibility method for legacy save_corpus calls (no-op)."""
        # This is intentionally empty for backwards compatibility
        return None

# Global instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get the singleton MemoryManager instance."""
    global _memory_manager  # pylint: disable=global-statement
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager

def get_kql_operator_syntax_guidance() -> str:
    """
    Get KQL operator syntax guidance for AI query generation.
    """
    return """
=== KQL GENERATION RULES (STRICT) ===
1. SCHEMA COMPLIANCE:
   - You MUST ONLY use columns that explicitly appear in the provided schema.
   - Do NOT hallucinate column names (e.g., do not assume 'EntityType', 'Target', 'Source' exist unless shown).
   - If a column is missing, use 'find' or 'search' instead of specific column references, or ask the user to refresh schema.

2. OPERATOR SYNTAX (CRITICAL):
   - Negation: Use '!=' (not '! ='), '!contains', '!in', '!has'. NO SPACES in negation operators.

   ✓ CORRECT Negation Syntax:
   - where Status != 'Active' (no space between ! and =)
   - where Name !contains 'test' (no space between ! and contains)
   - where Category !in ('A', 'B') (no space between ! and in)
   - where Title !has 'error' (no space between ! and has)

   ✗ WRONG Negation Syntax (DO NOT USE):
   - where Status ! = 'Active' (space between ! and =)
   - where Name ! contains 'test' (space between ! and contains)
   - where Category ! in ('A', 'B') (space between ! and in)
   - where Category !has_any ('A', 'B') (!has_any does not exist)

   List Operations:
   - Use 'in' for membership: where RuleName in ('Rule1', 'Rule2')
   - Use '!in' for exclusion: where RuleName !in ('Rule1', 'Rule2')
   - NEVER use '!has_any': !has_any does not exist in KQL

   Alternative Negation (using 'not' keyword):
   - where not (Status == 'Active')
   - where not (Name contains 'test')

   String Operators:
   - has: whole word/term matching (e.g., 'error' matches 'error log' but not 'errors')
   - contains: substring matching (e.g., 'test' matches 'testing')
   - startswith: prefix matching
   - endswith: suffix matching
   - All can be negated with ! prefix (NO SPACE): !has, !contains, !startswith, !endswith

3. BEST PRACTICES:
   - Always verify column names against the schema before generating the query.
   - Use 'take 10' for initial exploration if unsure about data volume.
   - Prefer 'where Column has "Value"' over 'where Column == "Value"' for text search unless exact match is required.
"""
