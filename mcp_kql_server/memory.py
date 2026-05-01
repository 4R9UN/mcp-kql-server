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
import re
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

RANKING_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "get",
    "how", "i", "in", "into", "is", "it", "last", "latest", "list", "me",
    "most", "of", "on", "or", "recent", "show", "table", "that", "the",
    "their", "them", "these", "those", "to", "top", "what", "which", "with"
}

RANKING_TOKEN_SYNONYMS = {
    "user": {"user", "users", "account", "accounts", "principal", "identity", "upn"},
    "signin": {"signin", "signins", "login", "logon", "auth", "authentication"},
    "alert": {"alert", "alerts", "severity", "incident"},
    "device": {"device", "devices", "host", "hosts", "computer", "machine"},
    "process": {"process", "processes", "command", "cmd"},
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
                except (OSError, RuntimeError, ValueError) as e:
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
        except (RuntimeError, ValueError, TypeError) as e:
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

            # Query Cache table: Stores result hashes with configurable TTL
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash TEXT PRIMARY KEY,
                    result_json TEXT,
                    timestamp TIMESTAMP,
                    row_count INTEGER,
                    query_type TEXT DEFAULT 'default',
                    ttl_seconds INTEGER DEFAULT 300,
                    cluster TEXT,
                    database TEXT
                )
            """)

            # Multi-cluster table registry
            conn.execute("""
                CREATE TABLE IF NOT EXISTS table_locations (
                    table_name TEXT,
                    cluster TEXT,
                    database TEXT,
                    last_seen TIMESTAMP,
                    PRIMARY KEY (table_name, cluster, database)
                )
            """)

            # Indexes for faster lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_schemas_db ON schemas(cluster, database)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_db ON queries(cluster, database)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_join_hints ON join_hints(table1, table2)")

            # Run schema migrations for existing databases
            self._migrate_schema(conn)

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

    def _migrate_schema(self, conn: sqlite3.Connection):
        """
        Migrate existing database schema to add new columns.
        This handles backwards compatibility for databases created before v2.1.1.
        """
        # Check for query_cache table columns and add if missing
        try:
            cursor = conn.execute("PRAGMA table_info(query_cache)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            # Add new columns to query_cache if they don't exist
            if 'query_type' not in existing_columns:
                conn.execute("ALTER TABLE query_cache ADD COLUMN query_type TEXT DEFAULT 'default'")
                logger.info("Migration: Added query_type column to query_cache")
            
            if 'ttl_seconds' not in existing_columns:
                conn.execute("ALTER TABLE query_cache ADD COLUMN ttl_seconds INTEGER DEFAULT 300")
                logger.info("Migration: Added ttl_seconds column to query_cache")
            
            if 'cluster' not in existing_columns:
                conn.execute("ALTER TABLE query_cache ADD COLUMN cluster TEXT")
                logger.info("Migration: Added cluster column to query_cache")
            
            if 'database' not in existing_columns:
                conn.execute("ALTER TABLE query_cache ADD COLUMN database TEXT")
                logger.info("Migration: Added database column to query_cache")
                
            conn.commit()
        except sqlite3.OperationalError as e:
            logger.debug("Migration check for query_cache: %s", e)

    def store_schema(self, cluster: str, database: str, table: str,
                     schema: Dict[str, Any], description: Optional[str] = None):
        """Store or update a table schema with embedding and description."""
        columns = schema.get("columns", {})
        if not columns and schema:
            # Accept either {"columns": {...}} or a raw {column_name: column_def} mapping.
            columns = schema

        normalized_cluster = self.normalize_cluster_uri(cluster)

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
                    (normalized_cluster, database, table)
                )
                row = cursor.fetchone()
                if row:
                    description = row[0]

            existing_row = conn.execute(
                """
                SELECT columns_json, description
                FROM schemas
                WHERE cluster=? AND database=? AND table_name=?
                """,
                (normalized_cluster, database, table)
            ).fetchone()

            existing_columns = json.loads(existing_row[0]) if existing_row and existing_row[0] else {}
            if not columns and existing_columns:
                # Avoid overwriting a rich cached schema with an empty placeholder.
                self.register_table_location(table, normalized_cluster, database)
                logger.debug("Skipping empty schema overwrite for %s in %s", table, database)
                return

            if existing_columns == columns and (description == (existing_row[1] if existing_row else description)):
                self.register_table_location(table, normalized_cluster, database)
                logger.debug("Schema already indexed for %s in %s, skipping reindex", table, database)
                return

        # Generate embedding only when the schema is genuinely changing.
        col_names = " ".join(columns.keys())
        embedding = self.semantic_search.encode(f"Table {table} contains columns: {col_names}")

        with self._lock, sqlite3.connect(self.db_path) as conn:

            conn.execute("""
                INSERT OR REPLACE INTO schemas
                (cluster, database, table_name, columns_json, embedding, description, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (normalized_cluster, database, table, json.dumps(columns),
                  embedding, description, datetime.now().isoformat()))

        cache_key = f"db_schema_{normalized_cluster}_{database}"
        if hasattr(self, '_schema_cache'):
            self._schema_cache.pop(cache_key, None)
        self.register_table_location(table, normalized_cluster, database)
        logger.debug("Stored schema for %s in %s", table, database)

    def add_successful_query(self, cluster: str, database: str, query: str,
                             description: str, execution_time_ms: float = 0.0):
        """Store a successful query with its description and embedding."""
        normalized_cluster = self.normalize_cluster_uri(cluster)
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
            """, (normalized_cluster, database, query, description, embedding,
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
        normalized_cluster = self.normalize_cluster_uri(cluster)
        query_embedding = self.semantic_search.encode(query)
        if query_embedding is None:
            return []

        query_vector = np.frombuffer(query_embedding, dtype=np.float32)
        results = []

        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT table_name, columns_json, embedding FROM schemas WHERE cluster = ? AND database = ?",
                (normalized_cluster, database)
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
        normalized_cluster = self.normalize_cluster_uri(cluster)
        query_embedding = self.semantic_search.encode(query)
        if query_embedding is None:
            return []

        # Rename to query_vector to avoid potential linter confusion with previous name
        query_vector = np.frombuffer(query_embedding, dtype=np.float32)
        results = []

        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT query, description, embedding FROM queries WHERE cluster = ? AND database = ?",
                (normalized_cluster, database)
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

    def _tokenize_ranking_text(self, text: str) -> List[str]:
        """Tokenize free text and identifiers for retrieval/ranking heuristics."""
        normalized = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text or "")
        tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', normalized.lower())
        return [token for token in tokens if len(token) > 1 and token not in RANKING_STOPWORDS]

    def _expand_query_tokens(self, tokens: List[str]) -> List[str]:
        """Expand query tokens with a few high-value domain synonyms."""
        expanded = set(tokens)
        for token in list(expanded):
            normalized = token[:-1] if token.endswith("s") else token
            expanded.add(normalized)
            for synonyms in RANKING_TOKEN_SYNONYMS.values():
                if token in synonyms or normalized in synonyms:
                    expanded.update(synonyms)
        return list(expanded)

    def _column_data_type(self, column_def: Any) -> str:
        """Get a normalized column data type string."""
        if isinstance(column_def, dict):
            return str(column_def.get("data_type") or column_def.get("type") or "string").lower()
        if isinstance(column_def, str):
            return column_def.lower()
        return "string"

    def rank_tables_for_query(
        self,
        cluster: str,
        database: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rank tables using hybrid lexical + semantic signals.

        This provides a more reliable fallback when embeddings are unavailable and
        keeps retrieval grounded in schema tokens and past successful queries.
        """
        schemas = self._get_database_schema(cluster, database)
        if not schemas:
            return []

        semantic_scores = {
            row["table"]: float(row.get("score", 0.0))
            for row in self.find_relevant_tables(cluster, database, query, limit=max(limit * 3, 8))
        }
        similar_queries = self.find_similar_queries(cluster, database, query, limit=5)
        query_tokens = set(self._expand_query_tokens(self._tokenize_ranking_text(query)))
        query_lower = query.lower()
        time_intent = any(
            marker in query_lower
            for marker in ("ago", "hour", "hours", "day", "days", "week", "weeks", "month",
                           "months", "today", "yesterday", "recent", "latest", "trend", "over time")
        )

        ranked_tables = []
        for schema in schemas:
            table_name = schema["table"]
            columns = schema.get("columns", {})
            table_tokens = set(self._tokenize_ranking_text(table_name))
            matched_columns = []
            datetime_columns = 0

            for column_name, column_def in columns.items():
                column_tokens = set(self._tokenize_ranking_text(column_name))
                overlap = query_tokens & column_tokens
                if overlap:
                    matched_columns.append(column_name)
                if self._column_data_type(column_def) == "datetime":
                    datetime_columns += 1

            direct_table_mention = table_name.lower() in query_lower
            semantic_score = semantic_scores.get(table_name, 0.0)
            lexical_overlap = len(query_tokens & table_tokens)
            column_overlap = len(matched_columns)
            history_boost = sum(
                1.0 for sq in similar_queries if table_name.lower() in sq.get("query", "").lower()
            )
            score = (
                (4.5 if direct_table_mention else 0.0) +
                lexical_overlap * 1.75 +
                min(column_overlap, 6) * 0.9 +
                semantic_score * 4.0 +
                min(history_boost, 2.0) * 0.8 +
                (0.75 if time_intent and datetime_columns else 0.0)
            )

            ranked_tables.append({
                "table": table_name,
                "columns": columns,
                "score": round(score, 4),
                "semantic_score": round(semantic_score, 4),
                "matched_columns": matched_columns[:8],
                "datetime_columns": datetime_columns,
            })

        ranked_tables.sort(
            key=lambda item: (item["score"], len(item["matched_columns"]), item["table"].lower()),
            reverse=True
        )
        return ranked_tables[:limit]

    def fingerprint_columns(
        self,
        nl_query: str,
        schema: Dict[str, Any],
        similar_queries: Optional[List[Dict[str, Any]]] = None,
        max_columns: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Rank the most relevant columns for an NL query.

        The output is intentionally compact so the generator can build smaller
        CAG prompts and more focused candidate queries.
        """
        columns = schema.get("columns", {}) or {}
        if not columns:
            return []

        query_tokens = set(self._expand_query_tokens(self._tokenize_ranking_text(nl_query)))
        query_lower = nl_query.lower()
        time_intent = any(
            marker in query_lower
            for marker in ("ago", "hour", "hours", "day", "days", "week", "weeks", "month",
                           "months", "today", "yesterday", "recent", "latest", "trend", "time")
        )
        aggregate_intent = any(
            marker in query_lower
            for marker in ("count", "how many", "total", "group", "per ", " by ", "top", "most", "trend")
        )

        history_hits: Dict[str, int] = {}
        for similar_query in similar_queries or []:
            for token in self._tokenize_ranking_text(similar_query.get("query", "")):
                history_hits[token] = history_hits.get(token, 0) + 1

        ranked_columns = []
        for column_name, column_def in columns.items():
            data_type = self._column_data_type(column_def)
            column_tokens = set(self._tokenize_ranking_text(column_name))
            overlap = query_tokens & column_tokens
            exact_match = column_name.lower() in query_lower
            history_score = history_hits.get(column_name.lower(), 0)
            score = (
                (5.0 if exact_match else 0.0) +
                len(overlap) * 1.8 +
                min(history_score, 3) * 0.75
            )

            if time_intent and (
                data_type == "datetime" or {"time", "date", "timestamp"} & column_tokens
            ):
                score += 2.5

            if aggregate_intent and data_type in {"string", "guid"}:
                score += 1.0

            # Schema-driven canonical-time boost (no hardcoded names).
            # A column qualifies as canonical-time when its name carries any
            # time token AND its declared data type is datetime.
            if data_type == "datetime" and {"time", "date", "timestamp"} & column_tokens:
                score += 1.5

            ranked_columns.append({
                "name": column_name,
                "data_type": data_type,
                "score": round(score, 4),
                "matched_terms": sorted(overlap),
            })

        ranked_columns.sort(
            key=lambda item: (item["score"], item["data_type"] == "datetime", item["name"].lower()),
            reverse=True
        )

        top_columns = ranked_columns[:max_columns]
        if not any(column["score"] > 0 for column in top_columns):
            return [
                {
                    "name": column_name,
                    "data_type": self._column_data_type(columns[column_name]),
                    "score": 0.0,
                    "matched_terms": [],
                }
                for column_name in list(columns.keys())[:max_columns]
            ]
        return top_columns

    def build_cag_bundle(
        self,
        cluster: str,
        database: str,
        user_query: str,
        max_tables: int = 3,
        max_columns: int = 12
    ) -> Dict[str, Any]:
        """
        Build compact CAG context for generation and ranking.

        Returns ranked tables, column fingerprints, similar queries, join hints,
        and a compact TOON context string.
        """
        similar_queries = self.find_similar_queries(cluster, database, user_query, limit=3)
        ranked_tables = self.rank_tables_for_query(cluster, database, user_query, limit=max_tables)

        compact_schemas = []
        enriched_tables = []
        for table_info in ranked_tables:
            fingerprinted_columns = self.fingerprint_columns(
                user_query,
                {"table": table_info["table"], "columns": table_info["columns"]},
                similar_queries=similar_queries,
                max_columns=max_columns,
            )
            selected_columns = {
                column["name"]: table_info["columns"][column["name"]]
                for column in fingerprinted_columns
                if column["name"] in table_info["columns"]
            }
            enriched_tables.append({
                **table_info,
                "fingerprinted_columns": fingerprinted_columns,
                "selected_columns": selected_columns,
            })
            compact_schemas.append({
                "table": table_info["table"],
                "columns": selected_columns,
            })

        join_hints = self.get_join_hints([table["table"] for table in ranked_tables]) if ranked_tables else []

        # Build a quick lookup of join keys (column names that appear in any
        # join hint touching this table). Schema-driven: we only retain names
        # that exist in the table's column dict, so no hardcoded keys leak in.
        def _extract_join_keys(table_name: str, columns: Dict[str, Any]) -> List[str]:
            keys: List[str] = []
            tname_lower = table_name.lower()
            for hint in join_hints or []:
                hint_text = hint if isinstance(hint, str) else str(hint.get("hint", ""))
                if tname_lower not in hint_text.lower():
                    continue
                # token extraction: any identifier in the hint that matches a real column
                for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", hint_text):
                    if token in columns and token not in keys:
                        keys.append(token)
            return keys[:6]

        # Decorate each enriched table with schema-driven hints so downstream
        # generators (heuristic OR future LLM) get the same compact context.
        for table_info in enriched_tables:
            cols = table_info.get("columns") or {}
            time_col = None
            for col_name, col_def in cols.items():
                dtype = col_def.get("data_type") if isinstance(col_def, dict) else col_def
                if str(dtype).lower() == "datetime":
                    time_col = col_name
                    break
            table_info["time_column"] = time_col
            table_info["join_keys"] = _extract_join_keys(table_info["table"], cols)

        return {
            "tables": enriched_tables,
            "similar_queries": similar_queries,
            "join_hints": join_hints,
            "context": self._to_toon(compact_schemas, similar_queries, join_hints),
        }

    def _build_cache_key(
        self,
        query: str,
        cluster: Optional[str] = None,
        database: Optional[str] = None,
        cache_namespace: str = "default",
    ) -> str:
        """Build a cache key scoped to query text and execution context."""
        import hashlib

        normalized_cluster = self.normalize_cluster_uri(cluster or "")
        payload = "||".join([
            cache_namespace or "default",
            normalized_cluster,
            database or "",
            query,
        ])
        return hashlib.sha256(payload.encode()).hexdigest()

    def cache_query_result(
        self,
        query: str,
        result_json: str,
        row_count: int,
        query_type: str = "default",
        ttl_seconds: int = 300,
        cluster: Optional[str] = None,
        database: Optional[str] = None,
        cache_namespace: str = "default",
    ):
        """
        Cache query result with configurable TTL per query type.
        
        Args:
            query: The KQL query string
            result_json: JSON-serialized query results
            row_count: Number of rows in result
            query_type: Type of query for TTL configuration (default, schema, aggregation, realtime)
            ttl_seconds: Time-to-live in seconds (default: 300)
            cluster: Cluster URL for multi-cluster tracking
            database: Database name for multi-cluster tracking
        
        Query Type TTL Guidelines:
            - 'schema': 3600s (1 hour) - table schemas change infrequently
            - 'aggregation': 600s (10 min) - aggregate queries can be cached longer
            - 'realtime': 60s (1 min) - time-sensitive queries
            - 'default': 300s (5 min) - standard caching
        """
        query_hash = self._build_cache_key(
            query,
            cluster=cluster,
            database=database,
            cache_namespace=cache_namespace,
        )

        # Apply query type TTL overrides
        ttl_map = {
            "schema": 3600,
            "aggregation": 600,
            "realtime": 60,
            "default": 300
        }
        effective_ttl = ttl_map.get(query_type, ttl_seconds)
        normalized_cluster = self.normalize_cluster_uri(cluster or "") if cluster else None

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO query_cache 
                (query_hash, result_json, timestamp, row_count, query_type, ttl_seconds, cluster, database)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (query_hash, result_json, datetime.now().isoformat(), row_count,
                  query_type, effective_ttl, normalized_cluster, database))

    def get_cached_result(
        self,
        query: str,
        ttl_seconds: Optional[int] = None,
        cluster: Optional[str] = None,
        database: Optional[str] = None,
        cache_namespace: str = "default",
    ) -> Optional[str]:
        """
        Get cached result if valid, respecting stored TTL.
        
        Args:
            query: The KQL query string
            ttl_seconds: Override TTL (if None, uses stored TTL from cache entry)
        
        Returns:
            Cached result JSON string, or None if not cached or expired
        """
        query_hash = self._build_cache_key(
            query,
            cluster=cluster,
            database=database,
            cache_namespace=cache_namespace,
        )

        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT result_json, timestamp, ttl_seconds FROM query_cache WHERE query_hash = ?",
                (query_hash,)
            )
            row = cursor.fetchone()
            if row:
                cached_time = datetime.fromisoformat(row[1])
                # Use provided TTL or stored TTL, default to 300
                effective_ttl = ttl_seconds if ttl_seconds is not None else (row[2] or 300)
                if (datetime.now() - cached_time).total_seconds() < effective_ttl:
                    return row[0]
        return None

    def clear_query_cache(
        self,
        cluster: Optional[str] = None,
        database: Optional[str] = None,
    ) -> int:
        """Clear cached query results, optionally scoped to a cluster/database."""
        clauses = []
        params: List[str] = []

        if cluster:
            clauses.append("cluster = ?")
            params.append(self.normalize_cluster_uri(cluster))
        if database:
            clauses.append("database = ?")
            params.append(database)

        delete_sql = "DELETE FROM query_cache"
        if clauses:
            delete_sql += " WHERE " + " AND ".join(clauses)

        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(delete_sql, params)
            removed = cursor.rowcount
            conn.commit()
        logger.info("Cleared %d cached query results", removed)
        return removed

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
            by_type = {}
            cursor = conn.execute(
                "SELECT query_type, COUNT(*), AVG(row_count) FROM query_cache GROUP BY query_type"
            )
            for row in cursor:
                by_type[row[0] or 'default'] = {"count": row[1], "avg_rows": round(row[2] or 0, 2)}
            
            # Count expired entries
            expired = conn.execute("""
                SELECT COUNT(*) FROM query_cache 
                WHERE (julianday('now') - julianday(timestamp)) * 86400 > COALESCE(ttl_seconds, 300)
            """).fetchone()[0]
            
            return {
                "total_cached": total,
                "expired_entries": expired,
                "by_query_type": by_type
            }

    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries and return count of removed entries."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM query_cache 
                WHERE (julianday('now') - julianday(timestamp)) * 86400 > COALESCE(ttl_seconds, 300)
            """)
            removed = cursor.rowcount
            conn.commit()
        logger.info("Removed %d expired cache entries", removed)
        return removed

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

        hints = []
        if not tables:
            return hints

        placeholders = ','.join(['?'] * len(tables))
        query = (
            "SELECT table1, table2, join_condition FROM join_hints "
            f"WHERE table1 IN ({placeholders}) OR table2 IN ({placeholders})"
        )
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, tables + tables)
            for row in cursor:
                hints.append(f"{row[0]} joins with {row[1]} on {row[2]}")
        return list(set(hints))

    def _get_database_schema(self, cluster: str, database: str) -> List[Dict[str, Any]]:
        """Get schema from SQLite with caching."""
        normalized_cluster = self.normalize_cluster_uri(cluster)
        cache_key = f"db_schema_{normalized_cluster}_{database}"
        # Simple in-memory cache check
        if hasattr(self, '_schema_cache') and cache_key in self._schema_cache:
            cached = self._schema_cache[cache_key]
            if (datetime.now() - cached['ts']).seconds < 300:  # 5 min TTL
                return cached['data']

        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT table_name, columns_json FROM schemas WHERE cluster = ? AND database = ?",
                (normalized_cluster, database)
            )
            schemas = [{"table": row[0], "columns": json.loads(row[1])} for row in cursor]

        # Cache result
        if not hasattr(self, '_schema_cache'):
            self._schema_cache = {}
        self._schema_cache[cache_key] = {'data': schemas, 'ts': datetime.now()}
        return schemas

    def get_relevant_context(
        self,
        cluster: str,
        database: str,
        user_query: str,
        max_tables: int = 20,
        max_columns: int = 12
    ) -> str:
        """
        Optimized CAG: Get schema + similar queries + join hints in TOON format.
        Limited to max_tables to prevent token overflow.
        """
        bundle = self.build_cag_bundle(
            cluster,
            database,
            user_query,
            max_tables=max_tables,
            max_columns=max_columns,
        )
        return bundle["context"]

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
                conn.execute("DELETE FROM query_cache")
                conn.execute("DELETE FROM table_locations")
                conn.execute("DELETE FROM join_hints")
                conn.execute("DELETE FROM learning_events")
            if hasattr(self, '_schema_cache'):
                self._schema_cache.clear()
            return True
        except (sqlite3.Error, OSError) as e:
            logger.error("Failed to clear memory: %s", e)
            return False

    # Use centralized normalize_cluster_uri from utils.py
    # Import at method level to avoid circular imports
    def normalize_cluster_uri(self, uri: str) -> str:
        """Normalize cluster URI - delegates to utils."""
        from .utils import normalize_cluster_uri as _normalize
        return _normalize(uri) if uri else ""

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory database including cache stats."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            schema_count = conn.execute("SELECT COUNT(*) FROM schemas").fetchone()[0]
            query_count = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
            learning_count = 0
            learning_by_type: Dict[str, int] = {}
            try:
                learning_count = conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0]
                cursor = conn.execute(
                    "SELECT execution_type, COUNT(*) FROM learning_events GROUP BY execution_type"
                )
                learning_by_type = {row[0]: row[1] for row in cursor.fetchall() if row[0]}
            except sqlite3.OperationalError:
                pass
            
            # Multi-cluster table count
            multi_cluster_count = 0
            try:
                multi_cluster_count = conn.execute(
                    "SELECT COUNT(DISTINCT table_name) FROM table_locations"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                pass
            
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        # Include cache stats
        cache_stats = self.get_cache_stats()

        return {
            "schema_count": schema_count,
            "query_count": query_count,
            "learning_count": learning_count,
            "learning_by_type": learning_by_type,
            "multi_cluster_tables": multi_cluster_count,
            "cache_stats": cache_stats,
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

    # ==================== Multi-Cluster Support Methods ====================

    def register_table_location(
        self, table_name: str, cluster: str, database: str
    ) -> bool:
        """
        Register a table's location (cluster/database) for multi-cluster support.
        
        Args:
            table_name: Name of the table
            cluster: Cluster URL
            database: Database name
            
        Returns:
            True if registered successfully, False otherwise
        """
        try:
            with self._lock, sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO table_locations 
                    (table_name, cluster, database, last_seen)
                    VALUES (?, ?, ?, ?)
                    """,
                    (table_name, cluster, database, datetime.now().isoformat())
                )
                conn.commit()
            return True
        except sqlite3.Error as e:
            logging.error("Failed to register table location: %s", e)
            return False

    def get_table_locations(self, table_name: str) -> List[Dict[str, str]]:
        """
        Get all known locations (cluster/database pairs) for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of dicts with 'cluster', 'database', 'last_seen' keys
        """
        try:
            with self._lock, sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT cluster, database, last_seen 
                    FROM table_locations 
                    WHERE table_name = ?
                    ORDER BY last_seen DESC
                    """,
                    (table_name,)
                )
                return [
                    {"cluster": row[0], "database": row[1], "last_seen": row[2]}
                    for row in cursor.fetchall()
                ]
        except sqlite3.Error as e:
            logging.error("Failed to get table locations: %s", e)
            return []

    def is_multi_cluster_table(self, table_name: str) -> bool:
        """
        Check if a table exists in multiple clusters.
        
        Args:
            table_name: Name of the table
            
        Returns:
            True if table is found in more than one cluster
        """
        try:
            with self._lock, sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT COUNT(DISTINCT cluster) 
                    FROM table_locations 
                    WHERE table_name = ?
                    """,
                    (table_name,)
                )
                count = cursor.fetchone()[0]
                return count > 1
        except sqlite3.Error as e:
            logging.error("Failed to check multi-cluster status: %s", e)
            return False

    def get_all_table_locations(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Get all registered tables and their locations.
        
        Returns:
            Dict mapping table names to lists of location dicts
        """
        try:
            with self._lock, sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT table_name, cluster, database, last_seen 
                    FROM table_locations 
                    ORDER BY table_name, last_seen DESC
                    """
                )
                result: Dict[str, List[Dict[str, str]]] = {}
                for row in cursor.fetchall():
                    table_name = row[0]
                    if table_name not in result:
                        result[table_name] = []
                    result[table_name].append({
                        "cluster": row[1],
                        "database": row[2],
                        "last_seen": row[3]
                    })
                return result
        except sqlite3.Error as e:
            logging.error("Failed to get all table locations: %s", e)
            return {}

    def remove_table_location(
        self, table_name: str, cluster: str, database: str
    ) -> bool:
        """
        Remove a specific table location entry.
        
        Args:
            table_name: Name of the table
            cluster: Cluster URL
            database: Database name
            
        Returns:
            True if removed successfully
        """
        try:
            with self._lock, sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    DELETE FROM table_locations 
                    WHERE table_name = ? AND cluster = ? AND database = ?
                    """,
                    (table_name, cluster, database)
                )
                conn.commit()
            return True
        except sqlite3.Error as e:
            logging.error("Failed to remove table location: %s", e)
            return False

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
