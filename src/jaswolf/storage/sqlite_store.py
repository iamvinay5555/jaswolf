"""SQLite storage backend.

Designed for development, tests, and small single-node deployments. Vector
search is exact brute-force over numpy (fast to ~100k memories); keyword
search uses FTS5 when available. Timestamps are stored as Unix epoch floats
so SQL comparisons are unambiguous.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..keywords import discriminative_tokens
from ..models import Memory, MemoryState, MemoryType, SweepReport, utcnow
from .base import LifecycleCutoffs, QueryScope

logger = logging.getLogger("jaswolf.storage.sqlite")

_COLUMNS = (
    "id, tenant_id, user_id, agent_id, session_id, namespace, content, content_hash, "
    "embedding, memory_type, state, importance, confidence, access_count, "
    "created_at, updated_at, last_accessed, expires_at, metadata"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT NOT NULL,
    agent_id TEXT,
    session_id TEXT,
    namespace TEXT NOT NULL DEFAULT 'default',
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding BLOB,
    memory_type TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'active',
    importance REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.8,
    access_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_accessed REAL,
    expires_at REAL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_scope
    ON memories (tenant_id, user_id, namespace, state, memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_hash
    ON memories (tenant_id, user_id, namespace, content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_expiry ON memories (expires_at);

CREATE TABLE IF NOT EXISTS memory_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    content TEXT NOT NULL,
    reason TEXT NOT NULL,
    payload TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_versions_memory ON memory_versions (memory_id);

CREATE TABLE IF NOT EXISTS memory_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rel_from ON memory_relationships (from_id);

CREATE TABLE IF NOT EXISTS memory_access_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    action TEXT NOT NULL,
    query TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_access_memory ON memory_access_logs (memory_id);

CREATE TABLE IF NOT EXISTS jaswolf_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, content='memories', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""

_MAX_VECTOR_CANDIDATES = 50_000


def _to_epoch(dt: datetime | None) -> float | None:
    return dt.timestamp() if dt else None


def _from_epoch(value: float | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _encode_vec(vec: list[float] | None) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def _decode_vec(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).tolist()


def validate_sqlite_snapshot(path: str) -> dict[str, Any]:
    """Inspect a backup file before restoring it: integrity, embedding
    fingerprint, and memory count. Read-only; never touches the live DB."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        integrity = conn.execute("PRAGMA quick_check").fetchone()[0]
        try:
            row = conn.execute(
                "SELECT value FROM jaswolf_meta WHERE key='embedding_fingerprint'"
            ).fetchone()
            fingerprint = row[0] if row else None
        except sqlite3.OperationalError:
            fingerprint = None
        try:
            count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        except sqlite3.OperationalError:
            count = None
    finally:
        conn.close()
    return {"integrity": integrity, "embedding_fingerprint": fingerprint, "memories": count}


class SQLiteStore:
    name = "sqlite"
    # discriminative-keyword thresholds; create_storage overrides from settings
    keyword_min_len = 3
    keyword_max_df_ratio = 0.10

    def __init__(self, path: str = "./jaswolf.db"):
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._fts = False
        self._df_cache: dict[str, int] = {}  # token -> corpus document frequency

    # -- lifecycle ---------------------------------------------------------

    async def init(self) -> None:
        await asyncio.to_thread(self._sync_init)

    def _sync_init(self) -> None:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        try:
            conn.executescript(_FTS_SCHEMA)
            self._fts = True
        except sqlite3.OperationalError:
            logger.warning("FTS5 unavailable — keyword search will use LIKE fallback")
            self._fts = False
        conn.commit()
        self._conn = conn

    async def close(self) -> None:
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    async def ping(self) -> bool:
        try:
            await asyncio.to_thread(self._sync_ping)
            return True
        except Exception:
            return False

    def _sync_ping(self) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute("SELECT 1").fetchone()

    async def integrity_check(self) -> str:
        return await asyncio.to_thread(self._sync_integrity_check)

    def _sync_integrity_check(self) -> str:
        with self._lock:
            assert self._conn is not None
            row = self._conn.execute("PRAGMA quick_check").fetchone()
        return row[0] if row else "no result"

    async def backup(self, dest_path: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync_backup, dest_path)

    def _sync_backup(self, dest_path: str) -> dict[str, Any]:
        # SQLite's online backup API takes a consistent snapshot even under WAL
        # and concurrent writes — unlike `cp`, which can capture a torn DB+WAL.
        with self._lock:
            assert self._conn is not None
            dest = sqlite3.connect(dest_path)
            try:
                self._conn.backup(dest)
            finally:
                dest.close()
        return {"path": dest_path, "bytes": os.path.getsize(dest_path)}

    # -- scope -> SQL ------------------------------------------------------

    def _scope_where(self, scope: QueryScope, now: float | None = None) -> tuple[str, list[Any]]:
        clauses = ["tenant_id = ?"]
        params: list[Any] = [scope.tenant_id]
        if scope.user_id is not None:
            clauses.append("user_id = ?")
            params.append(scope.user_id)
        if scope.agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(scope.agent_id)
        if scope.session_id is not None:
            clauses.append("session_id = ?")
            params.append(scope.session_id)
        if scope.namespaces:
            placeholders = ",".join("?" * len(scope.namespaces))
            clauses.append(f"namespace IN ({placeholders})")
            params.extend(scope.namespaces)
        elif scope.namespace is not None:
            clauses.append("namespace = ?")
            params.append(scope.namespace)
        if scope.memory_types:
            placeholders = ",".join("?" * len(scope.memory_types))
            clauses.append(f"memory_type IN ({placeholders})")
            params.extend(t.value for t in scope.memory_types)
        if scope.states:
            placeholders = ",".join("?" * len(scope.states))
            clauses.append(f"state IN ({placeholders})")
            params.extend(s.value for s in scope.states)
        if scope.min_importance is not None:
            clauses.append("importance >= ?")
            params.append(scope.min_importance)
        if now is not None:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(now)
        return " AND ".join(clauses), params

    @staticmethod
    def _row_to_memory(row: sqlite3.Row, include_embedding: bool = False) -> Memory:
        return Memory(
            id=row["id"],
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            namespace=row["namespace"],
            content=row["content"],
            content_hash=row["content_hash"],
            embedding=_decode_vec(row["embedding"]) if include_embedding else None,
            memory_type=MemoryType(row["memory_type"]),
            state=MemoryState(row["state"]),
            importance=row["importance"],
            confidence=row["confidence"],
            access_count=row["access_count"],
            created_at=_from_epoch(row["created_at"]),
            updated_at=_from_epoch(row["updated_at"]),
            last_accessed=_from_epoch(row["last_accessed"]),
            expires_at=_from_epoch(row["expires_at"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    # -- CRUD ---------------------------------------------------------------

    async def upsert(self, memory: Memory) -> None:
        await asyncio.to_thread(self._sync_upsert, memory)

    def _sync_upsert(self, memory: Memory) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                f"""INSERT INTO memories ({_COLUMNS})
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    content=excluded.content, content_hash=excluded.content_hash,
                    embedding=excluded.embedding, memory_type=excluded.memory_type,
                    state=excluded.state, importance=excluded.importance,
                    confidence=excluded.confidence, access_count=excluded.access_count,
                    updated_at=excluded.updated_at, last_accessed=excluded.last_accessed,
                    expires_at=excluded.expires_at, metadata=excluded.metadata""",
                (
                    memory.id, memory.tenant_id, memory.user_id, memory.agent_id,
                    memory.session_id, memory.namespace, memory.content, memory.content_hash,
                    _encode_vec(memory.embedding), memory.memory_type.value, memory.state.value,
                    memory.importance, memory.confidence, memory.access_count,
                    _to_epoch(memory.created_at), _to_epoch(memory.updated_at),
                    _to_epoch(memory.last_accessed), _to_epoch(memory.expires_at),
                    json.dumps(memory.metadata),
                ),
            )
            self._conn.commit()

    async def get(self, memory_id: str, tenant_id: str) -> Memory | None:
        return await asyncio.to_thread(self._sync_get, memory_id, tenant_id)

    def _sync_get(self, memory_id: str, tenant_id: str) -> Memory | None:
        with self._lock:
            assert self._conn is not None
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE id = ? AND tenant_id = ?",
                (memory_id, tenant_id),
            ).fetchone()
        return self._row_to_memory(row, include_embedding=True) if row else None

    async def delete(self, memory_id: str, tenant_id: str, hard: bool = False) -> bool:
        return await asyncio.to_thread(self._sync_delete, memory_id, tenant_id, hard)

    def _sync_delete(self, memory_id: str, tenant_id: str, hard: bool) -> bool:
        with self._lock:
            assert self._conn is not None
            if hard:
                cur = self._conn.execute(
                    "DELETE FROM memories WHERE id = ? AND tenant_id = ?", (memory_id, tenant_id)
                )
            else:
                cur = self._conn.execute(
                    "UPDATE memories SET state = ?, updated_at = ? WHERE id = ? AND tenant_id = ?",
                    (MemoryState.DELETED.value, utcnow().timestamp(), memory_id, tenant_id),
                )
            self._conn.commit()
            return cur.rowcount > 0

    async def list_memories(
        self,
        scope: QueryScope,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "recent",
        include_embeddings: bool = False,
    ) -> list[Memory]:
        return await asyncio.to_thread(
            self._sync_list, scope, limit, offset, order_by, include_embeddings
        )

    def _sync_list(
        self, scope: QueryScope, limit: int, offset: int, order_by: str, include_embeddings: bool
    ) -> list[Memory]:
        order_sql = {
            "recent": "COALESCE(last_accessed, updated_at) DESC",
            "importance": "importance DESC, COALESCE(last_accessed, updated_at) DESC",
            "created": "created_at DESC",
        }.get(order_by, "COALESCE(last_accessed, updated_at) DESC")
        where, params = self._scope_where(scope, now=utcnow().timestamp())
        with self._lock:
            assert self._conn is not None
            rows = self._conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE {where} ORDER BY {order_sql} LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [self._row_to_memory(r, include_embeddings) for r in rows]

    # -- search --------------------------------------------------------------

    async def search_vector(
        self, scope: QueryScope, query_vec: list[float], k: int
    ) -> list[tuple[Memory, float]]:
        return await asyncio.to_thread(self._sync_search_vector, scope, query_vec, k)

    def _sync_search_vector(
        self, scope: QueryScope, query_vec: list[float], k: int
    ) -> list[tuple[Memory, float]]:
        where, params = self._scope_where(scope, now=utcnow().timestamp())
        with self._lock:
            assert self._conn is not None
            rows = self._conn.execute(
                f"""SELECT {_COLUMNS} FROM memories
                WHERE {where} AND embedding IS NOT NULL
                ORDER BY created_at DESC LIMIT {_MAX_VECTOR_CANDIDATES}""",
                params,
            ).fetchall()
        if not rows:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        expected_bytes = q.shape[0] * 4
        usable = [r for r in rows if len(r["embedding"]) == expected_bytes]
        if len(usable) < len(rows):
            logger.warning(
                "vector search skipped %d/%d memories with mismatched embedding "
                "dimension — the embedding model changed after data was written; "
                "re-embed or revert the model (see TROUBLESHOOTING.md)",
                len(rows) - len(usable),
                len(rows),
            )
        if not usable:
            return []
        matrix = np.frombuffer(b"".join(r["embedding"] for r in usable), dtype=np.float32)
        matrix = matrix.reshape(len(usable), q.shape[0])
        sims = matrix @ q  # vectors are L2-normalized at creation -> dot == cosine
        top = np.argsort(-sims)[:k]
        return [(self._row_to_memory(usable[i], include_embedding=True), float(sims[i])) for i in top]

    async def search_keyword(
        self, scope: QueryScope, query: str, k: int
    ) -> list[tuple[Memory, float]]:
        return await asyncio.to_thread(self._sync_search_keyword, scope, query, k)

    def _token_df(self, token: str) -> int:
        """Corpus document frequency for a token (cached, FTS or LIKE)."""
        if token in self._df_cache:
            return self._df_cache[token]
        assert self._conn is not None
        if self._fts:
            row = self._conn.execute(
                "SELECT count(*) FROM memories_fts WHERE memories_fts MATCH ?", (f'"{token}"',)
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT count(*) FROM memories WHERE content LIKE ?", (f"%{token}%",)
            ).fetchone()
        df = int(row[0]) if row else 0
        self._df_cache[token] = df
        return df

    def _sync_search_keyword(
        self, scope: QueryScope, query: str, k: int
    ) -> list[tuple[Memory, float]]:
        where, params = self._scope_where(scope, now=utcnow().timestamp())
        with self._lock:
            assert self._conn is not None
            total = self._conn.execute("SELECT count(*) FROM memories").fetchone()[0]
            # only discriminative tokens count as lexical evidence — generic
            # words ("next", "week") must not exempt a memory from the gate
            tokens = discriminative_tokens(
                query, self._token_df, total,
                min_len=self.keyword_min_len, max_df_ratio=self.keyword_max_df_ratio,
            )
            if not tokens:
                return []
            if self._fts:
                qualified = ", ".join(f"memories.{c.strip()}" for c in _COLUMNS.split(","))
                match_expr = " OR ".join(f'"{t}"' for t in tokens)
                rows = self._conn.execute(
                    f"""SELECT {qualified}, bm25(memories_fts) AS rank
                    FROM memories_fts JOIN memories ON memories.rowid = memories_fts.rowid
                    WHERE memories_fts MATCH ? AND {where}
                    ORDER BY rank ASC LIMIT ?""",
                    (match_expr, *params, k),
                ).fetchall()
            else:
                hits_expr = " + ".join(
                    "(CASE WHEN content LIKE ? THEN 1 ELSE 0 END)" for _ in tokens
                )
                like_params = [f"%{t}%" for t in tokens]
                rows = self._conn.execute(
                    f"""SELECT {_COLUMNS}, ({hits_expr}) AS hits FROM memories
                    WHERE {where} AND ({hits_expr}) > 0
                    ORDER BY hits DESC LIMIT ?""",
                    (*like_params, *params, *like_params, k),
                ).fetchall()
        n = len(rows)
        # position-normalized score: 1.0 for the best hit, declining linearly
        return [
            (self._row_to_memory(r, include_embedding=True), 1.0 - (i / max(1, n)))
            for i, r in enumerate(rows)
        ]

    async def sample_embeddings(self, scope: QueryScope, k: int) -> list[list[float]]:
        return await asyncio.to_thread(self._sync_sample_embeddings, scope, k)

    def _sync_sample_embeddings(self, scope: QueryScope, k: int) -> list[list[float]]:
        where, params = self._scope_where(scope, now=utcnow().timestamp())
        with self._lock:
            assert self._conn is not None
            # ORDER BY id (uuid4) is a stable, representative sample of the
            # corpus — reproducible across runs, unlike ORDER BY random()
            rows = self._conn.execute(
                f"SELECT embedding FROM memories WHERE {where} AND embedding IS NOT NULL "
                "ORDER BY id LIMIT ?",
                (*params, k),
            ).fetchall()
        out = []
        for r in rows:
            vec = _decode_vec(r["embedding"])
            if vec is not None:
                out.append(vec)
        return out

    async def find_similar(
        self,
        scope: QueryScope,
        query_vec: list[float],
        threshold: float,
        memory_type: MemoryType | None = None,
    ) -> tuple[Memory, float] | None:
        if memory_type is not None:
            scope = QueryScope(**{**scope.__dict__, "memory_types": [memory_type]})
        hits = await self.search_vector(scope, query_vec, k=1)
        if hits and hits[0][1] >= threshold:
            return hits[0]
        return None

    async def get_by_hash(
        self, tenant_id: str, user_id: str, namespace: str, hash_value: str
    ) -> Memory | None:
        return await asyncio.to_thread(self._sync_get_by_hash, tenant_id, user_id, namespace, hash_value)

    def _sync_get_by_hash(
        self, tenant_id: str, user_id: str, namespace: str, hash_value: str
    ) -> Memory | None:
        with self._lock:
            assert self._conn is not None
            row = self._conn.execute(
                f"""SELECT {_COLUMNS} FROM memories
                WHERE tenant_id=? AND user_id=? AND namespace=? AND content_hash=?
                  AND state != 'deleted'
                ORDER BY updated_at DESC LIMIT 1""",
                (tenant_id, user_id, namespace, hash_value),
            ).fetchone()
        return self._row_to_memory(row, include_embedding=True) if row else None

    # -- bookkeeping ----------------------------------------------------------

    async def record_access(
        self, memory_ids: list[str], tenant_id: str, action: str, query: str | None = None
    ) -> None:
        if not memory_ids:
            return
        await asyncio.to_thread(self._sync_record_access, memory_ids, tenant_id, action, query)

    def _sync_record_access(
        self, memory_ids: list[str], tenant_id: str, action: str, query: str | None
    ) -> None:
        now = utcnow().timestamp()
        with self._lock:
            assert self._conn is not None
            placeholders = ",".join("?" * len(memory_ids))
            self._conn.execute(
                f"""UPDATE memories SET access_count = access_count + 1, last_accessed = ?
                WHERE id IN ({placeholders}) AND tenant_id = ?""",
                (now, *memory_ids, tenant_id),
            )
            self._conn.executemany(
                "INSERT INTO memory_access_logs (memory_id, tenant_id, action, query, created_at)"
                " VALUES (?,?,?,?,?)",
                [(mid, tenant_id, action, (query or "")[:200], now) for mid in memory_ids],
            )
            self._conn.commit()

    async def add_version(
        self, memory_id: str, content: str, reason: str, payload: dict[str, Any] | None = None
    ) -> None:
        await asyncio.to_thread(self._sync_add_version, memory_id, content, reason, payload)

    def _sync_add_version(
        self, memory_id: str, content: str, reason: str, payload: dict[str, Any] | None
    ) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT INTO memory_versions (memory_id, content, reason, payload, created_at)"
                " VALUES (?,?,?,?,?)",
                (memory_id, content, reason, json.dumps(payload or {}), utcnow().timestamp()),
            )
            self._conn.commit()

    async def get_versions(self, memory_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._sync_get_versions, memory_id)

    def _sync_get_versions(self, memory_id: str) -> list[dict[str, Any]]:
        with self._lock:
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT content, reason, payload, created_at FROM memory_versions"
                " WHERE memory_id = ? ORDER BY id DESC",
                (memory_id,),
            ).fetchall()
        return [
            {
                "content": r["content"],
                "reason": r["reason"],
                "payload": json.loads(r["payload"] or "{}"),
                "created_at": _from_epoch(r["created_at"]).isoformat(),
            }
            for r in rows
        ]

    async def add_relationship(self, from_id: str, to_id: str, relation: str) -> None:
        await asyncio.to_thread(self._sync_add_relationship, from_id, to_id, relation)

    def _sync_add_relationship(self, from_id: str, to_id: str, relation: str) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT INTO memory_relationships (from_id, to_id, relation, created_at)"
                " VALUES (?,?,?,?)",
                (from_id, to_id, relation, utcnow().timestamp()),
            )
            self._conn.commit()

    # -- lifecycle sweep -------------------------------------------------------

    async def apply_lifecycle(self, cutoffs: LifecycleCutoffs) -> SweepReport:
        return await asyncio.to_thread(self._sync_apply_lifecycle, cutoffs)

    def _sync_apply_lifecycle(self, cutoffs: LifecycleCutoffs) -> SweepReport:
        now = cutoffs.now.timestamp()
        report = SweepReport()
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                """UPDATE memories SET state='archived'
                WHERE state IN ('active','warm','cold') AND expires_at IS NOT NULL AND expires_at <= ?""",
                (now,),
            )
            report.expired_working = cur.rowcount
            # oldest stage first so a memory moves one stage per sweep
            transitions = [
                ("cold", "archived", cutoffs.archive_before.timestamp(), "cold_to_archived"),
                ("warm", "cold", cutoffs.cold_before.timestamp(), "warm_to_cold"),
                ("active", "warm", cutoffs.warm_before.timestamp(), "active_to_warm"),
            ]
            for from_state, to_state, cutoff, field_name in transitions:
                cur = self._conn.execute(
                    """UPDATE memories SET state=?
                    WHERE state=? AND COALESCE(last_accessed, updated_at) < ?""",
                    (to_state, from_state, cutoff),
                )
                setattr(report, field_name, cur.rowcount)
            self._conn.commit()
        return report

    # -- meta --------------------------------------------------------------------

    async def get_meta(self, key: str) -> str | None:
        return await asyncio.to_thread(self._sync_get_meta, key)

    def _sync_get_meta(self, key: str) -> str | None:
        with self._lock:
            assert self._conn is not None
            row = self._conn.execute("SELECT value FROM jaswolf_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    async def set_meta(self, key: str, value: str) -> None:
        await asyncio.to_thread(self._sync_set_meta, key, value)

    def _sync_set_meta(self, key: str, value: str) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT INTO jaswolf_meta (key, value, updated_at) VALUES (?,?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, utcnow().timestamp()),
            )
            self._conn.commit()

    # -- stats -------------------------------------------------------------------

    async def stats(self, tenant_id: str, user_id: str | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync_stats, tenant_id, user_id)

    def _sync_stats(self, tenant_id: str, user_id: str | None) -> dict[str, Any]:
        where = "tenant_id = ?"
        params: list[Any] = [tenant_id]
        if user_id:
            where += " AND user_id = ?"
            params.append(user_id)
        with self._lock:
            assert self._conn is not None
            total = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM memories WHERE {where}", params
            ).fetchone()["n"]
            by_state = self._conn.execute(
                f"SELECT state, COUNT(*) AS n FROM memories WHERE {where} GROUP BY state", params
            ).fetchall()
            by_type = self._conn.execute(
                f"SELECT memory_type, COUNT(*) AS n FROM memories WHERE {where} GROUP BY memory_type",
                params,
            ).fetchall()
        return {
            "total": total,
            "by_state": {r["state"]: r["n"] for r in by_state},
            "by_type": {r["memory_type"]: r["n"] for r in by_type},
        }
