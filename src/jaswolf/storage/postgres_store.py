"""PostgreSQL + pgvector storage backend (production).

Vector search uses an HNSW index with cosine distance; keyword search uses
Postgres full-text search. Migrations are plain SQL files shipped inside the
package and tracked in schema_migrations.
"""

from __future__ import annotations

import json
import logging
from importlib import resources
from typing import Any

from ..keywords import candidate_tokens, discriminative_tokens
from ..models import Memory, MemoryState, MemoryType, SweepReport, utcnow
from .base import LifecycleCutoffs, QueryScope

logger = logging.getLogger("jaswolf.storage.postgres")

_COLUMNS = (
    "id, tenant_id, user_id, agent_id, session_id, namespace, content, content_hash, "
    "embedding::text AS embedding, memory_type, state, importance, confidence, access_count, "
    "created_at, updated_at, last_accessed, expires_at, metadata"
)


def _vec_literal(vec: list[float] | None) -> str | None:
    if vec is None:
        return None
    return "[" + ",".join(f"{x:.8g}" for x in vec) + "]"


def _parse_vec(text: str | None) -> list[float] | None:
    if text is None:
        return None
    return json.loads(text)  # pgvector text format '[0.1,0.2]' is valid JSON


class PostgresStore:
    name = "postgres"
    keyword_min_len = 3
    keyword_max_df_ratio = 0.10

    def __init__(self, dsn: str, embedding_dim: int = 384):
        try:
            import asyncpg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL backend requires asyncpg: pip install jaswolf[postgres]"
            ) from exc
        self.dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
        self.embedding_dim = embedding_dim
        self._pool = None

    # -- lifecycle -----------------------------------------------------------

    async def init(self) -> None:
        import asyncpg

        async def _setup(conn) -> None:
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )

        self._pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10, init=_setup)
        await self._apply_migrations()

    async def _apply_migrations(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )"""
            )
            applied = {
                r["version"] for r in await conn.fetch("SELECT version FROM schema_migrations")
            }
            migration_files = sorted(
                f for f in resources.files("jaswolf.storage.migrations").iterdir()
                if f.name.endswith(".sql")
            )
            for migration in migration_files:
                if migration.name in applied:
                    continue
                sql = migration.read_text().replace("__EMBED_DIM__", str(self.embedding_dim))
                async with conn.transaction():
                    await conn.execute(sql)
                    await conn.execute(
                        "INSERT INTO schema_migrations (version) VALUES ($1)", migration.name
                    )
                logger.info("applied migration %s", migration.name)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def ping(self) -> bool:
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def integrity_check(self) -> str:
        # Postgres guarantees integrity at the engine level; a successful round
        # trip is the meaningful liveness signal here.
        return "ok" if await self.ping() else "unreachable"

    async def backup(self, dest_path: str) -> dict[str, Any]:
        raise NotImplementedError(
            "Postgres backups use pg_dump, not the JASWOLF backup command — "
            "e.g. `pg_dump $JASWOLF_DATABASE_URL > snapshot.sql` (see OPERATIONS.md)"
        )

    # -- helpers ---------------------------------------------------------------

    def _scope_where(
        self, scope: QueryScope, params: list[Any], with_expiry: bool = True
    ) -> str:
        clauses = []

        def bind(value: Any) -> str:
            params.append(value)
            return f"${len(params)}"

        clauses.append(f"tenant_id = {bind(scope.tenant_id)}")
        if scope.user_id is not None:
            clauses.append(f"user_id = {bind(scope.user_id)}")
        if scope.agent_id is not None:
            clauses.append(f"agent_id = {bind(scope.agent_id)}")
        if scope.session_id is not None:
            clauses.append(f"session_id = {bind(scope.session_id)}")
        if scope.namespaces:
            clauses.append(f"namespace = ANY({bind(list(scope.namespaces))})")
        elif scope.namespace is not None:
            clauses.append(f"namespace = {bind(scope.namespace)}")
        if scope.memory_types:
            clauses.append(f"memory_type = ANY({bind([t.value for t in scope.memory_types])})")
        if scope.states:
            clauses.append(f"state = ANY({bind([s.value for s in scope.states])})")
        if scope.min_importance is not None:
            clauses.append(f"importance >= {bind(scope.min_importance)}")
        if with_expiry:
            clauses.append(f"(expires_at IS NULL OR expires_at > {bind(utcnow())})")
        return " AND ".join(clauses)

    @staticmethod
    def _row_to_memory(row, include_embedding: bool = False) -> Memory:
        return Memory(
            id=str(row["id"]),
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            namespace=row["namespace"],
            content=row["content"],
            content_hash=row["content_hash"],
            embedding=_parse_vec(row["embedding"]) if include_embedding else None,
            memory_type=MemoryType(row["memory_type"]),
            state=MemoryState(row["state"]),
            importance=row["importance"],
            confidence=row["confidence"],
            access_count=row["access_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed=row["last_accessed"],
            expires_at=row["expires_at"],
            metadata=row["metadata"] or {},
        )

    # -- CRUD --------------------------------------------------------------------

    async def upsert(self, memory: Memory) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO memories (
                    id, tenant_id, user_id, agent_id, session_id, namespace, content,
                    content_hash, embedding, memory_type, state, importance, confidence,
                    access_count, created_at, updated_at, last_accessed, expires_at, metadata
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::vector,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)
                ON CONFLICT (id) DO UPDATE SET
                    content=EXCLUDED.content, content_hash=EXCLUDED.content_hash,
                    embedding=EXCLUDED.embedding, memory_type=EXCLUDED.memory_type,
                    state=EXCLUDED.state, importance=EXCLUDED.importance,
                    confidence=EXCLUDED.confidence, access_count=EXCLUDED.access_count,
                    updated_at=EXCLUDED.updated_at, last_accessed=EXCLUDED.last_accessed,
                    expires_at=EXCLUDED.expires_at, metadata=EXCLUDED.metadata""",
                memory.id, memory.tenant_id, memory.user_id, memory.agent_id,
                memory.session_id, memory.namespace, memory.content, memory.content_hash,
                _vec_literal(memory.embedding), memory.memory_type.value, memory.state.value,
                memory.importance, memory.confidence, memory.access_count,
                memory.created_at, memory.updated_at, memory.last_accessed,
                memory.expires_at, memory.metadata,
            )

    async def get(self, memory_id: str, tenant_id: str) -> Memory | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_COLUMNS} FROM memories WHERE id = $1 AND tenant_id = $2",
                memory_id, tenant_id,
            )
        return self._row_to_memory(row, include_embedding=True) if row else None

    async def delete(self, memory_id: str, tenant_id: str, hard: bool = False) -> bool:
        async with self._pool.acquire() as conn:
            if hard:
                result = await conn.execute(
                    "DELETE FROM memories WHERE id = $1 AND tenant_id = $2", memory_id, tenant_id
                )
            else:
                result = await conn.execute(
                    "UPDATE memories SET state = 'deleted', updated_at = $3"
                    " WHERE id = $1 AND tenant_id = $2",
                    memory_id, tenant_id, utcnow(),
                )
        return result.split()[-1] != "0"

    async def list_memories(
        self,
        scope: QueryScope,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "recent",
        include_embeddings: bool = False,
    ) -> list[Memory]:
        order_sql = {
            "recent": "COALESCE(last_accessed, updated_at) DESC",
            "importance": "importance DESC, COALESCE(last_accessed, updated_at) DESC",
            "created": "created_at DESC",
        }.get(order_by, "COALESCE(last_accessed, updated_at) DESC")
        params: list[Any] = []
        where = self._scope_where(scope, params)
        params.extend([limit, offset])
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_COLUMNS} FROM memories WHERE {where}"
                f" ORDER BY {order_sql} LIMIT ${len(params)-1} OFFSET ${len(params)}",
                *params,
            )
        return [self._row_to_memory(r, include_embeddings) for r in rows]

    # -- search --------------------------------------------------------------------

    async def search_vector(
        self, scope: QueryScope, query_vec: list[float], k: int
    ) -> list[tuple[Memory, float]]:
        params: list[Any] = []
        where = self._scope_where(scope, params)
        params.append(_vec_literal(query_vec))
        vec_param = f"${len(params)}"
        params.append(k)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT {_COLUMNS}, 1 - (embedding <=> {vec_param}::vector) AS sim
                FROM memories WHERE {where} AND embedding IS NOT NULL
                ORDER BY embedding <=> {vec_param}::vector LIMIT ${len(params)}""",
                *params,
            )
        return [(self._row_to_memory(r, include_embedding=True), float(r["sim"])) for r in rows]

    async def search_keyword(
        self, scope: QueryScope, query: str, k: int
    ) -> list[tuple[Memory, float]]:
        if not query.strip():
            return []
        candidates = candidate_tokens(query, self.keyword_min_len)
        if not candidates:
            return []
        async with self._pool.acquire() as conn:
            total = await conn.fetchval("SELECT count(*) FROM memories")
            df: dict[str, int] = {}
            for token in candidates:
                df[token] = await conn.fetchval(
                    "SELECT count(*) FROM memories "
                    "WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)",
                    token,
                )
            # only discriminative tokens are lexical evidence (Jasmine 2026-06-13)
            tokens = discriminative_tokens(
                query, lambda t: df.get(t, 0), total or 0,
                min_len=self.keyword_min_len, max_df_ratio=self.keyword_max_df_ratio,
            )
            if not tokens:
                return []
            params: list[Any] = []
            where = self._scope_where(scope, params)
            params.append(" | ".join(tokens))   # OR of discriminative terms
            q_param = f"${len(params)}"
            params.append(k)
            rows = await conn.fetch(
                f"""SELECT {_COLUMNS},
                    ts_rank_cd(to_tsvector('english', content), to_tsquery('english', {q_param})) AS rank
                FROM memories
                WHERE {where} AND to_tsvector('english', content) @@ to_tsquery('english', {q_param})
                ORDER BY rank DESC LIMIT ${len(params)}""",
                *params,
            )
        n = len(rows)
        return [
            (self._row_to_memory(r, include_embedding=True), 1.0 - (i / max(1, n)))
            for i, r in enumerate(rows)
        ]

    async def sample_embeddings(self, scope: QueryScope, k: int) -> list[list[float]]:
        params: list[Any] = []
        where = self._scope_where(scope, params)
        params.append(k)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT embedding::text AS embedding FROM memories "
                f"WHERE {where} AND embedding IS NOT NULL ORDER BY id LIMIT ${len(params)}",
                *params,
            )
        out: list[list[float]] = []
        for r in rows:
            vec = _parse_vec(r["embedding"])
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
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""SELECT {_COLUMNS} FROM memories
                WHERE tenant_id=$1 AND user_id=$2 AND namespace=$3 AND content_hash=$4
                  AND state != 'deleted'
                ORDER BY updated_at DESC LIMIT 1""",
                tenant_id, user_id, namespace, hash_value,
            )
        return self._row_to_memory(row, include_embedding=True) if row else None

    # -- bookkeeping ------------------------------------------------------------------

    async def record_access(
        self, memory_ids: list[str], tenant_id: str, action: str, query: str | None = None
    ) -> None:
        if not memory_ids:
            return
        now = utcnow()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE memories SET access_count = access_count + 1, last_accessed = $3
                WHERE id = ANY($1::uuid[]) AND tenant_id = $2""",
                memory_ids, tenant_id, now,
            )
            await conn.executemany(
                "INSERT INTO memory_access_logs (memory_id, tenant_id, action, query, created_at)"
                " VALUES ($1,$2,$3,$4,$5)",
                [(mid, tenant_id, action, (query or "")[:200], now) for mid in memory_ids],
            )

    async def add_version(
        self, memory_id: str, content: str, reason: str, payload: dict[str, Any] | None = None
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memory_versions (memory_id, content, reason, payload, created_at)"
                " VALUES ($1,$2,$3,$4,$5)",
                memory_id, content, reason, payload or {}, utcnow(),
            )

    async def get_versions(self, memory_id: str) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT content, reason, payload, created_at FROM memory_versions"
                " WHERE memory_id = $1 ORDER BY id DESC",
                memory_id,
            )
        return [
            {
                "content": r["content"],
                "reason": r["reason"],
                "payload": r["payload"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]

    async def add_relationship(self, from_id: str, to_id: str, relation: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memory_relationships (from_id, to_id, relation, created_at)"
                " VALUES ($1,$2,$3,$4)",
                from_id, to_id, relation, utcnow(),
            )

    # -- lifecycle sweep ------------------------------------------------------------------

    async def apply_lifecycle(self, cutoffs: LifecycleCutoffs) -> SweepReport:
        report = SweepReport()
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """UPDATE memories SET state='archived'
                WHERE state IN ('active','warm','cold') AND expires_at IS NOT NULL AND expires_at <= $1""",
                cutoffs.now,
            )
            report.expired_working = int(result.split()[-1])
            transitions = [
                ("cold", "archived", cutoffs.archive_before, "cold_to_archived"),
                ("warm", "cold", cutoffs.cold_before, "warm_to_cold"),
                ("active", "warm", cutoffs.warm_before, "active_to_warm"),
            ]
            for from_state, to_state, cutoff, field_name in transitions:
                result = await conn.execute(
                    f"""UPDATE memories SET state='{to_state}'
                    WHERE state='{from_state}' AND COALESCE(last_accessed, updated_at) < $1""",
                    cutoff,
                )
                setattr(report, field_name, int(result.split()[-1]))
        return report

    # -- meta ------------------------------------------------------------------------------

    async def get_meta(self, key: str) -> str | None:
        async with self._pool.acquire() as conn:
            return await conn.fetchval("SELECT value FROM jaswolf_meta WHERE key = $1", key)

    async def set_meta(self, key: str, value: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO jaswolf_meta (key, value, updated_at) VALUES ($1, $2, now())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()""",
                key,
                value,
            )

    # -- stats ---------------------------------------------------------------------------

    async def stats(self, tenant_id: str, user_id: str | None = None) -> dict[str, Any]:
        params: list[Any] = [tenant_id]
        where = "tenant_id = $1"
        if user_id:
            params.append(user_id)
            where += " AND user_id = $2"
        async with self._pool.acquire() as conn:
            total = await conn.fetchval(f"SELECT COUNT(*) FROM memories WHERE {where}", *params)
            by_state = await conn.fetch(
                f"SELECT state, COUNT(*) AS n FROM memories WHERE {where} GROUP BY state", *params
            )
            by_type = await conn.fetch(
                f"SELECT memory_type, COUNT(*) AS n FROM memories WHERE {where} GROUP BY memory_type",
                *params,
            )
        return {
            "total": total,
            "by_state": {r["state"]: r["n"] for r in by_state},
            "by_type": {r["memory_type"]: r["n"] for r in by_type},
        }
