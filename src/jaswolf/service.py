"""MemoryService — the composition root and main entry point.

The HTTP API, the Hermes provider (embedded mode), and the CLI are all thin
layers over this class. Library-first design: importing and using
MemoryService directly skips the network entirely.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import timedelta
from typing import Any

from .config import JaswolfSettings
from .consolidation import ConsolidationEngine
from .context_builder import ContextBuilder
from .extraction import ExtractionEngine
from .models import (
    ChatMessage,
    ConsolidationReport,
    ContextRequest,
    ContextResult,
    Memory,
    MemoryCreate,
    MemoryNotFound,
    MemoryState,
    MemoryType,
    MemoryUpdate,
    RelationType,
    ScoredMemory,
    SearchQuery,
    SweepReport,
    content_hash,
    default_expiry,
    utcnow,
)
from .retrieval import RetrievalEngine
from .scoring import importance_for
from .storage.base import LifecycleCutoffs, QueryScope, StorageBackend

logger = logging.getLogger("jaswolf.service")

# Correction language: a new statement carrying one of these markers and
# resembling an existing memory replaces it (the old one is archived with a
# SUPERSEDES relationship, never silently lost). Unmarked contradictions stay
# additive on purpose — guessing wrongly would destroy real memories.
_CORRECTION = re.compile(
    r"\b(?:actually|now|anymore|no\s+longer|instead|changed|moved\s+to|correction|"
    r"not\s+\w+\s+but|stopped|don'?t\s+call\s+me|from\s+now\s+on)\b",
    re.IGNORECASE,
)
# "User's <slot> is <value>" — same slot + different value = direct conflict
_SLOT = re.compile(r"^user'?s?\s+(.{2,40}?)\s+is\s+(.+)$", re.IGNORECASE)

_SUPERSEDABLE = (
    MemoryType.SEMANTIC,
    MemoryType.PREFERENCE,
    MemoryType.GOAL,
    MemoryType.RELATIONSHIP,
)


def _slot_of(content: str) -> tuple[str, str] | None:
    m = _SLOT.match(content.strip())
    if not m:
        return None
    slot = re.sub(r"\s+", " ", m.group(1).lower()).strip()
    value = re.sub(r"\s+", " ", m.group(2).lower()).strip().rstrip(".")
    return slot, value


def create_storage(settings: JaswolfSettings) -> StorageBackend:
    url = settings.database_url
    if url.startswith(("postgres://", "postgresql://", "postgresql+asyncpg://")):
        from .storage.postgres_store import PostgresStore

        store: StorageBackend = PostgresStore(url, embedding_dim=settings.embedding_dim)
    elif url.startswith("sqlite://"):
        # sqlalchemy-style: sqlite:///relative.db, sqlite:////absolute.db, sqlite:///:memory:
        path = url[len("sqlite://"):]
        if path.startswith("/"):
            path = path[1:]
        store = _sqlite(path or ":memory:")
    else:
        store = _sqlite(url)  # treat anything else as a literal sqlite path
    store.keyword_min_len = settings.keyword_min_token_len
    store.keyword_max_df_ratio = settings.keyword_max_df_ratio
    return store


def _sqlite(path: str):
    from .storage.sqlite_store import SQLiteStore

    return SQLiteStore(path)


class MemoryService:
    def __init__(
        self,
        settings: JaswolfSettings,
        storage: StorageBackend,
        embedder,
        cache,
        retrieval: RetrievalEngine,
        context: ContextBuilder,
        extraction: ExtractionEngine,
        consolidation: ConsolidationEngine,
    ):
        self.settings = settings
        self.storage = storage
        self.embedder = embedder
        self.cache = cache
        self.retrieval = retrieval
        self.context = context
        self.extraction = extraction
        self.consolidation = consolidation
        self._started = time.monotonic()
        # set in create() when the DB was stamped by a different embedding
        # provider/model than the one now configured (vector spaces differ)
        self._embedding_mismatch: str | None = None

    # -- construction ------------------------------------------------------------

    @classmethod
    async def create(cls, settings: JaswolfSettings | None = None) -> "MemoryService":
        settings = settings or JaswolfSettings()
        logging.basicConfig(level=settings.log_level.upper())

        from .cache.redis_cache import create_cache
        from .embeddings.base import create_embedder

        storage = create_storage(settings)
        await storage.init()
        cache = create_cache(settings.redis_url, max_entries=settings.embed_cache_size)
        embedder = create_embedder(settings, cache)
        if settings.embedding_prewarm:
            warm_start = time.monotonic()
            await embedder.embed(["jaswolf prewarm"])
            logger.info("embedder prewarmed in %.1fs", time.monotonic() - warm_start)
        retrieval = RetrievalEngine(storage, embedder, settings)
        context = ContextBuilder(storage, retrieval, settings)
        extraction = ExtractionEngine(settings)
        consolidation = ConsolidationEngine(storage, embedder, settings)
        service = cls(
            settings, storage, embedder, cache, retrieval, context, extraction, consolidation
        )

        # embedding fingerprint guard: a DB's vectors only mean anything under
        # the embedder that wrote them. Stamp the first embedder into the DB;
        # opening with a different one degrades health instead of silently
        # mixing vector spaces (provider names encode model and, for hashing,
        # dim — e.g. "st:BAAI/bge-small-en-v1.5", "hashing-384").
        stamped = await storage.get_meta("embedding_fingerprint")
        if stamped is None:
            await storage.set_meta("embedding_fingerprint", embedder.name)
        elif stamped != embedder.name:
            service._embedding_mismatch = stamped
            logger.warning(
                "EMBEDDING MISMATCH: this DB was written with '%s' but the configured "
                "embedder is '%s'. Existing vectors are incompatible with new ones — "
                "search/dedup/supersession against old memories will be unreliable. "
                "Use a fresh DB, revert the embedding config, or re-embed "
                "(see TROUBLESHOOTING.md). Health will report degraded.",
                stamped,
                embedder.name,
            )
        return service

    async def close(self) -> None:
        await self.extraction.close()
        await self.embedder.close()
        await self.cache.close()
        await self.storage.close()

    # -- write path -----------------------------------------------------------------

    async def add(
        self, payload: MemoryCreate, tenant_id: str = "default"
    ) -> tuple[Memory, bool]:
        """Store a memory. Returns (memory, created).

        Write-path dedup: an exact content match (hash) or a near-duplicate
        (cosine >= dedup_threshold, same type) reinforces the existing memory
        instead of inserting a clone — repeated statements make a memory
        stronger, not the database bigger.
        """
        text = payload.content.strip()
        if not text:
            raise ValueError("memory content is empty")
        digest = content_hash(text)

        exact = await self.storage.get_by_hash(tenant_id, payload.user_id, payload.namespace, digest)
        if exact is not None:
            return await self._reinforce(exact, payload, tenant_id), False

        embedding = await self.embedder.embed_one(text)
        scope = QueryScope(
            tenant_id=tenant_id, user_id=payload.user_id, namespace=payload.namespace
        )
        near = await self.storage.find_similar(
            scope, embedding, self.settings.dedup_threshold, memory_type=payload.memory_type
        )
        if near is not None:
            return await self._reinforce(near[0], payload, tenant_id), False

        superseded = await self._find_superseded(scope, payload, embedding, text)

        memory = Memory(
            tenant_id=tenant_id,
            user_id=payload.user_id,
            agent_id=payload.agent_id,
            session_id=payload.session_id,
            namespace=payload.namespace,
            content=text,
            content_hash=digest,
            embedding=embedding,
            memory_type=payload.memory_type,
            importance=(
                payload.importance
                if payload.importance is not None
                else importance_for(payload.memory_type, text)
            ),
            confidence=payload.confidence,
            metadata=payload.metadata,
            expires_at=default_expiry(
                payload.memory_type, payload.ttl_hours, self.settings.working_ttl_hours
            ),
        )
        if superseded is not None:
            memory.metadata = {**memory.metadata, "supersedes": superseded.id}
        await self.storage.upsert(memory)
        if superseded is not None:
            await self._supersede(superseded, memory)
        return memory, True

    async def _find_superseded(
        self, scope: QueryScope, payload: MemoryCreate, embedding: list[float], text: str
    ) -> Memory | None:
        """Detect whether this memory is a correction of an existing one.

        Conservative by design: requires correction language ("actually",
        "now", "no longer", …) AND either a slot conflict ("User's office is
        X" vs "… is Y") or sufficient similarity. Unmarked contradictions are
        stored additively — wrongly superseding a real memory is worse than
        keeping both until consolidation.
        """
        if not self.settings.supersession_enabled:
            return None
        if payload.memory_type not in _SUPERSEDABLE:
            return None
        if not _CORRECTION.search(text):
            return None

        type_scope = QueryScope(
            **{**scope.__dict__, "memory_types": [payload.memory_type]}
        )
        candidates = await self.storage.search_vector(type_scope, embedding, k=3)
        new_slot = _slot_of(text)
        best: tuple[Memory, float] | None = None
        for candidate, similarity in candidates:
            old_slot = _slot_of(candidate.content)
            slot_conflict = (
                new_slot is not None
                and old_slot is not None
                and new_slot[0] == old_slot[0]
                and new_slot[1] != old_slot[1]
            )
            if slot_conflict or similarity >= self.settings.supersession_threshold:
                if best is None or similarity > best[1]:
                    best = (candidate, similarity)
        return best[0] if best else None

    async def _supersede(self, old: Memory, new: Memory) -> None:
        await self.storage.add_version(
            old.id, old.content, reason="superseded", payload={"by": new.id}
        )
        await self.storage.add_relationship(new.id, old.id, RelationType.SUPERSEDES.value)
        old.state = MemoryState.ARCHIVED
        old.updated_at = utcnow()
        await self.storage.upsert(old)
        logger.info("memory %s superseded by %s", old.id[:8], new.id[:8])

    async def _reinforce(
        self, existing: Memory, payload: MemoryCreate, tenant_id: str
    ) -> Memory:
        existing.access_count += 1
        existing.last_accessed = utcnow()
        existing.importance = max(
            existing.importance,
            payload.importance
            if payload.importance is not None
            else importance_for(payload.memory_type, payload.content),
        )
        # repetition is evidence: nudge confidence up, capped
        existing.confidence = min(1.0, existing.confidence + 0.05)
        if payload.metadata:
            existing.metadata = {**existing.metadata, **payload.metadata}
        existing.updated_at = utcnow()
        # restatement is liveness evidence: revive an archived match, else the
        # exact-hash path reinforces a row search can never see again, while a
        # mere paraphrase would have created a fresh, visible memory
        if existing.state == MemoryState.ARCHIVED:
            existing.state = MemoryState.ACTIVE
            if existing.expires_at is not None and existing.expires_at <= utcnow():
                existing.expires_at = default_expiry(
                    existing.memory_type, payload.ttl_hours, self.settings.working_ttl_hours
                )
        await self.storage.upsert(existing)
        await self.storage.record_access([existing.id], tenant_id, "reinforce")
        return existing

    async def add_many(
        self, payloads: list[MemoryCreate], tenant_id: str = "default"
    ) -> list[tuple[Memory, bool]]:
        return [await self.add(p, tenant_id) for p in payloads]

    async def ingest_messages(
        self,
        user_id: str,
        messages: list[ChatMessage],
        agent_id: str | None = None,
        session_id: str | None = None,
        namespace: str = "default",
        tenant_id: str = "default",
    ) -> list[tuple[Memory, bool]]:
        """Extract memories from a conversation and store them."""
        items = await self.extraction.extract_messages(messages)
        payloads = [
            MemoryCreate(
                user_id=user_id,
                content=item.content,
                agent_id=agent_id,
                session_id=session_id,
                namespace=namespace,
                memory_type=item.memory_type,
                importance=item.importance,
                confidence=item.confidence,
                metadata={"extracted_by": item.source},
            )
            for item in items
        ]
        return await self.add_many(payloads, tenant_id)

    async def ingest_text(
        self,
        user_id: str,
        text: str,
        tenant_id: str = "default",
        **kwargs: Any,
    ) -> list[tuple[Memory, bool]]:
        return await self.ingest_messages(
            user_id, [ChatMessage(role="user", content=text)], tenant_id=tenant_id, **kwargs
        )

    # -- read path ------------------------------------------------------------------

    async def get(self, memory_id: str, tenant_id: str = "default") -> Memory:
        memory = await self.storage.get(memory_id, tenant_id)
        if memory is None:
            raise MemoryNotFound(memory_id)
        return memory

    async def update(
        self, memory_id: str, patch: MemoryUpdate, tenant_id: str = "default"
    ) -> Memory:
        memory = await self.get(memory_id, tenant_id)
        if patch.content is not None and patch.content.strip() != memory.content:
            await self.storage.add_version(memory.id, memory.content, reason="update")
            memory.content = patch.content.strip()
            memory.content_hash = content_hash(memory.content)
            memory.embedding = await self.embedder.embed_one(memory.content)
        if patch.memory_type is not None:
            memory.memory_type = patch.memory_type
        if patch.importance is not None:
            memory.importance = max(0.0, min(1.0, patch.importance))
        if patch.confidence is not None:
            memory.confidence = max(0.0, min(1.0, patch.confidence))
        if patch.state is not None:
            memory.state = patch.state
        if patch.metadata is not None:
            memory.metadata = patch.metadata
        if patch.expires_at is not None:
            memory.expires_at = patch.expires_at
        memory.updated_at = utcnow()
        await self.storage.upsert(memory)
        return memory

    async def delete(self, memory_id: str, tenant_id: str = "default", hard: bool = False) -> None:
        deleted = await self.storage.delete(memory_id, tenant_id, hard=hard)
        if not deleted:
            raise MemoryNotFound(memory_id)

    async def search(self, query: SearchQuery, tenant_id: str = "default") -> list[ScoredMemory]:
        return await self.retrieval.search(query, tenant_id)

    async def list_memories(
        self, scope: QueryScope, limit: int = 50, offset: int = 0, order_by: str = "recent"
    ) -> list[Memory]:
        return await self.storage.list_memories(scope, limit=limit, offset=offset, order_by=order_by)

    async def build_context(
        self, request: ContextRequest, tenant_id: str = "default"
    ) -> ContextResult:
        return await self.context.build(request, tenant_id)

    async def get_versions(self, memory_id: str, tenant_id: str = "default") -> list[dict[str, Any]]:
        await self.get(memory_id, tenant_id)  # 404 if not visible in this tenant
        return await self.storage.get_versions(memory_id)

    # -- maintenance --------------------------------------------------------------------

    async def consolidate(
        self,
        user_id: str,
        tenant_id: str = "default",
        namespace: str | None = None,
        memory_types: list[MemoryType] | None = None,
        dry_run: bool = False,
    ) -> ConsolidationReport:
        return await self.consolidation.consolidate(
            tenant_id, user_id, namespace=namespace, memory_types=memory_types, dry_run=dry_run
        )

    async def sweep(self) -> SweepReport:
        now = utcnow()
        cutoffs = LifecycleCutoffs(
            now=now,
            warm_before=now - timedelta(days=self.settings.active_to_warm_days),
            cold_before=now - timedelta(days=self.settings.warm_to_cold_days),
            archive_before=now - timedelta(days=self.settings.cold_to_archived_days),
        )
        report = await self.storage.apply_lifecycle(cutoffs)
        moved = (
            report.expired_working
            + report.active_to_warm
            + report.warm_to_cold
            + report.cold_to_archived
        )
        if moved:
            logger.info(
                "lifecycle sweep: %d expired, %d->warm, %d->cold, %d->archived",
                report.expired_working,
                report.active_to_warm,
                report.warm_to_cold,
                report.cold_to_archived,
            )
        return report

    # -- health / stats -----------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        storage_ok = await self.storage.ping()
        embedder_fallback = getattr(self.embedder, "is_fallback", False)
        status = "ok"
        reasons: list[str] = []
        if not storage_ok:
            status = "degraded"
            reasons.append("storage unreachable")
        try:
            integrity = await self.storage.integrity_check()
        except Exception as exc:  # never let the health probe itself crash
            integrity = f"error: {exc}"
        if integrity != "ok":
            status = "degraded"
            reasons.append(f"storage integrity: {integrity}")
        if embedder_fallback:
            status = "degraded"
            reasons.append(
                "hash embedder active via auto-fallback — not suitable for "
                "production retrieval quality"
            )
        if self._embedding_mismatch:
            status = "degraded"
            reasons.append(
                f"embedding mismatch: DB stamped with '{self._embedding_mismatch}' but "
                f"'{self.embedder.name}' is configured — use a fresh DB, revert the "
                "config, or re-embed (TROUBLESHOOTING.md)"
            )
        result: dict[str, Any] = {
            "status": status,
            "uptime_seconds": round(time.monotonic() - self._started, 1),
            "storage": {"backend": self.storage.name, "ok": storage_ok, "integrity": integrity},
            "embeddings": {
                "provider": self.embedder.name,
                "dim": self.embedder.dim,
                "fallback": embedder_fallback,
                "cache_hits": self.embedder.hits,
                "cache_misses": self.embedder.misses,
            },
            "cache": {"backend": self.cache.name},
        }
        if reasons:
            result["reasons"] = reasons
        return result

    async def stats(self, tenant_id: str = "default", user_id: str | None = None) -> dict[str, Any]:
        return await self.storage.stats(tenant_id, user_id)
