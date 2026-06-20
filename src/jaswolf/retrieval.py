"""Memory retrieval engine.

Five modes: semantic (vector), keyword (FTS), hybrid (RRF fusion of both),
recency, importance. All results are scored through the scoring engine so the
final ranking always blends importance, relevance, recency, and frequency.

Internally candidates travel as (memory, relevance, raw_similarity) triples:
relevance is what the scorer consumes (normalized per pool), raw_similarity is
the unnormalized cosine preserved for honest reporting — per-pool min-max can
show 1.0 for an off-topic query, raw cosine cannot.
"""

from __future__ import annotations

import logging
import time

from .config import JaswolfSettings
from .models import Memory, ScoredMemory, SearchMode, SearchQuery, SEARCHABLE_STATES
from .scoring import rrf_fuse, score_memory
from .storage.base import QueryScope, StorageBackend

logger = logging.getLogger("jaswolf.retrieval")

Candidate = tuple[Memory, float, float | None, bool]  # (memory, relevance, raw cosine, keyword hit)

_QUERY_MODES = (SearchMode.SEMANTIC, SearchMode.KEYWORD, SearchMode.HYBRID)


def _normalize_relevance(results: list[Candidate]) -> list[Candidate]:
    """Min-max normalize relevance within the candidate pool.

    Embedding models differ wildly in cosine calibration (bge "unrelated" can
    be 0.35 while another model's is 0.05). The final-score formula assumes
    relevance spans ~0..1, so we restore that spread per query. A pool whose
    scores are nearly uniform is left untouched — there the relative order
    carries no signal and other factors should decide. Raw similarity rides
    along untouched.
    """
    if len(results) < 2:
        return results
    values = [rel for _, rel, _, _ in results]
    lo, hi = min(values), max(values)
    if hi - lo < 0.05:
        return results
    return [(m, (rel - lo) / (hi - lo), raw, kw) for m, rel, raw, kw in results]


def scope_from_query(query: SearchQuery, tenant_id: str) -> QueryScope:
    return QueryScope(
        tenant_id=tenant_id,
        user_id=query.user_id,
        agent_id=query.agent_id,
        session_id=query.session_id,
        namespace=query.namespace,
        namespaces=query.namespaces,
        memory_types=query.memory_types,
        states=query.states or list(SEARCHABLE_STATES),
        min_importance=query.min_importance,
    )


class RetrievalEngine:
    def __init__(self, storage: StorageBackend, embedder, settings: JaswolfSettings):
        self.storage = storage
        self.embedder = embedder
        self.settings = settings
        self.last_latency_ms: float = 0.0

    async def search(self, query: SearchQuery, tenant_id: str = "default") -> list[ScoredMemory]:
        start = time.perf_counter()
        # an empty query in a query-driven mode used to silently degrade to a
        # recent-order listing with flat relevance 0.5 — the exact footgun that
        # made an eval pass look real. Fail loudly; listings are explicit modes.
        if query.mode in _QUERY_MODES and not query.query.strip():
            raise ValueError(
                f"query text is required for mode={query.mode.value}; "
                "use mode=recency or mode=importance for listings"
            )
        scope = scope_from_query(query, tenant_id)
        pool_k = max(query.top_k * 4, 24)

        if query.mode == SearchMode.SEMANTIC:
            results = _normalize_relevance(await self._semantic(scope, query.query, pool_k))
        elif query.mode == SearchMode.KEYWORD:
            results = _normalize_relevance(await self._keyword(scope, query.query, pool_k))
        elif query.mode == SearchMode.HYBRID:
            results = _normalize_relevance(await self._hybrid(scope, query.query, pool_k))
        elif query.mode == SearchMode.RECENCY:
            results = await self._listing(scope, pool_k, order_by="recent")
        elif query.mode == SearchMode.IMPORTANCE:
            results = await self._listing(scope, pool_k, order_by="importance")
        else:
            raise ValueError(f"unknown search mode: {query.mode}")

        scored: list[ScoredMemory] = []
        for memory, relevance, raw, kw in results:
            s = score_memory(memory, relevance, self.settings)
            s.similarity = raw
            s.keyword_match = kw
            scored.append(s)
        scored.sort(key=lambda s: s.final_score, reverse=True)
        if query.min_score is not None:
            scored = [s for s in scored if s.final_score >= query.min_score]
        scored = scored[: query.top_k]

        if query.record_access and scored:
            try:
                await self.storage.record_access(
                    [s.memory.id for s in scored], tenant_id, "search", query.query
                )
            except Exception as exc:
                logger.warning("access recording failed: %s", exc)

        self.last_latency_ms = (time.perf_counter() - start) * 1000
        return scored

    async def _semantic(self, scope: QueryScope, text: str, k: int) -> list[Candidate]:
        vec = await self.embedder.embed_one(text)
        hits = await self.storage.search_vector(scope, vec, k)
        floor = self.settings.min_relevance
        return [(m, max(0.0, sim), sim, False) for m, sim in hits if sim >= floor]

    async def _keyword(self, scope: QueryScope, text: str, k: int) -> list[Candidate]:
        hits = await self.storage.search_keyword(scope, text, k)
        return [(m, score, None, True) for m, score in hits]

    async def _hybrid(self, scope: QueryScope, text: str, k: int) -> list[Candidate]:
        vec = await self.embedder.embed_one(text)
        vector_hits = await self.storage.search_vector(scope, vec, k)
        # a textual match is its own evidence of relevance, so the floor only
        # applies to vector-sourced candidates
        vector_hits = [(m, s) for m, s in vector_hits if s >= self.settings.min_relevance]
        keyword_hits = await self.storage.search_keyword(scope, text, k)

        by_id: dict[str, Memory] = {}
        vec_sim: dict[str, float] = {}
        for memory, sim in vector_hits:
            by_id[memory.id] = memory
            vec_sim[memory.id] = max(0.0, sim)
        for memory, _ in keyword_hits:
            by_id.setdefault(memory.id, memory)

        keyword_ids = {m.id for m, _ in keyword_hits}
        fused = rrf_fuse(
            [[m.id for m, _ in vector_hits], [m.id for m, _ in keyword_hits]]
        )
        max_fused = max(fused.values(), default=1.0)
        results: list[Candidate] = []
        for memory_id, rrf_score in fused.items():
            # blend: RRF rank position (calibration-free) + raw cosine when known
            relevance = 0.6 * (rrf_score / max_fused) + 0.4 * vec_sim.get(memory_id, 0.0)
            results.append(
                (by_id[memory_id], relevance, vec_sim.get(memory_id), memory_id in keyword_ids)
            )
        results.sort(key=lambda c: c[1], reverse=True)
        return results[:k]

    async def _listing(self, scope: QueryScope, k: int, order_by: str) -> list[Candidate]:
        memories = await self.storage.list_memories(scope, limit=k, order_by=order_by)
        # neutral relevance: ranking driven by importance/recency/frequency
        return [(m, 0.5, None, False) for m in memories]
