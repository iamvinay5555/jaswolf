"""Context builder: assemble the optimal memory block for an LLM prompt.

Pipeline: derive query from conversation -> hybrid retrieval + pinned
preferences/goals -> cross-section dedup -> per-section token budgeting ->
greedy fill by final score -> render (markdown or xml).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .calibration import context_similarity_threshold
from .config import JaswolfSettings
from .models import (
    ContextRequest,
    ContextResult,
    ContextSection,
    MemoryState,
    MemoryType,
    ScoredMemory,
    SearchMode,
    SearchQuery,
)
from .retrieval import RetrievalEngine
from .scoring import score_memory
from .temporal import resolve_current_state
from .storage.base import QueryScope, StorageBackend
from .tokens import estimate_tokens, truncate_to_tokens

logger = logging.getLogger("jaswolf.context")

_SECTION_ORDER: list[tuple[MemoryType, str]] = [
    (MemoryType.PREFERENCE, "Preferences"),
    (MemoryType.GOAL, "Goals"),
    (MemoryType.RELATIONSHIP, "Relationships"),
    (MemoryType.SEMANTIC, "Facts"),
    (MemoryType.PROCEDURAL, "Procedures"),
    (MemoryType.EPISODIC, "Recent context"),
    (MemoryType.WORKING, "Active session notes"),
]

_QUERY_MESSAGE_WINDOW = 4
_MIN_TRUNCATED_TOKENS = 30

# Markers that identify staging/test memories which must never enter a live
# prompt. Narrow on purpose to avoid false positives on real content; the
# robust path is metadata {"test": true} / {"staging": true}, which test
# harnesses and the migration tool should set.
_TEST_CONTENT_MARKERS = ("staging_test", "jaswolf_test", "__smoke_test__")


def _is_test_memory(memory) -> bool:
    md = memory.metadata or {}
    if md.get("test") or md.get("staging"):
        return True
    low = memory.content.lower()
    return any(marker in low for marker in _TEST_CONTENT_MARKERS)


class ContextBuilder:
    def __init__(self, storage: StorageBackend, retrieval: RetrievalEngine, settings: JaswolfSettings):
        self.storage = storage
        self.retrieval = retrieval
        self.settings = settings
        self.last_latency_ms: float = 0.0
        self._bg_matrix: np.ndarray | None = None  # cached corpus background sample

    async def _background(self, scope: QueryScope) -> np.ndarray | None:
        """Cached sample of corpus embeddings for noise calibration. Staleness
        is fine — it estimates a distribution, not a fact."""
        if self._bg_matrix is None:
            vecs = await self.storage.sample_embeddings(
                scope, self.settings.context_background_sample
            )
            self._bg_matrix = np.asarray(vecs, dtype=np.float32) if vecs else np.empty((0, 0))
        return self._bg_matrix if self._bg_matrix.size else None

    async def similarity_gate(self, query_vec: list[float], scope: QueryScope) -> float:
        """Raw-cosine threshold a non-pinned candidate must clear to be injected.
        Single source of truth, shared by _gather and the eval harness."""
        return await context_similarity_threshold(
            self.retrieval.embedder,
            query_vec,
            await self._background(scope),
            noise_z=self.settings.context_noise_z,
            margin=self.settings.context_similarity_margin,
            min_background=self.settings.context_min_background,
        )

    async def build(self, request: ContextRequest, tenant_id: str = "default") -> ContextResult:
        start = time.perf_counter()
        budget = request.token_budget or self.settings.context_token_budget
        query_text = self._derive_query(request)

        candidates = await self._gather(request, query_text, tenant_id)
        if self.settings.temporal_resolution:
            candidates, stale = resolve_current_state(candidates)
            if stale:
                logger.debug(
                    "temporal: dropped %d stale same-slot fact(s) from context", len(stale)
                )
        candidates = self._dedupe(candidates)
        result = self._assemble(request, candidates, budget)

        self.last_latency_ms = (time.perf_counter() - start) * 1000
        return result

    # -- query derivation ----------------------------------------------------

    @staticmethod
    def _derive_query(request: ContextRequest) -> str:
        if request.query:
            return request.query
        if request.messages:
            user_turns = [m.content for m in request.messages if m.role == "user"]
            return "\n".join(user_turns[-_QUERY_MESSAGE_WINDOW:])[:2000]
        return ""

    # -- candidate gathering ----------------------------------------------------

    async def _gather(
        self, request: ContextRequest, query_text: str, tenant_id: str
    ) -> list[ScoredMemory]:
        types = [t for t, _ in _SECTION_ORDER if t != MemoryType.WORKING]
        if request.session_id:
            types.append(MemoryType.WORKING)

        # Multi-agent reads: an agent sees its own namespace + the shared one
        # (every-bot user facts), in one query. Writes/session stay single-scope.
        # Scoping is by NAMESPACE (the isolation axis); agent_id is provenance
        # (who wrote a memory), NOT a read filter. Filtering context reads by
        # agent_id silently hid non-pinned SHARED facts from any bot whose
        # agent_id differed from the writer's — the pin path already ignores
        # agent_id, so the two paths disagreed and the shared namespace only
        # half-worked. Namespace already isolates bots; agent_id stays available
        # for explicit provenance queries via the API. (2026-06-19 audit)
        own_ns = request.namespace or "default"
        read_namespaces = (
            [own_ns, request.shared_namespace]
            if request.shared_namespace and request.shared_namespace != own_ns
            else None
        )

        search = SearchQuery(
            user_id=request.user_id,
            query=query_text,
            namespace=request.namespace,
            namespaces=read_namespaces,
            memory_types=types,
            mode=SearchMode.HYBRID if query_text else SearchMode.RECENCY,
            top_k=self.settings.context_candidate_pool,
            record_access=True,
        )
        by_id: dict[str, ScoredMemory] = {
            s.memory.id: s for s in await self.retrieval.search(search, tenant_id)
        }

        # Context-boundary gate. Search may *rank* weak candidates — injecting
        # them into a prompt needs evidence. A non-pinned vector hit must clear
        # the query's calibrated similarity threshold (mean + z·std vs a corpus
        # background sample); bge-small scores arbitrary English ~0.6, so the
        # retrieval-side min_relevance floor cannot protect the prompt, and a
        # fixed anchor floor under-measures it. Keyword-evidenced hits carry
        # discriminative lexical proof (keywords.py) and are exempt. Pins and
        # session working notes are added below and stay exempt.
        if query_text and self.settings.context_noise_z > 0 and by_id:
            query_vec = await self.retrieval.embedder.embed_one(query_text)  # cache hit
            gate_scope = QueryScope(
                tenant_id=tenant_id, user_id=request.user_id,
                namespace=request.namespace, namespaces=read_namespaces,
            )
            gate = await self.similarity_gate(query_vec, gate_scope)
            dropped = [
                s for s in by_id.values()
                if not s.keyword_match and s.similarity is not None and s.similarity < gate
            ]
            for scored in dropped:
                del by_id[scored.memory.id]
            if dropped:
                logger.debug(
                    "context gate dropped %d candidates below %.3f", len(dropped), gate
                )

        # Force-pinned tier: only IDENTITY/SAFETY-grade preferences/goals belong
        # in EVERY context regardless of query — those explicitly marked
        # metadata.always_pin, OR at/above context_always_pin_importance — both
        # still clearing the confidence gate, capped by context_max_pins.
        # Lower-importance, unmarked preferences are NOT force-injected; they
        # appear only when the query-driven search above surfaced them. This
        # stops a stray/staging preference from dominating every turn
        # (2026-06-15 incident) while keeping true guardrails always present.
        pin_floor = self.settings.context_always_pin_importance

        def _force_pins(memory) -> bool:
            if memory.confidence < self.settings.pin_min_confidence:
                return False
            if (memory.metadata or {}).get("always_pin"):
                return True
            if self.settings.context_pin_requires_always_pin:
                return False  # strict: only the explicit flag force-pins
            return memory.importance >= pin_floor

        pinned_count = 0
        for pinned_type, limit in ((MemoryType.PREFERENCE, 8), (MemoryType.GOAL, 5)):
            if pinned_count >= self.settings.context_max_pins:
                break
            scope = QueryScope(
                tenant_id=tenant_id,
                user_id=request.user_id,
                namespace=request.namespace,
                namespaces=read_namespaces,
                memory_types=[pinned_type],
                min_importance=self.settings.pin_min_importance,  # fetch wide; filter below
            )
            candidates = await self.storage.list_memories(
                scope, limit=limit * 2, order_by="importance", include_embeddings=True
            )
            pinned = [m for m in candidates if _force_pins(m)][:limit]
            for memory in pinned:
                if pinned_count >= self.settings.context_max_pins:
                    break
                if memory.id not in by_id:
                    by_id[memory.id] = score_memory(memory, relevance=0.5, settings=self.settings)
                    pinned_count += 1

        # Working memories for the active session, newest first.
        if request.session_id:
            scope = QueryScope(
                tenant_id=tenant_id,
                user_id=request.user_id,
                session_id=request.session_id,
                namespace=request.namespace,
                memory_types=[MemoryType.WORKING],
                states=[MemoryState.ACTIVE],
            )
            for memory in await self.storage.list_memories(
                scope, limit=6, order_by="recent", include_embeddings=True
            ):
                if memory.id not in by_id:
                    by_id[memory.id] = score_memory(memory, relevance=0.6, settings=self.settings)

        # Provenance guard: staging/test memories must never reach a live prompt,
        # however they were typed (the 2026-06-15 incident was a STAGING_TEST_
        # preference pinned into every context).
        if self.settings.exclude_test_memories:
            for scored in [s for s in by_id.values() if _is_test_memory(s.memory)]:
                del by_id[scored.memory.id]

        return list(by_id.values())

    # -- dedup --------------------------------------------------------------------

    def _dedupe(self, candidates: list[ScoredMemory]) -> list[ScoredMemory]:
        if len(candidates) < 2:
            return candidates
        threshold = self.settings.context_dedup_threshold
        ordered = sorted(candidates, key=lambda s: s.final_score, reverse=True)
        kept: list[ScoredMemory] = []
        kept_vecs: list[np.ndarray] = []
        seen_hashes: set[str] = set()
        for scored in ordered:
            if scored.memory.content_hash and scored.memory.content_hash in seen_hashes:
                continue
            vec = (
                np.asarray(scored.memory.embedding, dtype=np.float32)
                if scored.memory.embedding
                else None
            )
            if vec is not None and any(
                vec.shape == kv.shape and float(vec @ kv) >= threshold for kv in kept_vecs
            ):
                continue
            kept.append(scored)
            if vec is not None:
                kept_vecs.append(vec)
            if scored.memory.content_hash:
                seen_hashes.add(scored.memory.content_hash)
        return kept

    # -- assembly --------------------------------------------------------------------

    def _assemble(
        self, request: ContextRequest, candidates: list[ScoredMemory], budget: int
    ) -> ContextResult:
        if not candidates:
            return ContextResult(text="", token_budget=budget)

        grouped: dict[MemoryType, list[ScoredMemory]] = {}
        for scored in candidates:
            grouped.setdefault(scored.memory.memory_type, []).append(scored)
        for members in grouped.values():
            members.sort(key=lambda s: s.final_score, reverse=True)

        shares = self.settings.context_shares()
        shares["working"] = 0.10
        active_shares = {
            mtype: shares.get(mtype.value, 0.1) for mtype, _ in _SECTION_ORDER if mtype in grouped
        }
        total_share = sum(active_shares.values()) or 1.0

        header_cost = 8  # block header overhead
        usable = max(0, budget - header_cost)
        selected: dict[MemoryType, list[tuple[ScoredMemory, str]]] = {}
        leftovers: list[tuple[ScoredMemory, str]] = []
        used_tokens = 0
        truncated = False

        # pass 1: per-section share
        for mtype, _title in _SECTION_ORDER:
            if mtype not in grouped:
                continue
            section_budget = int(usable * active_shares[mtype] / total_share)
            section_used = 0
            for scored in grouped[mtype]:
                line = self._format_line(scored, request.include_ids)
                cost = estimate_tokens(line) + 1
                if section_used + cost <= section_budget:
                    selected.setdefault(mtype, []).append((scored, line))
                    section_used += cost
                else:
                    leftovers.append((scored, line))
            used_tokens += section_used

        # pass 2: spend leftover budget on the best remaining, regardless of section
        leftovers.sort(key=lambda pair: pair[0].final_score, reverse=True)
        section_title_cost = 4
        committed = used_tokens + section_title_cost * len(selected)
        for scored, line in leftovers:
            cost = estimate_tokens(line) + 1
            if scored.memory.memory_type not in selected:
                cost += section_title_cost  # opening a new section costs its header
            if committed + cost <= usable:
                selected.setdefault(scored.memory.memory_type, []).append((scored, line))
                committed += cost
            elif usable - committed >= _MIN_TRUNCATED_TOKENS:
                short = truncate_to_tokens(line, usable - committed - section_title_cost - 1)
                selected.setdefault(scored.memory.memory_type, []).append((scored, short))
                committed += estimate_tokens(short) + 1
                truncated = True
            else:
                truncated = True

        # final guarantee: render, and evict lowest-scored lines until the
        # actual rendered size fits the budget (per-line estimates can
        # undercount the joined text slightly)
        result = self._render(request, selected, budget, truncated)
        while result.token_estimate > budget and any(selected.values()):
            worst_type, worst_idx = min(
                (
                    (mtype, idx)
                    for mtype, items in selected.items()
                    for idx in range(len(items))
                ),
                key=lambda loc: selected[loc[0]][loc[1]][0].final_score,
            )
            selected[worst_type].pop(worst_idx)
            if not selected[worst_type]:
                del selected[worst_type]
            result = self._render(request, selected, budget, truncated=True)
        return result

    @staticmethod
    def _format_line(scored: ScoredMemory, include_ids: bool) -> str:
        memory = scored.memory
        line = memory.content.strip()
        if memory.memory_type == MemoryType.EPISODIC:
            line = f"[{memory.created_at.date().isoformat()}] {line}"
        if include_ids:
            line += f" (mem:{memory.id[:8]})"
        return line

    def _render(
        self,
        request: ContextRequest,
        selected: dict[MemoryType, list[tuple[ScoredMemory, str]]],
        budget: int,
        truncated: bool,
    ) -> ContextResult:
        sections: list[ContextSection] = []
        memories_used: list[ScoredMemory] = []
        parts: list[str] = []
        xml = request.format.lower() == "xml"

        parts.append("<memories>" if xml else "# What I remember about this user")
        for mtype, title in _SECTION_ORDER:
            items = selected.get(mtype)
            if not items:
                continue
            tag = title.lower().replace(" ", "_")
            parts.append(f"<{tag}>" if xml else f"\n## {title}")
            section_tokens = 0
            for scored, line in items:
                rendered = f"<memory>{line}</memory>" if xml else f"- {line}"
                parts.append(rendered)
                section_tokens += estimate_tokens(rendered) + 1
                memories_used.append(scored)
            if xml:
                parts.append(f"</{tag}>")
            sections.append(
                ContextSection(title=title, memory_type=mtype, count=len(items), tokens=section_tokens)
            )
        if xml:
            parts.append("</memories>")

        text = "\n".join(parts) if memories_used else ""
        return ContextResult(
            text=text,
            memories=memories_used,
            sections=sections,
            token_estimate=estimate_tokens(text),
            token_budget=budget,
            truncated=truncated,
        )
