"""Memory consolidation: find clusters of near-duplicate memories and merge
them into one canonical memory, preserving history.

Merge strategy is deterministic by default (containment / sentence union);
when an LLM endpoint is configured it can rewrite the merged statement.
Losers are soft-deleted and linked to the canonical via merged_into.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from .config import JaswolfSettings
from .models import (
    ConsolidationMerge,
    ConsolidationReport,
    Memory,
    MemoryState,
    MemoryType,
    RelationType,
    content_hash,
    utcnow,
)
from .storage.base import QueryScope, StorageBackend

logger = logging.getLogger("jaswolf.consolidation")

# Working memories are transient; consolidating them is wasted effort.
_CONSOLIDATABLE = [t for t in MemoryType if t != MemoryType.WORKING]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def merge_contents(contents: list[str]) -> str:
    """Deterministic merge: if one statement contains another, keep the most
    informative; otherwise union distinct sentences."""
    ordered = sorted(contents, key=len, reverse=True)
    base = ordered[0]
    base_norm = re.sub(r"\W+", " ", base.lower())
    extras: list[str] = []
    for other in ordered[1:]:
        other_norm = re.sub(r"\W+", " ", other.lower())
        if other_norm in base_norm:
            continue
        # append sentences that add new information
        for sentence in re.split(r"(?<=[.!?])\s+", other):
            s_norm = re.sub(r"\W+", " ", sentence.lower()).strip()
            if s_norm and s_norm not in base_norm:
                extras.append(sentence.strip().rstrip("."))
                base_norm += " " + s_norm
    merged = base.rstrip(".")
    if extras:
        merged += ". " + ". ".join(extras)
    return merged + "."


class ConsolidationEngine:
    def __init__(self, storage: StorageBackend, embedder, settings: JaswolfSettings, llm_merge=None):
        self.storage = storage
        self.embedder = embedder
        self.settings = settings
        self.llm_merge = llm_merge  # optional async (list[str]) -> str

    async def consolidate(
        self,
        tenant_id: str,
        user_id: str,
        namespace: str | None = None,
        memory_types: list[MemoryType] | None = None,
        threshold: float | None = None,
        dry_run: bool = False,
    ) -> ConsolidationReport:
        threshold = threshold if threshold is not None else self.settings.consolidation_threshold
        report = ConsolidationReport(dry_run=dry_run)
        types = memory_types or _CONSOLIDATABLE

        for mtype in types:
            scope = QueryScope(
                tenant_id=tenant_id,
                user_id=user_id,
                namespace=namespace,
                memory_types=[mtype],
                states=[MemoryState.ACTIVE, MemoryState.WARM, MemoryState.COLD],
            )
            memories = await self.storage.list_memories(
                scope,
                limit=self.settings.consolidation_max_batch,
                order_by="created",
                include_embeddings=True,
            )
            memories = [m for m in memories if m.embedding]
            report.examined += len(memories)
            if len(memories) < 2:
                continue

            matrix = np.asarray([m.embedding for m in memories], dtype=np.float32)
            sims = matrix @ matrix.T  # normalized vectors -> cosine
            uf = _UnionFind(len(memories))
            pair_sim: dict[tuple[int, int], float] = {}
            rows, cols = np.where(np.triu(sims, k=1) >= threshold)
            for i, j in zip(rows.tolist(), cols.tolist()):
                uf.union(i, j)
                pair_sim[(i, j)] = float(sims[i, j])

            clusters: dict[int, list[int]] = {}
            for idx in range(len(memories)):
                clusters.setdefault(uf.find(idx), []).append(idx)

            for members in clusters.values():
                if len(members) < 2:
                    continue
                report.clusters_found += 1
                cluster = [memories[i] for i in members]
                merge = await self._merge_cluster(cluster, pair_sim, members, dry_run)
                report.merges.append(merge)
                report.memories_merged += len(merge.merged_ids)

        return report

    async def _merge_cluster(
        self,
        cluster: list[Memory],
        pair_sim: dict[tuple[int, int], float],
        member_indices: list[int],
        dry_run: bool,
    ) -> ConsolidationMerge:
        canonical = max(cluster, key=lambda m: (m.importance, m.access_count, m.created_at))
        others = [m for m in cluster if m.id != canonical.id]

        contents = [m.content for m in cluster]
        if self.llm_merge is not None:
            try:
                merged_content = await self.llm_merge(contents)
            except Exception as exc:
                logger.warning("LLM merge failed, using deterministic merge: %s", exc)
                merged_content = merge_contents(contents)
        else:
            merged_content = merge_contents(contents)

        cluster_sims = [
            s for (i, j), s in pair_sim.items() if i in member_indices and j in member_indices
        ]
        avg_sim = sum(cluster_sims) / len(cluster_sims) if cluster_sims else 1.0

        merge = ConsolidationMerge(
            canonical_id=canonical.id,
            merged_ids=[m.id for m in others],
            merged_content=merged_content,
            similarity=round(avg_sim, 4),
        )
        if dry_run:
            return merge

        # preserve the canonical's previous content before rewriting it
        await self.storage.add_version(
            canonical.id,
            canonical.content,
            reason="consolidation",
            payload={"merged_ids": merge.merged_ids, "similarity": merge.similarity},
        )
        canonical.content = merged_content
        canonical.content_hash = content_hash(merged_content)
        canonical.embedding = await self.embedder.embed_one(merged_content)
        canonical.importance = max(m.importance for m in cluster)
        total_conf_weight = sum(m.access_count + 1 for m in cluster)
        canonical.confidence = (
            sum(m.confidence * (m.access_count + 1) for m in cluster) / total_conf_weight
        )
        canonical.access_count = sum(m.access_count for m in cluster)
        canonical.updated_at = utcnow()
        await self.storage.upsert(canonical)

        for other in others:
            await self.storage.add_version(
                other.id, other.content, reason="merged_away", payload={"into": canonical.id}
            )
            await self.storage.add_relationship(
                other.id, canonical.id, RelationType.MERGED_INTO.value
            )
            await self.storage.delete(other.id, canonical.tenant_id, hard=False)

        return merge
