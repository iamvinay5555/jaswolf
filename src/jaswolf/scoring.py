"""Memory scoring engine.

final_score = w_i * importance + w_rel * relevance + w_rec * recency + w_f * frequency
(default weights 0.4 / 0.3 / 0.2 / 0.1, configurable).

Recency uses exponential decay with a configurable half-life; frequency uses
log saturation so a memory accessed 500 times doesn't drown out everything.
"""

from __future__ import annotations

import math
import re
from datetime import datetime

from .config import JaswolfSettings
from .models import Memory, MemoryType, ScoredMemory, utcnow

# Baseline importance by type: durable identity-shaping memories rank higher.
TYPE_BASE_IMPORTANCE: dict[MemoryType, float] = {
    MemoryType.PREFERENCE: 0.70,
    MemoryType.GOAL: 0.75,
    MemoryType.RELATIONSHIP: 0.65,
    MemoryType.SEMANTIC: 0.55,
    MemoryType.PROCEDURAL: 0.60,
    MemoryType.EPISODIC: 0.40,
    MemoryType.WORKING: 0.30,
}

_EMPHASIS = re.compile(
    r"\b(always|never|must|critical|important|love|hate|essential|favorite|favourite|strongly)\b|!",
    re.IGNORECASE,
)
_EXPLICIT_REMEMBER = re.compile(r"\b(remember|don't forget|note that|keep in mind)\b", re.IGNORECASE)


def importance_for(memory_type: MemoryType, content: str) -> float:
    """Heuristic importance when the caller doesn't supply one."""
    score = TYPE_BASE_IMPORTANCE.get(memory_type, 0.5)
    if _EMPHASIS.search(content):
        score += 0.10
    if _EXPLICIT_REMEMBER.search(content):
        score += 0.15
    return min(1.0, score)


def recency_score(last_activity: datetime, half_life_days: float, now: datetime | None = None) -> float:
    now = now or utcnow()
    age_days = max(0.0, (now - last_activity).total_seconds() / 86400.0)
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def frequency_score(access_count: int, saturation: int = 50) -> float:
    if access_count <= 0:
        return 0.0
    return min(1.0, math.log1p(access_count) / math.log1p(max(2, saturation)))


def final_score(
    importance: float,
    relevance: float,
    recency: float,
    frequency: float,
    settings: JaswolfSettings,
) -> float:
    total_weight = (
        settings.weight_importance
        + settings.weight_relevance
        + settings.weight_recency
        + settings.weight_frequency
    ) or 1.0
    raw = (
        settings.weight_importance * importance
        + settings.weight_relevance * relevance
        + settings.weight_recency * recency
        + settings.weight_frequency * frequency
    )
    return raw / total_weight


def score_memory(
    memory: Memory,
    relevance: float,
    settings: JaswolfSettings,
    now: datetime | None = None,
) -> ScoredMemory:
    rec = recency_score(memory.last_activity(), settings.recency_half_life_days, now)
    freq = frequency_score(memory.access_count, settings.frequency_saturation)
    return ScoredMemory(
        memory=memory,
        relevance=relevance,
        recency=rec,
        frequency=freq,
        final_score=final_score(memory.importance, relevance, rec, freq, settings),
    )


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over ranked id lists.

    Robust hybrid fusion without score calibration: each list contributes
    1/(k + rank). Returns id -> fused score.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return fused
