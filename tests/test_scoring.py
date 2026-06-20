from datetime import timedelta

from jaswolf.config import JaswolfSettings
from jaswolf.models import MemoryType, utcnow
from jaswolf.scoring import (
    final_score,
    frequency_score,
    importance_for,
    recency_score,
    rrf_fuse,
)


def test_recency_half_life():
    now = utcnow()
    assert recency_score(now, half_life_days=7.0, now=now) == 1.0
    week_old = recency_score(now - timedelta(days=7), half_life_days=7.0, now=now)
    assert abs(week_old - 0.5) < 1e-6
    fortnight_old = recency_score(now - timedelta(days=14), half_life_days=7.0, now=now)
    assert abs(fortnight_old - 0.25) < 1e-6


def test_frequency_saturates():
    assert frequency_score(0) == 0.0
    assert frequency_score(1) > 0.0
    assert frequency_score(10) > frequency_score(5)
    assert frequency_score(50, saturation=50) == 1.0
    assert frequency_score(5000, saturation=50) == 1.0  # capped


def test_final_score_uses_configured_weights():
    settings = JaswolfSettings(api_keys="")
    score = final_score(importance=1.0, relevance=0.0, recency=0.0, frequency=0.0, settings=settings)
    assert abs(score - 0.4) < 1e-6  # default importance weight
    perfect = final_score(1.0, 1.0, 1.0, 1.0, settings)
    assert abs(perfect - 1.0) < 1e-6


def test_importance_heuristics():
    plain_fact = importance_for(MemoryType.SEMANTIC, "User works at a bank")
    emphatic = importance_for(MemoryType.SEMANTIC, "User ALWAYS deploys on Fridays!")
    assert emphatic > plain_fact
    assert importance_for(MemoryType.PREFERENCE, "x") > importance_for(MemoryType.EPISODIC, "x")
    assert importance_for(MemoryType.GOAL, "remember that I want this") <= 1.0


def test_rrf_fusion_prefers_items_in_both_lists():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])
    assert fused["a"] > fused["c"]
    assert fused["b"] > fused["d"]
    assert set(fused) == {"a", "b", "c", "d"}
