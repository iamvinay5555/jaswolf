"""Regression tests for Jasmine's review (jasmine_feedback.md, 2026-06-10).

Her probe inputs are reproduced verbatim so the exact behaviors she flagged
can never silently regress.
"""

import pytest

from jaswolf.api.app import create_app
from jaswolf.config import JaswolfSettings
from jaswolf.extraction import RuleExtractor, apply_durability_gate
from jaswolf.models import (
    ExtractedItem,
    MemoryCreate,
    MemoryState,
    MemoryType,
    SearchQuery,
)
from jaswolf.service import MemoryService


# ---- 1. anti-ephemeral extraction gate -----------------------------------


@pytest.fixture
def engine_extract(service):
    async def run(text: str):
        return await service.extraction.extract_text(text)

    return run


async def test_transient_probe_messages(engine_extract):
    """Jasmine's exact probe set: nothing here may become durable memory."""
    probes = [
        "I want you to run the tests now.",
        "I hope this works today.",
        "Can you check if port 443 is open?",
    ]
    for probe in probes:
        items = await engine_extract(probe)
        assert items == [], f"extracted from transient message: {probe!r} -> {items}"


async def test_lunch_plan_becomes_working_memory_not_goal(engine_extract):
    items = await engine_extract("I plan to go for lunch in 10 minutes.")
    assert len(items) == 1
    assert items[0].memory_type == MemoryType.WORKING  # TTL-bound, not a durable goal
    assert items[0].confidence <= 0.6


async def test_compliment_reaction_is_dropped(engine_extract):
    assert await engine_extract("I like this answer, thanks honey.") == []
    assert await engine_extract("I love you") == []
    assert await engine_extract("I like that idea!") == []


async def test_durable_goals_and_preferences_still_extract(engine_extract):
    items = await engine_extract("I want to launch a SaaS by December.")
    assert items[0].memory_type == MemoryType.GOAL
    items = await engine_extract("I prefer Python for backend work.")
    assert items[0].memory_type == MemoryType.PREFERENCE


def test_gate_applies_to_llm_sourced_items_too():
    item = ExtractedItem(
        content="User likes this answer, thanks honey",
        memory_type=MemoryType.PREFERENCE,
        source="llm",
    )
    assert apply_durability_gate(item) is None
    plan = ExtractedItem(
        content="User wants to go for lunch in 10 minutes",
        memory_type=MemoryType.GOAL,
        source="llm",
    )
    routed = apply_durability_gate(plan)
    assert routed is not None and routed.memory_type == MemoryType.WORKING


def test_rule_extractor_raw_output_still_unfiltered():
    # the raw extractor stays gate-free by design (used for debugging);
    # the gate lives in ExtractionEngine
    items = RuleExtractor().extract("I plan to go for lunch in 10 minutes.")
    assert items and items[0].memory_type == MemoryType.GOAL


# ---- 2. supersession / correction handling ---------------------------------


async def test_preference_correction_supersedes(service):
    r1 = await service.ingest_text("alice", "I prefer tea in the morning.")
    r2 = await service.ingest_text("alice", "Actually I prefer coffee in the morning now.")
    old = r1[0][0]
    new = r2[0][0]
    assert r2[0][1] is True  # stored as a new memory…
    assert new.metadata.get("supersedes") == old.id  # …that supersedes the old one

    old_reloaded = await service.get(old.id)
    assert old_reloaded.state == MemoryState.ARCHIVED

    results = await service.search(SearchQuery(user_id="alice", query="morning drink preference"))
    contents = [s.memory.content for s in results]
    assert any("coffee" in c for c in contents)
    assert not any("tea" in c for c in contents)  # archived fact stays out of search

    versions = await service.get_versions(old.id)
    assert any(v["reason"] == "superseded" for v in versions)


async def test_fact_correction_supersedes(service):
    r1 = await service.ingest_text("alice", "My office is in Buona Vista.")
    r2 = await service.ingest_text("alice", "My office is in Changi now.")
    assert (await service.get(r1[0][0].id)).state == MemoryState.ARCHIVED
    assert (await service.get(r2[0][0].id)).state == MemoryState.ACTIVE


async def test_unmarked_contradiction_stays_additive(service):
    """Without correction language, both facts remain — wrongly superseding a
    real memory is worse than keeping both until consolidation."""
    await service.add(MemoryCreate(user_id="alice", content="User's friend is Sarah"))
    second, created = await service.add(MemoryCreate(user_id="alice", content="User's friend is Tom"))
    assert created is True
    assert "supersedes" not in second.metadata
    stats = await service.stats(user_id="alice")
    assert stats["by_state"].get("active", 0) == 2


async def test_supersession_can_be_disabled(service):
    service.settings.supersession_enabled = False
    r1 = await service.ingest_text("alice", "My office is in Buona Vista.")
    await service.ingest_text("alice", "My office is in Changi now.")
    assert (await service.get(r1[0][0].id)).state == MemoryState.ACTIVE


# ---- 3. identity-grade pinning ------------------------------------------------


async def test_low_confidence_preference_is_not_pinned(service):
    # single-shot extracted preference: confidence 0.75 < pin gate 0.8
    await service.ingest_text("pinuser", "I love oolong tea.")
    # unrelated, high-confidence explicit memory
    await service.add(
        MemoryCreate(
            user_id="pinuser",
            content="User prefers concise replies",
            memory_type=MemoryType.PREFERENCE,
            importance=0.9,
            confidence=0.9,
        )
    )
    from jaswolf.models import ContextRequest

    result = await service.build_context(
        ContextRequest(user_id="pinuser", query="how to configure postgres replication")
    )
    assert "concise replies" in result.text          # identity-grade: pinned
    assert "oolong" not in result.text               # single-shot: not pinned


async def test_reinforced_preference_retrievable_when_relevant(service):
    # Reinforcement raises confidence, but an everyday preference is NOT
    # force-pinned onto unrelated turns (2026-06-15 pin-tier change: only
    # identity-grade importance >= 0.9 force-pins). It surfaces on a RELEVANT
    # query instead — proving it's retained and retrievable, without polluting
    # off-topic context.
    await service.ingest_text("pinuser2", "I love oolong tea.")
    await service.ingest_text("pinuser2", "I love oolong tea.")
    from jaswolf.models import ContextRequest

    relevant = await service.build_context(
        ContextRequest(user_id="pinuser2", query="what kind of tea do I like?")
    )
    assert "oolong" in relevant.text  # surfaces when the query is about tea

    off_topic = await service.build_context(
        ContextRequest(user_id="pinuser2", query="how to configure postgres replication")
    )
    assert "oolong" not in off_topic.text  # no longer force-pinned onto unrelated turns


# ---- 4. hash embedder visibility ----------------------------------------------


async def test_auto_fallback_to_hash_reports_degraded(tmp_path, monkeypatch):
    # simulate sentence-transformers being absent so the test is deterministic
    # regardless of which extras the dev machine has installed (Jasmine's
    # 2026-06-11 finding: this failed on a box with local embeddings present)
    import jaswolf.embeddings.local as local_mod

    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise ImportError("sentence-transformers not installed (simulated)")

    monkeypatch.setattr(local_mod, "SentenceTransformerEmbedder", _Unavailable)
    settings = JaswolfSettings(
        database_url=f"sqlite:///{tmp_path}/fallback.db",
        embedding_provider="auto",  # no local model, no API key -> hash fallback
        openai_api_key=None,
        log_level="WARNING",
    )
    service = await MemoryService.create(settings)
    try:
        health = await service.health()
        assert health["status"] == "degraded"
        assert health["embeddings"]["fallback"] is True
        assert any("hash embedder" in r for r in health["reasons"])
    finally:
        await service.close()


async def test_explicit_hash_choice_is_ok(service):
    health = await service.health()
    assert health["status"] == "ok"
    assert health["embeddings"]["fallback"] is False


# ---- 6. security defaults ---------------------------------------------------------


def test_api_refuses_to_start_without_keys_or_dev_mode(tmp_path):
    settings = JaswolfSettings(
        database_url=f"sqlite:///{tmp_path}/sec.db",
        api_keys="",
        dev_open_mode=False,
    )
    with pytest.raises(RuntimeError, match="JASWOLF_API_KEYS"):
        create_app(settings=settings)


def test_api_starts_with_keys_or_dev_mode(tmp_path):
    base = dict(database_url=f"sqlite:///{tmp_path}/sec2.db", embedding_provider="hash")
    create_app(settings=JaswolfSettings(**base, api_keys="k1:t1"))
    create_app(settings=JaswolfSettings(**base, api_keys="", dev_open_mode=True))
