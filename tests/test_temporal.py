"""Read-time current-state resolution (mem0-inspired, lightweight, v0.6.0)."""

from datetime import timedelta

from jaswolf.models import (
    ContextRequest,
    Memory,
    MemoryCreate,
    MemoryType,
    ScoredMemory,
    utcnow,
)
from jaswolf.temporal import resolve_current_state


def _sm(content, *, minutes_ago=0, memory_type=MemoryType.SEMANTIC, importance=0.5):
    ts = utcnow() - timedelta(minutes=minutes_ago)
    mem = Memory(
        user_id="alice", content=content, memory_type=memory_type,
        importance=importance, created_at=ts, updated_at=ts,
    )
    return ScoredMemory(memory=mem, final_score=0.8)


# ---- pure function ---------------------------------------------------------


def test_singleton_conflict_keeps_freshest():
    stale = _sm("User's office is in Buona Vista", minutes_ago=60)
    fresh = _sm("User's office is in Changi", minutes_ago=1)
    kept, dropped = resolve_current_state([stale, fresh])
    assert [s.memory.content for s in kept] == ["User's office is in Changi"]
    assert dropped == [stale]


def test_multivalued_relation_is_not_collapsed():
    a = _sm("User's friend is Sarah", minutes_ago=60)
    b = _sm("User's friend is Tom", minutes_ago=1)
    kept, dropped = resolve_current_state([a, b])
    assert dropped == []
    assert len(kept) == 2  # "friend" is not a singleton slot — both survive


def test_same_value_restatements_all_kept():
    a = _sm("User's office is Changi", minutes_ago=60)
    b = _sm("User's office is Changi", minutes_ago=1)
    kept, dropped = resolve_current_state([a, b])
    assert dropped == []
    assert len(kept) == 2  # identical value -> dedup handles it, not temporal


def test_different_types_not_collapsed_across():
    pref = _sm("User's role is founder", minutes_ago=60, memory_type=MemoryType.PREFERENCE)
    sem = _sm("User's role is engineer", minutes_ago=1, memory_type=MemoryType.SEMANTIC)
    kept, dropped = resolve_current_state([pref, sem])
    assert dropped == []  # different memory types are different intents
    assert len(kept) == 2


def test_non_slot_memories_pass_through():
    a = _sm("Prefers dark roast coffee")
    b = _sm("Deploys with Docker Compose")
    kept, dropped = resolve_current_state([a, b])
    assert dropped == []
    assert len(kept) == 2


def test_three_way_conflict_keeps_only_newest():
    m1 = _sm("User's city is Singapore", minutes_ago=120)
    m2 = _sm("User's city is Tokyo", minutes_ago=60)
    m3 = _sm("User's city is Lisbon", minutes_ago=1)
    kept, dropped = resolve_current_state([m1, m2, m3])
    assert [s.memory.content for s in kept] == ["User's city is Lisbon"]
    assert len(dropped) == 2


# ---- end-to-end through the context builder --------------------------------


async def test_stale_singleton_fact_not_injected(service):
    # an UNMARKED contradiction: both active, no correction marker, so
    # write-time supersession leaves both — read-time resolution must hide stale
    old, _ = await service.add(MemoryCreate(
        user_id="alice", content="User's office is in Buona Vista",
        memory_type=MemoryType.SEMANTIC,
    ))
    mem = await service.get(old.id)
    mem.updated_at = utcnow() - timedelta(hours=2)
    mem.created_at = mem.updated_at
    await service.storage.upsert(mem)

    await service.add(MemoryCreate(
        user_id="alice", content="User's office is in Changi",
        memory_type=MemoryType.SEMANTIC,
    ))

    result = await service.build_context(
        ContextRequest(user_id="alice", query="where is the user's office")
    )
    assert "changi" in result.text.lower()
    assert "buona vista" not in result.text.lower()


async def test_disabling_temporal_resolution_keeps_both(service):
    service.settings.temporal_resolution = False
    old, _ = await service.add(MemoryCreate(
        user_id="alice", content="User's office is in Buona Vista",
        memory_type=MemoryType.SEMANTIC,
    ))
    mem = await service.get(old.id)
    mem.updated_at = utcnow() - timedelta(hours=2)
    await service.storage.upsert(mem)
    await service.add(MemoryCreate(
        user_id="alice", content="User's office is in Changi",
        memory_type=MemoryType.SEMANTIC,
    ))
    result = await service.build_context(
        ContextRequest(user_id="alice", query="where is the user's office")
    )
    assert "buona vista" in result.text.lower()  # both present when disabled
