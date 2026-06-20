"""Regression: restating an archived fact must revive it.

Before v0.2.1 the exact-hash dedup path matched archived rows and
reinforced them in place — the fact stayed invisible to search forever,
no matter how often the user repeated it verbatim, while a paraphrase
would have created a fresh visible memory.
"""

from datetime import timedelta

from jaswolf.models import MemoryCreate, MemoryState, MemoryType, SearchQuery, utcnow


async def _force_archive(service, memory_id: str) -> None:
    mem = await service.get(memory_id)
    mem.state = MemoryState.ARCHIVED
    await service.storage.upsert(mem)


async def test_restating_archived_fact_revives_it(service):
    first, _ = await service.add(
        MemoryCreate(
            user_id="alice",
            content="User prefers coffee in the morning",
            memory_type=MemoryType.PREFERENCE,
        )
    )
    await _force_archive(service, first.id)
    results = await service.search(SearchQuery(user_id="alice", query="coffee in the morning"))
    assert all(r.memory.id != first.id for r in results)  # archived = invisible

    second, created = await service.add(
        MemoryCreate(
            user_id="alice",
            content="User prefers coffee in the morning",
            memory_type=MemoryType.PREFERENCE,
        )
    )
    assert created is False
    assert second.id == first.id
    assert second.state == MemoryState.ACTIVE

    results = await service.search(SearchQuery(user_id="alice", query="coffee in the morning"))
    assert any(r.memory.id == first.id for r in results)


async def test_sweep_archive_restate_sweep_stays_active(service):
    note, _ = await service.add(
        MemoryCreate(
            user_id="alice",
            content="Working on the jaswolf shadow run",
            memory_type=MemoryType.WORKING,
        )
    )
    mem = await service.get(note.id)
    mem.expires_at = utcnow() - timedelta(minutes=1)
    await service.storage.upsert(mem)
    await service.sweep()
    assert (await service.get(note.id)).state == MemoryState.ARCHIVED

    again, created = await service.add(
        MemoryCreate(
            user_id="alice",
            content="Working on the jaswolf shadow run",
            memory_type=MemoryType.WORKING,
        )
    )
    assert created is False
    assert again.state == MemoryState.ACTIVE
    # the renewed TTL must survive the next sweep, not re-archive instantly
    assert again.expires_at is not None and again.expires_at > utcnow()
    await service.sweep()
    assert (await service.get(note.id)).state == MemoryState.ACTIVE


async def test_revived_durable_fact_clears_stale_ttl(service):
    fact, _ = await service.add(
        MemoryCreate(
            user_id="alice",
            content="User's office is in Changi",
            memory_type=MemoryType.SEMANTIC,
            ttl_hours=1.0,
        )
    )
    mem = await service.get(fact.id)
    mem.expires_at = utcnow() - timedelta(hours=2)
    mem.state = MemoryState.ARCHIVED
    await service.storage.upsert(mem)

    again, created = await service.add(
        MemoryCreate(
            user_id="alice",
            content="User's office is in Changi",
            memory_type=MemoryType.SEMANTIC,
        )
    )
    assert created is False
    assert again.state == MemoryState.ACTIVE
    assert again.expires_at is None  # durable restatement without TTL = permanent
