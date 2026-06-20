import pytest

from jaswolf.models import (
    MemoryCreate,
    MemoryNotFound,
    MemoryType,
    MemoryUpdate,
    SearchMode,
    SearchQuery,
)


async def test_add_and_get(service):
    memory, created = await service.add(
        MemoryCreate(user_id="alice", content="User prefers Python", memory_type=MemoryType.PREFERENCE)
    )
    assert created is True
    assert memory.importance > 0.5  # auto-scored, preference base
    loaded = await service.get(memory.id)
    assert loaded.content == "User prefers Python"


async def test_exact_duplicate_reinforces(service):
    first, created1 = await service.add(MemoryCreate(user_id="alice", content="User prefers Python"))
    second, created2 = await service.add(MemoryCreate(user_id="alice", content="user  prefers   PYTHON"))
    assert created1 is True
    assert created2 is False
    assert second.id == first.id
    assert second.access_count >= 1
    stats = await service.stats()
    assert stats["total"] == 1


async def test_near_duplicate_reinforces(service):
    service.settings.dedup_threshold = 0.80
    first, _ = await service.add(
        MemoryCreate(user_id="alice", content="User prefers Python for backend work")
    )
    second, created = await service.add(
        MemoryCreate(user_id="alice", content="User prefers Python for backend work today")
    )
    assert created is False
    assert second.id == first.id


async def test_different_users_do_not_dedupe(service):
    _, c1 = await service.add(MemoryCreate(user_id="alice", content="User prefers Python"))
    _, c2 = await service.add(MemoryCreate(user_id="bob", content="User prefers Python"))
    assert c1 and c2


async def test_working_memory_gets_ttl(service):
    memory, _ = await service.add(
        MemoryCreate(user_id="alice", content="current task: fix the deploy", memory_type=MemoryType.WORKING)
    )
    assert memory.expires_at is not None
    semantic, _ = await service.add(MemoryCreate(user_id="alice", content="some durable fact"))
    assert semantic.expires_at is None


async def test_ingest_text_extracts_and_stores(service):
    results = await service.ingest_text(
        "alice", "I love Python. Sarah is my cofounder. I want to launch a SaaS."
    )
    assert len(results) == 3
    types = {m.memory_type for m, _ in results}
    assert types == {MemoryType.PREFERENCE, MemoryType.RELATIONSHIP, MemoryType.GOAL}
    assert all(m.metadata.get("extracted_by") == "rules" for m, _ in results)


async def test_update_content_records_version_and_reembeds(service):
    memory, _ = await service.add(MemoryCreate(user_id="alice", content="User lives in Singapore"))
    old_embedding = list(memory.embedding)
    updated = await service.update(memory.id, MemoryUpdate(content="User lives in Tokyo now"))
    assert updated.content == "User lives in Tokyo now"
    assert updated.embedding != old_embedding
    versions = await service.get_versions(memory.id)
    assert len(versions) == 1
    assert versions[0]["content"] == "User lives in Singapore"


async def test_delete_and_not_found(service):
    memory, _ = await service.add(MemoryCreate(user_id="alice", content="temp"))
    await service.delete(memory.id)
    with pytest.raises(MemoryNotFound):
        await service.delete("does-not-exist")


async def test_search_modes(service):
    await service.add(
        MemoryCreate(user_id="alice", content="User deploys Hermes with Docker Compose on a VPS")
    )
    await service.add(
        MemoryCreate(user_id="alice", content="User's favorite tea is oolong", importance=0.95)
    )
    for mode in (SearchMode.SEMANTIC, SearchMode.KEYWORD, SearchMode.HYBRID):
        results = await service.search(
            SearchQuery(user_id="alice", query="docker deployment", mode=mode, top_k=5)
        )
        assert results, f"no results for {mode}"
        assert results[0].memory.content.startswith("User deploys Hermes")

    importance_first = await service.search(
        SearchQuery(user_id="alice", query="", mode=SearchMode.IMPORTANCE, top_k=5, record_access=False)
    )
    assert importance_first[0].memory.content.startswith("User's favorite tea")


async def test_search_records_access(service):
    memory, _ = await service.add(MemoryCreate(user_id="alice", content="User uses Neovim"))
    await service.search(SearchQuery(user_id="alice", query="neovim editor"))
    loaded = await service.get(memory.id)
    assert loaded.access_count >= 1


async def test_sweep_expires_working_memory(service):
    await service.add(
        MemoryCreate(
            user_id="alice",
            content="ephemeral note",
            memory_type=MemoryType.WORKING,
            ttl_hours=-1,  # already expired
        )
    )
    report = await service.sweep()
    assert report.expired_working == 1


async def test_health(service):
    health = await service.health()
    assert health["status"] == "ok"
    assert health["storage"]["backend"] == "sqlite"
    assert health["embeddings"]["provider"].startswith("hashing")
