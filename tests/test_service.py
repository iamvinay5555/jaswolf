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


async def test_hybrid_search_does_not_bury_exact_keyword_hit(service):
    # Several unrelated, high-importance memories that should NOT match the
    # query semantically or lexically.
    for content in [
        "Mom had an eye checkup today, needs follow-up tests next week",
        "Family Thailand trip is in August",
        "JasWolf smoke test: system health audit",
        "Crash-recovery test: write-ahead journal active",
    ]:
        await service.add(MemoryCreate(user_id="alice", content=content, importance=0.85))

    # One low-importance memory containing a rare exact phrase.
    rare, _ = await service.add(
        MemoryCreate(
            user_id="alice",
            content="Temporary exact retrieval probe phrase: zebra kumquat lighthouse",
            importance=0.20,
        )
    )

    results = await service.search(
        SearchQuery(
            user_id="alice",
            query="zebra kumquat lighthouse",
            mode=SearchMode.HYBRID,
            top_k=10,
        )
    )
    ids = [r.memory.id for r in results]
    assert rare.id in ids, "exact keyword hit must appear in hybrid results"
    rank = ids.index(rare.id)
    # must not be buried below the unrelated high-importance, non-keyword hits
    non_keyword_above = [
        r for r in results[:rank] if not r.keyword_match
    ]
    assert not non_keyword_above, (
        f"exact keyword hit ranked #{rank + 1}, behind non-keyword candidates: "
        f"{[c.memory.content[:40] for c in non_keyword_above]}"
    )


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
