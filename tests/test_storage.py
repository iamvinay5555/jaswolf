from datetime import timedelta

import pytest

from jaswolf.embeddings.hashing import HashingEmbedder
from jaswolf.models import Memory, MemoryState, MemoryType, content_hash, utcnow
from jaswolf.storage.base import LifecycleCutoffs, QueryScope
from jaswolf.storage.sqlite_store import SQLiteStore

EMBEDDER = HashingEmbedder(dim=128)


@pytest.fixture
async def store(tmp_path) -> SQLiteStore:
    s = SQLiteStore(str(tmp_path / "store.db"))
    await s.init()
    yield s
    await s.close()


async def make_memory(content: str, user_id: str = "u1", **kwargs) -> Memory:
    vec = (await EMBEDDER.embed([content]))[0]
    return Memory(
        user_id=user_id,
        content=content,
        content_hash=content_hash(content),
        embedding=vec,
        **kwargs,
    )


async def test_roundtrip_preserves_fields(store):
    memory = await make_memory(
        "User prefers Python",
        memory_type=MemoryType.PREFERENCE,
        importance=0.9,
        metadata={"source": "test", "nested": {"a": 1}},
    )
    await store.upsert(memory)
    loaded = await store.get(memory.id, "default")
    assert loaded is not None
    assert loaded.content == memory.content
    assert loaded.memory_type == MemoryType.PREFERENCE
    assert loaded.importance == pytest.approx(0.9)
    assert loaded.metadata == {"source": "test", "nested": {"a": 1}}
    assert loaded.embedding == pytest.approx(memory.embedding, abs=1e-6)
    assert abs((loaded.created_at - memory.created_at).total_seconds()) < 0.01


async def test_tenant_isolation(store):
    memory = await make_memory("secret fact")
    memory.tenant_id = "tenant_a"
    await store.upsert(memory)
    assert await store.get(memory.id, "tenant_a") is not None
    assert await store.get(memory.id, "tenant_b") is None
    scope_b = QueryScope(tenant_id="tenant_b", user_id="u1")
    assert await store.list_memories(scope_b) == []


async def test_soft_and_hard_delete(store):
    memory = await make_memory("to be deleted")
    await store.upsert(memory)
    assert await store.delete(memory.id, "default") is True
    loaded = await store.get(memory.id, "default")
    assert loaded.state == MemoryState.DELETED
    scope = QueryScope(tenant_id="default", user_id="u1")
    assert all(m.id != memory.id for m in await store.list_memories(scope))
    assert await store.delete(memory.id, "default", hard=True) is True
    assert await store.get(memory.id, "default") is None
    assert await store.delete("nonexistent", "default") is False


async def test_vector_search_ranks_exact_match_first(store):
    target = await make_memory("User deploys Hermes on a VPS with Docker")
    other1 = await make_memory("User likes espresso in the morning")
    other2 = await make_memory("The weather in Singapore is humid")
    for m in (target, other1, other2):
        await store.upsert(m)
    query_vec = (await EMBEDDER.embed(["User deploys Hermes on a VPS with Docker"]))[0]
    scope = QueryScope(tenant_id="default", user_id="u1")
    results = await store.search_vector(scope, query_vec, k=3)
    assert results[0][0].id == target.id
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert results[0][1] > results[1][1]


async def test_keyword_search_finds_term(store):
    hit = await make_memory("User's company migrated to Kubernetes last year")
    miss = await make_memory("User enjoys hiking on weekends")
    await store.upsert(hit)
    await store.upsert(miss)
    scope = QueryScope(tenant_id="default", user_id="u1")
    results = await store.search_keyword(scope, "kubernetes migration", k=5)
    assert [m.id for m, _ in results] == [hit.id]


async def test_expired_memories_invisible(store):
    expired = await make_memory("old working note", memory_type=MemoryType.WORKING)
    expired.expires_at = utcnow() - timedelta(hours=1)
    await store.upsert(expired)
    scope = QueryScope(tenant_id="default", user_id="u1")
    assert await store.list_memories(scope) == []
    vec = (await EMBEDDER.embed(["old working note"]))[0]
    assert await store.search_vector(scope, vec, k=5) == []


async def test_record_access_bumps_count(store):
    memory = await make_memory("frequently used fact")
    await store.upsert(memory)
    await store.record_access([memory.id], "default", "search", query="fact")
    await store.record_access([memory.id], "default", "search")
    loaded = await store.get(memory.id, "default")
    assert loaded.access_count == 2
    assert loaded.last_accessed is not None


async def test_get_by_hash(store):
    memory = await make_memory("User prefers tabs over spaces")
    await store.upsert(memory)
    found = await store.get_by_hash("default", "u1", "default", content_hash("user prefers TABS over   spaces"))
    assert found is not None and found.id == memory.id
    assert await store.get_by_hash("default", "u1", "default", "nope") is None


async def test_versions_and_relationships(store):
    memory = await make_memory("v1 content")
    await store.upsert(memory)
    await store.add_version(memory.id, "v1 content", reason="update", payload={"by": "test"})
    versions = await store.get_versions(memory.id)
    assert len(versions) == 1
    assert versions[0]["reason"] == "update"
    assert versions[0]["payload"] == {"by": "test"}
    await store.add_relationship(memory.id, "other-id", "merged_into")  # no exception


async def test_lifecycle_transitions(store):
    now = utcnow()
    stale_active = await make_memory("idle active memory")
    stale_active.last_accessed = now - timedelta(days=20)
    stale_active.updated_at = now - timedelta(days=20)
    fresh_active = await make_memory("fresh memory")
    expired = await make_memory("expired working", memory_type=MemoryType.WORKING)
    expired.expires_at = now - timedelta(hours=2)
    for m in (stale_active, fresh_active, expired):
        await store.upsert(m)

    cutoffs = LifecycleCutoffs(
        now=now,
        warm_before=now - timedelta(days=14),
        cold_before=now - timedelta(days=60),
        archive_before=now - timedelta(days=180),
    )
    report = await store.apply_lifecycle(cutoffs)
    assert report.expired_working == 1
    assert report.active_to_warm == 1
    assert (await store.get(stale_active.id, "default")).state == MemoryState.WARM
    assert (await store.get(fresh_active.id, "default")).state == MemoryState.ACTIVE
    assert (await store.get(expired.id, "default")).state == MemoryState.ARCHIVED


async def test_stats(store):
    await store.upsert(await make_memory("a", memory_type=MemoryType.PREFERENCE))
    await store.upsert(await make_memory("b", memory_type=MemoryType.SEMANTIC))
    stats = await store.stats("default")
    assert stats["total"] == 2
    assert stats["by_type"] == {"preference": 1, "semantic": 1}
    assert stats["by_state"] == {"active": 2}
