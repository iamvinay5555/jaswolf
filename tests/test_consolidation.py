from jaswolf.consolidation import merge_contents
from jaswolf.models import MemoryCreate, MemoryState, MemoryType


def test_merge_contents_containment():
    merged = merge_contents(["User likes Python", "User likes Python for backend development"])
    assert merged == "User likes Python for backend development."


def test_merge_contents_union():
    merged = merge_contents(["User prefers dark mode", "User prefers large fonts"])
    assert "dark mode" in merged
    assert "large fonts" in merged


async def _seed(service):
    a, _ = await service.add(
        MemoryCreate(
            user_id="alice",
            content="User prefers Python for backend work",
            memory_type=MemoryType.PREFERENCE,
            importance=0.9,
        )
    )
    b, _ = await service.add(
        MemoryCreate(
            user_id="alice",
            content="User prefers Python for backend development projects",
            memory_type=MemoryType.PREFERENCE,
            importance=0.6,
        )
    )
    unrelated, _ = await service.add(
        MemoryCreate(
            user_id="alice",
            content="Sarah is user's cofounder",
            memory_type=MemoryType.RELATIONSHIP,
        )
    )
    return a, b, unrelated


async def test_consolidation_merges_near_duplicates(service):
    # hash embeddings overlap less than real models: lower the bar accordingly
    service.settings.dedup_threshold = 0.995
    service.settings.consolidation_threshold = 0.60
    a, b, unrelated = await _seed(service)
    assert a.id != b.id  # below dedup threshold at write time

    report = await service.consolidate(user_id="alice")
    assert report.clusters_found == 1
    assert report.memories_merged == 1
    merge = report.merges[0]
    assert merge.canonical_id == a.id  # higher importance wins
    assert merge.merged_ids == [b.id]

    canonical = await service.get(a.id)
    assert canonical.importance == 0.9
    assert "backend" in canonical.content
    versions = await service.get_versions(a.id)
    assert any(v["reason"] == "consolidation" for v in versions)

    loser = await service.get(b.id)
    assert loser.state == MemoryState.DELETED
    untouched = await service.get(unrelated.id)
    assert untouched.state == MemoryState.ACTIVE


async def test_consolidation_dry_run_changes_nothing(service):
    service.settings.consolidation_threshold = 0.60
    a, b, _ = await _seed(service)
    report = await service.consolidate(user_id="alice", dry_run=True)
    assert report.dry_run is True
    assert report.clusters_found == 1
    assert (await service.get(a.id)).state == MemoryState.ACTIVE
    assert (await service.get(b.id)).state == MemoryState.ACTIVE
    assert await service.get_versions(a.id) == []
