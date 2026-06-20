"""Backup / restore / integrity — durability for a long-lived memory store."""

import sqlite3

from jaswolf.config import JaswolfSettings
from jaswolf.models import MemoryCreate
from jaswolf.service import MemoryService
from jaswolf.storage.sqlite_store import validate_sqlite_snapshot


async def test_integrity_check_ok(service):
    assert await service.storage.integrity_check() == "ok"


async def test_health_reports_integrity(service):
    health = await service.health()
    assert health["storage"]["integrity"] == "ok"


async def test_backup_is_consistent_point_in_time(service, tmp_path):
    # add A, snapshot, then add B — the snapshot must contain A and not B
    await service.add(MemoryCreate(user_id="alice", content="Fact A before backup"))
    snap = str(tmp_path / "snap.db")
    info = await service.storage.backup(snap)
    assert info["bytes"] > 0
    await service.add(MemoryCreate(user_id="alice", content="Fact B after backup"))

    conn = sqlite3.connect(snap)
    contents = {r[0] for r in conn.execute("SELECT content FROM memories")}
    conn.close()
    assert "Fact A before backup" in contents
    assert "Fact B after backup" not in contents  # snapshot froze at backup time


async def test_backup_restores_into_a_fresh_service(service, tmp_path):
    await service.add(MemoryCreate(user_id="alice", content="User's office is in Changi"))
    snap = str(tmp_path / "snap.db")
    await service.storage.backup(snap)

    # a brand-new DB pointed at the restored snapshot file sees the memory
    restored_path = tmp_path / "restored.db"
    import shutil
    shutil.copyfile(snap, restored_path)
    restored = await MemoryService.create(JaswolfSettings(
        database_url=f"sqlite:///{restored_path}", embedding_provider="hash",
    ))
    try:
        stats = await restored.stats(user_id="alice")
        assert stats["total"] >= 1
    finally:
        await restored.close()


async def test_validate_snapshot_reports_fingerprint_and_count(service, tmp_path):
    await service.add(MemoryCreate(user_id="alice", content="durable fact"))
    snap = str(tmp_path / "snap.db")
    await service.storage.backup(snap)

    info = validate_sqlite_snapshot(snap)
    assert info["integrity"] == "ok"
    assert info["embedding_fingerprint"] == "hashing-384"
    assert info["memories"] >= 1


async def test_validate_missing_snapshot_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        validate_sqlite_snapshot(str(tmp_path / "nope.db"))
