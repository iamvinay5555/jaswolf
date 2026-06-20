"""The DB remembers which embedder wrote it (Jasmine ask #5, 2026-06-11).

Vectors only mean anything under the embedder that produced them. Opening a
DB with a different provider/model must degrade health loudly instead of
silently mixing vector spaces — the failure mode the fresh-DB rule in
docs/HANDOFF.md guards against by hand.
"""

from jaswolf.config import JaswolfSettings
from jaswolf.service import MemoryService


def _settings(tmp_path, **overrides) -> JaswolfSettings:
    base = dict(
        database_url=f"sqlite:///{tmp_path}/fingerprint.db",
        embedding_provider="hash",
        embedding_dim=384,
        sweep_interval_seconds=3600,
        log_level="WARNING",
    )
    base.update(overrides)
    return JaswolfSettings(**base)


async def test_first_open_stamps_fingerprint(tmp_path):
    service = await MemoryService.create(_settings(tmp_path))
    try:
        assert await service.storage.get_meta("embedding_fingerprint") == "hashing-384"
        assert (await service.health())["status"] == "ok"
    finally:
        await service.close()


async def test_reopen_with_same_embedder_stays_ok(tmp_path):
    first = await MemoryService.create(_settings(tmp_path))
    await first.close()
    second = await MemoryService.create(_settings(tmp_path))
    try:
        health = await second.health()
        assert health["status"] == "ok"
        assert not any("mismatch" in r for r in health.get("reasons", []))
    finally:
        await second.close()


async def test_reopen_with_different_embedder_degrades(tmp_path):
    first = await MemoryService.create(_settings(tmp_path))
    await first.close()
    # same DB file, different embedder identity (hashing-256 vs hashing-384;
    # a hash->bge switch differs the same way: the provider name changes)
    second = await MemoryService.create(_settings(tmp_path, embedding_dim=256))
    try:
        health = await second.health()
        assert health["status"] == "degraded"
        assert any("hashing-384" in r and "hashing-256" in r for r in health["reasons"])
        # the original stamp is preserved — the mismatched opener must not overwrite it
        assert await second.storage.get_meta("embedding_fingerprint") == "hashing-384"
    finally:
        await second.close()
