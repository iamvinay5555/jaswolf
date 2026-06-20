"""Durable write-ahead journal — survives a crash before a write reaches JasWolf.

Models the 2026-06-15 incident: a memory write lost when the gateway was
force-restarted mid-turn. With a journal, the write is replayed on startup.
"""

from jaswolf.config import JaswolfSettings
from jaswolf.journal import WriteJournal
from jaswolf.models import SearchQuery
from jaswolf.providers.hermes import JaswolfMemoryProvider
from jaswolf.service import MemoryService


# ---- journal core --------------------------------------------------------------


def test_append_pending_done(tmp_path):
    j = WriteJournal(str(tmp_path / "j.jsonl"))
    a = j.append("observe", {"messages": [{"role": "user", "content": "hi"}]})
    b = j.append("add_memory", {"content": "x"})
    assert {e["id"] for e in j.pending()} == {a, b}
    j.mark_done(a)
    assert [e["id"] for e in j.pending()] == [b]  # order preserved
    j.mark_done(b)
    assert j.pending() == []


def test_tolerates_torn_final_line(tmp_path):
    path = tmp_path / "j.jsonl"
    j = WriteJournal(str(path))
    good = j.append("add_memory", {"content": "good"})
    with open(path, "a") as f:
        f.write('{"id": "torn", "op": "add_mem')  # crash mid-append, no newline
    pending = j.pending()
    assert [e["id"] for e in pending] == [good]  # torn line skipped, good survives


def test_auto_compacts_past_max_bytes(tmp_path):
    # steady-state append+done churn must not grow the log without bound
    j = WriteJournal(str(tmp_path / "j.jsonl"), max_bytes=2000)
    for i in range(500):
        eid = j.append("add_memory", {"content": f"fact {i} " + "x" * 50})
        j.mark_done(eid)  # triggers _maybe_compact once past max_bytes
    size = (tmp_path / "j.jsonl").stat().st_size
    assert size <= 4000  # bounded, not ~500*~120 bytes
    assert j.pending() == []  # all done -> compacted away


def test_compact_drops_done(tmp_path):
    path = tmp_path / "j.jsonl"
    j = WriteJournal(str(path))
    a = j.append("add_memory", {"content": "a"})
    b = j.append("add_memory", {"content": "b"})
    j.mark_done(a)
    j.compact()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1  # only the pending entry remains
    assert [e["id"] for e in j.pending()] == [b]


# ---- provider durability -------------------------------------------------------


def _settings(tmp_path):
    return JaswolfSettings(
        database_url=f"sqlite:///{tmp_path}/dur.db",
        embedding_provider="hash",
        sweep_interval_seconds=3600,
        log_level="WARNING",
    )


async def test_normal_write_leaves_no_pending(tmp_path):
    jpath = str(tmp_path / "j.jsonl")
    p = await JaswolfMemoryProvider.embedded(
        settings=_settings(tmp_path), user_id="u", auto_sweep=False, journal_path=jpath,
    )
    try:
        await p.add_memory("User prefers oat milk", memory_type="preference")
        assert WriteJournal(jpath).pending() == []  # confirmed + marked done
    finally:
        await p.close()


async def test_crash_pending_write_is_replayed_on_startup(tmp_path):
    jpath = str(tmp_path / "j.jsonl")
    # simulate a write that was journaled but the process died before it landed
    WriteJournal(jpath).append("add_memory", {
        "content": "Mom has an eye checkup on Tuesday",
        "memory_type": "semantic", "user_id": "u", "session_id": None,
        "importance": None, "metadata": None, "ttl_hours": None,
    })

    # next startup on the same DB + journal must replay it
    p = await JaswolfMemoryProvider.embedded(
        settings=_settings(tmp_path), user_id="u", auto_sweep=False, journal_path=jpath,
    )
    try:
        hits = await p.recall("mom eye checkup")
        assert any("eye checkup" in h.lower() for h in hits)   # recovered!
        assert WriteJournal(jpath).pending() == []             # flushed + compacted
    finally:
        await p.close()


async def test_replay_is_idempotent_via_dedup(tmp_path):
    # a write that landed but died before mark_done is replayed; dedup means no
    # duplicate row
    jpath = str(tmp_path / "j.jsonl")
    svc = await MemoryService.create(_settings(tmp_path))
    from jaswolf.models import MemoryCreate
    await svc.add(MemoryCreate(user_id="u", content="Alice lives in Singapore"))
    await svc.close()
    WriteJournal(jpath).append("add_memory", {
        "content": "Alice lives in Singapore", "memory_type": "semantic",
        "user_id": "u", "session_id": None, "importance": None,
        "metadata": None, "ttl_hours": None,
    })
    p = await JaswolfMemoryProvider.embedded(
        settings=_settings(tmp_path), user_id="u", auto_sweep=False, journal_path=jpath,
    )
    try:
        hits = await p._service.search(
            SearchQuery(user_id="u", query="where does Alice live", record_access=False)
        )
        same = [h for h in hits if "Singapore" in h.memory.content]
        assert len(same) == 1  # replay reinforced, did not duplicate
    finally:
        await p.close()
