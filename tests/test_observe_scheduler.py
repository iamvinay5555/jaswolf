"""Observe cadence: warm-up ramp, batching, idle flush, journal safety."""

import pytest

from jaswolf.providers.hermes import JaswolfMemoryProvider


def _turn(i: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"I really enjoy hobby number {i} on weekends"}]


@pytest.fixture
async def provider(settings):
    p = await JaswolfMemoryProvider.embedded(
        settings=settings, user_id="alice", auto_sweep=False
    )
    yield p
    await p.close()


async def test_default_cadence_extracts_every_call(provider):
    assert provider.observe_every_n == 1
    results = await provider.observe([{"role": "user", "content": "I prefer tea over coffee"}])
    assert results, "every_n=1 must behave exactly like classic observe"


async def test_warmup_ramp_1_2_4(settings):
    p = await JaswolfMemoryProvider.embedded(settings=settings, user_id="alice", auto_sweep=False)
    p.observe_every_n = 4
    p.observe_warmup = True
    try:
        # turn 1: threshold 2**0 = 1 -> flush immediately
        assert await p.observe(_turn(1), session_id="s1") != []
        # next threshold 2: turn 2 buffers, turn 3 flushes
        assert await p.observe(_turn(2), session_id="s1") == []
        assert await p.observe(_turn(3), session_id="s1") != []
        # next threshold 4: three buffer, fourth flushes
        assert await p.observe(_turn(4), session_id="s1") == []
        assert await p.observe(_turn(5), session_id="s1") == []
        assert await p.observe(_turn(6), session_id="s1") == []
        assert await p.observe(_turn(7), session_id="s1") != []
    finally:
        await p.close()


async def test_no_warmup_waits_for_n(settings):
    p = await JaswolfMemoryProvider.embedded(settings=settings, user_id="alice", auto_sweep=False)
    p.observe_every_n = 3
    p.observe_warmup = False
    try:
        assert await p.observe(_turn(1), session_id="s1") == []
        assert await p.observe(_turn(2), session_id="s1") == []
        assert await p.observe(_turn(3), session_id="s1") != []
    finally:
        await p.close()


async def test_buffers_are_per_session(settings):
    p = await JaswolfMemoryProvider.embedded(settings=settings, user_id="alice", auto_sweep=False)
    p.observe_every_n = 2
    p.observe_warmup = False
    try:
        assert await p.observe(_turn(1), session_id="a") == []
        assert await p.observe(_turn(2), session_id="b") == []
        # each session is one short of its threshold; a's second turn flushes only a
        assert await p.observe(_turn(3), session_id="a") != []
        assert p._buffers[("alice", "b")].messages
    finally:
        await p.close()


async def test_flush_observe_buffers_only_idle(settings):
    p = await JaswolfMemoryProvider.embedded(settings=settings, user_id="alice", auto_sweep=False)
    p.observe_every_n = 10
    p.observe_warmup = False
    p.observe_idle_flush_seconds = 3600
    try:
        await p.observe(_turn(1), session_id="s1")
        assert await p.flush_observe_buffers(only_idle=True) == []  # not idle yet
        p._buffers[("alice", "s1")].last_at -= 7200  # simulate an hour of silence
        flushed = await p.flush_observe_buffers(only_idle=True)
        assert flushed
        assert p._buffers[("alice", "s1")].messages == []
    finally:
        await p.close()


async def test_close_flushes_pending(settings):
    p = await JaswolfMemoryProvider.embedded(settings=settings, user_id="alice", auto_sweep=False)
    p.observe_every_n = 10
    p.observe_warmup = False
    await p.observe(
        [{"role": "user", "content": "I strongly prefer window seats on flights"}],
        session_id="s1",
    )
    await p.close()
    # reopen the same DB and confirm the memory landed
    p2 = await JaswolfMemoryProvider.embedded(settings=settings, user_id="alice", auto_sweep=False)
    try:
        hits = await p2.recall("window seats flights")
        assert any("window seat" in h.lower() for h in hits)
    finally:
        await p2.close()


async def test_buffered_turns_are_journaled_and_replayed(settings, tmp_path):
    journal_path = str(tmp_path / "observe_journal.jsonl")
    p = await JaswolfMemoryProvider.embedded(
        settings=settings, user_id="alice", auto_sweep=False, journal_path=journal_path
    )
    p.observe_every_n = 10
    p.observe_warmup = False
    await p.observe(
        [{"role": "user", "content": "My gym is in Tampines"}],
        session_id="s1",
    )
    # simulate a crash: drop the provider WITHOUT flushing
    pending = list(p._journal.pending())
    assert len(pending) == 1
    if p._service is not None:
        await p._service.close()

    # a fresh provider replays the journal on startup
    p2 = await JaswolfMemoryProvider.embedded(
        settings=settings, user_id="alice", auto_sweep=False, journal_path=journal_path
    )
    try:
        assert list(p2._journal.pending()) == []
        hits = await p2.recall("gym Tampines")
        assert hits
    finally:
        await p2.close()


async def test_flush_failure_restores_buffer(settings, monkeypatch):
    p = await JaswolfMemoryProvider.embedded(settings=settings, user_id="alice", auto_sweep=False)
    p.observe_every_n = 2
    p.observe_warmup = False
    try:
        await p.observe(_turn(1), session_id="s1")

        async def boom(**kwargs):
            raise RuntimeError("storage down")

        monkeypatch.setattr(p, "_observe_raw", boom)
        with pytest.raises(RuntimeError):
            await p.observe(_turn(2), session_id="s1")
        buffer = p._buffers[("alice", "s1")]
        assert len(buffer.messages) == 2, "failed flush must restore the buffer"
        assert buffer.flushes == 0
    finally:
        monkeypatch.undo()
        await p.close()
