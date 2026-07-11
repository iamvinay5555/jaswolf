"""Cold journal: archive-then-prune invariant, monthly files, failure safety."""

import gzip
import os
import stat
from datetime import timedelta

import pytest

from jaswolf.archive import read_archive_month
from jaswolf.config import JaswolfSettings
from jaswolf.models import ConversationMessage, utcnow
from jaswolf.service import MemoryService


def _old_turn(content: str, days_ago: float, **kwargs) -> ConversationMessage:
    return ConversationMessage(
        user_id=kwargs.pop("user_id", "alice"),
        role=kwargs.pop("role", "user"),
        content=content,
        created_at=utcnow() - timedelta(days=days_ago),
        **kwargs,
    )


@pytest.fixture
def journal_settings(tmp_path) -> JaswolfSettings:
    return JaswolfSettings(
        database_url=f"sqlite:///{tmp_path}/jaswolf_journal.db",
        embedding_provider="hash",
        dev_open_mode=True,
        conversation_capture=True,
        conversation_retention_days=90,
        conversation_archive_dir=str(tmp_path / "journal"),
        sweep_interval_seconds=3600,
        log_level="WARNING",
    )


@pytest.fixture
async def journal_service(journal_settings) -> MemoryService:
    svc = await MemoryService.create(journal_settings)
    yield svc
    await svc.close()


async def test_expiring_turns_are_archived_then_pruned(journal_service, journal_settings):
    rows = [
        _old_turn("ancient chat about kopi", days_ago=120),
        _old_turn("noted, kopi it is", days_ago=120, role="assistant"),
        _old_turn("fresh chat about lunch", days_ago=1),
    ]
    await journal_service.storage.add_conversation_messages(rows)

    report = await journal_service.sweep()
    assert report.archived_conversations == 2
    assert report.pruned_conversations == 2

    # pruned from the live DB, fresh row untouched
    hits = await journal_service.search_conversations(user_id="alice", query="kopi")
    assert hits == []
    assert await journal_service.search_conversations(user_id="alice", query="lunch")

    # every archived field survives in the journal file
    month = f"{rows[0].created_at.year:04d}-{rows[0].created_at.month:02d}"
    entries = read_archive_month(journal_settings.conversation_archive_dir, month)
    assert len(entries) == 2
    entry = next(e for e in entries if e["role"] == "user")
    assert entry["content"] == "ancient chat about kopi"
    assert entry["id"] == rows[0].id
    assert entry["user_id"] == "alice"
    assert entry["created_at"].startswith(str(rows[0].created_at.year))


async def test_turns_split_into_monthly_files(journal_service, journal_settings):
    await journal_service.storage.add_conversation_messages(
        [_old_turn("very old", days_ago=200), _old_turn("less old", days_ago=100)]
    )
    report = await journal_service.sweep()
    assert report.archived_conversations == 2
    files = sorted(os.listdir(journal_settings.conversation_archive_dir))
    assert len(files) == 2
    assert all(f.endswith(".jsonl.gz") for f in files)


async def test_append_across_sweeps_keeps_earlier_entries(journal_service, journal_settings):
    first = _old_turn("first expiring turn", days_ago=100)
    await journal_service.storage.add_conversation_messages([first])
    await journal_service.sweep()
    second = _old_turn("second expiring turn", days_ago=100)
    await journal_service.storage.add_conversation_messages([second])
    await journal_service.sweep()

    month = f"{first.created_at.year:04d}-{first.created_at.month:02d}"
    entries = read_archive_month(journal_settings.conversation_archive_dir, month)
    contents = [e["content"] for e in entries]
    assert "first expiring turn" in contents and "second expiring turn" in contents
    # multi-member gzip is also readable by plain gzip tooling
    path = os.path.join(journal_settings.conversation_archive_dir, f"{month}.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        assert len([ln for ln in f if ln.strip()]) == 2


async def test_unwritable_archive_refuses_to_prune(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the archive dir should be")
    settings = JaswolfSettings(
        database_url=f"sqlite:///{tmp_path}/jaswolf_blocked.db",
        embedding_provider="hash",
        dev_open_mode=True,
        conversation_capture=True,
        conversation_retention_days=90,
        conversation_archive_dir=str(blocker),  # makedirs will fail on a file
        log_level="ERROR",
    )
    svc = await MemoryService.create(settings)
    try:
        await svc.storage.add_conversation_messages([_old_turn("must not be lost", days_ago=120)])
        report = await svc.sweep()  # must not raise
        assert report.archived_conversations == 0
        assert report.pruned_conversations == 0
        # the expiring turn is still safely in the live DB
        hits = await svc.search_conversations(user_id="alice", query="lost")
        assert len(hits) == 1
    finally:
        await svc.close()


async def test_archive_dir_is_private(journal_service, journal_settings):
    await journal_service.storage.add_conversation_messages([_old_turn("private", days_ago=120)])
    await journal_service.sweep()
    mode = stat.S_IMODE(os.stat(journal_settings.conversation_archive_dir).st_mode)
    assert mode == 0o700


async def test_no_archive_dir_keeps_v014_prune_behavior(tmp_path):
    settings = JaswolfSettings(
        database_url=f"sqlite:///{tmp_path}/jaswolf_noarchive.db",
        embedding_provider="hash",
        dev_open_mode=True,
        conversation_capture=True,
        conversation_retention_days=90,
        log_level="WARNING",
    )
    svc = await MemoryService.create(settings)
    try:
        await svc.storage.add_conversation_messages([_old_turn("plain prune", days_ago=120)])
        report = await svc.sweep()
        assert report.pruned_conversations == 1
        assert report.archived_conversations == 0
    finally:
        await svc.close()


async def test_read_archive_month_missing_file(tmp_path):
    assert read_archive_month(str(tmp_path), "1999-01") == []


async def test_json_content_is_unicode_safe(journal_service, journal_settings):
    turn = _old_turn("排骨面 for lunch — läuft 👍", days_ago=120)
    await journal_service.storage.add_conversation_messages([turn])
    await journal_service.sweep()
    month = f"{turn.created_at.year:04d}-{turn.created_at.month:02d}"
    entries = read_archive_month(journal_settings.conversation_archive_dir, month)
    assert entries[0]["content"] == "排骨面 for lunch — läuft 👍"
