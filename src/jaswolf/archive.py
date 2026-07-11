"""Cold journal — archive-then-prune for the L0 conversation layer.

The live database keeps a rolling window of raw turns
(conversation_retention_days); this module is what happens at the edge of
that window when conversation_archive_dir is set: expiring turns are
exported to monthly JSONL.gz files BEFORE they are pruned, under one
invariant:

    a turn is only ever deleted after its archive write has been fsynced.

Enforced structurally, not by convention — deletion is by the exact ids of
the batch just archived (storage.delete_conversations), never by a blanket
time cutoff, and any write failure aborts the pass with everything not yet
archived still safely in the database. A crash, full disk, or unwritable
directory can cost a sweep cycle, never a memory.

Format: one gzip member appended per (month, pass) to
`<dir>/YYYY-MM.jsonl.gz` — concatenated gzip members are a valid gzip
stream, so the files read with zcat/zgrep/gzip.open forever. One JSON
object per line with every field needed to rebuild the moment. Flat files
on purpose: this is a life journal, and a deep archive should outlive the
software that wrote it.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from datetime import datetime

from .models import ConversationMessage
from .storage.base import StorageBackend

logger = logging.getLogger("jaswolf.archive")

_BATCH = 500


def _month_key(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def _line(message: ConversationMessage) -> bytes:
    return (
        json.dumps(
            {
                "id": message.id,
                "tenant_id": message.tenant_id,
                "user_id": message.user_id,
                "agent_id": message.agent_id,
                "session_id": message.session_id,
                "namespace": message.namespace,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            },
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _append_month(archive_dir: str, month: str, messages: list[ConversationMessage]) -> None:
    """Append one gzip member with these messages and fsync before returning.
    Raises on any failure — the caller must not delete anything then."""
    path = os.path.join(archive_dir, f"{month}.jsonl.gz")
    raw = open(path, "ab")
    try:
        with gzip.GzipFile(fileobj=raw, mode="ab") as gz:
            for message in messages:
                gz.write(_line(message))
        raw.flush()
        os.fsync(raw.fileno())
    finally:
        raw.close()


class ConversationArchiver:
    def __init__(self, storage: StorageBackend, archive_dir: str):
        self.storage = storage
        self.archive_dir = archive_dir

    async def archive_and_prune(self, before: datetime) -> tuple[int, int]:
        """Export every turn older than `before` to the archive, deleting each
        batch only after its write is durable. Returns (archived, pruned).

        Never raises: a failure logs loudly and returns the counts of what
        completed safely — un-archived rows stay in the database and the next
        sweep retries them.
        """
        archived = pruned = 0
        try:
            # private by default: this directory is a verbatim life transcript
            os.makedirs(self.archive_dir, mode=0o700, exist_ok=True)
            while True:
                batch = await self.storage.fetch_conversations_before(before, _BATCH)
                if not batch:
                    break
                by_month: dict[str, list[ConversationMessage]] = {}
                for message in batch:
                    by_month.setdefault(_month_key(message.created_at), []).append(message)
                for month in sorted(by_month):
                    _append_month(self.archive_dir, month, by_month[month])
                archived += len(batch)
                pruned += await self.storage.delete_conversations([m.id for m in batch])
        except Exception:
            logger.exception(
                "cold-journal archive to %s FAILED — pruning halted for this pass; "
                "expiring turns remain in the live database and will be retried "
                "next sweep. Fix the archive path/disk before they age further.",
                self.archive_dir,
            )
        return archived, pruned


def read_archive_month(archive_dir: str, month: str) -> list[dict]:
    """Read one month back out of the journal (timeline building, tests,
    restore tooling). Transparent to the multi-member gzip layout."""
    path = os.path.join(archive_dir, f"{month}.jsonl.gz")
    if not os.path.exists(path):
        return []
    out: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
