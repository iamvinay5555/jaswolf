"""Durable write-ahead journal for memory writes.

A force-restart (watchdog SIGKILL, OOM, reboot) between a turn ending and a
fire-and-forget `observe()`/`add_memory()` reaching JASWOLF silently loses that
memory — that is how Alice's "mom eye checkup" was lost on 2026-06-15. The
journal closes that gap: a write is appended to a local append-only log
*before* it is sent, and only marked done once JASWOLF confirms it. Anything
still pending is replayed on the next startup.

Append-only by design (no in-place rewrite), so a crash mid-append at worst
drops the last partial line; earlier entries stay intact. Replays are safe
because JASWOLF dedups by content hash — a write that landed but died before
`mark_done` is simply reinforced on replay, never duplicated.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Any


class WriteJournal:
    def __init__(self, path: str, fsync: bool = False, max_bytes: int = 1_000_000):
        self.path = path
        self._fsync = fsync  # True = survive power loss too (slower); False = survive process kill
        # steady state is append+done line pairs that are dead weight once done;
        # auto-compact past this size so a long-running gateway can't grow the
        # log without bound (it shrinks back to just the still-pending entries)
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _write_line(self, obj: dict[str, Any]) -> None:
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            if self._fsync:
                os.fsync(f.fileno())

    def append(self, op: str, payload: dict[str, Any]) -> str:
        """Record a pending write; returns its entry id."""
        entry_id = uuid.uuid4().hex
        self._write_line({"id": entry_id, "op": op, "payload": payload})
        return entry_id

    def mark_done(self, entry_id: str) -> None:
        self._write_line({"id": entry_id, "done": True})
        self._maybe_compact()

    def _maybe_compact(self) -> None:
        try:
            if os.path.getsize(self.path) > self._max_bytes:
                self.compact()  # drops done-marked lines; keeps pending
        except OSError:
            pass

    def pending(self) -> list[dict[str, Any]]:
        """Entries appended but not yet confirmed done, in original order.
        Tolerates a torn final line from a crash mid-append."""
        if not os.path.exists(self.path):
            return []
        entries: dict[str, dict[str, Any]] = {}
        done: set[str] = set()
        order: list[str] = []
        with self._lock, open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line from a crash — skip
                rid = rec.get("id")
                if not rid:
                    continue
                if rec.get("done"):
                    done.add(rid)
                else:
                    if rid not in entries:
                        order.append(rid)
                    entries[rid] = rec
        return [entries[rid] for rid in order if rid not in done]

    def compact(self) -> None:
        """Rewrite the log keeping only still-pending entries (call after a
        successful replay so the file doesn't grow without bound)."""
        remaining = self.pending()
        tmp = f"{self.path}.compact-{uuid.uuid4().hex}"
        with self._lock:
            with open(tmp, "w", encoding="utf-8") as f:
                for rec in remaining:
                    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                f.flush()
                if self._fsync:
                    os.fsync(f.fileno())
            os.replace(tmp, self.path)
