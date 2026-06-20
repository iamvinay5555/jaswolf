"""Temporal current-state resolution.

JASWOLF's write-path supersession is deliberately conservative: it only archives
an old fact when the new one carries a correction marker ("actually", "now",
…). Unmarked contradictions stay additive on purpose — guessing wrong at write
time would destroy a real memory. The cost is that two active facts can fill
the same slot ("User's office is Buona Vista" + a later "User's office is
Changi"), and a query can surface the stale one.

mem0's answer (v3) is to resolve this at *read* time — rank the right current
instance instead of mutating storage. We do the same, safely: when several
retrieved memories fill the same **singleton** slot (a thing a person has
exactly one current value of), keep the freshest and drop the stale ones from
the prompt. This never touches storage and never collapses **multi-valued**
relations (friends, pets, hobbies, languages) — those slots are simply not in
the singleton set, so they stay additive exactly as before.

Pure functions over ScoredMemory; the context builder is the only caller.
"""

from __future__ import annotations

import re

from .models import MemoryType, ScoredMemory

# "User's <slot> is <value>" — the same shape supersession keys on.
_SLOT = re.compile(r"^user'?s?\s+(.{2,40}?)\s+is\s+(.+)$", re.IGNORECASE)

# Relations a person has exactly ONE current value of — safe to collapse to the
# freshest. Matched as substrings, so "office location" and "home address"
# qualify. Multi-valued relations are deliberately absent: an unknown slot is
# left additive, never collapsed.
_SINGLETON_SLOTS = (
    "office", "home", "address", "location", "live", "based", "city", "country",
    "job", "work", "employer", "company", "role", "title", "position", "team",
    "phone", "email", "number", "handle", "birthday", "age", "timezone",
    "name", "surname",
)


def _slot(content: str) -> tuple[str, str] | None:
    m = _SLOT.match(content.strip())
    if not m:
        return None
    slot = re.sub(r"\s+", " ", m.group(1).lower()).strip()
    value = re.sub(r"\s+", " ", m.group(2).lower()).strip().rstrip(".")
    return slot, value


def _is_singleton(slot_key: str) -> bool:
    return any(token in slot_key for token in _SINGLETON_SLOTS)


def _freshness(scored: ScoredMemory) -> tuple:
    m = scored.memory
    # most-recently-asserted wins; updated_at is bumped on restate/reinforce
    return (m.updated_at, m.created_at, m.importance)


def resolve_current_state(
    scored: list[ScoredMemory],
) -> tuple[list[ScoredMemory], list[ScoredMemory]]:
    """Return (kept, dropped). Among memories of the same type filling the same
    singleton slot with *different* values, keep only the freshest. Everything
    else passes through untouched."""
    groups: dict[tuple[MemoryType, str], list[ScoredMemory]] = {}
    kept: list[ScoredMemory] = []
    for s in scored:
        slot = _slot(s.memory.content)
        if slot and _is_singleton(slot[0]):
            groups.setdefault((s.memory.memory_type, slot[0]), []).append(s)
        else:
            kept.append(s)

    dropped: list[ScoredMemory] = []
    for members in groups.values():
        distinct_values = {_slot(s.memory.content)[1] for s in members}  # type: ignore[index]
        if len(distinct_values) <= 1:
            kept.extend(members)  # same value (re-statements) — dedup handles it
            continue
        winner = max(members, key=_freshness)
        kept.append(winner)
        dropped.extend(s for s in members if s is not winner)
    return kept, dropped
