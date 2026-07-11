"""Persona compiler — the L3 "who is this user?" layer of the semantic pyramid.

Design stance (v0.3.0, after reviewing TencentDB-Agent-Memory): the persona
is a compiled VIEW over stored memories, not a second brain. Tencent
generates persona.md with a free-form LLM pass, which can invent narrative
the atom layer doesn't back; their only defense is drill-down to evidence.
JASWOLF inverts that: the render is deterministic — grouped, ranked, token-
capped — so the persona is exactly as trustworthy as the memories beneath
it, and every line carries its source memory id. An optional LLM polish can
be layered on later, but it must be fact-checked against these source ids.

Selection mirrors the context builder's identity gates (pin_min_confidence,
always_pin, importance ranking) so the persona and the prompt never disagree
about who the user is.
"""

from __future__ import annotations

import logging

from .config import JaswolfSettings
from .context_builder import PIN_TYPES, _is_test_memory, force_pin_eligible
from .models import Memory, MemoryType, PersonaDoc, utcnow
from .storage.base import QueryScope, StorageBackend
from .tokens import estimate_tokens

logger = logging.getLogger("jaswolf.persona")

# (memory_type, section title, fetch limit, min importance)
_PERSONA_SECTIONS: list[tuple[MemoryType, str, int, float]] = [
    (MemoryType.PREFERENCE, "Preferences", 12, 0.0),
    (MemoryType.GOAL, "Goals", 8, 0.0),
    (MemoryType.RELATIONSHIP, "Relationships", 8, 0.0),
    (MemoryType.SEMANTIC, "Key facts", 12, 0.7),
]


class PersonaCompiler:
    def __init__(self, storage: StorageBackend, settings: JaswolfSettings):
        self.storage = storage
        self.settings = settings

    async def compile(
        self,
        tenant_id: str,
        user_id: str,
        namespace: str | None = None,
        namespaces: list[str] | None = None,
        include_ids: bool = True,
        token_budget: int | None = None,
    ) -> PersonaDoc:
        budget = token_budget or self.settings.persona_token_budget

        # Tier 1 — Identity: the SAME unified force-pin pool the context
        # builder injects (force_pin_eligible, importance-ranked, capped at
        # context_max_pins), rendered before anything else. v0.3.0 filled
        # per-type sections in order, so a corpus with many pinned
        # preferences exhausted the budget before the Relationships section
        # was ever reached — the persona omitted pins the prompt was
        # injecting on every turn (the wedding-pin crowd-out shape again,
        # this time in the renderer; live report, 2026-07-11). One pool, ranked,
        # first — the persona and the prompt can no longer disagree about
        # identity.
        pin_candidates: dict[str, Memory] = {}
        for pinned_type in PIN_TYPES:
            scope = QueryScope(
                tenant_id=tenant_id,
                user_id=user_id,
                namespace=namespace,
                namespaces=namespaces,
                memory_types=[pinned_type],
                min_importance=self.settings.pin_min_importance,
            )
            for memory in await self.storage.list_memories(
                scope, limit=self.settings.context_max_pins * 2, order_by="importance"
            ):
                if force_pin_eligible(memory, self.settings) and not _is_test_memory(memory):
                    pin_candidates[memory.id] = memory
        identity = sorted(
            pin_candidates.values(),
            key=lambda m: (m.importance, m.confidence),
            reverse=True,
        )[: self.settings.context_max_pins]
        identity_ids = {m.id for m in identity}

        sections: list[tuple[str, list[Memory]]] = []
        if identity:
            sections.append(("Identity (always pinned)", identity))
        for mtype, title, limit, min_importance in _PERSONA_SECTIONS:
            scope = QueryScope(
                tenant_id=tenant_id,
                user_id=user_id,
                namespace=namespace,
                namespaces=namespaces,
                memory_types=[mtype],
                min_importance=min_importance or None,
            )
            rows = await self.storage.list_memories(scope, limit=limit, order_by="importance")
            rows = [
                m for m in rows
                if m.confidence >= self.settings.pin_min_confidence
                and not _is_test_memory(m)
                and m.id not in identity_ids
            ]
            if rows:
                sections.append((title, rows))
        return self._render(user_id, sections, budget, include_ids)

    def _render(
        self,
        user_id: str,
        sections: list[tuple[str, list[Memory]]],
        budget: int,
        include_ids: bool,
    ) -> PersonaDoc:
        header = f"# Persona: {user_id}\n> Compiled from stored memories — every line traces to a memory id. Regenerate anytime; edits belong in the memory layer, not this file."
        used = estimate_tokens(header)
        parts = [header]
        memory_ids: list[str] = []
        # Identity tier first: always_pin memories lead their section, so if
        # the budget truncates, the sacred facts survive.
        for title, rows in sections:
            ordered = sorted(
                rows,
                key=lambda m: (
                    bool((m.metadata or {}).get("always_pin")),
                    m.importance,
                    m.confidence,
                ),
                reverse=True,
            )
            section_lines: list[str] = []
            for memory in ordered:
                line = f"- {memory.content.strip()}"
                if include_ids:
                    line += f" `(mem:{memory.id[:8]})`"
                cost = estimate_tokens(line) + 1
                if used + cost > budget:
                    break
                section_lines.append(line)
                memory_ids.append(memory.id)
                used += cost
            if section_lines:
                title_line = f"\n## {title}"
                parts.append(title_line)
                parts.extend(section_lines)
                used += estimate_tokens(title_line)
        text = "\n".join(parts) if memory_ids else ""
        return PersonaDoc(
            text=text,
            memory_ids=memory_ids,
            token_estimate=estimate_tokens(text),
            compiled_at=utcnow(),
        )
