"""Memory extraction engine.

Turns raw conversation text into structured, third-person memory candidates.
Two extractors:

* RuleExtractor — deterministic regex patterns, zero cost, always available.
* LLMExtractor — any OpenAI-compatible /chat/completions endpoint (vLLM,
  Ollama, OpenAI, LiteLLM). Higher recall, used when configured.

Strategy "hybrid" runs both and de-duplicates.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from .config import JaswolfSettings
from .models import ChatMessage, ExtractedItem, MemoryType
from .scoring import importance_for

logger = logging.getLogger("jaswolf.extraction")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
# split compound sentences before a new first-person clause:
# "I love Python and I prefer dark mode" -> two clauses, each extracted cleanly
_CLAUSE_SPLIT = re.compile(r",?\s+(?:and|but|also|plus)\s+(?=(?:I|my|we)\b)|;\s*", re.IGNORECASE)
_TAIL_PRONOUNS = [
    (re.compile(r"\bmy\b", re.IGNORECASE), "user's"),
    (re.compile(r"\bour\b", re.IGNORECASE), "their"),
    (re.compile(r"\bmine\b", re.IGNORECASE), "user's"),
]

# --- durability gate -------------------------------------------------------
# Long-term memory must not absorb conversational chatter. These filters
# implement the routing hierarchy: reactions are dropped, short-horizon plans
# become TTL-bound working memory, durable facts pass through.

# time-bounded plans: "go for lunch in 10 minutes", "deploy tonight"
_EPHEMERAL_TIME = re.compile(
    r"\b(?:in\s+\d+\s+(?:seconds?|secs?|minutes?|mins?|hours?)|right\s+now|now|today|tonight|"
    r"tomorrow|later|in\s+a\s+(?:bit|moment|sec|while)|this\s+(?:morning|afternoon|evening))\b",
    re.IGNORECASE,
)
# short-horizon activities that are state, not goals
_EPHEMERAL_ACTIVITY = re.compile(
    r"\b(?:lunch|dinner|breakfast|coffee\s+break|a\s+break|a\s+nap|the\s+gym|a\s+walk|a\s+shower)\b",
    re.IGNORECASE,
)
# compliments / reactions about the conversation itself
_REACTION = re.compile(
    r"\b(?:th(?:anks|ank\s+you)|honey|babe|buddy|bro|dear)\b"
    r"|\b(?:this|that|your|the\s+last)\s+(?:answer|response|reply|message|suggestion|idea|one)\b",
    re.IGNORECASE,
)
# deictic or assistant-directed "preferences" that don't generalize
_NON_GENERALIZING = re.compile(
    r"^user\s+\w+\s+(?:this|that|it|you)\b", re.IGNORECASE
)
# Copied conversation transcripts / injected system notes — never a durable
# user fact, whatever type the extractor guessed. (2026-06-20: the greedy
# whole-message procedure capture stored 1500-char chat transcripts as
# `procedural` memories — they carried assistant turn-markers and model-switch
# notes that a real user fact never would.)
_TRANSCRIPT_MARKER = re.compile(
    r"🧠"                                       # assistant turn-marker emoji in copied chat
    r"|\bmodel was (?:just )?switched\b"          # injected model/runtime directive
    r"|\[note:[^\]]*\bmodel\b",                   # bracketed system note about the model
    re.IGNORECASE,
)


def apply_durability_gate(item: ExtractedItem) -> ExtractedItem | None:
    """Final routing decision for an extracted candidate.

    Returns None to drop, or the item (possibly downgraded to WORKING).
    Deterministic and conservative: only obvious chatter is filtered;
    ambiguous content stays, because consolidation and lifecycle can clean
    up later but a dropped memory is gone.
    """
    content = item.content
    if _TRANSCRIPT_MARKER.search(content):
        return None  # copied transcript / system note, not durable user memory
    if item.memory_type in (
        MemoryType.PREFERENCE,
        MemoryType.GOAL,
        MemoryType.SEMANTIC,
        MemoryType.PROCEDURAL,  # the whole-message capture bypassed this screen
    ):
        if _REACTION.search(content) or _NON_GENERALIZING.match(content):
            return None
    if item.memory_type == MemoryType.GOAL:
        if _EPHEMERAL_TIME.search(content) or _EPHEMERAL_ACTIVITY.search(content):
            return ExtractedItem(
                content=content,
                memory_type=MemoryType.WORKING,
                importance=min(item.importance, 0.35),
                confidence=min(item.confidence, 0.6),
                source=item.source,
            )
    return item
_PROCEDURE_HINT = re.compile(
    r"\b(deploy|install|setup|set up|configure|debug|build|release|migrate|procedure|workflow|process)\b",
    re.IGNORECASE,
)
_NUMBERED_STEPS = re.compile(r"(?m)^\s*(?:\d+[.)]|step\s+\d+)", re.IGNORECASE)

_IRREGULAR = {"am": "is", "have": "has", "do": "does", "go": "goes", "'m": "is"}


def _third_person(verb: str) -> str:
    verb = verb.lower()
    if verb in _IRREGULAR:
        return _IRREGULAR[verb]
    if verb.endswith(("s", "sh", "ch", "x", "z")):
        return verb + "es"
    if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def _clean_tail(text: str) -> str:
    text = text.strip().rstrip(".!,;").strip()
    for pattern, replacement in _TAIL_PRONOUNS:
        text = pattern.sub(replacement, text)
    return text


class RuleExtractor:
    """Deterministic extraction. Each rule: (compiled pattern, builder)."""

    def __init__(self) -> None:
        self._rules = [
            # --- preferences -------------------------------------------------
            (
                re.compile(
                    r"\bI\s+(?:really\s+|absolutely\s+|strongly\s+)?"
                    r"(love|like|prefer|enjoy|hate|dislike)\s+(?:to\s+)?(.{3,200})",
                    re.IGNORECASE,
                ),
                lambda m: (f"User {_third_person(m.group(1))} {_clean_tail(m.group(2))}", MemoryType.PREFERENCE),
            ),
            (
                re.compile(r"\bmy\s+favou?rite\s+([\w\s]{2,40}?)\s+is\s+(.{2,120})", re.IGNORECASE),
                lambda m: (
                    f"User's favorite {_clean_tail(m.group(1))} is {_clean_tail(m.group(2))}",
                    MemoryType.PREFERENCE,
                ),
            ),
            (
                re.compile(r"\bI\s+(always|never)\s+(.{3,200})", re.IGNORECASE),
                lambda m: (f"User {m.group(1).lower()} {_clean_tail(m.group(2))}", MemoryType.PREFERENCE),
            ),
            # --- goals --------------------------------------------------------
            (
                re.compile(
                    r"\bI\s+(?:want|plan|aim|intend|hope)\s+to\s+(.{3,200})|"
                    r"\bmy\s+goal\s+is\s+to\s+(.{3,200})|"
                    r"\bI'?m\s+(?:planning|trying|aiming|working)\s+(?:to|on)\s+(.{3,200})",
                    re.IGNORECASE,
                ),
                lambda m: (
                    f"User wants to {_clean_tail(next(g for g in m.groups() if g))}",
                    MemoryType.GOAL,
                ),
            ),
            (
                re.compile(r"\bwe\s+(?:want|plan|aim|intend)\s+to\s+(.{3,200})", re.IGNORECASE),
                lambda m: (f"User's team plans to {_clean_tail(m.group(1))}", MemoryType.GOAL),
            ),
            # --- relationships --------------------------------------------------
            (
                re.compile(
                    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+is\s+my\s+"
                    r"(?!favou?rite\b|main\b|primary\b|preferred\b|go-to\b)"
                    r"([\w-]{2,30}(?:\s+[\w-]{2,30})?)\b"
                ),
                lambda m: (
                    f"{m.group(1)} is user's {_clean_tail(m.group(2))}",
                    MemoryType.RELATIONSHIP,
                ),
            ),
            (
                re.compile(
                    r"\bmy\s+(wife|husband|partner|boss|manager|cofounder|co-founder|friend|"
                    r"brother|sister|mom|mum|dad|mentor|colleague|cto|ceo)\s+is\s+(?:named\s+|called\s+)?"
                    r"([A-Z][a-z]+)",
                    re.IGNORECASE,
                ),
                lambda m: (
                    f"User's {m.group(1).lower()} is {m.group(2)}",
                    MemoryType.RELATIONSHIP,
                ),
            ),
            # --- facts ---------------------------------------------------------
            (
                re.compile(
                    r"\b(?:my\s+company|my\s+team|we)\s+(use|run|deploy|host)s?\s+(.{2,150})",
                    re.IGNORECASE,
                ),
                lambda m: (
                    f"User's company {_third_person(m.group(1))} {_clean_tail(m.group(2))}",
                    MemoryType.SEMANTIC,
                ),
            ),
            (
                re.compile(
                    r"\bI\s+(use|run|work at|work for|live in|own|maintain|develop|host)\s+(.{2,150})",
                    re.IGNORECASE,
                ),
                lambda m: (
                    f"User {_third_person(m.group(1).split()[0])}"
                    f"{' ' + ' '.join(m.group(1).split()[1:]) if len(m.group(1).split()) > 1 else ''}"
                    f" {_clean_tail(m.group(2))}",
                    MemoryType.SEMANTIC,
                ),
            ),
            (
                re.compile(r"\bI\s*(?:am|'m)\s+an?\s+(.{2,100})", re.IGNORECASE),
                lambda m: (f"User is a {_clean_tail(m.group(1))}", MemoryType.SEMANTIC),
            ),
            (
                re.compile(r"\bmy\s+([\w\s]{2,40}?)\s+is\s+(.{2,120})", re.IGNORECASE),
                lambda m: (
                    f"User's {_clean_tail(m.group(1))} is {_clean_tail(m.group(2))}",
                    MemoryType.SEMANTIC,
                ),
            ),
        ]

    def extract(self, text: str) -> list[ExtractedItem]:
        items: list[ExtractedItem] = []

        # whole-message check: numbered procedure blocks
        if _NUMBERED_STEPS.search(text) and _PROCEDURE_HINT.search(text):
            content = text.strip()[:1500]
            items.append(
                ExtractedItem(
                    content=content,
                    memory_type=MemoryType.PROCEDURAL,
                    importance=importance_for(MemoryType.PROCEDURAL, content),
                    confidence=0.7,
                    source="rules",
                )
            )

        for raw_sentence in _SENTENCE_SPLIT.split(text):
            sentence = raw_sentence.strip()
            if len(sentence) < 8 or len(sentence) > 400 or sentence.endswith("?"):
                continue
            for clause in _CLAUSE_SPLIT.split(sentence):
                clause = clause.strip()
                if len(clause) < 8:
                    continue
                for pattern, builder in self._rules:
                    m = pattern.search(clause)
                    if not m:
                        continue
                    try:
                        content, mtype = builder(m)
                    except StopIteration:
                        continue
                    content = content.strip()
                    if len(content) < 8:
                        continue
                    items.append(
                        ExtractedItem(
                            content=content,
                            memory_type=mtype,
                            importance=importance_for(mtype, clause),
                            confidence=0.75,
                            source="rules",
                        )
                    )
                    break  # one memory per clause
        return _dedupe(items)


_LLM_SYSTEM_PROMPT = """You extract long-term memories from conversation text for an AI agent.
Return ONLY a JSON array. Each element:
{"content": "<concise third-person statement about the user>",
 "type": "preference|semantic|goal|relationship|procedural|episodic",
 "importance": <0.0-1.0>, "confidence": <0.0-1.0>}

Rules:
- Extract durable facts, preferences, goals, relationships, and procedures. Skip chit-chat.
- Write content in third person ("User prefers...", "User's cofounder is...").
- Be concise: one fact per element, max 25 words each.
- Return [] if nothing is worth remembering."""


class LLMExtractor:
    def __init__(self, settings: JaswolfSettings, client: httpx.AsyncClient | None = None):
        if not settings.llm_base_url:
            raise RuntimeError("LLM extraction requires JASWOLF_LLM_BASE_URL")
        self.base_url = settings.llm_base_url.rstrip("/")
        self.model = settings.llm_model
        headers = {}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        self._client = client or httpx.AsyncClient(
            timeout=settings.llm_timeout_seconds, headers=headers
        )

    async def extract(self, text: str) -> list[ExtractedItem]:
        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": text[:8000]},
                    ],
                },
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return self._parse(raw)
        except Exception as exc:
            logger.warning("LLM extraction failed, continuing without it: %s", exc)
            return []

    @staticmethod
    def _parse(raw: str) -> list[ExtractedItem]:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        items: list[ExtractedItem] = []
        type_map = {t.value: t for t in MemoryType}
        for entry in data:
            if not isinstance(entry, dict) or not entry.get("content"):
                continue
            mtype = type_map.get(str(entry.get("type", "semantic")).lower(), MemoryType.SEMANTIC)
            items.append(
                ExtractedItem(
                    content=str(entry["content"]).strip()[:500],
                    memory_type=mtype,
                    importance=max(0.0, min(1.0, float(entry.get("importance", 0.5)))),
                    confidence=max(0.0, min(1.0, float(entry.get("confidence", 0.8)))),
                    source="llm",
                )
            )
        return items

    async def close(self) -> None:
        await self._client.aclose()


def _dedupe(items: list[ExtractedItem]) -> list[ExtractedItem]:
    seen: set[str] = set()
    out: list[ExtractedItem] = []
    for item in items:
        key = re.sub(r"\W+", " ", item.content.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


class ExtractionEngine:
    def __init__(self, settings: JaswolfSettings, llm_client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.rules = RuleExtractor()
        self.llm: LLMExtractor | None = None
        if settings.extraction_strategy in ("llm", "hybrid") and settings.llm_base_url:
            self.llm = LLMExtractor(settings, client=llm_client)

    async def extract_text(self, text: str) -> list[ExtractedItem]:
        strategy = self.settings.extraction_strategy
        items: list[ExtractedItem] = []
        if strategy in ("rules", "hybrid") or self.llm is None:
            items.extend(self.rules.extract(text))
        if self.llm is not None and strategy in ("llm", "hybrid"):
            items.extend(await self.llm.extract(text))
        # durability gate applies to every candidate regardless of extractor:
        # reactions dropped, short-horizon plans downgraded to working memory
        gated = (apply_durability_gate(item) for item in items)
        return _dedupe([item for item in gated if item is not None])

    async def extract_messages(
        self, messages: list[ChatMessage], roles: tuple[str, ...] = ("user",)
    ) -> list[ExtractedItem]:
        text = "\n".join(m.content for m in messages if m.role in roles)
        if not text.strip():
            return []
        return await self.extract_text(text)

    async def close(self) -> None:
        if self.llm is not None:
            await self.llm.close()
