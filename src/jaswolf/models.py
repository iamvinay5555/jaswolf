"""Core domain models for JasWolf.

Everything that moves between subsystems is one of these types. Storage
backends translate them to rows; the API layer translates them to JSON.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


def content_hash(content: str) -> str:
    """Stable hash of normalized content, used for exact-duplicate detection."""
    normalized = re.sub(r"\s+", " ", content.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class MemoryType(str, Enum):
    WORKING = "working"          # short-lived, TTL-bound (current tasks, scratch state)
    EPISODIC = "episodic"        # experiences: "user deployed Hermes on 2026-06-01"
    SEMANTIC = "semantic"        # facts: "user's company runs Kubernetes"
    PREFERENCE = "preference"    # durable user preferences
    PROCEDURAL = "procedural"    # learned workflows and how-tos
    GOAL = "goal"                # active objectives
    RELATIONSHIP = "relationship"  # people/org relationships
    TASTE = "taste"              # judgment/steering rules: what good looks like,
    #                              what to avoid, where it should shape future work.
    #                              Explicit-signal-only: never passively extracted,
    #                              never auto-consolidated or superseded, injected
    #                              into context only for a declared task_type.


# The closed vocabulary shared by taste capture (metadata.where_to_apply) and
# task-aware retrieval (ContextRequest.task_type). Exact-match on purpose:
# free-text apply-scopes matched semantically would reintroduce the embedding
# calibration problem inside the taste lane.
TASTE_TASK_TYPES = frozenset(
    {"writing", "product", "design", "research", "architecture", "agent_behavior"}
)


class MemoryState(str, Enum):
    ACTIVE = "active"
    WARM = "warm"
    COLD = "cold"
    ARCHIVED = "archived"
    DELETED = "deleted"


# States that participate in retrieval by default.
SEARCHABLE_STATES = (MemoryState.ACTIVE, MemoryState.WARM, MemoryState.COLD)


class RelationType(str, Enum):
    MERGED_INTO = "merged_into"
    SUPERSEDES = "supersedes"
    DERIVED_FROM = "derived_from"
    RELATED = "related"


class SearchMode(str, Enum):
    SEMANTIC = "semantic"      # vector similarity only
    KEYWORD = "keyword"        # full-text only
    HYBRID = "hybrid"          # vector + keyword fused with RRF
    RECENCY = "recency"        # most recently active first
    IMPORTANCE = "importance"  # highest importance first


class Memory(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(default_factory=new_id)
    tenant_id: str = "default"
    user_id: str
    agent_id: str | None = None
    session_id: str | None = None
    namespace: str = "default"
    content: str
    content_hash: str = ""
    embedding: list[float] | None = None
    memory_type: MemoryType = MemoryType.SEMANTIC
    state: MemoryState = MemoryState.ACTIVE
    importance: float = 0.5
    confidence: float = 0.8
    access_count: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    last_accessed: datetime | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("importance", "confidence")
    @classmethod
    def _clamp_unit(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    def last_activity(self) -> datetime:
        """Most recent touch — used for recency scoring and lifecycle."""
        candidates = [self.created_at, self.updated_at]
        if self.last_accessed:
            candidates.append(self.last_accessed)
        return max(candidates)


class MemoryCreate(BaseModel):
    user_id: str
    content: str
    agent_id: str | None = None
    session_id: str | None = None
    namespace: str = "default"
    memory_type: MemoryType = MemoryType.SEMANTIC
    importance: float | None = None  # None -> scored automatically
    confidence: float = 0.8
    metadata: dict[str, Any] = Field(default_factory=dict)
    ttl_hours: float | None = None  # explicit TTL; working memory gets a default


class MemoryUpdate(BaseModel):
    content: str | None = None
    memory_type: MemoryType | None = None
    importance: float | None = None
    confidence: float | None = None
    state: MemoryState | None = None
    metadata: dict[str, Any] | None = None
    expires_at: datetime | None = None


class SearchQuery(BaseModel):
    # forbid unknown fields: SearchQuery(text=...) must fail loudly, not fall
    # through to an empty-query listing (Jasmine's 24h-eval probe bug)
    model_config = ConfigDict(extra="forbid")

    user_id: str
    query: str = ""
    agent_id: str | None = None
    session_id: str | None = None
    namespace: str | None = None
    # read across several namespaces (e.g. own + shared); overrides namespace
    namespaces: list[str] | None = None
    memory_types: list[MemoryType] | None = None
    states: list[MemoryState] | None = None
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = 8
    min_score: float | None = None
    min_importance: float | None = None
    record_access: bool = True


class ScoredMemory(BaseModel):
    memory: Memory
    relevance: float = 0.0
    recency: float = 0.0
    frequency: float = 0.0
    final_score: float = 0.0
    # raw cosine from the vector path, before per-pool min-max normalization;
    # None for keyword-only/listing hits. Normalized relevance shows 1.0 even
    # for off-topic queries — gate decisions should read this field instead.
    similarity: float | None = None
    # True when the FTS/keyword path matched this memory — lexical overlap is
    # its own relevance evidence, independent of vector calibration
    keyword_match: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ConversationMessage(BaseModel):
    """One raw conversation turn — the L0 evidence layer.

    Memories are distilled claims; these are the ground truth they were
    distilled from. Stored only when conversation_capture is enabled, and
    immutable once written: corrections happen in the memory layer, never by
    rewriting history.
    """

    id: str = Field(default_factory=new_id)
    tenant_id: str = "default"
    user_id: str
    agent_id: str | None = None
    session_id: str | None = None
    namespace: str = "default"
    role: str
    content: str
    created_at: datetime = Field(default_factory=utcnow)


class ConversationHit(BaseModel):
    message: ConversationMessage
    score: float = 0.0


class MemoryExplanation(BaseModel):
    """Full drill-down for one memory: the row, its history, its graph
    edges, and the raw conversation turns it was extracted from. Answers
    "where did the agent get this idea?" deterministically."""

    memory: Memory
    versions: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[ConversationMessage] = Field(default_factory=list)


class PersonaDoc(BaseModel):
    """Compiled at-a-glance user profile (L3). A deterministic VIEW over
    stored memories — every line traces to a memory id, so the persona can
    never claim something the atom layer doesn't back."""

    text: str
    memory_ids: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    compiled_at: datetime = Field(default_factory=utcnow)


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    query: str | None = None
    messages: list[ChatMessage] | None = None
    agent_id: str | None = None
    session_id: str | None = None
    namespace: str | None = None  # the agent's own (write) namespace
    # also read from this namespace (shared user facts every bot should see)
    shared_namespace: str | None = None
    token_budget: int | None = None
    format: str = "markdown"  # "markdown" | "xml"
    include_ids: bool = False
    # declared kind of work the agent is about to do; when set, active taste
    # memories whose where_to_apply includes it are injected as a Taste section
    task_type: str | None = None

    @field_validator("task_type")
    @classmethod
    def _known_task_type(cls, v: str | None) -> str | None:
        if v is not None and v not in TASTE_TASK_TYPES:
            raise ValueError(
                f"unknown task_type {v!r}; expected one of {sorted(TASTE_TASK_TYPES)}"
            )
        return v


class ContextSection(BaseModel):
    title: str
    memory_type: MemoryType
    count: int
    tokens: int


class ContextResult(BaseModel):
    text: str
    memories: list[ScoredMemory] = Field(default_factory=list)
    sections: list[ContextSection] = Field(default_factory=list)
    token_estimate: int = 0
    token_budget: int = 0
    truncated: bool = False


class ExtractedItem(BaseModel):
    content: str
    memory_type: MemoryType
    importance: float = 0.5
    confidence: float = 0.75
    source: str = "rules"  # "rules" | "llm"


class ConsolidationMerge(BaseModel):
    canonical_id: str
    merged_ids: list[str]
    merged_content: str
    similarity: float


class ConsolidationReport(BaseModel):
    examined: int = 0
    clusters_found: int = 0
    memories_merged: int = 0
    merges: list[ConsolidationMerge] = Field(default_factory=list)
    dry_run: bool = False


class SweepReport(BaseModel):
    expired_working: int = 0
    active_to_warm: int = 0
    warm_to_cold: int = 0
    cold_to_archived: int = 0
    pruned_conversations: int = 0
    # cold journal: turns exported to the archive before pruning (archive.py).
    # With an archive dir configured, pruned can never exceed archived.
    archived_conversations: int = 0
    swept_at: datetime = Field(default_factory=utcnow)


class MemoryNotFound(Exception):
    def __init__(self, memory_id: str):
        self.memory_id = memory_id
        super().__init__(f"memory not found: {memory_id}")


def default_expiry(memory_type: MemoryType, ttl_hours: float | None, working_ttl_hours: float) -> datetime | None:
    """Working memory always expires; other types only when an explicit TTL is given."""
    if ttl_hours is not None:
        return utcnow() + timedelta(hours=ttl_hours)
    if memory_type == MemoryType.WORKING:
        return utcnow() + timedelta(hours=working_ttl_hours)
    return None
