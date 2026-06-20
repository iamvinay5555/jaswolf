"""REST API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..models import (
    ChatMessage,
    ContextSection,
    Memory,
    MemoryState,
    MemoryType,
    ScoredMemory,
    SearchMode,
)


class MemoryIn(BaseModel):
    user_id: str
    content: str = Field(min_length=1, max_length=20_000)
    agent_id: str | None = None
    session_id: str | None = None
    namespace: str = "default"
    memory_type: MemoryType = MemoryType.SEMANTIC
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    ttl_hours: float | None = Field(default=None, gt=0)


class MemoryPatch(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=20_000)
    memory_type: MemoryType | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    state: MemoryState | None = None
    metadata: dict[str, Any] | None = None
    expires_at: datetime | None = None


class MemoryOut(BaseModel):
    id: str
    user_id: str
    agent_id: str | None
    session_id: str | None
    namespace: str
    content: str
    memory_type: MemoryType
    state: MemoryState
    importance: float
    confidence: float
    access_count: int
    created_at: datetime
    updated_at: datetime
    last_accessed: datetime | None
    expires_at: datetime | None
    metadata: dict[str, Any]
    embedding: list[float] | None = None

    @classmethod
    def from_memory(cls, memory: Memory, include_embedding: bool = False) -> "MemoryOut":
        data = memory.model_dump(exclude={"tenant_id", "content_hash", "embedding"})
        return cls(**data, embedding=memory.embedding if include_embedding else None)


class CreateMemoryResponse(BaseModel):
    memory: MemoryOut
    created: bool  # False -> deduplicated into an existing memory (reinforced)


class ExtractIn(BaseModel):
    user_id: str
    text: str | None = None
    messages: list[ChatMessage] | None = None
    agent_id: str | None = None
    session_id: str | None = None
    namespace: str = "default"


class ExtractResponse(BaseModel):
    extracted: int
    results: list[CreateMemoryResponse]


class SearchIn(BaseModel):
    user_id: str
    query: str = ""
    agent_id: str | None = None
    session_id: str | None = None
    namespace: str | None = None
    memory_types: list[MemoryType] | None = None
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = Field(default=8, ge=1, le=100)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    min_importance: float | None = Field(default=None, ge=0.0, le=1.0)
    include_embeddings: bool = False
    record_access: bool = True


class ScoredMemoryOut(BaseModel):
    memory: MemoryOut
    relevance: float
    recency: float
    frequency: float
    final_score: float

    @classmethod
    def from_scored(cls, scored: ScoredMemory, include_embedding: bool = False) -> "ScoredMemoryOut":
        return cls(
            memory=MemoryOut.from_memory(scored.memory, include_embedding),
            relevance=round(scored.relevance, 4),
            recency=round(scored.recency, 4),
            frequency=round(scored.frequency, 4),
            final_score=round(scored.final_score, 4),
        )


class SearchResponse(BaseModel):
    results: list[ScoredMemoryOut]
    count: int
    latency_ms: float


class ContextIn(BaseModel):
    user_id: str
    query: str | None = None
    messages: list[ChatMessage] | None = None
    agent_id: str | None = None
    session_id: str | None = None
    namespace: str | None = None
    shared_namespace: str | None = None  # also read shared user facts
    token_budget: int | None = Field(default=None, ge=50, le=32_000)
    format: str = "markdown"
    include_ids: bool = False


class ContextResponse(BaseModel):
    text: str
    token_estimate: int
    token_budget: int
    truncated: bool
    sections: list[ContextSection]
    memory_ids: list[str]
    latency_ms: float


class ConsolidateIn(BaseModel):
    user_id: str
    namespace: str | None = None
    memory_types: list[MemoryType] | None = None
    dry_run: bool = False
