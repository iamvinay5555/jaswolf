"""Storage backend protocol.

Backends translate domain models to rows. Two implementations ship with JASWOLF:
SQLiteStore (zero-infra dev/edge) and PostgresStore (production, pgvector).
Both speak the same protocol, so every other subsystem is backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from ..models import Memory, MemoryState, MemoryType, SweepReport, SEARCHABLE_STATES


@dataclass
class QueryScope:
    """Row-visibility scope. tenant_id is the isolation boundary; everything
    else narrows within it."""

    tenant_id: str = "default"
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    namespace: str | None = None
    # multi-namespace read scope (e.g. ["shared", "jasmine"]); when set it
    # takes precedence over `namespace` so one agent can read its own + shared
    # memory in a single query. `namespace` stays the single write scope.
    namespaces: tuple[str, ...] | list[str] | None = None
    memory_types: list[MemoryType] | None = None
    states: tuple[MemoryState, ...] | list[MemoryState] = field(default_factory=lambda: list(SEARCHABLE_STATES))
    min_importance: float | None = None


@dataclass
class LifecycleCutoffs:
    """Absolute datetimes: anything with last activity before the cutoff moves."""

    now: datetime
    warm_before: datetime      # active -> warm
    cold_before: datetime      # warm -> cold
    archive_before: datetime   # cold -> archived


@runtime_checkable
class StorageBackend(Protocol):
    name: str

    async def init(self) -> None: ...

    async def close(self) -> None: ...

    async def ping(self) -> bool: ...

    async def integrity_check(self) -> str: ...

    async def backup(self, dest_path: str) -> dict[str, Any]: ...

    async def upsert(self, memory: Memory) -> None: ...

    async def get(self, memory_id: str, tenant_id: str) -> Memory | None: ...

    async def delete(self, memory_id: str, tenant_id: str, hard: bool = False) -> bool: ...

    async def list_memories(
        self,
        scope: QueryScope,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "recent",  # recent | importance | created
        include_embeddings: bool = False,
    ) -> list[Memory]: ...

    async def search_vector(
        self, scope: QueryScope, query_vec: list[float], k: int
    ) -> list[tuple[Memory, float]]: ...

    async def search_keyword(
        self, scope: QueryScope, query: str, k: int
    ) -> list[tuple[Memory, float]]: ...

    async def sample_embeddings(self, scope: QueryScope, k: int) -> list[list[float]]: ...

    async def find_similar(
        self,
        scope: QueryScope,
        query_vec: list[float],
        threshold: float,
        memory_type: MemoryType | None = None,
    ) -> tuple[Memory, float] | None: ...

    async def get_by_hash(
        self, tenant_id: str, user_id: str, namespace: str, hash_value: str
    ) -> Memory | None: ...

    async def record_access(
        self, memory_ids: list[str], tenant_id: str, action: str, query: str | None = None
    ) -> None: ...

    async def add_version(
        self, memory_id: str, content: str, reason: str, payload: dict[str, Any] | None = None
    ) -> None: ...

    async def get_versions(self, memory_id: str) -> list[dict[str, Any]]: ...

    async def add_relationship(self, from_id: str, to_id: str, relation: str) -> None: ...

    async def apply_lifecycle(self, cutoffs: LifecycleCutoffs) -> SweepReport: ...

    async def stats(self, tenant_id: str, user_id: str | None = None) -> dict[str, Any]: ...

    async def get_meta(self, key: str) -> str | None: ...

    async def set_meta(self, key: str, value: str) -> None: ...
