"""JaswolfMemoryProvider — drop-in long-term memory for Hermes agents.

Two deployment modes behind one interface:

* **embedded** — runs the full memory engine in-process. No network hop, no
  extra service to operate; retrieval is a function call. Ideal for a single
  Hermes instance on a VPS.

      provider = await JaswolfMemoryProvider.embedded(user_id="alice")

* **remote** — talks to a shared JASWOLF API over HTTP. Use when several agents
  or machines share one memory store.

      provider = JaswolfMemoryProvider.remote(
          "http://localhost:8400", api_key="...", user_id="alice"
      )

Every method returns plain dicts/strings in both modes, so Hermes code never
changes when the deployment does.

Typical agent loop:

    context = await provider.build_context(messages=conversation)
    if context:
        system_prompt += "\\n\\n" + context          # before the LLM call
    ...
    await provider.observe(new_turns)                # after the turn (auto-extract)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from ..config import JaswolfSettings
from ..models import (
    ChatMessage,
    ContextRequest,
    Memory,
    MemoryCreate,
    MemoryNotFound,
    MemoryUpdate,
    ScoredMemory,
    SearchQuery,
)
from ..service import MemoryService

logger = logging.getLogger("jaswolf.provider")


def _memory_to_dict(memory: Memory) -> dict[str, Any]:
    data = memory.model_dump(exclude={"embedding", "content_hash", "tenant_id"}, mode="json")
    return data


def _scored_to_dict(scored: ScoredMemory) -> dict[str, Any]:
    return {
        "memory": _memory_to_dict(scored.memory),
        "relevance": round(scored.relevance, 4),
        "recency": round(scored.recency, 4),
        "frequency": round(scored.frequency, 4),
        "final_score": round(scored.final_score, 4),
    }


class JaswolfMemoryProvider:
    """Memory provider for Hermes. Construct via `embedded()` or `remote()`."""

    def __init__(
        self,
        *,
        service: MemoryService | None = None,
        client: Any | None = None,
        user_id: str = "default",
        agent_id: str | None = "hermes",
        namespace: str = "default",
        shared_namespace: str | None = None,
        session_id: str | None = None,
        journal_path: str | None = None,
    ):
        if (service is None) == (client is None):
            raise ValueError("provide exactly one of service= or client=")
        self._service = service
        self._client = client
        self.user_id = user_id
        self.agent_id = agent_id
        self.namespace = namespace
        # also read these shared user facts in build_context (multi-agent setups)
        self.shared_namespace = shared_namespace
        self.session_id = session_id
        self._sweeper: asyncio.Task | None = None
        # durable write-ahead journal: writes survive a crash before they reach
        # JASWOLF, replayed by flush_journal() on startup (journal.py)
        from .. import journal as _journal_mod

        self._journal = _journal_mod.WriteJournal(journal_path) if journal_path else None

    async def _journaled(self, op: str, payload: dict[str, Any]) -> Any:
        raw = {"add_memory": self._add_memory_raw, "observe": self._observe_raw}[op]
        if self._journal is None:
            return await raw(**payload)
        entry_id = self._journal.append(op, payload)
        result = await raw(**payload)  # if this raises, entry stays pending -> replayed
        self._journal.mark_done(entry_id)
        return result

    async def flush_journal(self) -> int:
        """Replay writes that were journaled but never confirmed (e.g. lost to a
        crash). Safe to call on startup — JASWOLF dedups, so replays reinforce, not
        duplicate. Stops at the first failure (JASWOLF likely down) to retry later."""
        if self._journal is None:
            return 0
        flushed = 0
        for entry in self._journal.pending():
            raw = {"add_memory": self._add_memory_raw, "observe": self._observe_raw}.get(entry["op"])
            if raw is None:
                continue
            try:
                await raw(**entry["payload"])
            except Exception as exc:
                logger.warning("journal replay paused at %s: %s", entry["id"], exc)
                break
            self._journal.mark_done(entry["id"])
            flushed += 1
        self._journal.compact()  # always prune done-markers, even if nothing replayed
        if flushed:
            logger.info("journal: replayed %d pending write(s) on startup", flushed)
        return flushed

    # -- construction ---------------------------------------------------------

    @classmethod
    async def embedded(
        cls,
        settings: JaswolfSettings | None = None,
        user_id: str = "default",
        agent_id: str | None = "hermes",
        namespace: str = "default",
        shared_namespace: str | None = None,
        session_id: str | None = None,
        auto_sweep: bool = True,
        journal_path: str | None = None,
    ) -> "JaswolfMemoryProvider":
        service = await MemoryService.create(settings)
        provider = cls(
            service=service,
            user_id=user_id,
            agent_id=agent_id,
            namespace=namespace,
            shared_namespace=shared_namespace,
            session_id=session_id,
            journal_path=journal_path,
        )
        await provider.flush_journal()  # replay anything a prior crash left pending
        if auto_sweep:
            provider._sweeper = asyncio.create_task(
                provider._sweep_loop(service.settings.sweep_interval_seconds)
            )
        return provider

    @classmethod
    def remote(
        cls,
        base_url: str,
        api_key: str | None = None,
        user_id: str = "default",
        agent_id: str | None = "hermes",
        namespace: str = "default",
        shared_namespace: str | None = None,
        session_id: str | None = None,
        timeout: float = 15.0,
        journal_path: str | None = None,
    ) -> "JaswolfMemoryProvider":
        from ..sdk.client import AsyncJaswolfClient

        return cls(
            client=AsyncJaswolfClient(base_url, api_key=api_key, timeout=timeout),
            user_id=user_id,
            agent_id=agent_id,
            namespace=namespace,
            shared_namespace=shared_namespace,
            session_id=session_id,
            journal_path=journal_path,
        )

    async def _sweep_loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            with contextlib.suppress(Exception):
                await self._service.sweep()

    async def close(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper
        if self._service is not None:
            await self._service.close()
        if self._client is not None:
            await self._client.close()

    def _ids(self, user_id: str | None, session_id: str | None) -> tuple[str, str | None]:
        return user_id or self.user_id, session_id or self.session_id

    # ======================================================================
    # Core provider interface (per JASWOLF spec)
    # ======================================================================

    async def add_memory(
        self,
        content: str,
        memory_type: str = "semantic",
        user_id: str | None = None,
        session_id: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_hours: float | None = None,
    ) -> dict[str, Any]:
        """Store one memory. Returns {"memory": ..., "created": bool} —
        created=False means it reinforced an existing duplicate. Durable when a
        journal is configured: the write survives a crash before it reaches JASWOLF."""
        uid, sid = self._ids(user_id, session_id)
        payload = {
            "content": content, "memory_type": memory_type, "user_id": uid,
            "session_id": sid, "importance": importance, "metadata": metadata,
            "ttl_hours": ttl_hours,
        }
        return await self._journaled("add_memory", payload)

    async def _add_memory_raw(
        self, content, memory_type, user_id, session_id, importance, metadata, ttl_hours
    ) -> dict[str, Any]:
        if self._service is not None:
            memory, created = await self._service.add(
                MemoryCreate(
                    user_id=user_id,
                    content=content,
                    agent_id=self.agent_id,
                    session_id=session_id,
                    namespace=self.namespace,
                    memory_type=memory_type,
                    importance=importance,
                    metadata=metadata or {},
                    ttl_hours=ttl_hours,
                )
            )
            return {"memory": _memory_to_dict(memory), "created": created}
        return await self._client.add_memory(
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            agent_id=self.agent_id,
            session_id=session_id,
            namespace=self.namespace,
            importance=importance,
            metadata=metadata or {},
            ttl_hours=ttl_hours,
        )

    async def search_memory(
        self,
        query: str,
        top_k: int = 8,
        memory_types: list[str] | None = None,
        mode: str = "hybrid",
        user_id: str | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic/hybrid search. Returns scored results, best first."""
        uid, _ = self._ids(user_id, None)
        if self._service is not None:
            results = await self._service.search(
                SearchQuery(
                    user_id=uid,
                    query=query,
                    namespace=self.namespace,
                    memory_types=memory_types,
                    mode=mode,
                    top_k=top_k,
                    min_score=min_score,
                )
            )
            return [_scored_to_dict(s) for s in results]
        response = await self._client.search(
            user_id=uid,
            query=query,
            namespace=self.namespace,
            memory_types=memory_types,
            mode=mode,
            top_k=top_k,
            min_score=min_score,
        )
        return response["results"]

    async def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        if self._service is not None:
            try:
                return _memory_to_dict(await self._service.get(memory_id))
            except MemoryNotFound:
                return None
        from ..sdk.client import JaswolfError

        try:
            return await self._client.get_memory(memory_id)
        except JaswolfError as exc:
            if exc.status_code == 404:
                return None
            raise

    async def update_memory(self, memory_id: str, **fields: Any) -> dict[str, Any]:
        if self._service is not None:
            memory = await self._service.update(memory_id, MemoryUpdate(**fields))
            return _memory_to_dict(memory)
        return await self._client.update_memory(memory_id, **fields)

    async def delete_memory(self, memory_id: str, hard: bool = False) -> bool:
        if self._service is not None:
            try:
                await self._service.delete(memory_id, hard=hard)
                return True
            except MemoryNotFound:
                return False
        from ..sdk.client import JaswolfError

        try:
            await self._client.delete_memory(memory_id, hard=hard)
            return True
        except JaswolfError as exc:
            if exc.status_code == 404:
                return False
            raise

    async def build_context(
        self,
        messages: list[dict[str, str]] | None = None,
        query: str | None = None,
        token_budget: int | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        format: str = "markdown",
    ) -> str:
        """Build the memory block to inject into the system prompt.
        Returns "" when there is nothing worth injecting."""
        uid, sid = self._ids(user_id, session_id)
        if self._service is not None:
            result = await self._service.build_context(
                ContextRequest(
                    user_id=uid,
                    query=query,
                    messages=[ChatMessage(**m) for m in messages] if messages else None,
                    agent_id=self.agent_id,
                    session_id=sid,
                    namespace=self.namespace,
                    shared_namespace=self.shared_namespace,
                    token_budget=token_budget,
                    format=format,
                )
            )
            return result.text
        response = await self._client.build_context(
            user_id=uid,
            query=query,
            messages=messages,
            agent_id=self.agent_id,
            session_id=sid,
            namespace=self.namespace,
            shared_namespace=self.shared_namespace,
            token_budget=token_budget,
            format=format,
        )
        return response["text"]

    async def consolidate_memories(
        self, user_id: str | None = None, dry_run: bool = False
    ) -> dict[str, Any]:
        uid, _ = self._ids(user_id, None)
        if self._service is not None:
            report = await self._service.consolidate(
                user_id=uid, namespace=self.namespace, dry_run=dry_run
            )
            return report.model_dump(mode="json")
        return await self._client.consolidate(
            user_id=uid, namespace=self.namespace, dry_run=dry_run
        )

    async def health_check(self) -> dict[str, Any]:
        if self._service is not None:
            return await self._service.health()
        return await self._client.health()

    # ======================================================================
    # Agent-loop conveniences
    # ======================================================================

    async def observe(
        self,
        messages: list[dict[str, str]],
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Post-turn hook: extract and store memories from new conversation
        turns. Call after each exchange; duplicates are reinforced, not
        re-stored."""
        uid, sid = self._ids(user_id, session_id)
        return await self._journaled(
            "observe", {"messages": messages, "user_id": uid, "session_id": sid}
        )

    async def _observe_raw(self, messages, user_id, session_id) -> list[dict[str, Any]]:
        if self._service is not None:
            results = await self._service.ingest_messages(
                user_id=user_id,
                messages=[ChatMessage(**m) for m in messages],
                agent_id=self.agent_id,
                session_id=session_id,
                namespace=self.namespace,
            )
            return [
                {"memory": _memory_to_dict(m), "created": created} for m, created in results
            ]
        response = await self._client.extract(
            user_id=user_id,
            messages=messages,
            agent_id=self.agent_id,
            session_id=session_id,
            namespace=self.namespace,
        )
        return response["results"]

    async def remember(self, content: str, **kwargs: Any) -> dict[str, Any]:
        """Explicit "remember this" — stores with a boosted importance floor."""
        kwargs.setdefault("importance", 0.85)
        return await self.add_memory(content, **kwargs)

    async def recall(self, query: str, top_k: int = 5) -> list[str]:
        """Lightweight recall: just the remembered statements, best first."""
        results = await self.search_memory(query, top_k=top_k)
        return [r["memory"]["content"] for r in results]

    async def inject_context(
        self, messages: list[dict[str, str]], token_budget: int | None = None
    ) -> list[dict[str, str]]:
        """Return a copy of `messages` with the memory block appended to the
        system message (or prepended as one if none exists)."""
        block = await self.build_context(messages=messages, token_budget=token_budget)
        if not block:
            return list(messages)
        out = [dict(m) for m in messages]
        for message in out:
            if message.get("role") == "system":
                message["content"] = f"{message['content']}\n\n{block}"
                return out
        return [{"role": "system", "content": block}, *out]
