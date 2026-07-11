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
import time
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


class _ObserveBuffer:
    """Per-session staging area for the observe cadence."""

    __slots__ = ("messages", "journal_ids", "last_at", "flushes")

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []
        self.journal_ids: list[str] = []
        self.last_at = time.monotonic()
        self.flushes = 0


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
        observe_every_n: int = 1,
        observe_warmup: bool = True,
        observe_idle_flush_seconds: float = 600.0,
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
        # observe cadence (Tencent-inspired, v0.3.0): 1 = extract every call
        # (classic). N>1 buffers turns per session and extracts in batches,
        # with a warm-up ramp (1, 2, 4, … up to N) so fresh sessions learn
        # from turn one, and an idle flush so a finished session's tail is
        # never left unextracted. Buffered turns are journaled individually
        # when a journal is configured, so a crash replays them on startup.
        self.observe_every_n = max(1, observe_every_n)
        self.observe_warmup = observe_warmup
        self.observe_idle_flush_seconds = observe_idle_flush_seconds
        self._buffers: dict[tuple[str, str | None], _ObserveBuffer] = {}
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
            observe_every_n=service.settings.observe_every_n,
            observe_warmup=service.settings.observe_warmup,
            observe_idle_flush_seconds=service.settings.observe_idle_flush_seconds,
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
            with contextlib.suppress(Exception):
                await self.flush_observe_buffers(only_idle=True)

    async def close(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper
        # don't sit on unextracted turns; journaled entries survive a failure here
        with contextlib.suppress(Exception):
            await self.flush_observe_buffers()
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
        """Semantic/hybrid search. Returns scored results, best first.

        Reads the same namespace surface as build_context (own + shared):
        search answering "do you remember X?" from a different universe than
        the context injector was the silent dual-brain found 2026-07-09.
        """
        uid, _ = self._ids(user_id, None)
        read_namespaces = (
            [self.namespace, self.shared_namespace]
            if self.shared_namespace and self.shared_namespace != self.namespace
            else None
        )
        if self._service is not None:
            results = await self._service.search(
                SearchQuery(
                    user_id=uid,
                    query=query,
                    namespace=self.namespace,
                    namespaces=read_namespaces,
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
            namespaces=read_namespaces,
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
        task_type: str | None = None,
    ) -> str:
        """Build the memory block to inject into the system prompt.
        Returns "" when there is nothing worth injecting.

        task_type declares the kind of work this turn is (models.TASTE_TASK_TYPES);
        it unlocks the Taste section for matching where_to_apply rules."""
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
                    task_type=task_type,
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
            task_type=task_type,
        )
        return response["text"]

    async def search_conversations(
        self,
        query: str,
        top_k: int = 10,
        user_id: str | None = None,
        session_id: str | None = None,
        since_days: float | None = None,
    ) -> list[dict[str, Any]]:
        """Full-text search over the raw L0 conversation archive (requires
        conversation_capture). Finds what extraction missed; empty query
        returns the most recent turns. Reads own + shared namespaces, same
        surface as search_memory."""
        uid, _ = self._ids(user_id, None)
        read_namespaces = (
            [self.namespace, self.shared_namespace]
            if self.shared_namespace and self.shared_namespace != self.namespace
            else None
        )
        if self._service is not None:
            hits = await self._service.search_conversations(
                user_id=uid,
                query=query,
                namespace=None if read_namespaces else self.namespace,
                namespaces=read_namespaces,
                session_id=session_id,
                top_k=top_k,
                since_days=since_days,
            )
            return [
                {
                    "role": h.message.role,
                    "content": h.message.content,
                    "session_id": h.message.session_id,
                    "created_at": h.message.created_at.isoformat(),
                    "score": h.score,
                }
                for h in hits
            ]
        response = await self._client.search_conversations(
            user_id=uid,
            query=query,
            namespace=None if read_namespaces else self.namespace,
            namespaces=read_namespaces,
            session_id=session_id,
            top_k=top_k,
            since_days=since_days,
        )
        return response["results"]

    async def explain_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Provenance drill-down: the memory, its version history, its graph
        edges (supersedes/merged/derived), and the raw turns it came from."""
        if self._service is not None:
            try:
                explanation = await self._service.explain(memory_id)
            except MemoryNotFound:
                return None
            return explanation.model_dump(
                mode="json", exclude={"memory": {"embedding", "content_hash", "tenant_id"}}
            )
        from ..sdk.client import JaswolfError

        try:
            return await self._client.explain_memory(memory_id)
        except JaswolfError as exc:
            if exc.status_code == 404:
                return None
            raise

    async def get_persona(
        self,
        user_id: str | None = None,
        token_budget: int | None = None,
        include_ids: bool = True,
    ) -> dict[str, Any]:
        """Compiled at-a-glance profile of the user (deterministic L3 view).
        Returns {"text": markdown, "memory_ids": [...], ...}."""
        uid, _ = self._ids(user_id, None)
        read_namespaces = (
            [self.namespace, self.shared_namespace]
            if self.shared_namespace and self.shared_namespace != self.namespace
            else None
        )
        if self._service is not None:
            doc = await self._service.compile_persona(
                user_id=uid,
                namespace=None if read_namespaces else self.namespace,
                namespaces=read_namespaces,
                include_ids=include_ids,
                token_budget=token_budget,
            )
            return doc.model_dump(mode="json")
        return await self._client.get_persona(
            user_id=uid,
            namespace=None if read_namespaces else self.namespace,
            namespaces=read_namespaces,
            include_ids=include_ids,
            token_budget=token_budget,
        )

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
        re-stored.

        With observe_every_n > 1, turns are buffered per session and
        extracted in batches on a warm-up ramp (1, 2, 4, … up to N). A call
        that only buffered returns [] — the memories arrive on the flushing
        call, the idle sweep, or close(). Every buffered call is journaled
        first when a journal is configured, so a crash mid-buffer replays
        instead of forgetting.
        """
        uid, sid = self._ids(user_id, session_id)
        if self.observe_every_n <= 1:
            return await self._journaled(
                "observe", {"messages": messages, "user_id": uid, "session_id": sid}
            )
        key = (uid, sid)
        buffer = self._buffers.setdefault(key, _ObserveBuffer())
        # a buffer that idled past the flush window belongs to a *previous*
        # sitting — flush it before this turn joins, so stale context never
        # batches with a fresh conversation (also the remote-mode idle path,
        # which has no sweep loop)
        if (
            buffer.messages
            and time.monotonic() - buffer.last_at >= self.observe_idle_flush_seconds
        ):
            await self._flush_buffer(key)
        if self._journal is not None:
            buffer.journal_ids.append(
                self._journal.append(
                    "observe", {"messages": messages, "user_id": uid, "session_id": sid}
                )
            )
        buffer.messages.extend(messages)
        buffer.last_at = time.monotonic()
        if len(buffer.messages) >= self._observe_threshold(buffer):
            return await self._flush_buffer(key)
        return []

    def _observe_threshold(self, buffer: _ObserveBuffer) -> int:
        """Messages needed before the next extraction pass. Warm-up doubles
        1 -> 2 -> 4 … so a brand-new session is never amnesiac for its first
        N turns; steady state is observe_every_n."""
        if self.observe_warmup:
            return min(self.observe_every_n, 2 ** buffer.flushes)
        return self.observe_every_n

    async def _flush_buffer(self, key: tuple[str, str | None]) -> list[dict[str, Any]]:
        buffer = self._buffers.get(key)
        if buffer is None or not buffer.messages:
            return []
        uid, sid = key
        messages, journal_ids = buffer.messages, buffer.journal_ids
        buffer.messages, buffer.journal_ids = [], []
        buffer.flushes += 1
        buffer.last_at = time.monotonic()
        try:
            results = await self._observe_raw(messages=messages, user_id=uid, session_id=sid)
        except Exception:
            # restore so a transient failure (JASWOLF down) retries on the next
            # turn; journal entries stay pending for the crash case
            buffer.messages = messages + buffer.messages
            buffer.journal_ids = journal_ids + buffer.journal_ids
            buffer.flushes -= 1
            raise
        if self._journal is not None:
            for journal_id in journal_ids:
                self._journal.mark_done(journal_id)
        return results

    async def flush_observe_buffers(self, only_idle: bool = False) -> list[dict[str, Any]]:
        """Flush pending observe buffers (all, or just the idle ones). Called
        by the sweep loop and close(); also available to hosts that want an
        explicit end-of-session flush."""
        flushed: list[dict[str, Any]] = []
        for key, buffer in list(self._buffers.items()):
            if not buffer.messages:
                continue
            if (
                only_idle
                and time.monotonic() - buffer.last_at < self.observe_idle_flush_seconds
            ):
                continue
            flushed.extend(await self._flush_buffer(key))
        return flushed

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
