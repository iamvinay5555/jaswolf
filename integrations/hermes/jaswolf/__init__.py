"""Hermes memory-provider plugin backed by JASWOLF.

Drop this into Hermes at  plugins/memory/jaswolf/  and activate with
`memory.provider: jaswolf`. It implements Hermes' host-driven MemoryProvider
ABC and delegates to a running JASWOLF REST server (`jaswolf serve`) via the JASWOLF
SDK — so context is injected before each turn and writes happen after each
turn, exactly like the Mem0 provider path.

Design priorities (in order): never block or crash a chat turn, then recall
quality. Every JASWOLF call runs on a background event loop with a short
timeout; on timeout/error the turn proceeds with NO memory rather than
hanging — a JASWOLF outage degrades to "no long-term memory this turn", never a
stuck or failed turn.

⚠ The base-class import below must match your Hermes tree. Confirm against an
existing plugin (e.g. plugins/memory/mem0/__init__.py); adjust if needed.

Config (env on the Hermes process):
  JASWOLF_API_URL        default http://127.0.0.1:8400   (the `jaswolf serve` REST API)
  JASWOLF_API_KEY        optional; matches the server's api_keys
  JASWOLF_MEMORY_USER_ID default default             (MUST match the DB's user_id)
  JASWOLF_MEMORY_TIMEOUT default 3.0  seconds for a prefetch before degrading
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Dict, List, Optional

# Hermes base class — same import the bundled plugins use (per mem0/__init__.py).
from agent.memory_provider import MemoryProvider  # type: ignore

logger = logging.getLogger(__name__)


class JaswolfProvider(MemoryProvider):
    """Host-driven memory provider delegating to a JASWOLF REST server."""

    @property
    def name(self) -> str:
        return "jaswolf"

    # -- availability: config/deps only, no network (per ABC contract) -------

    def is_available(self) -> bool:
        try:
            import jaswolf
            from jaswolf import JaswolfMemoryProvider
        except Exception:
            logger.warning("jaswolf provider unavailable: `pip install jas0` in the Hermes venv")
            return False
        # Capability guard (2026-06-15 incident): a stale SDK in the Hermes venv
        # lacked `shared_namespace`, so the provider activated green but every
        # turn crashed with a cryptic AttributeError. Fail closed + loud instead:
        # refuse to activate so Hermes falls back to built-in memory, with an
        # actionable message, rather than running broken.
        import inspect

        params = inspect.signature(JaswolfMemoryProvider.remote).parameters
        if "shared_namespace" not in params:
            logger.error(
                "jaswolf SDK %s is too old for this plugin (no `shared_namespace`). "
                "Upgrade the Hermes venv: pip install --upgrade jaswolf. "
                "Refusing to activate to avoid a green-but-broken provider.",
                getattr(jaswolf, "__version__", "?"),
            )
            return False
        return True

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        from jaswolf import JaswolfMemoryProvider

        self._user_id = (
            kwargs.get("user_id")
            or os.environ.get("JASWOLF_MEMORY_USER_ID")
            or "default"
        )
        # per-bot scope (multi-agent): each bot writes to its own namespace and
        # reads its own + the shared namespace. Defaults keep single-bot setups
        # working unchanged.
        self._agent_id = (
            os.environ.get("JASWOLF_MEMORY_AGENT_ID")
            or kwargs.get("agent_identity")
            or "hermes"
        )
        self._namespace = os.environ.get("JASWOLF_MEMORY_NAMESPACE", "default")
        self._shared_namespace = os.environ.get("JASWOLF_MEMORY_SHARED_NAMESPACE") or None
        self._base_url = os.environ.get("JASWOLF_API_URL", "http://127.0.0.1:8400")
        api_key = os.environ.get("JASWOLF_API_KEY") or None
        # 6s default: a busy CPU VPS exceeded 3s under load (2026-06-15), which
        # showed up as silent "degraded" turns. Background prefetch + per-session
        # cache still keep the live path fast.
        self._timeout = float(os.environ.get("JASWOLF_MEMORY_TIMEOUT", "6.0"))
        # durable write-ahead journal: a write survives a mid-turn gateway kill
        # (2026-06-15 lost-memory incident) — replayed on next startup. Default
        # per profile so bots don't share a journal.
        journal_path = os.environ.get("JASWOLF_MEMORY_JOURNAL") or None
        self._cache: dict[str, str] = {}

        # dedicated background event loop so the sync ABC methods never block
        # the agent's main thread on network I/O
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="jaswolf-memory", daemon=True
        )
        self._thread.start()
        self._provider = JaswolfMemoryProvider.remote(
            self._base_url,
            api_key=api_key,
            user_id=self._user_id,
            agent_id=self._agent_id,
            namespace=self._namespace,
            shared_namespace=self._shared_namespace,
            journal_path=journal_path,
        )
        logger.info(
            "jaswolf memory provider ready (user=%s agent=%s ns=%s shared=%s journal=%s base=%s)",
            self._user_id, self._agent_id, self._namespace, self._shared_namespace,
            bool(journal_path), self._base_url,
        )
        if journal_path:  # replay anything a prior crash left pending
            try:
                n = self._submit(self._provider.flush_journal()).result(30)
                if n:
                    logger.info("jaswolf journal: replayed %d pending write(s) on startup", n)
            except Exception as exc:
                logger.warning("jaswolf journal flush failed (will retry next start): %s", exc)

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _degraded(self, op: str, exc: Exception) -> None:
        # structured so a blank '' exception message can't hide the cause
        # (2026-06-15: the old log line printed nothing after the colon)
        logger.warning(
            "jaswolf %s degraded (no memory this turn): %s: %r "
            "[timeout=%ss base=%s user=%s agent=%s ns=%s shared=%s]",
            op, type(exc).__name__, str(exc), self._timeout, self._base_url,
            self._user_id, self._agent_id, self._namespace, self._shared_namespace,
        )

    def system_prompt_block(self) -> str:
        return ""  # recall is injected via prefetch(); no static block needed

    # -- recall (host-driven context injection) ------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        key = session_id or "_"
        cached = self._cache.pop(key, None)
        if cached is not None:
            return cached
        try:
            return self._submit(self._provider.build_context(query=query)).result(self._timeout)
        except Exception as exc:
            self._degraded("prefetch", exc)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        key = session_id or "_"
        future = self._submit(self._provider.build_context(query=query))

        def _store(f) -> None:
            try:
                self._cache[key] = f.result()
            except Exception as exc:  # cache miss next turn -> live prefetch
                logger.debug("jaswolf background prefetch failed: %s", exc)

        future.add_done_callback(_store)

    # -- write (host-driven, non-blocking) -----------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        turn = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        future = self._submit(self._provider.observe(turn))

        def _log(f) -> None:
            try:
                f.result()
            except Exception as exc:
                self._degraded("sync_turn", exc)

        future.add_done_callback(_log)

    # host-driven provider: no model-facing tools (most reliable for the pilot)
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def shutdown(self) -> None:
        try:
            self._submit(self._provider.close()).result(5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


# Hermes plugin discovery hook — same shape as mem0's register(). The
# PluginManager calls this; register_memory_provider routes us through the
# exclusive memory-provider path. (Having `register_memory_provider` in this
# file's source is also what auto-coerces the plugin to kind="exclusive".)
def register(ctx) -> None:
    ctx.register_memory_provider(JaswolfProvider())
