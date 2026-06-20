"""JASWOLF as an MCP memory server — `jaswolf mcp`.

Exposes the JASWOLF memory engine over the Model Context Protocol so any
MCP-capable agent (Hermes, Claude Desktop, …) can use it as its long-term
memory provider, the same way memory servers like mempalace/agentmemory plug
in. This is the clean replacement for a bespoke per-host memory plugin: the
contract is the stable MCP tool surface, not Hermes internals.

The tool surface deliberately supports BOTH integration styles:

* **host-driven** (most reliable): the host injects `build_memory_context`
  before each turn and calls `record_conversation` after. JASWOLF's own
  extraction/pinning/supersession does the work — the agent LLM doesn't have
  to remember to manage memory.
* **model-driven**: the agent's LLM calls `recall` / `remember` / `forget`
  as tools when it decides to.

Run a single long-lived server with the embedder prewarmed (cold start is
~seconds on CPU), so no live turn ever pays the model-load cost:

    jaswolf mcp                       # stdio (host launches & supervises it)
    jaswolf mcp --transport http      # streamable-HTTP on mcp_host:mcp_port

Tool *logic* lives in `_impl`-style functions that take an explicit provider,
so it is unit-testable without an MCP client; `create_server` is the thin
FastMCP wiring.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from .config import JaswolfSettings
from .providers.hermes import JaswolfMemoryProvider

logger = logging.getLogger("jaswolf.mcp")

Message = dict[str, str]  # {"role": "...", "content": "..."}


# ---------------------------------------------------------------------------
# Tool logic — provider-explicit, no MCP dependency, unit-testable
# ---------------------------------------------------------------------------

async def remember_impl(
    provider: JaswolfMemoryProvider,
    content: str,
    memory_type: str = "semantic",
    importance: float | None = None,
) -> dict[str, Any]:
    imp = importance if importance is not None else 0.85  # explicit "remember" boosts
    result = await provider.add_memory(content, memory_type=memory_type, importance=imp)
    mem = result["memory"]
    return {
        "stored": True,
        "created": result["created"],  # False => reinforced an existing memory
        "id": mem["id"],
        "type": mem["memory_type"],
    }


async def recall_impl(
    provider: JaswolfMemoryProvider, query: str, limit: int = 5
) -> list[str]:
    return await provider.recall(query, top_k=limit)


async def search_memory_impl(
    provider: JaswolfMemoryProvider, query: str, top_k: int = 8, mode: str = "hybrid"
) -> list[dict[str, Any]]:
    rows = await provider.search_memory(query, top_k=top_k, mode=mode)
    return [
        {
            "content": r["memory"]["content"],
            "type": r["memory"]["memory_type"],
            "score": r["final_score"],
        }
        for r in rows
    ]


async def build_memory_context_impl(
    provider: JaswolfMemoryProvider,
    query: str | None = None,
    recent_messages: list[Message] | None = None,
    token_budget: int | None = None,
) -> str:
    return await provider.build_context(
        messages=recent_messages, query=query, token_budget=token_budget
    )


async def record_conversation_impl(
    provider: JaswolfMemoryProvider, messages: list[Message]
) -> dict[str, Any]:
    results = await provider.observe(messages)
    stored = sum(1 for r in results if r.get("created"))
    return {"observed": len(messages), "stored": stored, "reinforced": len(results) - stored}


async def forget_impl(
    provider: JaswolfMemoryProvider, memory_id: str, hard: bool = False
) -> dict[str, Any]:
    return {"deleted": await provider.delete_memory(memory_id, hard=hard)}


async def memory_health_impl(provider: JaswolfMemoryProvider) -> dict[str, Any]:
    return await provider.health_check()


async def healthz_payload(
    provider: JaswolfMemoryProvider | None,
) -> tuple[dict[str, Any], int]:
    """Body + HTTP status for the /healthz endpoint. 200 only when the engine
    is fully ok; 503 while starting, on error, or when degraded — so an
    external monitor/systemd/load-balancer can act on it."""
    if provider is None:
        return {"status": "starting"}, 503
    try:
        health = await provider.health_check()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}, 503
    return health, (200 if health.get("status") == "ok" else 503)


# ---------------------------------------------------------------------------
# FastMCP wiring
# ---------------------------------------------------------------------------

_INSTRUCTIONS = """JASWOLF long-term memory for this agent.

Host-driven (preferred): call build_memory_context before answering and
record_conversation after each exchange. Model-driven: call recall to look
something up, remember to store an important durable fact, forget to remove
one. Memory is scoped to a single user; do not paste another user's data in.
"""


def create_server(settings: JaswolfSettings | None = None, *, fastmcp_lifespan: bool = True):
    """Build the FastMCP server. Imports mcp lazily so the core package does
    not depend on it.

    For streamable-HTTP, FastMCP session termination can run the FastMCP
    lifespan cleanup when an MCP client sends DELETE /mcp. The HTTP server has
    its own Starlette app lifespan below, so disable the FastMCP lifespan there
    and keep the provider process-wide until server shutdown.
    """
    from mcp.server.fastmcp import FastMCP

    settings = settings or JaswolfSettings()
    # a memory server is long-lived — always boot warm so no turn pays cold start
    warm_settings = settings.model_copy(update={"embedding_prewarm": True})
    # idempotent provider init: stdio runs the FastMCP lifespan at process start,
    # HTTP inits via run()'s startup hook; tools also ensure it lazily. The lock
    # makes concurrent first-callers safe.
    state: dict[str, JaswolfMemoryProvider | None] = {"provider": None}
    lock = asyncio.Lock()

    async def ensure_provider() -> JaswolfMemoryProvider:
        if state["provider"] is None:
            async with lock:
                if state["provider"] is None:
                    state["provider"] = await JaswolfMemoryProvider.embedded(
                        settings=warm_settings,
                        user_id=settings.mcp_user_id,
                        agent_id=settings.mcp_agent_id,
                        namespace=settings.mcp_namespace,
                    )
                    logger.info(
                        "JASWOLF MCP provider ready (user=%s, db=%s)",
                        settings.mcp_user_id, settings.database_url,
                    )
        return state["provider"]

    async def close_provider() -> None:
        if state["provider"] is not None:
            await state["provider"].close()
            state["provider"] = None

    @asynccontextmanager
    async def lifespan(_server):
        await ensure_provider()  # stdio: warm at process start
        try:
            yield
        finally:
            await close_provider()

    mcp_kwargs: dict[str, Any] = {
        "instructions": _INSTRUCTIONS,
        "host": settings.mcp_host,
        "port": settings.mcp_port,
    }
    if fastmcp_lifespan:
        mcp_kwargs["lifespan"] = lifespan
    mcp = FastMCP("jaswolf-memory", **mcp_kwargs)
    # run() needs these for the HTTP startup/shutdown hooks
    mcp._jaswolf_ensure_provider = ensure_provider  # type: ignore[attr-defined]
    mcp._jaswolf_close_provider = close_provider    # type: ignore[attr-defined]

    @mcp.tool()
    async def build_memory_context(
        query: str | None = None,
        recent_messages: list[Message] | None = None,
        token_budget: int | None = None,
    ) -> str:
        """Return the memory block to prepend to the system prompt for this
        turn. Pass the recent conversation as recent_messages, or an explicit
        query. Returns "" when nothing is worth injecting."""
        return await build_memory_context_impl(await ensure_provider(), query, recent_messages, token_budget)

    @mcp.tool()
    async def record_conversation(messages: list[Message]) -> dict[str, Any]:
        """Post-turn: extract and store durable memories from new turns.
        Ephemeral chatter is dropped and corrections supersede old facts
        automatically — safe to call after every exchange."""
        return await record_conversation_impl(await ensure_provider(), messages)

    @mcp.tool()
    async def recall(query: str, limit: int = 5) -> list[str]:
        """Look up what is remembered about the user, best match first."""
        return await recall_impl(await ensure_provider(), query, limit)

    @mcp.tool()
    async def remember(
        content: str, memory_type: str = "semantic", importance: float | None = None
    ) -> dict[str, Any]:
        """Explicitly store a durable fact/preference (e.g. the user asked you
        to remember it). memory_type: semantic|preference|goal|procedural|
        episodic|relationship."""
        return await remember_impl(await ensure_provider(), content, memory_type, importance)

    @mcp.tool()
    async def search_memory(
        query: str, top_k: int = 8, mode: str = "hybrid"
    ) -> list[dict[str, Any]]:
        """Search memories with scores. mode: hybrid|semantic|keyword."""
        return await search_memory_impl(await ensure_provider(), query, top_k, mode)

    @mcp.tool()
    async def forget(memory_id: str, hard: bool = False) -> dict[str, Any]:
        """Delete a memory by id (soft by default)."""
        return await forget_impl(await ensure_provider(), memory_id, hard)

    @mcp.tool()
    async def memory_health() -> dict[str, Any]:
        """Engine health: provider, embedding fingerprint, fallback, status."""
        return await memory_health_impl(await ensure_provider())

    # plain HTTP health endpoint for systemd / monitors / a Hermes pre-start
    # check. Reports current state only — never triggers a slow init, so a probe
    # can't make itself time out (provider is warmed at startup by run()).
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        body, code = await healthz_payload(state["provider"])
        return JSONResponse(body, status_code=code)

    return mcp


def run(settings: JaswolfSettings | None = None, transport: str = "stdio") -> None:
    settings = settings or JaswolfSettings()
    if transport in ("http", "streamable-http"):
        # FastMCP's HTTP app doesn't run our FastMCP lifespan at uvicorn
        # startup, so wrap it: init+prewarm the provider BEFORE serving (health
        # is only reachable once warm), then run the session manager's lifespan.
        # Disable FastMCP's own provider lifespan in HTTP mode so DELETE /mcp
        # closes only the client session, not the process-wide memory provider.
        server = create_server(settings, fastmcp_lifespan=False)
        from contextlib import asynccontextmanager as _acm

        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount

        inner = server.streamable_http_app()

        @_acm
        async def lifespan(_app):
            await server._jaswolf_ensure_provider()  # type: ignore[attr-defined]
            async with inner.router.lifespan_context(inner):
                yield
            await server._jaswolf_close_provider()   # type: ignore[attr-defined]

        outer = Starlette(lifespan=lifespan, routes=[Mount("/", app=inner)])
        uvicorn.run(outer, host=settings.mcp_host, port=settings.mcp_port)
    else:
        server = create_server(settings)
        server.run(transport="stdio")
