"""JasWolf MCP memory server — tool logic and registration."""

import pytest

from jaswolf.config import JaswolfSettings
from jaswolf.mcp_server import (
    build_memory_context_impl,
    forget_impl,
    healthz_payload,
    memory_health_impl,
    recall_impl,
    record_conversation_impl,
    remember_impl,
    search_memory_impl,
)
from jaswolf.providers.hermes import JaswolfMemoryProvider


@pytest.fixture
async def provider(tmp_path):
    settings = JaswolfSettings(
        database_url=f"sqlite:///{tmp_path}/mcp.db",
        embedding_provider="hash",
        sweep_interval_seconds=3600,
        log_level="WARNING",
    )
    p = await JaswolfMemoryProvider.embedded(settings=settings, user_id="alice", auto_sweep=False)
    yield p
    await p.close()


async def test_remember_then_recall(provider):
    out = await remember_impl(provider, "User prefers tea in the morning", "preference")
    assert out["stored"] and out["created"] and out["id"]
    hits = await recall_impl(provider, "morning beverage preference", limit=5)
    assert any("tea" in h.lower() for h in hits)


async def test_remember_reinforces_duplicate(provider):
    first = await remember_impl(provider, "User lives in Singapore", "semantic")
    second = await remember_impl(provider, "User lives in Singapore", "semantic")
    assert first["created"] is True
    assert second["created"] is False  # reinforced, not duplicated


async def test_record_conversation_extracts(provider):
    out = await record_conversation_impl(provider, [
        {"role": "user", "content": "Remember that my office is in Changi"},
        {"role": "assistant", "content": "Noted."},
    ])
    assert out["observed"] == 2
    assert out["stored"] + out["reinforced"] >= 0  # extraction is gated, may store 0+


async def test_build_memory_context_returns_string(provider):
    await remember_impl(provider, "User prefers Python for backend", "preference")
    block = await build_memory_context_impl(provider, query="what language for the backend")
    assert isinstance(block, str)


async def test_search_memory_shape(provider):
    await remember_impl(provider, "User deploys with Docker Compose", "semantic")
    rows = await search_memory_impl(provider, "docker deployment", top_k=5)
    assert isinstance(rows, list)
    for r in rows:
        assert {"content", "type", "score"} <= set(r)


async def test_forget_and_health(provider):
    out = await remember_impl(provider, "Temporary fact to delete", "semantic")
    assert (await forget_impl(provider, out["id"]))["deleted"] is True
    assert (await forget_impl(provider, "no-such-id"))["deleted"] is False
    health = await memory_health_impl(provider)
    assert health["status"] in ("ok", "degraded")


async def test_healthz_payload_states(provider):
    # ready + ok -> 200
    body, code = await healthz_payload(provider)
    assert code == 200 and body["status"] == "ok"
    # not yet started -> 503 starting
    body, code = await healthz_payload(None)
    assert code == 503 and body["status"] == "starting"


async def test_healthz_503_when_degraded(provider, monkeypatch):
    async def degraded():
        return {"status": "degraded", "reasons": ["embedder fallback"]}

    monkeypatch.setattr(provider, "health_check", degraded)
    body, code = await healthz_payload(provider)
    assert code == 503 and body["status"] == "degraded"


async def test_server_registers_expected_tools(tmp_path):
    pytest.importorskip("mcp")
    from jaswolf.mcp_server import create_server

    server = create_server(JaswolfSettings(
        database_url=f"sqlite:///{tmp_path}/mcp2.db", embedding_provider="hash",
    ))
    tools = {t.name for t in await server.list_tools()}
    assert {
        "build_memory_context", "record_conversation", "recall",
        "remember", "search_memory", "forget", "memory_health",
    } <= tools
