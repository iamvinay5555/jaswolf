import httpx
import pytest

from jaswolf.api.app import create_app
from jaswolf.providers.hermes import JaswolfMemoryProvider
from jaswolf.sdk.client import AsyncJaswolfClient, JaswolfError


@pytest.fixture
async def sdk(settings, service):
    app = create_app(settings=settings, service=service)
    client = AsyncJaswolfClient(
        base_url="http://test", transport=httpx.ASGITransport(app=app)
    )
    yield client
    await client.close()


async def test_sdk_full_flow(sdk):
    created = await sdk.add_memory(
        user_id="alice", content="User prefers Python", memory_type="preference"
    )
    assert created["created"] is True
    memory_id = created["memory"]["id"]

    got = await sdk.get_memory(memory_id)
    assert got["content"] == "User prefers Python"

    updated = await sdk.update_memory(memory_id, importance=0.99)
    assert updated["importance"] == pytest.approx(0.99)

    results = await sdk.search(user_id="alice", query="python preference")
    assert results["count"] >= 1

    context = await sdk.build_context(user_id="alice", query="what language?")
    assert "Python" in context["text"]

    health = await sdk.health()
    assert health["status"] == "ok"

    await sdk.delete_memory(memory_id)
    with pytest.raises(JaswolfError) as exc_info:
        await sdk.update_memory("00000000-0000-0000-0000-000000000000", importance=0.5)
    assert exc_info.value.status_code == 404


async def test_provider_embedded_full_flow(settings):
    provider = await JaswolfMemoryProvider.embedded(
        settings=settings, user_id="alice", auto_sweep=False
    )
    try:
        added = await provider.add_memory(
            "User prefers Python for backend work", memory_type="preference"
        )
        assert added["created"] is True

        results = await provider.search_memory("python backend", top_k=3)
        assert results
        assert results[0]["memory"]["content"].startswith("User prefers Python")

        recalled = await provider.recall("python")
        assert any("Python" in r for r in recalled)

        observed = await provider.observe(
            [
                {"role": "user", "content": "I want to launch a SaaS by December."},
                {"role": "assistant", "content": "Great goal!"},
            ]
        )
        assert len(observed) == 1
        assert observed[0]["memory"]["memory_type"] == "goal"

        context = await provider.build_context(query="planning my product launch")
        assert "SaaS" in context

        messages = [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "What should I work on?"},
        ]
        injected = await provider.inject_context(messages)
        assert injected[0]["role"] == "system"
        assert "What I remember" in injected[0]["content"]
        assert messages[0]["content"] == "You are Hermes."  # original untouched

        memory_id = added["memory"]["id"]
        fetched = await provider.get_memory(memory_id)
        assert fetched is not None
        updated = await provider.update_memory(memory_id, importance=1.0)
        assert updated["importance"] == 1.0
        assert await provider.delete_memory(memory_id) is True
        assert await provider.get_memory("00000000-0000-0000-0000-000000000000") is None

        report = await provider.consolidate_memories(dry_run=True)
        assert "clusters_found" in report

        health = await provider.health_check()
        assert health["status"] == "ok"
    finally:
        await provider.close()


async def test_provider_remote_matches_embedded_shapes(settings, service):
    app = create_app(settings=settings, service=service)
    provider = JaswolfMemoryProvider(
        client=AsyncJaswolfClient(base_url="http://test", transport=httpx.ASGITransport(app=app)),
        user_id="alice",
    )
    try:
        added = await provider.add_memory("User prefers dark mode", memory_type="preference")
        assert added["created"] is True
        assert added["memory"]["content"] == "User prefers dark mode"

        results = await provider.search_memory("dark mode")
        assert results[0]["memory"]["content"] == "User prefers dark mode"
        assert "final_score" in results[0]

        context = await provider.build_context(query="ui preferences")
        assert "dark mode" in context

        health = await provider.health_check()
        assert health["status"] == "ok"
    finally:
        await provider.close()
