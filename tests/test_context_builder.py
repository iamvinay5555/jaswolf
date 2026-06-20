from jaswolf.models import ContextRequest, ChatMessage, MemoryCreate, MemoryType


async def _seed(service, user_id="alice"):
    rows = [
        ("User prefers Python for backend development", MemoryType.PREFERENCE, 0.95),
        ("User prefers concise answers without fluff", MemoryType.PREFERENCE, 0.9),
        ("User wants to launch a SaaS by December", MemoryType.GOAL, 0.85),
        ("User's company runs Kubernetes on Hetzner", MemoryType.SEMANTIC, 0.7),
        ("User deploys Hermes agent with Docker Compose", MemoryType.SEMANTIC, 0.7),
        ("User asked about pgvector index tuning yesterday", MemoryType.EPISODIC, 0.4),
    ]
    for content, mtype, importance in rows:
        await service.add(
            MemoryCreate(user_id=user_id, content=content, memory_type=mtype, importance=importance)
        )


async def test_context_includes_pinned_preferences_even_off_topic(service):
    await _seed(service)
    result = await service.build_context(
        ContextRequest(user_id="alice", query="how do I tune pgvector indexes?")
    )
    assert result.text
    assert "## Preferences" in result.text
    assert "Python for backend" in result.text
    assert result.token_estimate <= result.token_budget


async def test_context_respects_token_budget(service):
    await _seed(service)
    result = await service.build_context(
        ContextRequest(user_id="alice", query="deployment", token_budget=60)
    )
    assert result.token_estimate <= 60
    full = await service.build_context(
        ContextRequest(user_id="alice", query="deployment", token_budget=4000)
    )
    assert len(full.memories) >= len(result.memories)


async def test_context_from_messages(service):
    await _seed(service)
    result = await service.build_context(
        ContextRequest(
            user_id="alice",
            messages=[
                ChatMessage(role="system", content="You are Hermes."),
                ChatMessage(role="user", content="Help me deploy the agent to the VPS"),
            ],
        )
    )
    assert "Docker Compose" in result.text


async def test_empty_user_returns_empty_context(service):
    result = await service.build_context(ContextRequest(user_id="nobody", query="anything"))
    assert result.text == ""
    assert result.memories == []


async def test_xml_format(service):
    await _seed(service)
    result = await service.build_context(
        ContextRequest(user_id="alice", query="python", format="xml")
    )
    assert result.text.startswith("<memories>")
    assert result.text.endswith("</memories>")
    assert "<memory>" in result.text


async def test_context_deduplicates_similar_memories(service):
    await service.add(
        MemoryCreate(user_id="dup", content="User prefers Python", memory_type=MemoryType.PREFERENCE)
    )
    # near-identical content, stored separately by forcing a high dedup threshold off
    service.settings.dedup_threshold = 1.01
    await service.add(
        MemoryCreate(user_id="dup", content="User prefers Python!", memory_type=MemoryType.PREFERENCE)
    )
    service.settings.context_dedup_threshold = 0.85
    result = await service.build_context(ContextRequest(user_id="dup", query="python preference"))
    assert result.text.count("prefers Python") == 1


async def test_include_ids(service):
    await _seed(service)
    result = await service.build_context(
        ContextRequest(user_id="alice", query="python", include_ids=True)
    )
    assert "(mem:" in result.text
