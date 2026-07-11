"""Persona compiler: deterministic L3 view, traced lines, budget, gates."""

from jaswolf.models import MemoryCreate, MemoryType


async def _seed(service, user_id="alice"):
    rows = [
        # importance 0.85 keeps this below the non-strict pin floor (0.9), so
        # it exercises the ordinary Preferences section, not the Identity tier
        ("User prefers Python over JavaScript", MemoryType.PREFERENCE, 0.85, None),
        ("Never call him Mr Smith", MemoryType.PREFERENCE, 0.95, {"always_pin": True}),
        ("Ship Project Atlas v1 by August", MemoryType.GOAL, 0.85, None),
        ("Dana is Alice's cofounder and daily collaborator", MemoryType.RELATIONSHIP, 0.8, None),
        ("User runs a nine-person cloud infrastructure team", MemoryType.SEMANTIC, 0.85, None),
    ]
    created = {}
    for content, mtype, importance, metadata in rows:
        memory, _ = await service.add(
            MemoryCreate(
                user_id=user_id, content=content, memory_type=mtype,
                importance=importance, confidence=0.9, metadata=metadata or {},
            )
        )
        created[content] = memory
    return created


async def test_persona_renders_all_sections_with_ids(service):
    created = await _seed(service)
    doc = await service.compile_persona(user_id="alice")
    assert doc.text.startswith("# Persona: alice")
    for title in ("Identity (always pinned)", "Preferences", "Goals", "Relationships", "Key facts"):
        assert f"## {title}" in doc.text
    assert "User prefers Python over JavaScript" in doc.text
    # every line traces: short ids present, and full ids in memory_ids
    pin = created["Never call him Mr Smith"]
    assert f"(mem:{pin.id[:8]})" in doc.text
    assert pin.id in doc.memory_ids


async def test_persona_identity_tier_leads_and_dedupes(service):
    await _seed(service)
    doc = await service.compile_persona(user_id="alice")
    identity = doc.text.split("## Identity (always pinned)")[1].split("\n## ")[0]
    assert "Mr Smith" in identity, "always_pin memory belongs in the Identity tier"
    # a pin renders once: not repeated in its type section below
    assert doc.text.count("Never call him Mr Smith") == 1


async def test_persona_identity_pool_matches_context_pins(service):
    """The renderer bug found live 2026-07-11: the first pyramid release
    filled per-type sections in order, so many pinned preferences exhausted
    the budget before Relationships — the persona omitted pins the context
    builder injected every turn. The Identity tier is the SAME ranked pool,
    rendered first."""
    from jaswolf.models import MemoryCreate

    # over-subscribe the pin budget with flagged preferences…
    for i in range(service.settings.context_max_pins + 4):
        await service.add(
            MemoryCreate(
                user_id="alice",
                content=f"Pinned working style rule number {i} for daily behavior",
                memory_type=MemoryType.PREFERENCE,
                importance=0.9,
                confidence=0.95,
                metadata={"always_pin": True},
            )
        )
    # …plus one higher-importance sacred relationship pin
    sacred, _ = await service.add(
        MemoryCreate(
            user_id="alice",
            content="Married to Jamie; wedding anniversary is April 2",
            memory_type=MemoryType.RELATIONSHIP,
            importance=1.0,
            confidence=1.0,
            metadata={"always_pin": True},
        )
    )
    doc = await service.compile_persona(user_id="alice", token_budget=150)
    assert "wedding anniversary" in doc.text, (
        "sacred relationship pin crowded out of the persona by preference pins"
    )
    assert sacred.id in doc.memory_ids


async def test_persona_empty_when_no_memories(service):
    doc = await service.compile_persona(user_id="ghost")
    assert doc.text == ""
    assert doc.memory_ids == []


async def test_persona_respects_token_budget(service):
    await _seed(service)
    small = await service.compile_persona(user_id="alice", token_budget=60)
    full = await service.compile_persona(user_id="alice")
    assert small.token_estimate <= 60
    assert len(small.memory_ids) < len(full.memory_ids)
    # sacred pin survives even a tiny budget
    assert "Mr Smith" in small.text


async def test_persona_excludes_low_confidence_and_test_memories(service):
    await _seed(service)
    await service.add(
        MemoryCreate(
            user_id="alice", content="Unverified rumor about user",
            memory_type=MemoryType.PREFERENCE, importance=0.9, confidence=0.5,
        )
    )
    await service.add(
        MemoryCreate(
            user_id="alice", content="staging pref for the pipeline",
            memory_type=MemoryType.PREFERENCE, importance=0.95, confidence=0.95,
            metadata={"staging": True},
        )
    )
    doc = await service.compile_persona(user_id="alice")
    assert "Unverified rumor" not in doc.text
    assert "staging pref" not in doc.text


async def test_persona_semantic_needs_high_importance(service):
    await _seed(service)
    await service.add(
        MemoryCreate(
            user_id="alice", content="User mentioned it might rain",
            memory_type=MemoryType.SEMANTIC, importance=0.3, confidence=0.9,
        )
    )
    doc = await service.compile_persona(user_id="alice")
    assert "might rain" not in doc.text
    assert "cloud infrastructure team" in doc.text


async def test_persona_no_ids_mode(service):
    await _seed(service)
    doc = await service.compile_persona(user_id="alice", include_ids=False)
    assert "(mem:" not in doc.text
    assert doc.memory_ids  # ids still reported for programmatic use


async def test_persona_deterministic(service):
    await _seed(service)
    a = await service.compile_persona(user_id="alice")
    b = await service.compile_persona(user_id="alice")
    assert a.text == b.text
    assert a.memory_ids == b.memory_ids
