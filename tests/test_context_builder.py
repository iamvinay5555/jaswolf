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


async def test_always_pin_is_type_open_off_topic(service):
    # Strict pin mode (live VPS config): ONLY the explicit flag force-pins.
    # An always_pin RELATIONSHIP must appear in every context, exactly like an
    # always_pin preference — the 2026-07-09 wedding-pin finding: the old pin
    # loop fetched only preferences/goals, so sacred relationship facts
    # silently vanished from off-topic prompts.
    service.settings.context_pin_requires_always_pin = True
    await service.add(
        MemoryCreate(
            user_id="alice",
            content="Married to Jamie; wedding anniversary is April 2",
            memory_type=MemoryType.RELATIONSHIP,
            importance=1.0,
            metadata={"always_pin": True, "category": "sacred"},
        )
    )
    await service.add(
        MemoryCreate(
            user_id="alice",
            content="Knows the barista at the corner coffee shop",
            memory_type=MemoryType.RELATIONSHIP,
            importance=0.95,  # high importance but NOT flagged
        )
    )
    result = await service.build_context(
        ContextRequest(user_id="alice", query="how do I tune pgvector indexes?")
    )
    assert "wedding anniversary is April 2" in result.text
    assert "## Relationships" in result.text
    # the flag pins; importance alone never pins non-pref/goal types
    assert "barista" not in result.text


async def test_always_pin_importance_floor_stays_pref_goal_only(service):
    # Non-strict mode: the context_always_pin_importance floor force-pins
    # preferences/goals, but must NOT start pinning every high-importance
    # episodic/semantic/relationship row (that would be a behavior change far
    # beyond the always_pin contract fix).
    service.settings.context_pin_requires_always_pin = False
    await service.add(
        MemoryCreate(
            user_id="alice",
            content="Attended a Kubernetes meetup in Singapore last month",
            memory_type=MemoryType.EPISODIC,
            importance=0.95,  # above the pin floor, but unflagged + wrong type
        )
    )
    result = await service.build_context(
        ContextRequest(user_id="alice", query="what pasta should I cook tonight?")
    )
    assert "Kubernetes meetup" not in result.text


async def test_always_pin_taste_stays_in_its_lane(service):
    # An always_pin TASTE row must still never inject without its task_type —
    # taste is a steering lane, not an identity pin.
    service.settings.context_pin_requires_always_pin = True
    await service.add(
        MemoryCreate(
            user_id="alice",
            content="Prefer brutally short openings over throat-clearing",
            memory_type=MemoryType.TASTE,
            importance=1.0,
            metadata={
                "always_pin": True,
                "explicit_signal": True,
                "why_useful": "keeps writing tight",
                "where_to_apply": ["writing"],
            },
        )
    )
    off_topic = await service.build_context(
        ContextRequest(user_id="alice", query="how do I tune pgvector indexes?")
    )
    assert "throat-clearing" not in off_topic.text
    declared = await service.build_context(
        ContextRequest(
            user_id="alice", query="help me open this essay", task_type="writing"
        )
    )
    assert "throat-clearing" in declared.text


async def test_always_pin_respects_max_pins_cap(service):
    service.settings.context_pin_requires_always_pin = True
    service.settings.context_max_pins = 2
    for i in range(4):
        await service.add(
            MemoryCreate(
                user_id="alice",
                content=f"Sacred fact number {i} about the user's family",
                memory_type=MemoryType.RELATIONSHIP,
                importance=1.0,
                metadata={"always_pin": True},
            )
        )
    result = await service.build_context(
        ContextRequest(user_id="alice", query="completely unrelated astronomy question")
    )
    pinned_lines = [ln for ln in result.text.splitlines() if "Sacred fact" in ln]
    assert len(pinned_lines) == 2


async def test_always_pin_relationship_wins_slot_over_lower_pref(service):
    # The live 2026-07-09 case: more always_pin preferences than
    # context_max_pins, PLUS a higher-importance always_pin relationship.
    # The relationship must still be injected off-topic — importance decides
    # the budget, not "preferences are gathered first". (Pre-rework, the pref
    # loop filled all pins and the relationship pass was skipped entirely.)
    service.settings.context_pin_requires_always_pin = True
    service.settings.context_max_pins = 6
    for i in range(9):
        await service.add(
            MemoryCreate(
                user_id="alice",
                content=f"User preference number {i}",
                memory_type=MemoryType.PREFERENCE,
                importance=0.90,
                confidence=0.95,
                metadata={"always_pin": True},
            )
        )
    await service.add(
        MemoryCreate(
            user_id="alice",
            content="Married to Jamie; anniversary April 2 (sacred)",
            memory_type=MemoryType.RELATIONSHIP,
            importance=1.0,
            confidence=1.0,
            metadata={"always_pin": True},
        )
    )
    result = await service.build_context(
        ContextRequest(user_id="alice", query="completely unrelated astronomy trivia")
    )
    assert "anniversary April 2" in result.text, "sacred relationship crowded out by prefs"
    # budget still respected: exactly context_max_pins force-pins present
    pin_lines = [ln for ln in result.text.splitlines() if ln.startswith("- ")]
    assert len(pin_lines) <= 6
