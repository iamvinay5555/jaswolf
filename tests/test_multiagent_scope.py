"""Multi-agent / multi-namespace memory scoping (Jasmine 2026-06-15 architecture).

Each bot writes to its own namespace and reads its own + a shared namespace.
A bot must not see another bot's private memory; shared facts are visible to
all; the always_pin flag force-pins identity facts from the shared scope.
"""

from jaswolf.models import ContextRequest, MemoryCreate, MemoryType


async def _seed(service):
    # shared user facts (every bot should see)
    await service.add(MemoryCreate(
        user_id="alice", namespace="shared",
        content="Do not call Alice Mr Smith.", memory_type=MemoryType.PREFERENCE,
        importance=0.95, confidence=0.95, metadata={"always_pin": True},
    ))
    # Jasmine-private
    await service.add(MemoryCreate(
        user_id="alice", namespace="jasmine",
        content="Jasmine main bot uses a warm Telegram tone.", memory_type=MemoryType.SEMANTIC,
    ))
    # Freya-private
    await service.add(MemoryCreate(
        user_id="alice", namespace="freya",
        content="Freya delivers terse world-event briefings.", memory_type=MemoryType.SEMANTIC,
    ))


async def test_shared_pin_visible_to_an_agent(service):
    await _seed(service)
    text = (await service.build_context(ContextRequest(
        user_id="alice", namespace="jasmine", shared_namespace="shared",
        query="anything at all",
    ))).text
    assert "Mr Smith" in text  # shared always-pin reaches the jasmine bot


async def test_agent_cannot_see_other_agents_private_memory(service):
    await _seed(service)
    service.settings.context_noise_z = 0  # disable gate so any in-scope hit could show
    # Jasmine reads jasmine + shared, never freya
    jas = (await service.build_context(ContextRequest(
        user_id="alice", namespace="jasmine", shared_namespace="shared",
        query="briefing tone style",
    ))).text
    assert "Freya delivers" not in jas
    # Freya reads freya + shared, never jasmine
    fre = (await service.build_context(ContextRequest(
        user_id="alice", namespace="freya", shared_namespace="shared",
        query="telegram tone style",
    ))).text
    assert "Jasmine main bot" not in fre


async def test_shared_namespace_search_reaches_agent(service):
    # a shared SEMANTIC fact (not pinned) surfaces for a relevant query via the
    # multi-namespace read
    await service.add(MemoryCreate(
        user_id="alice", namespace="shared",
        content="Alice's WARP proxy is SOCKS5 on 127.0.0.1:40000, never full tunnel.",
        memory_type=MemoryType.SEMANTIC,
    ))
    text = (await service.build_context(ContextRequest(
        user_id="alice", namespace="jasmine", shared_namespace="shared",
        query="how does the WARP proxy SOCKS5 work",
    ))).text
    assert "SOCKS5" in text


async def test_shared_nonpinned_fact_crosses_agents_despite_agent_id(service):
    # Regression (2026-06-19 audit): the live plugin passes agent_id (hermes for
    # main, freya for Freya). A NON-pinned shared fact written by one bot
    # (agent_id=hermes) must still reach another bot reading with a different
    # agent_id (freya). Reads are scoped by NAMESPACE; agent_id is provenance,
    # not a read filter. Before the fix the agent_id filter dropped this hit and
    # the shared namespace only worked for force-pinned facts.
    service.settings.context_noise_z = 0  # don't let the gate hide the in-scope hit
    await service.add(MemoryCreate(
        user_id="alice", agent_id="hermes", namespace="shared",
        content="Alice's JASX is a paper-only SG Pools betting tracker.",
        memory_type=MemoryType.SEMANTIC,
    ))
    text = (await service.build_context(ContextRequest(
        user_id="alice", agent_id="freya", namespace="freya", shared_namespace="shared",
        query="what is the JASX betting tracker",
    ))).text
    assert "JASX" in text


async def test_eval_runs_in_a_bot_scope(settings, service, tmp_path):
    # cutover-preflight path: probes run in freya+shared and see shared facts,
    # not other bots' private memory
    await _seed(service)
    await service.close()
    import json as _json

    from jaswolf.evals import load_probes, run_eval

    probes_path = tmp_path / "p.json"
    probes_path.write_text(_json.dumps({"probes": [
        {"id": "naik", "kind": "context", "query": "what should I call Alice",
         "expect_any": ["mr naik"], "high_salience": True},
        {"id": "no-jasmine-leak", "kind": "context", "query": "telegram tone",
         "forbid": ["jasmine main bot"]},
    ]}))
    report = await run_eval(
        settings, load_probes(str(probes_path)), user_id="alice",
        namespace="freya", shared_namespace="shared",
    )
    assert report["namespace"] == "freya"
    assert report["shared_namespace"] == "shared"
    assert report["verdict"] == "GO_PILOT"  # shared pin seen, no cross-bot leak


async def test_no_shared_namespace_keeps_single_scope(service):
    # without shared_namespace, an agent sees only its own namespace
    await _seed(service)
    text = (await service.build_context(ContextRequest(
        user_id="alice", namespace="freya", query="anything",
    ))).text
    assert "Mr Smith" not in text  # shared not read when shared_namespace unset
