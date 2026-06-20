"""Context-boundary similarity gate (Jasmine handoff 2026-06-13).

Search may rank weak candidates; prompt injection must not. Off-topic
build_context used to inject 26 non-pinned memories on the BGE shadow DB
because bge scores arbitrary English ~0.6 raw cosine and the only floor
was retrieval's min_relevance=0.1. The gate: non-pinned vector candidates
must beat the query's noise floor (median similarity to unrelated anchors)
by `context_similarity_margin`; keyword hits and pins are exempt.
"""

from jaswolf.models import (
    ContextRequest,
    Memory,
    MemoryCreate,
    MemoryType,
    ScoredMemory,
)

# the gate the builder computes for these queries; candidates above it are
# injected, below are dropped (Jasmine's measured BGE world: floor ~0.60)
_FIXED_GATE = 0.68


def _scored(content: str, similarity: float | None, memory_type=MemoryType.SEMANTIC,
            keyword_match: bool = False):
    memory = Memory(user_id="alice", content=content, memory_type=memory_type)
    return ScoredMemory(memory=memory, relevance=0.9, final_score=0.8,
                        similarity=similarity, keyword_match=keyword_match)


async def _pin_name_warning(service):
    # identity/safety-grade => importance >= context_always_pin_importance (0.9)
    # so it force-pins into every context (the whole point of a name guardrail)
    await service.add(MemoryCreate(
        user_id="alice",
        content="Never call Alice 'Mr Smith' — just Alice",
        memory_type=MemoryType.PREFERENCE,
        importance=0.95,
        confidence=0.95,
    ))


def _mock_bge_world(monkeypatch, service, candidates, gate: float = _FIXED_GATE):
    """Patch the calibrated gate to a fixed value and feed fixed search hits,
    so these tests exercise the drop/keep/exempt logic independently of the
    calibration math (covered separately in test_calibration.py)."""

    async def fake_gate(query_vec, scope):
        return gate

    async def fake_search(query, tenant_id="default"):
        return candidates

    monkeypatch.setattr(service.context, "similarity_gate", fake_gate)
    monkeypatch.setattr(service.context.retrieval, "search", fake_search)


async def test_off_topic_context_injects_zero_non_pinned(service, monkeypatch):
    await _pin_name_warning(service)
    # her observed off-topic candidates: raw cosine 0.63-0.64, all semantic
    _mock_bge_world(monkeypatch, service, [
        _scored("VPS proxy notes", 0.642),
        _scored("JASX betting universe notes", 0.632),
        _scored("MRT route history", 0.61),
    ])
    result = await service.build_context(
        ContextRequest(user_id="alice", query="weather forecast Lisbon next week")
    )
    non_pinned = [m for m in result.memories
                  if m.memory.memory_type not in (MemoryType.PREFERENCE, MemoryType.GOAL)]
    assert non_pinned == []


async def test_pinned_preference_survives_off_topic_gate(service, monkeypatch):
    await _pin_name_warning(service)
    _mock_bge_world(monkeypatch, service, [_scored("unrelated noise", 0.63)])
    result = await service.build_context(
        ContextRequest(user_id="alice", query="weather forecast Lisbon next week")
    )
    assert "Mr Smith" in result.text  # pin injected regardless of query topic


async def test_on_topic_memory_clears_the_gate(service, monkeypatch):
    _mock_bge_world(monkeypatch, service, [
        _scored("Return route home goes via HarborFront interchange", 0.78),  # real match
        _scored("loosely related commute trivia", 0.63),                       # noise-level
    ])
    result = await service.build_context(
        ContextRequest(user_id="alice", query="which MRT route home")
    )
    texts = [m.memory.content for m in result.memories]
    assert any("HarborFront" in t for t in texts)
    assert not any("trivia" in t for t in texts)


async def test_keyword_evidenced_hit_bypasses_similarity_gate(service, monkeypatch):
    # lexical overlap is its own evidence — even when the vector side scored
    # the same memory below the gate (found by both paths, weak cosine)
    _mock_bge_world(monkeypatch, service, [
        _scored("HarborFront interchange exit B is faster at peak hour", 0.62,
                keyword_match=True),
        _scored("vector-only noise", 0.61),
    ])
    result = await service.build_context(
        ContextRequest(user_id="alice", query="HarborFront interchange")
    )
    texts = [m.memory.content for m in result.memories]
    assert any("exit B" in t for t in texts)
    assert not any("noise" in t for t in texts)


async def test_noise_z_zero_disables_gate(service, monkeypatch):
    service.settings.context_noise_z = 0.0  # gate off: similarity_gate never consulted
    _mock_bge_world(monkeypatch, service, [_scored("borderline memory", 0.61)])
    result = await service.build_context(
        ContextRequest(user_id="alice", query="anything at all")
    )
    assert any("borderline" in m.memory.content for m in result.memories)
