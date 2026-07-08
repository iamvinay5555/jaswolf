"""Taste Index MVP (2026-07-08 proposal, Jasmine + Claude review).

Taste = explicit judgment/steering rules: what good looks like, what to
avoid, where it should shape future work. The invariants under test:

1. explicit-signal-only capture — the engine rejects hollow taste entries
2. no passive ingestion — the extractor can never emit taste
3. task-scoped retrieval — taste appears ONLY for a declared task_type,
   and only for matching where_to_apply values
4. anti-patterns render first and are marked AVOID
5. taste is exempt from auto-consolidation and supersession
"""

import pytest

from jaswolf.models import (
    ContextRequest,
    MemoryCreate,
    MemoryState,
    MemoryType,
    SearchQuery,
    TASTE_TASK_TYPES,
)


def _taste(content, apply_to, anti_pattern=False, importance=0.8):
    return MemoryCreate(
        user_id="alice",
        content=content,
        memory_type=MemoryType.TASTE,
        importance=importance,
        confidence=0.95,
        metadata={
            "explicit_signal": True,
            "why_useful": "steers future work quality",
            "where_to_apply": apply_to,
            "anti_pattern": anti_pattern,
        },
    )


# ---- capture validation -----------------------------------------------------


async def test_taste_requires_explicit_signal(service):
    with pytest.raises(ValueError, match="explicit_signal"):
        await service.add(MemoryCreate(
            user_id="alice", content="good demos show workflow",
            memory_type=MemoryType.TASTE,
            metadata={"why_useful": "x", "where_to_apply": ["product"]},
        ))


async def test_taste_requires_why_useful(service):
    with pytest.raises(ValueError, match="why_useful"):
        await service.add(MemoryCreate(
            user_id="alice", content="good demos show workflow",
            memory_type=MemoryType.TASTE,
            metadata={"explicit_signal": True, "where_to_apply": ["product"]},
        ))


async def test_taste_requires_valid_where_to_apply(service):
    with pytest.raises(ValueError, match="where_to_apply"):
        await service.add(MemoryCreate(
            user_id="alice", content="good demos show workflow",
            memory_type=MemoryType.TASTE,
            metadata={"explicit_signal": True, "why_useful": "x"},
        ))
    with pytest.raises(ValueError, match="unknown where_to_apply"):
        await service.add(_taste("rule", ["cooking"]))


async def test_taste_where_to_apply_string_normalized_to_list(service):
    memory, created = await service.add(MemoryCreate(
        user_id="alice", content="show workflow not just output",
        memory_type=MemoryType.TASTE,
        metadata={"explicit_signal": True, "why_useful": "x", "where_to_apply": "product"},
    ))
    assert created
    assert memory.metadata["where_to_apply"] == ["product"]


async def test_valid_taste_capture(service):
    memory, created = await service.add(
        _taste("Prefer boring, official, predictable infra over custom brittle setups",
               ["architecture"])
    )
    assert created
    assert memory.memory_type == MemoryType.TASTE
    assert memory.importance == 0.8


# ---- no passive ingestion ---------------------------------------------------


async def test_extractor_can_never_emit_taste(service):
    """Guard the invariant forever: ordinary chat about quality/judgment must
    not create taste entries — capture is explicit-only."""
    results = await service.ingest_text(
        "alice",
        "I love this article about product demos. This is exactly what good "
        "looks like. I want demos to show workflow, not just output.",
    )
    assert all(m.memory_type != MemoryType.TASTE for m, _ in results)


# ---- task-scoped retrieval --------------------------------------------------


async def _seed_taste(service):
    await service.add(_taste(
        "Good product demos show the workflow, not just the output", ["product"]))
    await service.add(_taste(
        "Write plainly; cut hype words and unverified claims", ["writing"]))
    await service.add(_taste(
        "Do not propose generic same-app-different-noun demo ideas",
        ["product"], anti_pattern=True, importance=0.6))
    await service.add(MemoryCreate(
        user_id="alice", content="User's favorite tea is oolong",
        memory_type=MemoryType.PREFERENCE, importance=0.7))


async def test_no_task_type_means_no_taste(service):
    await _seed_taste(service)
    result = await service.build_context(
        ContextRequest(user_id="alice", query="what tea does the user like")
    )
    assert all(s.memory.memory_type != MemoryType.TASTE for s in result.memories)


async def test_task_type_injects_matching_taste_only(service):
    await _seed_taste(service)
    result = await service.build_context(
        ContextRequest(user_id="alice", query="draft the launch post",
                       task_type="writing")
    )
    taste = [s for s in result.memories if s.memory.memory_type == MemoryType.TASTE]
    contents = [s.memory.content for s in taste]
    assert any("Write plainly" in c for c in contents)
    assert not any("product demos" in c for c in contents)  # wrong task type


async def test_taste_never_rides_in_on_vector_similarity(service):
    """Even a query semantically identical to a taste rule must not surface it
    without a declared task_type."""
    await _seed_taste(service)
    result = await service.build_context(
        ContextRequest(user_id="alice",
                       query="product demos workflow output good demos")
    )
    assert all(s.memory.memory_type != MemoryType.TASTE for s in result.memories)


async def test_anti_pattern_renders_first_with_avoid_prefix(service):
    await _seed_taste(service)
    result = await service.build_context(
        ContextRequest(user_id="alice", query="brainstorm new app ideas",
                       task_type="product")
    )
    taste_lines = [
        line for line in result.text.splitlines()
        if "demo" in line.lower()
    ]
    assert taste_lines, "product taste must be present"
    # the anti-pattern (lower importance!) still renders before the positive rule
    assert taste_lines[0].startswith("- AVOID:")


async def test_unknown_task_type_rejected(service):
    with pytest.raises(ValueError, match="unknown task_type"):
        ContextRequest(user_id="alice", query="x", task_type="cooking")
    assert "writing" in TASTE_TASK_TYPES


async def test_archived_taste_not_injected(service):
    memory, _ = await service.add(_taste("old rule", ["writing"]))
    from jaswolf.models import MemoryUpdate
    await service.update(memory.id, MemoryUpdate(state=MemoryState.ARCHIVED))
    result = await service.build_context(
        ContextRequest(user_id="alice", query="write something", task_type="writing")
    )
    assert all(s.memory.id != memory.id for s in result.memories)


# ---- lifecycle exemptions ---------------------------------------------------


async def test_taste_excluded_from_consolidation(service):
    service.settings.dedup_threshold = 0.995
    service.settings.consolidation_threshold = 0.60
    a, _ = await service.add(_taste("Prefer boring infra over custom setups", ["architecture"]))
    b, _ = await service.add(_taste("Prefer boring infra over fancy custom setups", ["architecture"]))
    assert a.id != b.id
    report = await service.consolidate(user_id="alice")
    merged_ids = {mid for m in report.merges for mid in m.merged_ids}
    assert a.id not in merged_ids and b.id not in merged_ids
    assert (await service.get(a.id)).state == MemoryState.ACTIVE
    assert (await service.get(b.id)).state == MemoryState.ACTIVE


async def test_taste_never_superseded_by_correction_language(service):
    rule, _ = await service.add(_taste("Prefer boring infra over custom setups", ["architecture"]))
    await service.add(MemoryCreate(
        user_id="alice",
        content="Actually the infra now runs custom setups instead",
        memory_type=MemoryType.SEMANTIC,
    ))
    assert (await service.get(rule.id)).state == MemoryState.ACTIVE


# ---- search still works on the lane -----------------------------------------


async def test_taste_searchable_when_explicitly_scoped(service):
    await _seed_taste(service)
    hits = await service.search(SearchQuery(
        user_id="alice", query="demo workflow",
        memory_types=[MemoryType.TASTE], record_access=False,
    ))
    assert hits
    assert all(h.memory.memory_type == MemoryType.TASTE for h in hits)


# ---- update-path invariants (Jasmine review point 1) --------------------------


async def test_patch_cannot_hollow_out_taste(service):
    from jaswolf.models import MemoryUpdate

    memory, _ = await service.add(_taste("show workflow not output", ["product"]))
    # clearing why_useful via metadata replacement must fail loudly
    with pytest.raises(ValueError, match="why_useful"):
        await service.update(memory.id, MemoryUpdate(
            metadata={"explicit_signal": True, "where_to_apply": ["product"]}
        ))
    # corrupting where_to_apply must fail loudly
    with pytest.raises(ValueError, match="where_to_apply"):
        await service.update(memory.id, MemoryUpdate(
            metadata={"explicit_signal": True, "why_useful": "x", "where_to_apply": []}
        ))
    # the stored memory is untouched after failed patches
    loaded = await service.get(memory.id)
    assert loaded.metadata["why_useful"]
    assert loaded.metadata["where_to_apply"] == ["product"]


async def test_retype_into_taste_requires_taste_metadata(service):
    from jaswolf.models import MemoryUpdate

    memory, _ = await service.add(MemoryCreate(
        user_id="alice", content="prefer boring infra", memory_type=MemoryType.PREFERENCE,
    ))
    with pytest.raises(ValueError, match="explicit_signal"):
        await service.update(memory.id, MemoryUpdate(memory_type=MemoryType.TASTE))
    assert (await service.get(memory.id)).memory_type == MemoryType.PREFERENCE


async def test_where_to_apply_deduped_and_stable(service):
    memory, _ = await service.add(MemoryCreate(
        user_id="alice", content="cite official sources",
        memory_type=MemoryType.TASTE,
        metadata={"explicit_signal": True, "why_useful": "x",
                  "where_to_apply": ["writing", "research", "writing"]},
    ))
    assert memory.metadata["where_to_apply"] == ["research", "writing"]


# ---- truncation pressure (Jasmine review point 2) -----------------------------


async def test_anti_patterns_survive_tight_taste_budget(service):
    long_pad = " because " + "detail " * 40  # make positive rules expensive
    for i in range(3):
        await service.add(_taste(
            f"Positive product guidance number {i}{long_pad}", ["product"],
            importance=0.95))
    await service.add(_taste(
        "Do not ship generic wrapper demos", ["product"],
        anti_pattern=True, importance=0.4))
    await service.add(_taste(
        "Do not invent user research data", ["product"],
        anti_pattern=True, importance=0.4))

    result = await service.build_context(ContextRequest(
        user_id="alice", query="demo ideas", task_type="product",
        token_budget=160,  # tight: cannot fit everything
    ))
    text = result.text
    assert result.token_estimate <= 160
    # both cheap anti-patterns survive; at least one fat positive rule did not
    assert "AVOID: Do not ship generic wrapper demos" in text
    assert "AVOID: Do not invent user research data" in text
    assert sum(1 for i in range(3) if f"guidance number {i}" in text) < 3


# ---- mixed-lane isolation lock (Jasmine review point 3) ------------------------


async def test_mixed_lane_isolation_end_to_end(service):
    """One test that locks every lane boundary at once."""
    await _seed_taste(service)  # product + writing taste, one preference
    await service.add(MemoryCreate(
        user_id="alice", content="User's favorite editor is Neovim",
        memory_type=MemoryType.PREFERENCE, importance=0.95, confidence=0.95,
    ))

    # 1. no task_type => zero taste, normal recall works
    plain = await service.build_context(
        ContextRequest(user_id="alice", query="which editor does the user like"))
    assert all(s.memory.memory_type != MemoryType.TASTE for s in plain.memories)
    assert any("Neovim" in s.memory.content for s in plain.memories)

    # 2. writing task => writing taste only
    writing = await service.build_context(
        ContextRequest(user_id="alice", query="draft a post", task_type="writing"))
    w_taste = [s.memory.content for s in writing.memories
               if s.memory.memory_type == MemoryType.TASTE]
    assert any("Write plainly" in c for c in w_taste)
    assert not any("product demos" in c for c in w_taste)

    # 3. semantically-adjacent product taste still excluded from writing task
    adjacent = await service.build_context(
        ContextRequest(user_id="alice",
                       query="write about product demos and workflow output",
                       task_type="writing"))
    a_taste = [s.memory.content for s in adjacent.memories
               if s.memory.memory_type == MemoryType.TASTE]
    assert not any("product demos" in c for c in a_taste)

    # 4. preference/pin behavior unchanged by taste presence
    assert any("Neovim" in s.memory.content for s in plain.memories)
