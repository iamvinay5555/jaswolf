"""Deterministic shadow evaluator + query guardrails (Jasmine handoff 2026-06-12)."""

import json

import pytest
from pydantic import ValidationError

from jaswolf.evals import load_probes, run_eval
from jaswolf.models import MemoryCreate, MemoryType, SearchQuery


async def _seed(service):
    await service.add(MemoryCreate(
        user_id="alice",
        content="Never call Alice 'Mr Smith' — he prefers just Alice",
        memory_type=MemoryType.PREFERENCE,
        importance=0.85,
        confidence=0.9,
    ))
    await service.add(MemoryCreate(
        user_id="alice",
        content="Return route home goes via HarborFront interchange",
        memory_type=MemoryType.SEMANTIC,
    ))


def _write_probes(tmp_path, probes):
    path = tmp_path / "probes.json"
    path.write_text(json.dumps({"probes": probes}))
    return str(path)


# ---- SearchQuery guardrails ----------------------------------------------------


def test_searchquery_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        SearchQuery(user_id="alice", text="the 24h-eval footgun")


async def test_empty_query_fails_loudly_in_query_modes(service):
    with pytest.raises(ValueError, match="query text is required"):
        await service.search(SearchQuery(user_id="alice", query="   "))


async def test_listing_modes_still_allow_empty_query(service):
    from jaswolf.models import SearchMode

    await _seed(service)
    hits = await service.search(
        SearchQuery(user_id="alice", query="", mode=SearchMode.RECENCY, record_access=False)
    )
    assert hits


async def test_search_hits_carry_raw_similarity(service):
    await _seed(service)
    hits = await service.search(
        SearchQuery(user_id="alice", query="HarborFront route", record_access=False)
    )
    assert any(h.similarity is not None for h in hits)
    for h in hits:
        if h.similarity is not None:
            assert -1.0 <= h.similarity <= 1.0


# ---- probe loading ---------------------------------------------------------------


def test_load_probes_rejects_unknown_fields_and_empty_query(tmp_path):
    with pytest.raises(ValueError, match="unknown fields"):
        load_probes(_write_probes(tmp_path, [{"id": "x", "query": "q", "querry_typo": 1}]))
    with pytest.raises(ValueError, match="empty query"):
        load_probes(_write_probes(tmp_path, [{"id": "x", "query": "  "}]))


# ---- end-to-end eval -------------------------------------------------------------


async def test_eval_passes_and_verdict_gates(settings, service, tmp_path):
    await _seed(service)
    await service.close()  # run_eval opens its own service on the same DB

    probes_path = _write_probes(tmp_path, [
        {"id": "name-warning", "kind": "search", "query": "what should I call Alice",
         "expect_any": ["mr naik"], "high_salience": True},
        {"id": "route", "kind": "context", "query": "which route home",
         "expect_any": ["harborfront"]},
    ])
    report = await run_eval(settings, load_probes(probes_path), user_id="alice")
    assert report["golden_probe_score"] == 1.0
    assert report["irrelevant_injection_count"] == 0
    assert report["verdict"] == "GO_PILOT"
    assert report["embedding_fingerprint"] == "hashing-384"
    assert report["sqlite_quick_check"] == "ok"
    assert report["cold_latency_ms"] is not None
    assert report["warm_repeats"] == 5
    assert report["latency_samples"] == {"search": 5, "context": 5}
    assert report["system_load"]["start"]["cpu_count"] is not None
    per_probe = {probe["id"]: probe for probe in report["per_probe"]}
    assert per_probe["name-warning"]["samples"] == 5
    assert per_probe["route"]["samples"] == 5


async def test_eval_forbidden_content_forces_no_go(settings, service, tmp_path):
    await _seed(service)
    await service.close()

    probes_path = _write_probes(tmp_path, [
        {"id": "stale-route", "kind": "search", "query": "route home HarborFront",
         "forbid": ["harborfront"]},  # the "stale" fact IS present -> must flag
    ])
    report = await run_eval(settings, load_probes(probes_path), user_id="alice")
    assert report["verdict"] == "NO_GO"
    assert any("forbidden" in r for r in report["no_go_reasons"])


async def test_eval_high_salience_failure_continues_shadow(settings, service, tmp_path):
    await _seed(service)
    await service.close()

    probes_path = _write_probes(tmp_path, [
        {"id": "missing-fact", "kind": "search", "query": "favourite programming font",
         "expect_any": ["definitely-not-stored-keyword"], "high_salience": True},
        {"id": "route", "kind": "search", "query": "route home HarborFront",
         "expect_any": ["harborfront"]},
    ])
    report = await run_eval(settings, load_probes(probes_path), user_id="alice")
    assert report["verdict"] == "CONTINUE_SHADOW"
    assert report["high_salience_failures"] == ["missing-fact"]
