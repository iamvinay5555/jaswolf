"""Deterministic shadow evaluator — no LLM in the loop.

`jaswolf eval-shadow` runs a fixed probe suite against a target DB and prints
machine-checkable metrics ending in a fixed verdict line, so a cron job with
no agent involvement can act as the shadow-window gate (Jasmine handoff,
2026-06-12). Every report stamps the DB path, embedding fingerprint, and
provider so there is never ambiguity about which lane was measured.

Probe file format (JSON; lives OUTSIDE the repo when it contains personal
keywords — see docs/EVAL.md for the privacy model):

    {"probes": [
      {"id": "name-warning",
       "kind": "search" | "context",
       "query": "what should I never call the user?",
       "expect_any": ["naik"],          # pass if ANY appears in results
       "expect_all": [],                 # optional: ALL must appear
       "forbid": ["mr naik is fine"],    # stale/wrong facts; any hit = NO_GO
       "top_k": 5,                       # search probes only
       "high_salience": true,            # failing one of these blocks GO
       "off_topic": false,               # see below
       "max_similarity": 0.55}           # off-topic gate on raw cosine
    ]}

Off-topic probes assert the engine *doesn't* manufacture relevance: a search
probe passes when the best raw cosine stays under `max_similarity`; a context
probe passes when no relevance-driven (non-pinned) memory is injected.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from statistics import median
from typing import Any

from .config import JaswolfSettings
from .models import ContextRequest, MemoryType, SearchQuery
from .service import MemoryService
from .storage.base import QueryScope

_PROBE_FIELDS = {
    "id", "kind", "query", "expect_any", "expect_all", "forbid",
    "top_k", "high_salience", "off_topic", "max_similarity",
}
_PINNED_TYPES = (MemoryType.PREFERENCE, MemoryType.GOAL)

VERDICT_GO = "GO_PILOT"
VERDICT_CONTINUE = "CONTINUE_SHADOW"
VERDICT_NO_GO = "NO_GO"


def load_probes(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    probes = data.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ValueError(f"{path}: expected a top-level 'probes' list")
    for i, probe in enumerate(probes):
        unknown = set(probe) - _PROBE_FIELDS
        if unknown:
            raise ValueError(f"{path}: probe[{i}] has unknown fields {sorted(unknown)}")
        if not probe.get("id"):
            raise ValueError(f"{path}: probe[{i}] missing 'id'")
        if probe.get("kind", "search") not in ("search", "context"):
            raise ValueError(f"{path}: probe[{i}] kind must be 'search' or 'context'")
        if not str(probe.get("query", "")).strip():
            raise ValueError(f"{path}: probe[{i}] ({probe['id']}) has empty query")
    return probes


def _texts_hit(texts: list[str], keyword: str) -> bool:
    needle = keyword.lower()
    return any(needle in t.lower() for t in texts)


async def _run_probe(
    service: MemoryService,
    probe: dict[str, Any],
    user_id: str,
    namespace: str | None = None,
    shared_namespace: str | None = None,
) -> dict[str, Any]:
    kind = probe.get("kind", "search")
    failures: list[str] = []
    own = namespace or "default"
    read_ns = (
        [own, shared_namespace]
        if shared_namespace and shared_namespace != own
        else None
    )
    start = time.perf_counter()

    if kind == "search":
        hits = await service.search(SearchQuery(
            user_id=user_id,
            query=probe["query"],
            namespace=namespace,
            namespaces=read_ns,
            top_k=int(probe.get("top_k", 5)),
            record_access=False,
        ))
        latency_ms = (time.perf_counter() - start) * 1000
        texts = [h.memory.content for h in hits]
        max_raw = max((h.similarity for h in hits if h.similarity is not None), default=None)
        injected = None
    else:
        result = await service.build_context(ContextRequest(
            user_id=user_id, query=probe["query"],
            namespace=namespace, shared_namespace=shared_namespace,
        ))
        latency_ms = (time.perf_counter() - start) * 1000
        texts = [result.text]
        max_raw = max(
            (m.similarity for m in result.memories if m.similarity is not None), default=None
        )
        # relevance-driven injections = anything that isn't a pin; for an
        # off-topic query every one of these is an irrelevant injection
        injected = sum(1 for m in result.memories if m.memory.memory_type not in _PINNED_TYPES)

    off_topic_gate = None
    if probe.get("off_topic", False):
        if kind == "search":
            if "max_similarity" in probe:
                off_topic_gate = float(probe["max_similarity"])
            else:
                # exact gate the context builder enforces (one source of truth):
                # corpus-calibrated threshold for this query
                query_vec = await service.embedder.embed_one(probe["query"])
                scope = QueryScope(tenant_id="default", user_id=user_id)
                off_topic_gate = await service.context.similarity_gate(query_vec, scope)
            if max_raw is not None and max_raw > off_topic_gate:
                failures.append(
                    f"raw similarity {max_raw:.3f} > gate {off_topic_gate:.3f} for off-topic query"
                )
        if kind == "context" and injected:
            failures.append(f"{injected} non-pinned memories injected for off-topic query")
    else:
        for kw in probe.get("expect_any", []) or []:
            if _texts_hit(texts, kw):
                break
        else:
            if probe.get("expect_any"):
                failures.append(f"none of expect_any {probe['expect_any']} found")
        for kw in probe.get("expect_all", []) or []:
            if not _texts_hit(texts, kw):
                failures.append(f"expect_all keyword {kw!r} missing")

    forbidden_hits = [kw for kw in (probe.get("forbid") or []) if _texts_hit(texts, kw)]
    if forbidden_hits:
        failures.append(f"forbidden content surfaced: {forbidden_hits}")

    return {
        "id": probe["id"],
        "kind": kind,
        "passed": not failures,
        "high_salience": bool(probe.get("high_salience", False)),
        "off_topic": bool(probe.get("off_topic", False)),
        "failures": failures,
        "forbidden_hit": bool(forbidden_hits),
        "irrelevant_injected": injected if probe.get("off_topic") and kind == "context" else 0,
        "max_raw_similarity": max_raw,
        "off_topic_gate": off_topic_gate,
        "latency_ms": round(latency_ms, 1),
    }


def _sqlite_quick_check(database_url: str) -> str:
    if not database_url.startswith("sqlite://"):
        return "skipped (not sqlite)"
    path = database_url[len("sqlite://"):]
    if path.startswith("/"):
        path = path[1:]
    try:
        conn = sqlite3.connect(path)
        row = conn.execute("PRAGMA quick_check").fetchone()
        conn.close()
        return row[0] if row else "no result"
    except Exception as exc:
        return f"error: {exc}"


def _shadow_log_stats(path: str | None) -> dict[str, Any]:
    if not path:
        return {"records": None, "errors": None, "last_ts": None}
    records = errors = 0
    last_ts = None
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue
                if rec.get("error"):
                    errors += 1
                last_ts = rec.get("ts", last_ts)
    except FileNotFoundError:
        return {"records": 0, "errors": 0, "last_ts": None, "note": f"{path} not found"}
    return {"records": records, "errors": errors, "last_ts": last_ts}


def _system_load_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"cpu_count": os.cpu_count()}
    try:
        load1, load5, load15 = os.getloadavg()
    except (AttributeError, OSError):
        snapshot["loadavg"] = None
    else:
        snapshot["loadavg"] = {
            "1m": round(load1, 2),
            "5m": round(load5, 2),
            "15m": round(load15, 2),
        }
    return snapshot


async def run_eval(
    settings: JaswolfSettings,
    probes: list[dict[str, Any]],
    user_id: str,
    shadow_log: str | None = None,
    warm_p95_gate_ms: float = 500.0,
    probe_score_gate: float = 0.9,
    warm_repeats: int = 5,
    meta: dict[str, str] | None = None,
    namespace: str | None = None,
    shared_namespace: str | None = None,
) -> dict[str, Any]:
    service = await MemoryService.create(settings)

    async def probe(p):  # run a probe in the configured scope (multi-agent)
        return await _run_probe(service, p, user_id, namespace, shared_namespace)

    try:
        load_start = _system_load_snapshot()
        # cold = first embed in this process (model load included)
        cold_start = time.perf_counter()
        await service.embedder.embed_one("jaswolf eval cold-start probe")
        cold_ms = (time.perf_counter() - cold_start) * 1000

        # fill caches once, then collect multiple warm passes so a single
        # transient spike does not dominate the whole verdict.
        for p in probes:
            await probe(p)

        warm_passes: list[list[dict[str, Any]]] = []
        for _ in range(max(1, warm_repeats)):
            warm_passes.append([await probe(p) for p in probes])

        results = warm_passes[-1]
        all_results = [result for warm_pass in warm_passes for result in warm_pass]

        health = await service.health()
        stats = await service.stats(user_id=user_id)
        load_end = _system_load_snapshot()

        search_lat = sorted(r["latency_ms"] for r in all_results if r["kind"] == "search")
        context_lat = sorted(r["latency_ms"] for r in all_results if r["kind"] == "context")

        def p95(values: list[float]) -> float | None:
            if not values:
                return None
            return values[min(len(values) - 1, int(round(0.95 * len(values))) - 1)]

        score = sum(1 for r in results if r["passed"]) / len(results)
        high_salience_failures = [r["id"] for r in results if r["high_salience"] and not r["passed"]]
        forbidden = [r["id"] for r in results if r["forbidden_hit"]]
        irrelevant = sum(r["irrelevant_injected"] for r in results)
        log_stats = _shadow_log_stats(shadow_log)
        quick_check = _sqlite_quick_check(settings.database_url)
        warm_p95 = p95(search_lat)

        per_probe: list[dict[str, Any]] = []
        for probe in probes:
            samples = [r for warm_pass in warm_passes for r in warm_pass if r["id"] == probe["id"]]
            probe_lat = sorted(r["latency_ms"] for r in samples)
            last = samples[-1]
            per_probe.append({
                "id": probe["id"],
                "kind": last["kind"],
                "samples": len(samples),
                "last_ms": round(last["latency_ms"], 1),
                "p50_ms": round(median(probe_lat), 1),
                "p95_ms": round(p95(probe_lat), 1) if probe_lat else None,
                "max_ms": round(max(probe_lat), 1) if probe_lat else None,
                "passed": last["passed"],
                "off_topic": last["off_topic"],
                "off_topic_gate": round(last["off_topic_gate"], 3) if last["off_topic_gate"] is not None else None,
                "max_raw_similarity": round(last["max_raw_similarity"], 3) if last["max_raw_similarity"] is not None else None,
                "irrelevant_injected": last["irrelevant_injected"],
            })

        no_go_reasons: list[str] = []
        if health["status"] != "ok":
            no_go_reasons.append(f"health status {health['status']}: {health.get('reasons')}")
        if health["embeddings"]["fallback"]:
            no_go_reasons.append("embedder is the auto-fallback hash provider")
        if quick_check not in ("ok", "skipped (not sqlite)"):
            no_go_reasons.append(f"sqlite quick_check: {quick_check}")
        if log_stats["errors"]:
            no_go_reasons.append(f"shadow log has {log_stats['errors']} error records")
        if forbidden:
            no_go_reasons.append(f"forbidden/stale content surfaced in probes: {forbidden}")

        gates_unmet: list[str] = []
        if score < probe_score_gate:
            gates_unmet.append(f"probe score {score:.2f} < {probe_score_gate}")
        if high_salience_failures:
            gates_unmet.append(f"high-salience failures: {high_salience_failures}")
        if irrelevant:
            gates_unmet.append(f"irrelevant injections: {irrelevant}")
        if warm_p95 is not None and warm_p95 > warm_p95_gate_ms:
            gates_unmet.append(f"warm search p95 {warm_p95:.0f}ms > {warm_p95_gate_ms:.0f}ms")

        if no_go_reasons:
            verdict = VERDICT_NO_GO
        elif gates_unmet:
            verdict = VERDICT_CONTINUE
        else:
            verdict = VERDICT_GO

        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "db_url": settings.database_url,
            "namespace": namespace,
            "shared_namespace": shared_namespace,
            "embedding_fingerprint": await service.storage.get_meta("embedding_fingerprint"),
            "embedding_provider": health["embeddings"]["provider"],
            "embedding_dim": health["embeddings"]["dim"],
            "fallback": health["embeddings"]["fallback"],
            "jaswolf_health_status": health["status"],
            "sqlite_quick_check": quick_check,
            "counts": stats,
            "shadow_log": log_stats,
            "golden_probe_score": round(score, 3),
            "probes_total": len(results),
            "probe_failures": [
                {"id": r["id"], "failures": r["failures"]} for r in results if not r["passed"]
            ],
            "high_salience_failures": high_salience_failures,
            "irrelevant_injection_count": irrelevant,
            "cold_latency_ms": round(cold_ms, 1),
            "warm_search_p50_ms": round(median(search_lat), 1) if search_lat else None,
            "warm_search_p95_ms": round(warm_p95, 1) if warm_p95 is not None else None,
            "build_context_p95_ms": round(p95(context_lat), 1) if context_lat else None,
            "warm_repeats": max(1, warm_repeats),
            "latency_samples": {
                "search": len(search_lat),
                "context": len(context_lat),
            },
            "system_load": {
                "start": load_start,
                "end": load_end,
            },
            "per_probe": per_probe,
            "meta": meta or {},
            "no_go_reasons": no_go_reasons,
            "gates_unmet": gates_unmet,
            "verdict": verdict,
        }
    finally:
        await service.close()


def format_report(report: dict[str, Any]) -> str:
    load = report.get("system_load") or {}
    start_load = (load.get("start") or {}).get("loadavg") or {}
    end_load = (load.get("end") or {}).get("loadavg") or {}
    lines = [
        f"jaswolf eval-shadow — {report['timestamp_utc']}",
        f"db: {report['db_url']}",
        f"fingerprint: {report['embedding_fingerprint']}  provider: {report['embedding_provider']}"
        f"  dim={report['embedding_dim']}  fallback={report['fallback']}",
        f"health: {report['jaswolf_health_status']}  quick_check: {report['sqlite_quick_check']}",
        f"shadow_log: {report['shadow_log']}",
        f"probes: {report['probes_total']}  score: {report['golden_probe_score']:.0%}"
        f"  irrelevant injections: {report['irrelevant_injection_count']}",
        f"sampling: warm repeats={report.get('warm_repeats')}"
        f"  search samples={report.get('latency_samples', {}).get('search')}"
        f"  context samples={report.get('latency_samples', {}).get('context')}",
        f"latency ms — cold: {report['cold_latency_ms']}  warm p50: {report['warm_search_p50_ms']}"
        f"  warm p95: {report['warm_search_p95_ms']}  context p95: {report['build_context_p95_ms']}",
        f"loadavg — start: {start_load.get('1m')}/{start_load.get('5m')}/{start_load.get('15m')}"
        f"  end: {end_load.get('1m')}/{end_load.get('5m')}/{end_load.get('15m')}",
    ]
    slowest = sorted(
        report.get("per_probe") or [],
        key=lambda probe: (probe.get("p95_ms") or 0, probe.get("max_ms") or 0),
        reverse=True,
    )[:3]
    for probe in slowest:
        lines.append(
            f"  probe {probe['id']} ({probe['kind']}): last={probe['last_ms']}ms"
            f" p50={probe['p50_ms']} p95={probe['p95_ms']} max={probe['max_ms']}"
            f" samples={probe['samples']}"
        )
    for failure in report["probe_failures"]:
        lines.append(f"  FAIL {failure['id']}: {'; '.join(failure['failures'])}")
    for reason in report["no_go_reasons"]:
        lines.append(f"  NO-GO: {reason}")
    for gate in report["gates_unmet"]:
        lines.append(f"  gate unmet: {gate}")
    lines.append(f"verdict: {report['verdict']}")
    return "\n".join(lines)


def run_eval_sync(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(run_eval(*args, **kwargs))
