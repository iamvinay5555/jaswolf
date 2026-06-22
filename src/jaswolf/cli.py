"""JASWOLF command line: serve the API, run maintenance, inspect stats.

    jaswolf serve --host 0.0.0.0 --port 8400
    jaswolf sweep
    jaswolf consolidate --user-id alice
    jaswolf stats [--user-id alice]
    jaswolf diagnose [--user-id alice]   # paste-ready bug-report diagnostics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys

from .config import JaswolfSettings
from .models import ContextRequest, SearchQuery
from .service import MemoryService


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "jaswolf.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=JaswolfSettings().log_level.lower(),
    )


async def _with_service(fn) -> None:
    service = await MemoryService.create()
    try:
        await fn(service)
    finally:
        await service.close()


def _sweep(args: argparse.Namespace) -> None:
    async def run(service: MemoryService) -> None:
        report = await service.sweep()
        print(report.model_dump_json(indent=2))

    asyncio.run(_with_service(run))


def _consolidate(args: argparse.Namespace) -> None:
    async def run(service: MemoryService) -> None:
        report = await service.consolidate(
            user_id=args.user_id, namespace=args.namespace, dry_run=args.dry_run
        )
        print(report.model_dump_json(indent=2))

    asyncio.run(_with_service(run))


def _stats(args: argparse.Namespace) -> None:
    async def run(service: MemoryService) -> None:
        print(json.dumps(await service.stats(user_id=args.user_id), indent=2))

    asyncio.run(_with_service(run))


def _redact(value: str | None) -> str:
    """Mask credentials in connection URLs; pass everything else through."""
    if not value:
        return "(unset)"
    return re.sub(r"://([^/@]*):[^@]+@", r"://\1:***@", value)


def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    """Best-effort check that something is listening (e.g. a running jaswolf serve)."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _scope_label(args: argparse.Namespace) -> str:
    """' ns=<own>[+<shared>]' suffix for the probe line, or '' when unscoped."""
    ns = getattr(args, "namespace", None)
    if not ns:
        return ""
    shared = getattr(args, "shared_namespace", None)
    return f" ns={ns}+{shared}" if shared else f" ns={ns}"


def _diagnose_remote(args: argparse.Namespace) -> None:
    """Diagnose a RUNNING jaswolf service over REST — the exact path Hermes uses.

    This avoids spinning up a separate embedded provider (wrong DB, cold model
    load, HuggingFace metadata calls) and reports the engine's own warm latency.
    """
    import os
    import time

    from .sdk.client import JaswolfClient, JaswolfError

    api_key = args.api_key or os.environ.get("JASWOLF_API_KEY")
    try:
        with JaswolfClient(args.api_url, api_key=api_key, timeout=args.timeout) as client:
            health = client.health()
            stats = client.stats(user_id=args.user_id)
            emb = health.get("embeddings", {})
            store = health.get("storage", {})
            lines = [
                "## JasWolf diagnostic report (remote)",
                f"- target: {args.api_url} · status: {health.get('status', '?')}",
                f"- storage: {store.get('backend')} (ok={store.get('ok')}"
                f" integrity={store.get('integrity', '?')})",
                f"- embeddings: {emb.get('provider')} dim={emb.get('dim')}"
                f" fallback={emb.get('fallback')}"
                f" · cache {emb.get('cache_hits', '?')}h/{emb.get('cache_misses', '?')}m",
                f"- memories: total={stats['total']} by_state={stats['by_state']}"
                f" by_type={stats['by_type']}",
            ]
            if args.user_id:
                search_kw: dict[str, object] = {"record_access": False}
                ctx_kw: dict[str, object] = {}
                if args.namespace:  # mirror a bot's production read surface
                    search_kw["namespace"] = args.namespace
                    ctx_kw["namespace"] = args.namespace
                if args.shared_namespace:
                    ctx_kw["shared_namespace"] = args.shared_namespace
                t0 = time.perf_counter()
                hits = client.search(
                    user_id=args.user_id, query="diagnostic probe", **search_kw
                )
                rtt_search = (time.perf_counter() - t0) * 1000
                ctx = client.build_context(
                    user_id=args.user_id, query="diagnostic probe", **ctx_kw
                )
                # server-side latency is the true warm-path number (no network/cold load)
                lines.append(
                    f"- live probe (user={args.user_id}{_scope_label(args)}): search"
                    f" {hits.get('latency_ms', '?')}ms-engine/{rtt_search:.0f}ms-rtt"
                    f"/{hits.get('count', 0)} hits · context"
                    f" {ctx.get('latency_ms', '?')}ms-engine/{ctx.get('token_estimate', 0)} tokens"
                )
            print("\n".join(lines))
    except JaswolfError as exc:
        print(f"remote diagnose failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # connection refused, DNS, timeout
        print(f"remote diagnose unreachable at {args.api_url}: {exc}", file=sys.stderr)
        sys.exit(1)


def _diagnose(args: argparse.Namespace) -> None:
    if args.api_url:
        _diagnose_remote(args)
        return

    async def run(service: MemoryService) -> None:
        import platform
        import time

        from . import __version__

        health = await service.health()
        stats = await service.stats(user_id=args.user_id)
        s = service.settings
        lines = [
            "## JasWolf diagnostic report",
            f"- jaswolf {__version__} · python {platform.python_version()} · {platform.platform()}",
            f"- storage: {health['storage']['backend']} (ok={health['storage']['ok']}"
            f" integrity={health['storage'].get('integrity', '?')}) · url: {_redact(s.database_url)}",
            f"- embeddings: {health['embeddings']['provider']} dim={health['embeddings']['dim']}"
            f" · cache {health['embeddings']['cache_hits']}h/{health['embeddings']['cache_misses']}m",
            f"- cache: {health['cache']['backend']} · redis: {_redact(s.redis_url)}",
            f"- extraction: {s.extraction_strategy} · llm: {_redact(s.llm_base_url)}",
            f"- thresholds: dedup={s.dedup_threshold} consolidation={s.consolidation_threshold}"
            f" min_relevance={s.min_relevance}",
            f"- weights: importance={s.weight_importance} relevance={s.weight_relevance}"
            f" recency={s.weight_recency} frequency={s.weight_frequency}",
            f"- memories: total={stats['total']} by_state={stats['by_state']} by_type={stats['by_type']}",
        ]

        # False-empty guard: a repo-checkout diagnose silently inspects the local
        # default DB, not the live service. If that DB is the default AND empty
        # while a server is listening on :8400, the operator is almost certainly
        # reading the WRONG DB (2026-06-19 audit reported total=0 + a 125s probe).
        default_db = JaswolfSettings.model_fields["database_url"].default
        if s.database_url == default_db:
            warn = ["", "⚠ Using the DEFAULT local DB (sqlite:///./jaswolf.db), not a running service."]
            if stats["total"] == 0:
                warn.append("  This DB is EMPTY (total=0) — likely the WRONG DB if your data is live.")
            if _port_open("127.0.0.1", 8400):
                warn.append("  A service IS listening on 127.0.0.1:8400 — diagnose IT instead:")
                warn.append("    jaswolf diagnose --api-url http://127.0.0.1:8400 --user-id <uid>")
            warn.append("  Or target the live file: JASWOLF_DATABASE_URL=sqlite:////abs/path.db jaswolf diagnose ...")
            lines[1:1] = warn  # surface right under the title

        if args.user_id:
            sq_kw: dict[str, object] = {}
            cr_kw: dict[str, object] = {}
            if args.namespace:  # mirror a bot's production read surface
                sq_kw["namespace"] = args.namespace
                cr_kw["namespace"] = args.namespace
            if args.shared_namespace:
                cr_kw["shared_namespace"] = args.shared_namespace
            t0 = time.perf_counter()
            hits = await service.search(
                SearchQuery(
                    user_id=args.user_id, query="diagnostic probe", record_access=False, **sq_kw
                )
            )
            search_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            ctx = await service.build_context(
                ContextRequest(user_id=args.user_id, query="diagnostic probe", **cr_kw)
            )
            context_ms = (time.perf_counter() - t0) * 1000
            lines.append(
                f"- live probe (user={args.user_id}{_scope_label(args)}): "
                f"search {search_ms:.1f}ms/{len(hits)} hits"
                f" · context {context_ms:.1f}ms/{ctx.token_estimate} tokens"
            )
        print("\n".join(lines))

    asyncio.run(_with_service(run))


def _eval_shadow(args: argparse.Namespace) -> None:
    from .evals import format_report, load_probes, run_eval

    overrides: dict[str, object] = {"embedding_prewarm": False}  # cold timing is part of the report
    if args.db:
        overrides["database_url"] = args.db
    if args.embedding_provider:
        overrides["embedding_provider"] = args.embedding_provider
    if args.embedding_model:
        overrides["embedding_model"] = args.embedding_model
    settings = JaswolfSettings(**overrides)

    meta = {}
    for item in args.meta or []:
        key, _, value = item.partition("=")
        meta[key] = value

    probes = load_probes(args.probes)
    report = asyncio.run(run_eval(
        settings,
        probes,
        user_id=args.user_id,
        shadow_log=args.shadow_log,
        warm_p95_gate_ms=args.warm_p95_gate,
        probe_score_gate=args.probe_score_gate,
        warm_repeats=args.warm_repeats,
        meta=meta,
        namespace=args.namespace,
        shared_namespace=args.shared_namespace,
    ))
    print(json.dumps(report, indent=2, default=str) if args.json else format_report(report))
    sys.exit(0 if report["verdict"] != "NO_GO" else 1)


def _cutover_preflight(args: argparse.Namespace) -> None:
    """Run the golden probes through ONE bot's exact scope (namespace + shared)
    and print a per-profile GO/NO-GO. This is the gate to run before pointing a
    bot at JASWOLF — it tests the real multi-agent read path, not a generic eval."""
    from .evals import format_report, load_probes, run_eval

    overrides: dict[str, object] = {"embedding_prewarm": False}
    if args.db:
        overrides["database_url"] = args.db
    if args.embedding_provider:
        overrides["embedding_provider"] = args.embedding_provider
    settings = JaswolfSettings(**overrides)
    report = asyncio.run(run_eval(
        settings,
        load_probes(args.probes),
        user_id=args.user_id,
        warm_p95_gate_ms=args.warm_p95_gate,
        warm_repeats=args.warm_repeats,
        meta={"profile": args.profile or args.namespace or "default"},
        namespace=args.namespace,
        shared_namespace=args.shared_namespace,
    ))
    label = args.profile or args.namespace or "default"
    print(f"=== cutover-preflight: profile={label} "
          f"(namespace={args.namespace} shared={args.shared_namespace}) ===")
    print(json.dumps(report, indent=2, default=str) if args.json else format_report(report))
    sys.exit(0 if report["verdict"] == "GO_PILOT" else 1)  # strict: GO only on full pass


def _sqlite_path(url: str) -> str | None:
    """Resolve the on-disk file for a sqlite URL, else None."""
    if url.startswith("sqlite://"):
        path = url[len("sqlite://"):]
        if path.startswith("/"):
            path = path[1:]
        return path or ":memory:"
    if "://" not in url:
        return url
    return None


def _backup(args: argparse.Namespace) -> None:
    import glob
    import os
    from datetime import datetime

    out = args.out or f"jaswolf_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

    async def run(service: MemoryService) -> None:
        info = await service.storage.backup(out)
        print(json.dumps(info, indent=2))

    asyncio.run(_with_service(run))

    if args.keep and args.out is None:  # rotate the default-named series
        backups = sorted(glob.glob(os.path.join(os.path.dirname(out) or ".", "jaswolf_backup_*.db")))
        for stale in backups[: max(0, len(backups) - args.keep)]:
            os.remove(stale)
            print(f"rotated out {stale}")


def _restore(args: argparse.Namespace) -> None:
    import os
    import shutil

    from .storage.sqlite_store import validate_sqlite_snapshot

    info = validate_sqlite_snapshot(args.source)
    print("snapshot:", json.dumps(info))
    if info["integrity"] != "ok":
        print("refusing: snapshot failed integrity check", file=sys.stderr)
        sys.exit(1)

    target = _sqlite_path(JaswolfSettings().database_url)
    if target is None or target == ":memory:":
        print("restore supports sqlite file DBs only (Postgres: use pg_restore)", file=sys.stderr)
        sys.exit(1)
    if not args.yes:
        print(f"would overwrite {target} with {args.source} — re-run with --yes to proceed")
        print("⚠ stop any running jaswolf process first (server/MCP) so the file isn't open")
        return
    shutil.copyfile(args.source, target)
    for sidecar in ("-wal", "-shm"):  # stale WAL would mask the restored data
        if os.path.exists(target + sidecar):
            os.remove(target + sidecar)
    print(f"restored {target} from {args.source}")


def _mcp_health(args: argparse.Namespace) -> None:
    """Probe a running MCP HTTP server's /healthz. Exit 0 if ok, 1 otherwise —
    for systemd watchdog timers, cron alerts, or a Hermes pre-start gate."""
    import httpx

    s = JaswolfSettings()
    url = args.url or f"http://{s.mcp_host}:{s.mcp_port}/healthz"
    try:
        resp = httpx.get(url, timeout=args.timeout)
    except Exception as exc:
        print(f"unreachable: {exc}", file=sys.stderr)
        sys.exit(1)
    print(resp.text)
    sys.exit(0 if resp.status_code == 200 else 1)


def _mcp(args: argparse.Namespace) -> None:
    from .mcp_server import run

    overrides: dict[str, object] = {}
    if args.db:
        overrides["database_url"] = args.db
    if args.user_id:
        overrides["mcp_user_id"] = args.user_id
    if args.host:
        overrides["mcp_host"] = args.host
    if args.port:
        overrides["mcp_port"] = args.port
    settings = JaswolfSettings(**overrides)
    transport = "streamable-http" if args.transport == "http" else "stdio"
    run(settings, transport=transport)


def _update(args: argparse.Namespace) -> None:
    from .models import MemoryState, MemoryType, MemoryUpdate

    async def run(service: MemoryService) -> None:
        patch = MemoryUpdate(
            memory_type=MemoryType(args.type) if args.type else None,
            importance=args.importance,
            confidence=args.confidence,
            state=MemoryState(args.state) if args.state else None,
        )
        memory = await service.update(args.id, patch)
        print(memory.model_dump_json(indent=2))

    asyncio.run(_with_service(run))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jaswolf", description="JASWOLF memory engine")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8400)
    serve.add_argument("--workers", type=int, default=1)
    serve.set_defaults(fn=_serve)

    sweep = sub.add_parser("sweep", help="run one lifecycle sweep")
    sweep.set_defaults(fn=_sweep)

    consolidate = sub.add_parser("consolidate", help="merge duplicate memories")
    consolidate.add_argument("--user-id", required=True)
    consolidate.add_argument("--namespace", default=None)
    consolidate.add_argument("--dry-run", action="store_true")
    consolidate.set_defaults(fn=_consolidate)

    stats = sub.add_parser("stats", help="memory counts by state/type")
    stats.add_argument("--user-id", default=None)
    stats.set_defaults(fn=_stats)

    diagnose = sub.add_parser("diagnose", help="print a paste-ready diagnostic report")
    diagnose.add_argument("--user-id", default=None, help="also run a live search/context probe")
    diagnose.add_argument(
        "--api-url",
        default=None,
        help="diagnose a RUNNING service via REST (e.g. http://127.0.0.1:8400) — the path Hermes "
        "uses; avoids the embedded cold-load/wrong-DB pitfalls",
    )
    diagnose.add_argument("--api-key", default=None, help="bearer token for --api-url (else $JASWOLF_API_KEY)")
    diagnose.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout for --api-url mode")
    diagnose.add_argument(
        "--namespace",
        default=None,
        help="scope the live probe to a bot's own namespace — reflects production recall "
        "instead of the whole corpus (e.g. when a large 'shadow' namespace dominates raw search)",
    )
    diagnose.add_argument(
        "--shared-namespace",
        default=None,
        help="also read this shared namespace in the context probe (e.g. shared)",
    )
    diagnose.set_defaults(fn=_diagnose)

    eval_shadow = sub.add_parser(
        "eval-shadow",
        help="deterministic golden-probe evaluation with a fixed verdict line (cron-safe, no LLM)",
    )
    eval_shadow.add_argument("--probes", required=True, help="JSON probe file (see docs/EVAL.md)")
    eval_shadow.add_argument("--user-id", required=True)
    eval_shadow.add_argument("--db", default=None, help="database URL override")
    eval_shadow.add_argument("--embedding-provider", default=None)
    eval_shadow.add_argument("--embedding-model", default=None)
    eval_shadow.add_argument("--shadow-log", default=None, help="shadow_log.jsonl to summarize")
    eval_shadow.add_argument("--warm-p95-gate", type=float, default=500.0)
    eval_shadow.add_argument("--probe-score-gate", type=float, default=0.9)
    eval_shadow.add_argument(
        "--warm-repeats",
        type=int,
        default=5,
        help="number of warm passes to measure after the initial cache-fill pass",
    )
    eval_shadow.add_argument("--meta", action="append", help="k=v pairs echoed into the report")
    eval_shadow.add_argument("--namespace", default=None, help="probe in this bot's namespace")
    eval_shadow.add_argument("--shared-namespace", default=None, help="also read this shared namespace")
    eval_shadow.add_argument("--json", action="store_true")
    eval_shadow.set_defaults(fn=_eval_shadow)

    preflight = sub.add_parser(
        "cutover-preflight",
        help="GO/NO-GO gate for one bot's exact scope (namespace+shared) before cutover",
    )
    preflight.add_argument("--probes", required=True, help="JSON probe file (see docs/EVAL.md)")
    preflight.add_argument("--user-id", required=True)
    preflight.add_argument("--namespace", required=True, help="the bot's own namespace, e.g. freya")
    preflight.add_argument("--shared-namespace", default="shared")
    preflight.add_argument("--profile", default=None, help="label for the report (e.g. freya)")
    preflight.add_argument("--db", default=None, help="database URL override")
    preflight.add_argument("--embedding-provider", default=None)
    preflight.add_argument("--warm-p95-gate", type=float, default=500.0)
    preflight.add_argument("--warm-repeats", type=int, default=5)
    preflight.add_argument("--json", action="store_true")
    preflight.set_defaults(fn=_cutover_preflight)

    backup = sub.add_parser("backup", help="consistent snapshot of the memory DB (online, safe)")
    backup.add_argument("--out", default=None, help="snapshot path (default: jaswolf_backup_<ts>.db)")
    backup.add_argument("--keep", type=int, default=None, help="rotate: keep N default-named backups")
    backup.set_defaults(fn=_backup)

    restore = sub.add_parser("restore", help="restore the DB from a snapshot (stop the server first)")
    restore.add_argument("--from", dest="source", required=True, help="snapshot file to restore")
    restore.add_argument("--yes", action="store_true", help="confirm overwrite of the live DB")
    restore.set_defaults(fn=_restore)

    mcp = sub.add_parser("mcp", help="serve JASWOLF as an MCP memory provider (for Hermes etc.)")
    mcp.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                     help="stdio (host-launched) or http (long-running streamable-HTTP)")
    mcp.add_argument("--db", default=None, help="database URL override")
    mcp.add_argument("--user-id", default=None, help="identity the server operates as")
    mcp.add_argument("--host", default=None, help="http transport bind host")
    mcp.add_argument("--port", type=int, default=None, help="http transport port")
    mcp.set_defaults(fn=_mcp)

    mcp_health = sub.add_parser(
        "mcp-health", help="probe a running MCP server's /healthz (exit 0=ok, 1=down/degraded)"
    )
    mcp_health.add_argument("--url", default=None, help="override (default: mcp_host:mcp_port/healthz)")
    mcp_health.add_argument("--timeout", type=float, default=5.0)
    mcp_health.set_defaults(fn=_mcp_health)

    update = sub.add_parser("update", help="retype/rescore a single memory by id")
    update.add_argument("--id", required=True)
    update.add_argument("--type", default=None, help="e.g. preference | goal | semantic")
    update.add_argument("--importance", type=float, default=None)
    update.add_argument("--confidence", type=float, default=None)
    update.add_argument("--state", default=None, help="e.g. active")
    update.set_defaults(fn=_update)

    args = parser.parse_args(argv)
    try:
        args.fn(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
