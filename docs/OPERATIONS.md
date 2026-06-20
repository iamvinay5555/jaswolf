# Operating JASWOLF

## Configuration reference

All settings are env vars prefixed `JASWOLF_` (or a local `.env`). The
essentials:

| Variable | Default | Notes |
| --- | --- | --- |
| `JASWOLF_DATABASE_URL` | `sqlite:///./jaswolf.db` | `postgresql://…` for prod |
| `JASWOLF_REDIS_URL` | *(unset)* | unset → in-process LRU cache |
| `JASWOLF_API_KEYS` | *(empty)* | `key:tenant,key2:tenant2` — required for the HTTP API |
| `JASWOLF_DEV_OPEN_MODE` | `false` | explicit opt-in to run the API without auth (dev only) |
| `JASWOLF_RATE_LIMIT_PER_MINUTE` | `600` | per key; `0` disables |
| `JASWOLF_EMBEDDING_PROVIDER` | `auto` | `local` / `openai` / `hash` |
| `JASWOLF_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | sentence-transformers id |
| `JASWOLF_EMBEDDING_DIM` | `384` | fixed at first migration |
| `JASWOLF_OPENAI_BASE_URL` / `_API_KEY` / `_EMBEDDING_MODEL` | — | OpenAI-compatible embeddings |
| `JASWOLF_LLM_BASE_URL` / `_API_KEY` / `_MODEL` | — | extraction/merge LLM (optional) |
| `JASWOLF_EXTRACTION_STRATEGY` | `rules` | `rules` / `llm` / `hybrid` |
| `JASWOLF_DEDUP_THRESHOLD` | `0.95` | write-path near-dup reinforcement |
| `JASWOLF_CONSOLIDATION_THRESHOLD` | `0.88` | merge clustering |
| `JASWOLF_MIN_RELEVANCE` | `0.1` | vector candidate floor |
| `JASWOLF_SUPERSESSION_ENABLED` | `true` | corrections archive the contradicted memory |
| `JASWOLF_SUPERSESSION_THRESHOLD` | `0.5` | min similarity for a correction to supersede |
| `JASWOLF_PIN_MIN_IMPORTANCE` | `0.7` | identity-grade pinning gate (context builder) |
| `JASWOLF_PIN_MIN_CONFIDENCE` | `0.8` | second pinning gate; single-shot extractions don't pin |
| `JASWOLF_WORKING_TTL_HOURS` | `24` | working-memory lifetime |
| `JASWOLF_ACTIVE_TO_WARM_DAYS` / `WARM_TO_COLD_DAYS` / `COLD_TO_ARCHIVED_DAYS` | `14/60/180` | lifecycle ladder |
| `JASWOLF_SWEEP_INTERVAL_SECONDS` | `300` | background sweeper period |
| `JASWOLF_CONTEXT_TOKEN_BUDGET` | `1500` | default context size |
| `JASWOLF_CORS_ORIGINS` | *(empty = disabled)* | comma-separated explicit origins |
| `JASWOLF_LOG_LEVEL` | `INFO` | |

Scoring weights: `JASWOLF_WEIGHT_IMPORTANCE/RELEVANCE/RECENCY/FREQUENCY`
(0.4/0.3/0.2/0.1). Context section shares:
`JASWOLF_CONTEXT_SHARE_PREFERENCE/GOAL/SEMANTIC/PROCEDURAL/EPISODIC/RELATIONSHIP`.

## Monitoring

`GET /metrics` (Prometheus). Key series:

| Metric | Watch for |
| --- | --- |
| `jaswolf_search_latency_seconds` | p95 < 0.1 (target) |
| `jaswolf_context_latency_seconds` | p95 < 0.15 |
| `jaswolf_request_latency_seconds{route=…}` | per-endpoint health |
| `jaswolf_memories_created_total` vs `jaswolf_memories_reinforced_total` | reinforcement ratio rising = healthy dedup |
| `jaswolf_embed_cache_total{result=…}` | hit ratio (also in `/health`) |
| `jaswolf_requests_total{status=…}` | 401/429/500 rates |

`GET /health` returns storage/embedder/cache status — wire it to your
uptime checker. The compose file ships Prometheus + Grafana under
`--profile monitoring`.

## Maintenance

```bash
jaswolf sweep                          # one lifecycle pass (also runs automatically)
jaswolf consolidate --user-id alice    # merge duplicates (add --dry-run first)
jaswolf stats [--user-id alice]        # counts by state/type
```

Suggested cron on the VPS:

```cron
0 4 * * * /opt/jaswolf/.venv/bin/jaswolf consolidate --user-id alice >> /var/log/jas0-consolidate.log 2>&1
```

## Backups & recovery

A long-lived memory store is only as durable as its backups — one bad disk
or stray `rm` otherwise loses everything. The `memories` table is the source
of truth (versions/access logs reconstruct it only partially), so back up the
whole DB.

**SQLite — use `jaswolf backup`, not `cp`.** It uses SQLite's online backup API,
so the snapshot is consistent even while the server/MCP process is running and
writing (a plain `cp` can capture a torn DB+WAL):

```bash
jaswolf backup --out /backups/jaswolf.db            # one consistent snapshot
jaswolf backup --keep 7                          # default-named, rotate to 7
# nightly cron (server can stay up):
0 3 * * *  cd /home/jaswolf/.hermes && JASWOLF_DATABASE_URL=sqlite:////.../memory.db \
           /path/to/jaswolf backup --out /backups/jaswolf_backup_$(date +\%Y\%m\%d).db --keep 14
```

**Restore** validates the snapshot (integrity + embedding fingerprint) before
overwriting, and is sqlite-only. **Stop the server/MCP process first** so the
file isn't open:

```bash
jaswolf restore --from /backups/jaswolf_backup_20260613.db        # dry run: shows snapshot info
jaswolf restore --from /backups/jaswolf_backup_20260613.db --yes  # actually overwrite
```

If the snapshot's `embedding_fingerprint` differs from your configured model,
the fingerprint guard will flag it as degraded on next start (restore won't
silently mix vector spaces).

**Postgres**: `pg_dump $JASWOLF_DATABASE_URL` (embeddings are ordinary columns;
compose volume `jaswolf_pgdata`). `jaswolf backup` is sqlite-only by design.

**Integrity is now in the health signal.** `health()` / `memory_health` (MCP)
/ `jaswolf diagnose` report `storage.integrity` (SQLite `quick_check`); a
non-`ok` value degrades health, so corruption surfaces as a rollback trigger
instead of silently serving a damaged DB.

## Scaling path

1. Single VPS, embedded provider, SQLite — zero ops.
2. Same VPS, `docker compose up`: Postgres + Redis, several agent processes
   share the API.
3. Bigger: scale API replicas horizontally (stateless), Postgres holds
   state; pgvector HNSW handles 10M+ rows. Tune
   `hnsw.ef_search` for recall/latency trade-offs at large scale.

## Upgrades

Migrations run automatically at startup and are tracked in
`schema_migrations`; they're append-only SQL files in
`src/jaswolf/storage/migrations/`. Read new ones before deploying to prod.
