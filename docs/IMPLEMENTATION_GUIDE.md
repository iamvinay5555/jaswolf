# JASWOLF Implementation Guide (v0.7.0)

The single "read this one" reference for what JASWOLF is, everything built into
it, and how to run/deploy each capability. Deeper dives are linked per
section. For the live Hermes cutover, follow
[the MCP runbook](claude_handoffs/2026-06-13-claude-mcp-cutover-runbook.md).

---

## 1. What JASWOLF is now

A self-hosted, local-first long-term memory engine for the Hermes agent —
fast (~65 ms warm search vs mem0's ~1 s), deterministic, and operationally
honest. Library-first: `MemoryService` is the composition root; the HTTP API,
the SDK client, the `JaswolfMemoryProvider`, and the **MCP server** are thin
layers over it. SQLite (dev/single-node) or Postgres+pgvector (scale).

**Status:** v0.7.0, 130 tests pass (`PYTHONPATH=src .venv/bin/pytest`),
`ruff check` clean. Engine and protocol surface are cutover-ready; the
remaining work is Hermes-side MCP wiring + the Mem0→JASWOLF migration.

---

## 2. Architecture at a glance

```
            Hermes (agent)
                │  MCP (stdio / streamable-HTTP)
        ┌───────▼─────────┐
        │  jaswolf mcp server │   ← src/jaswolf/mcp_server.py (7 tools)
        └───────┬─────────┘
        ┌───────▼─────────┐
        │ JaswolfMemoryProvider│  ← providers/hermes.py (embedded | remote)
        └───────┬─────────┘
        ┌───────▼─────────────────────────────────┐
        │ MemoryService (composition root)         │
        │  write path · retrieval · context build  │
        │  extraction · scoring · consolidation    │
        └───────┬─────────────────────────────────┘
        ┌───────▼─────────┐   embeddings: local bge-small | OpenAI-compat | hash
        │ storage backend  │   SQLite (numpy + FTS5) | Postgres (pgvector HNSW)
        └─────────────────┘
```

---

## 3. Capabilities built (by theme)

### A. Trustworthy write path
- **Durability gate** — reactions/chatter/short-horizon plans don't become
  durable memories; only lasting facts are stored. Keeps the store small.
- **Dedup + reinforcement** — restating a fact reinforces one row (confidence
  +, importance max) instead of inserting duplicates.
- **Supersession** (write-time, marker-gated) — "actually/now/no longer …"
  archives the contradicted fact with a SUPERSEDES link + version record.
  Conservative on purpose: unmarked contradictions stay additive (see §3C).
- **Revival** (v0.2.1) — restating a fact whose memory was archived (idle
  ladder / TTL / supersession) brings it back to active with a fresh TTL.

### B. Retrieval quality
- **Hybrid search** — vector (cosine) + FTS keyword fused with RRF, then
  scored by importance·relevance·recency·frequency.
- **Context-boundary gate** (v0.4.x) — search *ranks*; the prompt builder
  *injects*, which needs a higher bar. A non-pinned vector candidate enters
  the prompt only if its raw cosine clears a per-query threshold
  **calibrated against the corpus** (`mean + context_noise_z·std` of the
  query's similarity to a background sample). Anisotropy-invariant — bge-small
  scores arbitrary English ~0.6 where hash scores ~0.0. (`calibration.py`)
- **Discriminative keyword evidence** (v0.4.1) — a keyword hit exempts a
  memory from the gate only if the term is non-stopword, long enough, and
  rare in the corpus (IDF-style cut). Generic words like "week" no longer
  smuggle unrelated memories into the prompt. (`keywords.py`)
- **Identity pinning** — high-importance, high-confidence preferences/goals
  are injected regardless of query relevance; single-shot facts aren't pinned
  until reinforced.

### C. Temporal current-state resolution (v0.6.0, mem0-inspired)
Write-time supersession only fires on a correction marker, so *unmarked*
contradictions ("office is X" then later "office is Y", no marker) leave both
active. At **read time**, when retrieved memories of the same type fill the
same **singleton** slot ("User's `<slot>` is `<value>`", slot ∈ a curated
single-valued set: office/job/address/city/role/phone/…) with *different*
values, only the freshest (`updated_at`) is injected. Never mutates storage;
**multi-valued** relations (friend/pet/hobby) are deliberately excluded and
stay additive; raw `search()` still returns history. (`temporal.py`, toggle
`temporal_resolution`). See §5 for the mem0 comparison.

### D. Operational safety
- **Embedding fingerprint guard** (v0.2.2) — each DB is stamped with its
  embedder at first open; opening it with a different model/dim degrades
  health loudly instead of silently mixing incompatible vector spaces.
- **Prewarm** — `JASWOLF_EMBEDDING_PREWARM=true` loads the model at startup so no
  live turn pays the seconds-long CPU cold start. The MCP server prewarms
  automatically.
- **Health + integrity** (v0.7.0) — `health()` reports storage reachability,
  embedding provider/fallback, and `quick_check` **integrity**; any problem
  degrades status. Surfaced via MCP `memory_health` and `jaswolf diagnose`.

### E. Deterministic evaluation
- **`jaswolf eval-shadow`** — cron-safe, no LLM in the loop. Runs a golden-probe
  suite, multi-pass warm latency sampling, system-load snapshot, and prints a
  fixed verdict line `GO_PILOT | CONTINUE_SHADOW | NO_GO` with an exit code.
  Probe file is private (personal keywords live off-repo). Full spec:
  [EVAL.md](EVAL.md).

### F. Durability (v0.7.0)
- **`jaswolf backup`** — consistent online snapshot (see §6).
- **`jaswolf restore`** — validated restore (integrity + fingerprint).

---

## 4. The MCP memory server (the cutover path)

Full doc: [MCP.md](MCP.md). This is how Hermes uses JASWOLF as its memory
provider — through the stable MCP tool contract, not Hermes internals.

### Install & run
```bash
pip install "jaswolf[mcp,local-embeddings] @ git+https://github.com/iamvinay5555/jaswolf.git"
# CPU-only VPS: install CPU torch FIRST (see INSTALL.md) to avoid the CUDA wheel

# long-lived, prewarmed; pick one transport:
JASWOLF_DATABASE_URL=sqlite:////home/jaswolf/.hermes/jaswolf_shadow/data/jaswolf_shadow_bge_small.db \
JASWOLF_EMBEDDING_PROVIDER=local JASWOLF_MCP_USER_ID=<USER_ID> \
jaswolf mcp                              # stdio (host launches it)
jaswolf mcp --transport http --port 8765 # streamable-HTTP (persistent, stays warm)
```
Prefer **HTTP** if Hermes would otherwise spawn a stdio subprocess per
session (per-session stdio reloads the model each time).

### The 7 tools
| Tool | Purpose | Style |
| --- | --- | --- |
| `build_memory_context(query?, recent_messages?, token_budget?)` | memory block for the system prompt | host-driven |
| `record_conversation(messages)` | post-turn extract/store (gated, supersession-aware) | host-driven |
| `recall(query, limit?)` | remembered statements, best first | model-driven |
| `remember(content, memory_type?, importance?)` | explicit durable store | model-driven |
| `search_memory(query, top_k?, mode?)` | scored search | model-driven |
| `forget(memory_id, hard?)` | delete | model-driven |
| `memory_health()` | provider/fingerprint/fallback/integrity/status | ops |

### ⚠️ The one thing to confirm before cutover
How does Hermes consume an MCP memory provider?
- **Host-driven** (preferred): Hermes auto-injects `build_memory_context` each
  turn and calls `record_conversation` after — reliable, JASWOLF is a drop-in.
- **Model-driven**: Hermes exposes the tools to its LLM, which must choose to
  call them — needs a system-prompt instruction + day-one prompt tuning.

Confirm by reading how mempalace/agentmemory are registered in the Hermes
config on the box, and copy that shape for `jaswolf-memory`. Until that's
confirmed and smoke-tested, the cutover isn't real.

### Run as a managed service
systemd units (boot-persistent, auto-restart on crash *and* hang) + the honest
failure-semantics and monitoring guide are in
[deploy/README.md](../deploy/README.md). `GET /healthz` (200 ok / 503
starting|degraded) and `jaswolf mcp-health` (exit 0/1) drive systemd, cron
alerts, and a Hermes pre-start gate. The provider is prewarmed at HTTP startup,
so `/healthz` only reports ok once the model is loaded.

### Migration & rollback (no dual-write)
1. **Migrate Mem0 → JASWOLF first** (export + import, typed where possible) or
   the agent starts amnesiac.
2. **Freeze, don't delete, Mem0** — it's the rollback target. Not dual-write:
   nothing writes to Mem0 during the pilot; rollback = flip the provider
   config back.
3. Verify the live user_id (the BGE memories were imported under a specific
   `user_id` — confirm it from the DB; wrong value = silent empty context).

---

## 5. mem0 comparison — what was borrowed, what was skipped

Reviewed the mem0 v3 repo (April 2026). Key facts: mem0 puts an **LLM on the
write path** (≈0.88–1.09 s p50), uses Qdrant + OpenAI embeddings, and gets
quality from LLM fact-extraction, multi-signal retrieval (semantic + BM25 +
entity), and read-time temporal reasoning.

**Borrowed (lightweight):** read-time temporal current-state resolution
(§3C) — JASWOLF's biggest quality-per-effort win, no LLM, no latency cost.

**Deliberately NOT adopted** (the "heaviness" to avoid):
- per-write LLM calls — would erase JASWOLF's ~15× speed advantage;
- graph store (Neo4j/entity graph) — operational burden for marginal gain;
- Qdrant dependency — SQLite+numpy / pgvector is lighter and sufficient.

**Optional future borrows (not blockers):** lightweight deterministic entity
tagging + retrieval boost; adopting LoCoMo / LongMemEval as eval targets.

---

## 6. Backup & recovery (v0.7.0)

Full guide + cron recipe: [OPERATIONS.md](OPERATIONS.md). A long-lived memory
store is only as durable as its backups.

```bash
# consistent snapshot — safe even while the server/MCP process is running
# (uses SQLite's online backup API; a plain `cp` can capture a torn DB+WAL)
jaswolf backup --out /backups/jaswolf.db
jaswolf backup --keep 14                  # default-named series, rotate to 14

# restore (sqlite only) — validates integrity + embedding fingerprint first;
# STOP the server/MCP process so the file isn't open
jaswolf restore --from /backups/jaswolf_backup_20260613.db        # dry run, shows info
jaswolf restore --from /backups/jaswolf_backup_20260613.db --yes  # overwrite

# nightly cron (server can stay up):
0 3 * * *  /path/to/jaswolf backup --out /backups/jaswolf_$(date +\%Y\%m\%d).db --keep 14
```
Postgres: use `pg_dump $JASWOLF_DATABASE_URL` (embeddings are ordinary columns).
Recovery loop verified end-to-end: seed → backup → delete DB → restore → back.

**Add the nightly backup cron before cutover** — it's the last durability
piece and an ops step, not code.

---

## 7. CLI reference

```text
jaswolf serve [--host --port --workers]      run the HTTP API
jaswolf sweep                                one lifecycle sweep (age/expire)
jaswolf consolidate --user-id U [--dry-run]  merge near-duplicates
jaswolf stats [--user-id U]                  counts by state/type
jaswolf diagnose [--user-id U]               paste-ready report (+integrity, live probe)
jaswolf eval-shadow --probes F --user-id U   deterministic gate (verdict + exit code)
jaswolf update --id ID [--type --importance --confidence --state]   retype/rescore a memory
jaswolf mcp [--transport stdio|http --db --user-id --host --port]   MCP memory server
jaswolf mcp-health [--url --timeout]         probe /healthz (exit 0=ok, 1=down/degraded)
jaswolf backup [--out PATH --keep N]         consistent DB snapshot
jaswolf restore --from PATH [--yes]          restore from snapshot (sqlite, validated)
```

---

## 8. Settings reference (env prefix `JASWOLF_`)

Set via env vars or `.env`; constructor kwargs override env (verify with
`jaswolf diagnose`). Highlights of what was added across v0.2–0.7:

```text
# embeddings / startup
JASWOLF_EMBEDDING_PROVIDER     auto|local|openai|hash   (use 'local' explicitly for bge)
JASWOLF_EMBEDDING_MODEL        BAAI/bge-small-en-v1.5
JASWOLF_EMBEDDING_PREWARM      true   → load model at startup, no per-turn cold start

# context-boundary gate (prompt-injection safety)
JASWOLF_CONTEXT_NOISE_Z        3.5    → higher = stricter; 0 disables the gate
JASWOLF_CONTEXT_BACKGROUND_SAMPLE  256
JASWOLF_CONTEXT_MIN_BACKGROUND 24     → below this, fall back to fixed anchors
JASWOLF_CONTEXT_SIMILARITY_MARGIN  0.08  (anchor-fallback only)

# keyword evidence
JASWOLF_KEYWORD_MIN_TOKEN_LEN  3
JASWOLF_KEYWORD_MAX_DF_RATIO   0.10   → token in >10% of corpus isn't evidence

# temporal current-state resolution
JASWOLF_TEMPORAL_RESOLUTION    true

# MCP server
JASWOLF_MCP_USER_ID            <the user_id the memories live under>
JASWOLF_MCP_AGENT_ID           hermes
JASWOLF_MCP_NAMESPACE          default
JASWOLF_MCP_HOST / JASWOLF_MCP_PORT   127.0.0.1 / 8765
```
Lifecycle, scoring weights, dedup/supersession thresholds, and pinning gates
are in [OPERATIONS.md](OPERATIONS.md).

---

## 9. Cutover path (summary)

1. **Phase 0 (blocker):** confirm how Hermes consumes MCP memory (§4).
2. Install v0.7.0 with the `mcp` extra (CPU-safe torch).
3. Confirm the real `user_id` from the BGE DB.
4. Migrate Mem0 → JASWOLF (typed where possible).
5. Boot `jaswolf mcp` warm; verify `memory_health` ok + non-empty recall.
6. Register `jaswolf-memory` in Hermes; smoke-test live (store→restart→recall,
   off-topic silence, correction supersedes).
7. Add the nightly backup cron.
8. Freeze Mem0, flip the provider, watch the first hour, eval-shadow hourly.

Step-by-step with exact commands:
[claude_handoffs/2026-06-13-claude-mcp-cutover-runbook.md](claude_handoffs/2026-06-13-claude-mcp-cutover-runbook.md).

---

## 10. Version history

| Ver | What |
| --- | --- |
| 0.2.0 | hardened write path per Jasmine's review (durability gate, supersession, pinning, security defaults) |
| 0.2.1 | archived-memory revival on restatement |
| 0.2.2 | embedding fingerprint guard; CPU-only install path |
| 0.3.0 | deterministic `eval-shadow`; SearchQuery/ContextRequest guardrails; raw similarity exposed; prewarm; `update` CLI |
| 0.4.0 | context-boundary gate (anchor noise-floor) |
| 0.4.1 | corpus-calibrated gate + discriminative keyword evidence |
| 0.5.0 | **MCP memory server** (`jaswolf mcp`) |
| 0.6.0 | temporal current-state resolution (mem0-inspired) |
| 0.7.0 | **backup/restore + integrity in health** |

---

## 11. Verifying an install

```bash
python -c "import jaswolf; print(jaswolf.__version__)"      # 0.7.0
PYTHONPATH=src .venv/bin/pytest -q                     # 130 passed
jaswolf diagnose --user-id <USER_ID>                      # provider, fingerprint, integrity, probe
jaswolf eval-shadow --probes <probes.json> --user-id <USER_ID> --db <bge.db> \
  --embedding-provider local                           # verdict: GO_PILOT
```
