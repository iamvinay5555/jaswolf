# Deterministic shadow evaluation — `jaswolf eval-shadow`

The shadow-window gate must be a script, not an agent: deterministic
metrics, a fixed verdict line, exit code for cron. This replaces the
LLM-driven report generator (Jasmine handoff, 2026-06-12). Run it with
`no_agent=True` from cron; an agent may *interpret* the report afterwards,
but never generates it.

```bash
jaswolf eval-shadow \
  --db sqlite:////home/jaswolf/.hermes/jaswolf_shadow/data/jaswolf_shadow_bge_small.db \
  --embedding-provider local \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --probes /home/jaswolf/.hermes/jaswolf_shadow/golden_probes.json \
  --user-id default \
  --shadow-log /home/jaswolf/.hermes/jaswolf_shadow/shadow_log.jsonl \
  --meta lane=bge --json
```

Every report stamps `db_url`, `embedding_fingerprint`, and provider — there
is never ambiguity about which lane was measured. Exit code: 0 unless the
verdict is `NO_GO`.

## Probe file format

JSON, top-level `"probes"` list (template:
[evals/golden_probes.example.json](evals/golden_probes.example.json)):

| Field | Meaning |
| --- | --- |
| `id` | stable name, shown in reports |
| `kind` | `search` (raw recall) or `context` (the full build_context gate) |
| `query` | the probe question — must be non-empty |
| `expect_any` | pass if ANY keyword appears (case-insensitive substring) |
| `expect_all` | every keyword must appear |
| `forbid` | stale/wrong facts; ANY hit forces verdict `NO_GO` |
| `top_k` | search pool to scan (default 5) |
| `high_salience` | failing one of these blocks `GO_PILOT` |
| `off_topic` | inverted probe — see below |
| `max_similarity` | optional absolute override for the off-topic search gate |

**Off-topic probes** assert the engine doesn't manufacture relevance:
a `search` probe passes when the best *raw* cosine stays under the gate;
a `context` probe passes when zero non-pinned memories are injected
(pinned preferences/goals legitimately appear everywhere).

**The gate is corpus-calibrated** (v0.4.1). Absolute cosine thresholds are
model-dependent (bge-small scores arbitrary unrelated English ~0.55–0.65;
the hash embedder ~0.0), and fixed "unrelated anchor" sentences *under*-
measure the floor — exotic anchors (basalt, falconry) score ~0.39 against a
query while the user's own off-topic memories score ~0.63. So the gate is
calibrated against the **actual corpus**: `mean + context_noise_z·std` of
the query's cosine to a background sample of the user's own memory
embeddings (`calibration.py`). Because it measures separation in units of
the corpus's own spread, it is anisotropy-invariant — an off-topic query's
top hit is just the upper tail of that background, while a real hit stands
out. This is the *same* gate the context builder enforces at prompt
assembly: non-pinned vector candidates below it are never injected.
Fixed anchors remain only as a fallback for corpora too small
(`context_min_background`) to estimate a distribution. Each probe result
records the computed `off_topic_gate`; set `max_similarity` only to pin an
absolute number.

**Keyword exemption requires discriminative evidence** (v0.4.1). A keyword
hit exempts a memory from the gate only when the matched term is
discriminative — not a stopword, ≥ `keyword_min_token_len`, and present in
at most `keyword_max_df_ratio` of the corpus (an IDF-style cut, with an
absolute-count floor so it can't misfire on tiny corpora). This closes the
leak where generic tokens like "next"/"week" in an off-topic query matched
unrelated memories and bypassed the gate.

**Why raw cosine:** search relevance is min-max normalized per query pool,
so the top hit shows ~1.0 even for "weather in Lisbon". Normalized
relevance ranks *within* a query; it says nothing *across* queries. Since
v0.3.0 every search hit carries `similarity` (raw, pre-normalization) —
gates and reports must read that field, never normalized relevance.

## Privacy model

The probe file contains personal keywords (routes, names, preferences) —
it lives on the VPS **outside the repo** (e.g.
`~/.hermes/jaswolf_shadow/golden_probes.json`). The repo carries only the
placeholder template. Reports reference probes by `id` and quote only the
keyword that failed, never memory contents — safe to commit
`--json` output into `docs/reports/`.

## `jaswolf cutover-preflight` — the per-bot GO/NO-GO gate (v0.11.0)

Before pointing a bot at JASWOLF, run the golden probes through **that bot's exact
scope** (its namespace + the shared namespace) — the real multi-agent read
path, not a generic eval:

```bash
jaswolf cutover-preflight --db <bge.db> --embedding-provider local \
  --probes <golden_probes.json> --user-id default \
  --namespace freya --shared-namespace shared --profile freya
```
Exit 0 only on `GO_PILOT` (strict). It asserts: shared/identity facts surface
in that scope, off-topic injects nothing, no test data, health ok. `eval-shadow`
also accepts `--namespace`/`--shared-namespace` for the same scoped probing.
Run it once per bot (freya, then main) before cutover, and after any gate
re-tuning.

## Verdict semantics

| Verdict | Meaning | Triggers |
| --- | --- | --- |
| `NO_GO` | something is broken — investigate before more shadow time | health ≠ ok · hash-fallback active · sqlite quick_check failure · shadow-log errors · **any forbidden/stale fact surfaced** |
| `CONTINUE_SHADOW` | healthy, gates not met yet | probe score < 0.9 · any high-salience failure · irrelevant injections > 0 · warm search p95 > 500 ms |
| `GO_PILOT` | all gates green — eligible for the pilot review | everything above passes |

## Gates: shadow → pilot → cutover

**`GO_PILOT` (this tool decides):** verdict `GO_PILOT` on ≥ 3 consecutive
daily runs over a ≥ 72 h window in which at least one real live write
window appeared in the shadow log, fingerprint unchanged throughout.

**Pilot = comparison, not replacement:** JASWOLF builds candidate context
per real turn; Mem0 stays primary and keeps receiving writes; divergences
logged with winner judgments (SHADOW_MODE.md cutover section).

**Production cutover (humans decide, evidence required):** ≥ 7 days clean
pilot · personal corrections, supersession, and anti-ephemeral verified by
high-salience probes · warm-path latency stable with prewarm in place ·
rollback rehearsed · Mem0 export confirmed.

## Cold start / prewarm

The first embed in a process loads the model (seconds to tens of seconds
on CPU). Production shape, in order of preference:

1. **Long-running process + `JASWOLF_EMBEDDING_PREWARM=true`** — the load
   happens at boot, never on a live turn. This is the pilot answer.
2. **`jaswolf serve` as a warm HTTP service** + `JaswolfMemoryProvider.remote()`
   — when multiple short-lived processes need the same store.
3. Never: per-turn process spawn with embedded mode — every turn pays the
   model load.

`eval-shadow` reports `cold_latency_ms` (deliberately unprewarmed) so the
boot cost stays visible.
