# Shadow Mode: running JASWOLF beside Mem0

Per the review in `jasmine_feedback.md`: JASWOLF must not become the primary
memory provider by enthusiasm — it earns cutover through logged evidence.
This guide is the concrete protocol.

## Architecture

```
            agent turn
                │
        ┌───────▼────────┐      writes + recalls (logged, compared)
        │  ShadowMemory  │ ───────────────────────────► JASWOLF (embedded)
        └───────┬────────┘                                   │
                │ authoritative path                    shadow_log.jsonl
                ▼
              Mem0  ──► live prompt context
```

* Mem0 remains primary; **nothing from JASWOLF enters the live prompt**.
* Every candidate write goes to both; every recall queries both.
* Each operation appends a JSONL record for offline scoring.

`examples/shadow_mode.py` ships the `ShadowMemory` wrapper and the record
schemas. Adapt the `StubPrimary` to your Mem0 client (two methods:
`add(text)`, `search(query) -> list[str]`).

## Protocol (3–7 days)

1. Wire `ShadowMemory` into the agent loop where Mem0 is called today.
2. Use a dedicated JASWOLF DB (`sqlite:////var/lib/jasmine/jaswolf_shadow.db`) —
   never the live store.
3. Run with a real embedding model (`jaswolf[local-embeddings]` or an
   OpenAI-compatible endpoint) — hash-embedder evidence doesn't count.
4. Let it run across restarts, model switches, long sessions, and noisy
   chats. Restart persistence is itself one of the things under test.
5. Daily: skim `shadow_log.jsonl`, fill `human_label` / `winner` on a sample
   (10–20 records is enough), and run:

   ```bash
   jaswolf diagnose --user-id alice          # health, counts, latency probe
   python benchmarks/eval_retrieval.py --provider local   # golden metrics
   ```

## Scoring the log

For **writes**, label a sample: `correct | pollution | wrong_type |
too_verbose | sensitive | duplicate | superseded_old_fact`. Pollution rate =
labels in {pollution, wrong_type} / total durable writes.

For **recalls**, compare result lists side by side and set `winner`. Track:

* helpful hits (memories that would have improved the reply)
* harmful injections (irrelevant / stale / superseded content)

A memory engine that recalls too much irrelevant information is worse than
one that recalls slightly less — weigh harmful injections heavily.

## Go / no-go gates

Cut over only when, on your logged sample:

| Gate | Threshold |
| --- | --- |
| durable-write pollution | ≤ 2–5%, zero obvious chatter ("thanks", lunch plans) |
| recall quality | ≥ 90% of Mem0 baseline (Recall@5 / MRR on the same queries) |
| harmful injections | ≤ Mem0 baseline; superseded facts: 0 |
| token budget violations | 0 |
| restart persistence | no memory loss across gateway/process restarts |
| health honesty | degraded states actually reported degraded |

`benchmarks/eval_retrieval.py` enforces the structural gates (superseded
injections = 0, budget violations = 0) and reports the quality numbers;
extend its golden dataset with sanitized examples from your real workload as
you label the shadow log.

## Cutover (gradual)

1. Keep Mem0 writes on; start injecting JASWOLF context for **low-stakes task
   types only** (e.g. coding), comparing reply quality.
2. Expand task types over days. Mem0 keeps receiving writes the whole time.
3. Full cutover: JASWOLF primary, Mem0 in read-only retention.

## Rollback (at any point, in minutes)

1. Flip the agent's provider flag back to Mem0 (writes never stopped — no
   data gap).
2. Keep the JASWOLF DB file; it's the shadow evidence either way.
3. File what went wrong per [BUG_REPORTS.md](BUG_REPORTS.md) — a failed
   cutover with a good report still moves the project forward.
