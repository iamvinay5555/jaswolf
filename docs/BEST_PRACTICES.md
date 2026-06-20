# JASWOLF Best Practices

## Choosing memory types

* `preference` / `goal` are **pinned**: the context builder includes the top
  ones even when off-topic. Reserve them for genuinely identity-shaping
  facts, or context fills with noise.
* `working` is for *state*, not knowledge ("current task: …"). It expires —
  never put durable facts there.
* `episodic` is cheap history; let the lifecycle sweeper age it out. Don't
  fight churn with high importance — use `semantic` for things that must
  persist.
* `procedural` memories can be long (runbooks). They cost tokens: keep one
  good version, not five drafts — consolidation helps.

## Writing good memory content

Third person, one fact per memory, self-contained:

* ✅ `User prefers Python for backend development`
* ❌ `He said he likes it` (no referent, no fact)

`observe()`/extraction produces this shape automatically; follow the same
style for manual `add_memory()` calls.

## Token budgets

The default 1500 fits a typical 8k-context agent comfortably. Guidance:

| Agent context | Suggested budget |
| --- | --- |
| 8k | 800–1500 |
| 32k | 1500–3000 |
| 128k+ | 2000–4000 (more isn't better — relevance dilutes) |

Watch `truncated: true` in context responses: chronic truncation means the
budget is too small or the store needs consolidation.

## Thresholds that matter

| Setting | Default | Raise it when… | Lower it when… |
| --- | --- | --- | --- |
| `dedup_threshold` | 0.95 | distinct facts get merged at write | duplicates pile up |
| `consolidation_threshold` | 0.88 | consolidation over-merges | store stays noisy |
| `min_relevance` | 0.10 | irrelevant results appear | recall feels too strict |
| `recency_half_life_days` | 7 | agent forgets too fast | stale facts dominate |

Note: thresholds are calibrated for real embedding models. The hash
embedder produces lower similarities — fine for tests, but don't tune
production thresholds against it.

## Hygiene schedule

* **Sweeper** runs automatically (5 min default) — it only changes states.
* **Consolidation**: run daily or weekly per active user
  (`jaswolf consolidate --user-id …` or the API). Always dry-run after changing
  thresholds.
* **Hard deletes**: soft-deleted rows keep ids resolvable and history
  intact. Purge with `?hard=true` only for data-removal requests.

## Multi-tenancy

One tenant per trust boundary (per customer, per deployment). Within a
tenant, use `namespace` to partition domains (e.g. `coding`, `personal`) and
`user_id` per human. Search never crosses tenant; namespace crossing is
opt-in (omit `namespace` in the query).

## Embedding model choice

* `bge-small-en-v1.5` (default): 384-dim, CPU-friendly, English. Right for a
  VPS.
* OpenAI `text-embedding-3-small`: better multilingual; set
  `JASWOLF_EMBEDDING_DIM=1536` *before* first migration.
* Changing dimension later = new migration + re-embed. Decide early.
