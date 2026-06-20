# JASWOLF Architecture

## Design stance

JASWOLF is **library-first**: every capability lives in plain Python classes
composed by `MemoryService` (`service.py`). The REST API, the CLI, and the
Hermes provider are thin adapters over that one object. This is the main
deviation from a classic "microservice" memory store, and it is deliberate:
an autonomous agent on a VPS gets sub-millisecond memory calls in embedded
mode, and you only pay for HTTP when several processes must share a store.

```
                 ┌──────────────┐  ┌──────────┐  ┌─────────────────┐
   entry points  │ FastAPI /v1  │  │   CLI    │  │ JaswolfMemoryProvider │
                 └──────┬───────┘  └────┬─────┘  └──────┬──────────┘
                        └───────────────┼───────────────┘
                                 ┌──────▼───────┐
                                 │ MemoryService │   composition root
                                 └──────┬───────┘
        ┌──────────────┬───────────────┼────────────────┬─────────────┐
        ▼              ▼               ▼                ▼             ▼
  ExtractionEngine RetrievalEngine ContextBuilder ConsolidationEngine sweep()
        │              │               │                │
        └──────┬───────┴───────┬───────┴────────────────┘
               ▼               ▼
        CachedEmbedder   StorageBackend (protocol)
               │               ├── SQLiteStore     (dev/edge, exact search)
        CacheBackend           └── PostgresStore   (prod, pgvector HNSW)
        ├── InMemoryCache
        └── RedisCache
```

Every box behind `MemoryService` is swappable: storage, cache, and embedder
are protocols; extraction and merging accept injected clients.

## Write path

`service.add()`:

1. Normalize + hash content (`content_hash`).
2. **Exact-dup fast path**: same hash within (tenant, user, namespace) →
   *reinforce* the existing memory (access_count +1, confidence +0.05 capped,
   importance = max). No embedding computed at all.
3. Embed (through the cache — repeated text never re-embeds).
4. **Near-dup check**: cosine ≥ `dedup_threshold` (default 0.95) against the
   same memory type → reinforce instead of insert.
5. Insert with auto-scored importance when none supplied
   (type baseline + emphasis cues).

Reinforcement-instead-of-duplication is what keeps an always-on agent's
store small: restating "I prefer Python" for the 50th time makes one memory
stronger, not 50 rows.

Reinforcement also **revives**: the exact-hash path can match an *archived*
row (idle ladder, TTL expiry, or supersession), and restating a fact is
liveness evidence — the row returns to `active`, with a fresh TTL if the
old one had lapsed (v0.2.1). Without this, a verbatim repeat reinforced a
row search could never see again, while a paraphrase — landing in the
near-dup path, which only scans searchable states — created a fresh visible
memory. Near-dup matches never touch archived rows, so an archived fact's
paraphrase still becomes a new memory; only exact restatement revives the
original with its history intact.

## Current-state resolution (read-time)

Write-path supersession is conservative — it archives an old fact only on a
correction marker ("actually", "now"), so **unmarked** contradictions
("User's office is Buona Vista" then later "… is Changi", no marker) leave
both active and a query could surface the stale one. Rather than guess
destructively at write time, JASWOLF resolves this at read time (`temporal.py`,
v0.6.0): when retrieved memories of the same type fill the same **singleton**
slot ("User's <slot> is <value>" where the slot is a thing a person has one
current value of — office, job, address, …) with *different* values, only the
freshest (by `updated_at`) is injected into context. It never mutates storage,
and **multi-valued** relations (friend, pet, hobby — slots deliberately absent
from the singleton set) stay additive. Applies in the context builder only;
raw `search()` still returns history. Disable with `temporal_resolution=false`.

## Read path

`RetrievalEngine.search()` supports five modes. `hybrid` (default) runs
vector and full-text search, fuses with **Reciprocal Rank Fusion** (no score
calibration needed), then blends in raw cosine:

```
relevance = 0.6 · rrf_normalized + 0.4 · cosine
```

Two guards make ranking robust across embedding models:

* **Relevance floor** (`min_relevance`, default 0.1): vector-only candidates
  below the floor are dropped — brute-force top-K always returns *something*,
  and without the floor an important-but-irrelevant memory could surface.
* **Per-query min-max normalization**: embedding models differ wildly in
  cosine calibration (bge's "unrelated" ≈ 0.35; other models ≈ 0.05). The
  final-score formula assumes relevance spans ~0..1, so the spread is
  restored per query. Pools with near-uniform scores are left untouched.

Final ranking (weights configurable):

```
final = 0.4·importance + 0.3·relevance + 0.2·recency + 0.1·frequency
recency  = 0.5^(age_days / half_life)          # default half-life 7d
frequency = log1p(access_count)/log1p(50)       # saturating
```

Searches record access (batched UPDATE + audit log), which feeds the
Search *ranks*; the context builder *injects* — and injection carries a
stricter burden of proof (v0.4.0, hardened in v0.4.1). Non-pinned vector
candidates enter the prompt only if their raw cosine clears a per-query
threshold calibrated against the **actual corpus**: `mean + context_noise_z
·std` of the query's cosine to a background sample of the user's own memory
embeddings (`calibration.py`). Measuring separation in units of the
corpus's own spread makes it anisotropy-invariant — bge-small parks
arbitrary English ~0.6 where the hash embedder sits ~0.0, and an off-topic
query's top hit is merely the upper tail of that background. (Fixed
unrelated-anchor sentences, the v0.4.0 approach, under-measured this floor
and are kept only as a small-corpus fallback.) Keyword hits are exempt only
when the matched term is *discriminative* — non-stopword, long enough, and
rare in the corpus (`keywords.py`); generic tokens like "week" no longer
buy an exemption. Pinned preferences/goals and session working notes are
exempt by design, their purpose being to appear regardless of the query.

frequency signal and the lifecycle sweeper — retrieval literally keeps
memories alive.

## Context builder

1. Derive a query from the last user turns (or accept one).
2. One hybrid search across all types + **pinned** top preferences/goals
   (identity-defining memories belong in context even when off-topic).
3. Cross-candidate dedup (cosine ≥ 0.92 keeps the higher-scored one).
4. Two-pass token budgeting: per-section shares first (preferences 20%,
   facts 30%, …), then a global leftover pass by score; oversized items are
   sentence-truncated.
5. Render markdown (default) or XML, then a final verify-and-evict loop
   guarantees `token_estimate ≤ budget`.

## Memory evolution

* **Consolidation** (`consolidation.py`): per type, cluster stored
  embeddings with union-find over a cosine threshold (0.88 default), pick a
  canonical (importance → access_count → age), merge contents
  deterministically (containment/sentence-union) or via an optional LLM,
  re-embed, soft-delete losers with `merged_into` relationships, and record
  `memory_versions` for every change. Dry-run supported.
* **Lifecycle** (`sweep()`): expired working memories archive; idle memories
  walk active → warm → cold → archived (oldest stage first, one stage per
  sweep). Transitions never touch `updated_at`, so recency scoring stays
  honest. Archived/deleted memories leave search but remain addressable.
* **Versioning**: every content rewrite (update, consolidation, merge-away)
  appends to `memory_versions` — full audit trail.

## Storage

One protocol (`storage/base.py`), two backends:

| | SQLiteStore | PostgresStore |
|---|---|---|
| vectors | float32 BLOBs, exact numpy search | `vector(dim)`, HNSW cosine |
| keyword | FTS5 (LIKE fallback) | `to_tsvector` + GIN |
| timestamps | epoch floats | timestamptz |
| good for | dev, tests, ≤~50k memories | production, 10M+ |

Migrations are plain SQL shipped inside the package
(`storage/migrations/*.sql`), tracked in `schema_migrations`, with
`__EMBED_DIM__` substituted at apply time.

### Deviations from the original plan (and why)

* **No SQLAlchemy.** Hand-written SQL via `sqlite3`/`asyncpg` keeps the hot
  path allocation-light and the pgvector/FTS SQL explicit. The storage
  protocol is small; an ORM would add a layer without removing complexity.
* **No users/agents/sessions tables.** Identity belongs to the agent
  platform; JASWOLF treats `user_id`/`agent_id`/`session_id` as opaque strings.
  Tenant isolation is enforced by API key → `tenant_id` scoping on every
  query.
* **Added `goal` and `relationship` memory types** — the plan extracted
  them but had nowhere to store them.
* **API keys instead of JWT** for v0.1: one fewer moving part on a VPS;
  the auth dependency is a single function to swap when JWT is needed.
* **memory_summaries table dropped** — summaries are a context-builder
  concern; versions + relationships already capture history.

## Security model

API key → tenant mapping (`JASWOLF_API_KEYS=key:tenant,…`), constant-time key
comparison, per-key rate limiting (cache-backed buckets), tenant_id scoping
in every storage query, access audit log (`memory_access_logs`). Open mode
(no keys) is for local development and logs a warning.

## Observability

Prometheus metrics (request/search/context latency histograms, created/
reinforced counters, embed-cache hit counters) with a no-op shim when
prometheus-client is absent; `X-Response-Time-Ms` on every response;
`/health` reports storage/embedder/cache state and embed-cache hit ratio.
