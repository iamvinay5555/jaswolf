# Troubleshooting JASWOLF

## "Refusing to start the HTTP API without authentication"

New in v0.2.0: the API requires `JASWOLF_API_KEYS` to be set, or an explicit
`JASWOLF_DEV_OPEN_MODE=true` for local development. Embedded mode
(`JaswolfMemoryProvider.embedded()`) is unaffected.

## Health says "degraded" with a hash-embedder reason

`embedding_provider=auto` found neither sentence-transformers nor an OpenAI
key and fell back to the hash embedder. That fallback is fine for tests and
deliberately ugly in production. Install `jaswolf[local-embeddings]` or
configure an endpoint; or set `JASWOLF_EMBEDDING_PROVIDER=hash` explicitly if
this is a dev box (explicit choice reports "ok").

## A correction didn't replace the old fact

Supersession is marker-gated: it triggers on correction language
("actually", "now", "no longer", "instead", …) plus similarity or a slot
conflict. "My office is in Changi." without any marker stays additive by
design — run consolidation, or phrase corrections as corrections. See
ARCHITECTURE.md for the rationale.

## "No embedding model available — using deterministic hashing embedder"

Informational in dev. For production retrieval quality install one of:

```bash
pip install "jaswolf[local-embeddings]"          # sentence-transformers
# or
export JASWOLF_EMBEDDING_PROVIDER=openai JASWOLF_OPENAI_API_KEY=sk-...
```

## Search returns nothing / misses obvious memories

1. Scope mismatch — search is scoped by `user_id` (+ `namespace` when set).
   `jaswolf stats --user-id …` to confirm where memories actually live.
2. Hash embedder in production — similarities are weaker than real models;
   check `/health` → `embeddings.provider`.
3. Memories archived — long-idle memories leave the searchable states.
   Inspect with `GET /v1/memories/{id}`; resurrect via
   `PATCH {"state": "active"}`.
4. `min_relevance` too high for your embedding model — try `0.0` and
   re-test.

## Duplicates keep appearing

`dedup_threshold` (0.95) only reinforces *very* close matches at write time.
Differently-phrased repeats are merged by consolidation, not at write:

```bash
jaswolf consolidate --user-id alice --dry-run
```

If dry-run finds nothing, lower `JASWOLF_CONSOLIDATION_THRESHOLD` (e.g. 0.82)
and dry-run again before committing.

## Context block is empty

`build_context` returns `""` when nothing clears the relevance bar — that's
by design (don't inject noise). If it persists with a populated store, the
query may be too short to carry signal; pass `messages` (last turns) instead
of a one-word `query`.

## 401 / 404 surprises with multiple keys

A memory created with key A is invisible to key B by design (tenant
isolation). 404-on-existing-id almost always means the wrong API key.

## `expected N dimensions, got M` (Postgres) or silently empty vector search (SQLite)

The embedding dimension changed after data was written (model switch).
Either revert `JASWOLF_EMBEDDING_DIM`/model, or re-embed: export contents,
reinsert under the new model. Decide the model before production
(see BEST_PRACTICES).

## "EMBEDDING MISMATCH" warning / health degraded with an embedding-mismatch reason

Since v0.2.2 every DB is stamped at first open with the embedder that
writes it (`jaswolf_meta` table, e.g. `st:BAAI/bge-small-en-v1.5` or
`hashing-384`). Opening the DB with a *different* provider/model logs this
warning and degrades health — even when the dimensions happen to match,
because cosine similarity across two models' vector spaces is meaningless:
old memories become invisible to search and dedup/supersession misfire
against them. Remedies, in order of preference: point the config back at
the stamped model; start a fresh DB for the new model (shadow runs); or
re-embed everything under the new model (no built-in command yet — planned).
Writes are not blocked; the stamp is never overwritten by a mismatched
opener.

## Slow context builds on SQLite

SQLite vector search is exact brute-force per user. Beyond ~20–50k memories
per tenant, move to Postgres:

```bash
export JASWOLF_DATABASE_URL=postgresql://jaswolf:***@localhost:5432/jaswolf
```

## macOS dev: `ModuleNotFoundError: No module named 'jaswolf'` after `pip install -e .`

CPython ≥ 3.12.6 skips `.pth` files carrying the macOS *hidden* flag, and
iCloud-synced folders (Desktop/Documents) sometimes re-flag freshly written
files. Check with `python -v -c "import jaswolf" 2>&1 | grep pth` — if you see
"Skipping hidden .pth file":

```bash
chflags nohidden .venv/lib/python3.12/site-packages/__editable__.jaswolf*.pth
# or avoid the issue entirely:
pip install .            # regular install (copies files; immune)
# or keep the repo outside iCloud-synced folders
```

## Rate limit (429) in legitimate use

Raise `JASWOLF_RATE_LIMIT_PER_MINUTE` (it's per key). Remember every
`inject_context` + `observe` pair is ~2 requests per agent turn in remote
mode; embedded mode has no rate limit.

## Background sweeper didn't run

The sweeper lives in the API process (or embedded provider with
`auto_sweep=True`). If you only ever run one-off scripts, states never
transition — run `jaswolf sweep` from cron instead.
