# Catching and Reporting Bugs

This guide is for both humans and agents operating JASWOLF. The goal of every
report is the same: enough signal that a fix can be designed without access
to your machine. A report with a failing test gets fixed almost immediately;
a report saying "search feels off" cannot be acted on.

## Part 1 — Catching bugs

### Continuous signals (cheap, always on)

| Signal | How | Healthy looks like |
| --- | --- | --- |
| engine health | `GET /health` or `await provider.health_check()` | `status: "ok"` |
| latency | `/metrics`: `jaswolf_search_latency_seconds`, `jaswolf_context_latency_seconds` | p95 < 0.1s / 0.15s |
| error rate | `/metrics`: `jaswolf_requests_total{status="500"}` | flat at 0 |
| store growth | `jaswolf stats` weekly | reinforced/created ratio rising; total not exploding |
| dedup health | `jaswolf_memories_reinforced_total` vs `_created_total` | reinforcements happen at all |

### Symptom → where to look first

| Symptom | Most likely subsystem | First check |
| --- | --- | --- |
| relevant memory not recalled | retrieval / embeddings | same `user_id`+`namespace`? `/health` shows real embedder (not `hashing-*`)? memory state archived? |
| irrelevant memories in context | scoring / thresholds | raise `JASWOLF_MIN_RELEVANCE`; check importance inflation (everything 0.9+) |
| duplicates accumulating | write-path dedup / consolidation | `jaswolf consolidate --dry-run`; thresholds vs your embedding model |
| context block empty | relevance floor / scope | pass `messages` not a 1-word query; try `min_relevance=0` |
| wrong facts extracted | extraction rules | reproduce with `RuleExtractor().extract(text)` directly |
| 500 responses | API / storage | server log has the full traceback (every 500 is logged) |
| slow at scale | storage backend | SQLite past ~50k memories → move to Postgres |

### Turn up visibility when investigating

```bash
export JASWOLF_LOG_LEVEL=DEBUG     # subsystem loggers: jaswolf.retrieval, jaswolf.extraction, …
jaswolf diagnose --user-id alice   # snapshot: versions, config, counts, live latency probe
```

### The golden rule: shrink it to a deterministic repro

The hash embedder + a temp SQLite file make everything reproducible (no
model downloads, no network, stable vectors). Start from this template and
delete everything not needed to show the bug:

```python
# tests/test_repro.py — run with: pytest tests/test_repro.py -q
from jaswolf import JaswolfSettings, MemoryService
from jaswolf.models import MemoryCreate, SearchQuery

async def test_repro(tmp_path):
    service = await MemoryService.create(JaswolfSettings(
        database_url=f"sqlite:///{tmp_path}/repro.db",
        embedding_provider="hash",
        log_level="DEBUG",
    ))
    try:
        await service.add(MemoryCreate(user_id="u", content="User prefers Python"))
        results = await service.search(SearchQuery(user_id="u", query="python"))
        assert results, "expected a hit"          # <- state what SHOULD happen
    finally:
        await service.close()
```

If the bug only appears with a real embedding model or Postgres, say so in
the report and include which one — that locates the bug to the
calibration-sensitive or backend-specific code paths.

## Part 2 — Reporting bugs

### Report template

```markdown
## Summary
One sentence: what's wrong.

## Expected
What should have happened.

## Actual
What happened instead (exact output, not a paraphrase).

## Reproduction
The minimal test/snippet from Part 1, or exact API calls in order.

## Diagnostics
Output of `jaswolf diagnose --user-id <uid>` (secrets are auto-redacted).

## Logs / traceback
Relevant lines with JASWOLF_LOG_LEVEL=DEBUG; full traceback for any exception.

## Impact
blocking / degraded / cosmetic — and any workaround found.
```

### Where to send it

1. **GitHub issue** (preferred — keeps history searchable):

   ```bash
   gh issue create --repo iamalice5555/jaswolf \
     --title "search: relevant memory not recalled after consolidation" \
     --body-file report.md
   ```

2. **Ask Claude for the fix.** Alice's workflow: paste the filled template
   into a Claude Code session in the `jaswolf` repo and ask for a diagnosis
   and fix plan. The template is designed so that a session with no memory
   of writing this code can still act on it. Include the repro test —
   "make `tests/test_repro.py` pass without breaking the other tests" is
   the ideal, fully-specified ask.

### Rules

* **Never include secrets.** `jaswolf diagnose` redacts DB/Redis passwords and
  prints no API keys; don't paste raw `.env` contents into issues.
* One bug per report. Two symptoms = two reports (they may share a cause;
  let the investigation establish that).
* Report what you observed, not your theory of the cause — include the
  theory separately if you have one.
* After a fix: the repro test joins the permanent suite. That's how this
  codebase ratchets — bugs die once.

### Worked example (the kind of report that gets fixed in minutes)

> **Summary**: exact-duplicate detection misses when content differs only by
> trailing punctuation.
> **Expected**: `add("User prefers Python")` then `add("User prefers Python!")`
> reinforces (created=False).
> **Actual**: second call returns created=True; `stats` shows 2 memories.
> **Reproduction**: two `service.add()` calls as above, hash embedder,
> fresh SQLite. **Diagnostics**: jaswolf 0.1.0, sqlite, hashing-384,
> dedup=0.95. **Impact**: degraded — duplicates accumulate slowly;
> consolidation cleans them up later.

(That one is real, by the way: `content_hash` normalizes whitespace/case
but keeps punctuation, and `!` shifts the hash embedding enough to stay
under 0.95. With a real embedding model the near-dup check catches it —
with the hash embedder it does not. Known, accepted for v0.1, documented
here so it isn't re-reported.)
