# The Semantic Pyramid (v0.3.0)

*What we adopted from TencentDB-Agent-Memory, what we deliberately did not,
and how the pieces fit JASWOLF's existing physics.*

## Background

In July 2026 we reviewed [TencentDB-Agent-Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory)
(Tencent's open-source agent memory plugin, ~8.3k stars) against JASWOLF. The
verdict, confirmed by two independent reviews: **JASWOLF is ahead on atom
hygiene** — durability gating, reinforcement-not-duplication, supersession,
temporal current-state resolution, corpus-calibrated injection gates, journal
durability, multi-tenant isolation (Tencent's isolation gap is their open
issue #111). **Tencent is ahead on macro structure**: their "semantic
pyramid" (L0 conversations → L1 atoms → L2 scenes → L3 persona) gives agents
provenance, an at-a-glance profile, and drill-down from any claim back to
ground truth.

v0.3.0 puts Tencent's pyramid **on top of** JASWOLF's atom physics — never in
place of them.

```
L3  persona          ← compile   (persona.py — deterministic VIEW, v0.3.0)
L2  scenarios        ← aggregate (roadmap — Phase C, see below)
L1  typed memories   ← extract   (existing JASWOLF physics, unchanged)
L0  conversations    ← capture   (conversation_messages, v0.3.0, opt-in)
```

## What we built

### 1. L0 conversation archive (`conversation_capture`, default off)

Raw turns are stored in `conversation_messages` before extraction runs.
Immutable once written — corrections happen in the memory layer, never by
rewriting history. FTS-indexed (`conversations_fts` / `to_tsvector`), pruned
after `conversation_retention_days` (default 90; 0 = keep forever). Memories
extracted from pruned turns are unaffected — only the raw evidence expires.

`search_conversations` (service / REST / MCP / provider) searches transcripts
directly: the "what did we discuss last Tuesday?" query that extraction may
have missed. Recall-first on purpose: cheap stopword filter, no corpus-DF
gate — an agent searching its own transcripts wants hits, not precision.

**Cold journal (v0.4.0).** With `conversation_archive_dir` set, expiring
turns are exported to monthly `YYYY-MM.jsonl.gz` files BEFORE pruning, under
the invariant *a turn is only deleted after its archive write is fsynced* —
deletion is by the exact archived ids, never a blanket time cutoff, and any
write failure halts pruning for that pass (`archive.py`). The live DB stays
a rolling window; the journal keeps everything, forever, in flat files that
outlive the software (readable with `zgrep`/`gzip.open` in any decade).
This is the substrate for the future timeline layer: life, month by month,
rebuildable from `read_archive_month()`.

### 2. Provenance + `explain`

With capture on, every extracted memory carries
`metadata.source_message_ids`. `explain(memory_id)` walks the full chain:

    memory → versions → relationships (supersedes/merged/derived) → source turns

Surface: `GET /v1/memories/{id}/explain`, MCP `explain_memory`, CLI
`jaswolf explain --id …`, provider `explain_memory()`. This is the "why do you
think that?" tool — debugging recall stops being vector-score archaeology
and becomes a deterministic walk.

### 3. Compiled persona (L3) — deterministic, never generative

`jaswolf persona --user-id alice --out persona.md` renders "who this user is"
from authoritative rows: preferences, goals, relationships, high-importance
facts — `always_pin` first, importance-ranked, token-capped
(`persona_token_budget`, default 400), every line tagged with its source
memory id.

**The key deviation from Tencent**: their persona.md is written by a
free-form LLM pass told to find "narrative coherence" and "connecting
threads" — an invitation to hallucinate structure, with drill-down as the
only defense. JASWOLF's persona is a **compiled view**: it can only say what
the atom layer already says, and identity-grade gates apply (confidence ≥
`pin_min_confidence`, semantic facts need importance ≥ 0.7, test/staging
rows excluded). Single-shot extracted facts don't qualify until reinforced
— same bar as context pinning, so the persona and the prompt never disagree.
An optional LLM polish can be layered later; it must be fact-checked against
the source ids.

### 4. Observe cadence (warm-up + idle flush)

Tencent's extraction scheduling, adapted for the provider:
`observe_every_n` (default 1 = classic per-turn), warm-up ramp 1→2→4→…→N so
fresh sessions learn from turn one, and an idle flush
(`observe_idle_flush_seconds`, default 600) so a finished session's tail is
never left unextracted. Buffered turns are journaled individually when a
journal is configured — a crash replays them instead of forgetting. A failed
flush restores the buffer for retry.

## What we deliberately did NOT take

* **Their retrieval/dedup/storage** — JASWOLF's are stronger (calibrated gates,
  reinforcement, revival, discriminative-keyword evidence).
* **LLM-generated persona/scenes with file tools** — their scene extractor
  needs filename normalizers, `[DELETED]` soft-delete markers, and
  engineering fallbacks for when the model misbehaves. A memory system that
  must not lie shouldn't have a generative component in its trust path.
* **L0–L3 labels replacing typed memories** — preference/goal/taste/working
  carry more semantics than "atom".
* **Mermaid short-term offload** — real idea, wrong repo. It's an in-task
  context-compression problem owned by the agent host (Hermes), not the
  long-term store. Tracked as a separate Hermes-side experiment.
* **Running their plugin alongside JASWOLF** — two writers, two half-truths.
  One engine of record.

## Roadmap

* **Phase C — L2 scenarios**: deterministic clustering (consolidation
  machinery already does the hard half) into navigable scene docs
  (title, summary, atom_ids, time_span). Gated on golden-probe evals:
  project-context recall up, off-topic injection still zero.
* **LLM persona polish** (flagged, fact-checked against source ids).
* **Skill distillation** from repeated procedural clusters — proposals only,
  human-approved.

## Config quick reference

```bash
JASWOLF_CONVERSATION_CAPTURE=true        # opt in to L0
JASWOLF_CONVERSATION_RETENTION_DAYS=90   # 0 = keep forever (live DB)
JASWOLF_CONVERSATION_ARCHIVE_DIR=~/.hermes/jaswolf_journal_archive  # cold journal
JASWOLF_PERSONA_TOKEN_BUDGET=400
JASWOLF_OBSERVE_EVERY_N=4                # 1 = classic per-turn extraction
JASWOLF_OBSERVE_WARMUP=true
JASWOLF_OBSERVE_IDLE_FLUSH_SECONDS=600
```

All default to off/classic — an untouched deployment behaves exactly like
v0.2.x.
