# The Taste Index — judgment memory for agents

Most agent memory answers *"what is true about the user?"* The Taste Index
answers a different question: *"how does the user judge quality?"*

A fact like `Alice's office is in the city centre` is biography. A rule like
`Good product demos show the workflow, not just the output` is **judgment** —
a reusable steering rule that should shape how the agent writes, designs,
researches, and builds. JasWolf models these as a first-class memory type:
`taste`.

The design follows one principle throughout: **memory is not judgment, and
judgment must never be inferred.** Taste entries are explicit, selective, and
task-scoped — a small, sharp set of rules, not another pile of text.

## The three rules

### 1. Explicit capture only — enforced by the engine

A taste memory is rejected (HTTP 422) unless it carries:

| Field | Meaning |
|---|---|
| `metadata.explicit_signal: true` | the user deliberately asked to save this |
| `metadata.why_useful` (non-empty) | why this should steer future work — the judgment is the asset, not the artifact |
| `metadata.where_to_apply` (list) | which kinds of work should consult it |

`where_to_apply` values come from a **closed vocabulary**, matched exactly:

```
writing · product · design · research · architecture · agent_behavior
```

No semantic matching of apply-scopes, no free text — a taste rule can never
leak into unrelated work because an embedding model thought it was similar.

The extractor can never emit taste (regression-guarded): ordinary
conversation about quality does not create taste entries, however
enthusiastic it sounds. Temporary excitement is not durable judgment.

These invariants hold on **update too**: a PATCH that would clear
`why_useful`, corrupt `where_to_apply`, or retype another memory into taste
fails before anything is written.

### 2. One entrance: task-aware retrieval

Taste never rides into context on incidental vector similarity — it is
excluded from the generic retrieval pool entirely. It enters through exactly
one door: the caller declares what kind of work is about to happen.

```json
POST /v1/memories/context
{
  "user_id": "alice",
  "query": "draft the launch post",
  "task_type": "writing"
}
```

Active taste entries whose `where_to_apply` includes the declared
`task_type` render as a compact, budget-capped `## Taste` section. No
`task_type` → zero taste, always. Unknown `task_type` → 422.

Taste is also **not** identity: it never force-pins, so it cannot dominate
every turn the way identity/safety memories deliberately do.

### 3. Anti-patterns are first-class

Negative rules ("do not invent user data", "no generic wrapper demos")
often generalize better than positive examples. Mark them with
`metadata.anti_pattern: true` and they:

- sort **first** in the Taste section, regardless of importance
- render with an `AVOID:` prefix
- survive token-budget truncation ahead of softer positive guidance

## Capturing a rule

```json
POST /v1/memories
{
  "user_id": "alice",
  "content": "Good product demos show the workflow, not just the output",
  "memory_type": "taste",
  "importance": 0.85,
  "metadata": {
    "explicit_signal": true,
    "why_useful": "keeps demo ideas concrete instead of wrapper fluff",
    "where_to_apply": ["product"],
    "anti_pattern": false,
    "source": "taste-command"
  }
}
```

`where_to_apply` is stored deduped and stable-sorted. A single string is
normalized to a one-element list.

## Lifecycle: taste evolves only by hand

Taste is exempt from two automations that are safe for facts but dangerous
for judgment:

- **No auto-consolidation.** Sentence-union merging two distinct steering
  rules can garble both into mush. Near-duplicate taste rules stay separate
  until a human merges them.
- **No supersession.** A correction-shaped sentence in ordinary chat
  ("actually, now we...") must never silently archive a steering rule.

To evolve a rule: update its content explicitly, or archive it
(`state: archived` — archived taste is never injected).

## Recommended agent-side pattern

The engine is deliberately only the wall. The conversational half belongs in
your agent layer:

1. **Trigger phrases**, not inference: "save this as taste",
   "remember this style", "don't do this again" (offer `anti_pattern`).
2. **Collect the missing fields conversationally** — ask "why should this
   go in your Taste Index?" rather than saving something hollow. The
   engine's 422s tell you exactly what's missing.
3. **Preview + explicit confirmation** before the write.
4. **Declare `task_type` explicitly** when building context for a known
   kind of work — a command flag or agent convention, not semantic
   guessing (at least until real usage proves inference trustworthy).
5. Keep the set small. If everything is taste, nothing steers.

## Configuration

| Setting | Default | Meaning |
|---|---|---|
| `JASWOLF_CONTEXT_SHARE_TASTE` | `0.12` | Taste section's share of the context token budget (idle unless task_type is set) |
| `JASWOLF_CONTEXT_MAX_TASTE` | `6` | Max taste entries injected per context build |

## Evaluation

The `eval-shadow` probe format accepts `"task_type"` on context probes, so
golden probes can assert both directions:

```json
{"id": "writing-taste-present", "kind": "context", "task_type": "writing",
 "query": "draft a post", "expect_any": ["write plainly"]},
{"id": "no-task-no-taste", "kind": "context",
 "query": "what is the user's favorite tea", "forbid": ["write plainly"]}
```
