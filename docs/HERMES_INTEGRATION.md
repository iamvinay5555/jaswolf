# Hermes Integration Guide

`JaswolfMemoryProvider` is the Hermes-facing interface. One class, two
deployment modes, identical method signatures and return shapes — your agent
code never changes when the deployment does.

## Choose a mode

```python
from jaswolf import JaswolfMemoryProvider, JaswolfSettings

# Embedded: engine runs inside the agent process. Fastest, zero ops.
provider = await JaswolfMemoryProvider.embedded(
    settings=JaswolfSettings(database_url="sqlite:///./hermes_memory.db"),
    user_id="alice",
    agent_id="hermes-main",
)

# Remote: shared JASWOLF service (several agents / machines, one memory).
# .remote(...) is synchronous — do NOT await it; only its methods are async.
provider = JaswolfMemoryProvider.remote(
    base_url="http://localhost:8400", api_key="my-key", user_id="alice"
)
```

Embedded mode starts a background lifecycle sweeper (`auto_sweep=True`).
Call `await provider.close()` on shutdown.

## The three-line agent loop

```python
async def turn(history, user_input):
    history.append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    messages = await provider.inject_context(messages, token_budget=1200)   # 1

    reply = await hermes_inference(messages)                                # 2
    history.append({"role": "assistant", "content": reply})

    await provider.observe(history[-2:])                                    # 3
    return reply
```

1. **inject_context** retrieves relevant + pinned memories, budgets tokens,
   and appends the block to your system message. Returns the messages
   unchanged when nothing is worth injecting.
2. Your inference call, untouched.
3. **observe** extracts durable facts/preferences/goals from the new turns.
   Repeats reinforce existing memories instead of duplicating.

## Full method reference

Core interface (spec methods):

| Method | Purpose |
| --- | --- |
| `add_memory(content, memory_type=…)` | store one memory → `{"memory": …, "created": bool}` |
| `search_memory(query, top_k, mode, memory_types)` | scored results, best first |
| `get_memory(id)` | fetch or `None` |
| `update_memory(id, **fields)` | patch content/importance/state/… |
| `delete_memory(id, hard=False)` | soft delete by default |
| `build_context(messages=…/query=…, token_budget=…)` | LLM-ready block (`""` if empty) |
| `consolidate_memories(dry_run=False)` | merge near-duplicates |
| `health_check()` | engine health dict |

Agent-loop sugar:

| Method | Purpose |
| --- | --- |
| `observe(messages)` | post-turn extraction + storage |
| `remember(content)` | explicit save with importance floor 0.85 |
| `recall(query, top_k)` | just the remembered strings |
| `inject_context(messages, token_budget)` | messages with memory block in the system prompt |

## Patterns

**Explicit memory commands.** Wire "remember that …" tool calls (or detect
the phrase) to `provider.remember(...)` — it stores with high importance and
skips extraction heuristics.

**Session working memory.** Pass `session_id` and store scratch state as
`working` memories — they expire automatically (24h default) and only show
up in that session's context:

```python
await provider.add_memory("current task: fix the deploy pipeline",
                          memory_type="working", session_id="sess-42")
```

**Periodic hygiene.** Run consolidation occasionally (cron, or after long
sessions):

```python
report = await provider.consolidate_memories()
```

**Multiple humans, one agent.** Pass `user_id=` per call — the default is
just a convenience: `await provider.build_context(query=…, user_id="alice")`.

**Citing memories.** `build_context(..., format="xml")` or
`include_ids=True` (via the service/API) exposes memory ids so the agent can
reference or update specific memories.

## Self-hosted extraction with your own Hermes model

Rule-based extraction is the zero-cost default. To use your own inference
endpoint (vLLM/Ollama serving Hermes is OpenAI-compatible) for higher-recall
extraction:

```bash
export JASWOLF_LLM_BASE_URL=http://localhost:8000/v1
export JASWOLF_LLM_MODEL=Hermes-3-Llama-3.1-8B
export JASWOLF_EXTRACTION_STRATEGY=hybrid    # rules + LLM, deduplicated
```

Extraction failures degrade gracefully to rules — the agent never blocks on
the extractor.
