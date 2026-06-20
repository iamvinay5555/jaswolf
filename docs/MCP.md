# JASWOLF as an MCP memory provider

`jaswolf mcp` serves the memory engine over the Model Context Protocol, so
Hermes uses JASWOLF as its memory provider the same way it would use
mempalace/agentmemory — through the stable MCP tool surface, not a bespoke
plugin against Hermes internals. This is the supported cutover path.

## Install

```bash
pip install "jaswolf[mcp] @ git+https://github.com/iamalice5555/jaswolf.git"
# CPU-only VPS: install CPU torch first (see INSTALL.md) before local-embeddings
```

## Run it (long-lived and warm)

A memory server must be **one long-running, prewarmed process** — cold start
is seconds on CPU, and you never want a live turn to pay it. `jaswolf mcp`
prewarms the embedder at startup automatically.

```bash
# stdio: the host (Hermes) launches and supervises the process
JASWOLF_DATABASE_URL=sqlite:////home/jaswolf/.hermes/jaswolf_shadow/data/jaswolf_shadow_bge_small.db \
JASWOLF_EMBEDDING_PROVIDER=local \
JASWOLF_MCP_USER_ID=alice \
jaswolf mcp

# streamable-HTTP: one persistent server, Hermes connects as a client
jaswolf mcp --transport http --host 127.0.0.1 --port 8765
```

Prefer **HTTP for production** if Hermes would otherwise spawn a stdio
subprocess per session — a persistent HTTP server stays warm across sessions;
per-session stdio reloads the model each time.

### Run it as a managed service (boot-persistent, self-healing, observable)

For a real deployment — starts before Hermes, survives reboots, auto-restarts
on crash *and* hang, and is monitorable — use the systemd units and the honest
failure-semantics guide in **[deploy/README.md](../deploy/README.md)**. In
short:

- `jaswolf-mcp.service` runs the HTTP server prewarmed with `Restart=always`.
- `jaswolf-mcp-health.timer` probes `/healthz` every 2 min and restarts a hung
  server (which `Restart=always` alone can't catch).
- `GET /healthz` → 200 only when fully ok; 503 while starting / degraded /
  on integrity or embedder-fallback. `jaswolf mcp-health` wraps it for cron and
  the watchdog (exit 0=ok, 1=down/degraded).
- Health, on crash/hang, and "does it fall back?" are answered in detail in
  deploy/README.md — summary: JASWOLF self-heals in seconds (crash) or ~2 min
  (hang); during a gap a turn has no long-term memory; make Hermes degrade,
  not crash (short client timeout). No automatic Mem0 fallback by design.

## Tools exposed

| Tool | Purpose | Integration style |
| --- | --- | --- |
| `build_memory_context(query?, recent_messages?, token_budget?)` | the memory block to prepend to the system prompt | host-driven |
| `record_conversation(messages)` | post-turn extract/store (gated, supersession-aware) | host-driven |
| `recall(query, limit?)` | look up remembered statements | model-driven |
| `remember(content, memory_type?, importance?)` | explicit durable store | model-driven |
| `search_memory(query, top_k?, mode?)` | scored search | model-driven |
| `forget(memory_id, hard?)` | delete | model-driven |
| `memory_health()` | provider/fingerprint/fallback/status | ops |

The surface supports both styles on purpose (see below).

## ⚠️ The one thing to confirm before cutover: how Hermes consumes MCP memory

This is the open integration question — answer it from the box before relying
on JASWOLF for a live turn. There are two models and they behave very
differently:

1. **Host-driven memory provider.** Hermes has a memory-provider abstraction
   that, each turn, automatically injects context and records the exchange.
   JASWOLF maps perfectly: wire Hermes to call `build_memory_context` pre-turn
   and `record_conversation` post-turn. **Reliable** — memory works without
   the agent LLM choosing to use it. This is what you want.
2. **Model-driven tools.** Hermes just exposes the MCP tools to its LLM and
   the model calls `recall`/`remember` when it decides to. **Less reliable** —
   recall quality now depends on the model remembering to call the tool every
   time, and on prompt discipline.

**Action:** look at how the mempalace/agentmemory MCP servers are registered
in the Hermes config on the box (the config block that points Hermes at an
MCP memory server). Copy that exact shape to register `jaswolf-memory`. If that
mechanism is host-driven (model 1), JASWOLF is a drop-in. If it is model-driven
(model 2), expect to add a system-prompt instruction telling the agent to
call `build_memory_context`/`recall` every turn, and treat the first day as
prompt-tuning, not just memory evaluation.

Until that config shape is confirmed and a smoke test passes, the cutover is
not real — the same integration-reality gap that blocked the bespoke-plugin
plan.

## Safe cutover sequence (no dual-write)

You want JASWOLF primary and Mem0 gone. That does **not** mean deleting Mem0
tonight — it means a clean switch with a working escape hatch:

1. **Migrate first.** Export existing Mem0 memories and import them into the
   JASWOLF DB *before* cutover, or the agent starts amnesiac. Type them where
   possible (identity-grade facts → `preference`/`goal`) so pinning works —
   the current corpus is ~all `semantic`, which is a real gap (see HANDOFF).
2. **Freeze, don't delete, Mem0.** Keep the Mem0 store intact and read-only as
   the rollback target. This is not dual-write — nothing writes to Mem0 during
   the pilot — it just means "flip the provider config back" is a real,
   data-complete rollback if something breaks.
3. **Pre-cutover gate** (must all hold): `jaswolf eval-shadow` verdict
   `GO_PILOT`; `memory_health()` over the MCP server returns `status: ok`,
   `fallback: false`, correct fingerprint; the Hermes→JASWOLF MCP smoke test
   passes (below).
4. **Smoke test the live path** (not the shadow DB): through Hermes, store a
   fact → restart the JASWOLF MCP server → recall it (proves persistence +
   warm restart); ask one off-topic question → confirm no irrelevant memory
   is injected; confirm context is non-empty for the real user_id (the #1
   silent integration bug).
5. **Cut over**, watch the first hour attended, keep the eval cron running
   hourly, and keep the kill switch (config flip back to Mem0) one command
   away.

## Rollback triggers (in addition to the human-feel ones)

Automate these — an overnight window can't rely on a human reading every
reply:

- `memory_health()` `fallback` becomes true, or `status != ok`
- empty context returned for N consecutive turns (silent-memory failure)
- MCP tool error rate / exceptions above a small threshold
- `eval-shadow` cron drops below `GO_PILOT`
- sqlite quick_check failure or DB growth/corruption
- process restart storms (repeated cold starts → OOM signal)
