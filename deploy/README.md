# Running JASWOLF MCP as a managed local service

Goal: the JASWOLF MCP memory server starts **before** Hermes, **survives
reboots**, **self-heals** on crash *and* hang, and is **observable**.

## Files

| File | Role |
| --- | --- |
| `jaswolf-mcp.service` | the long-lived, prewarmed MCP server; `Restart=always` |
| `jaswolf-mcp-health.service` | watchdog: probes `/healthz`, restarts if down/hung |
| `jaswolf-mcp-health.timer` | runs the watchdog every 2 min |

## Install (system service)

```bash
sudo cp deploy/jaswolf-mcp.service deploy/jaswolf-mcp-health.service \
        deploy/jaswolf-mcp-health.timer /etc/systemd/system/
# edit jaswolf-mcp.service: User, WorkingDirectory, the venv path, and JASWOLF_* env
sudo systemctl daemon-reload
sudo systemctl enable --now jaswolf-mcp.service
sudo systemctl enable --now jaswolf-mcp-health.timer
```

## Start before Hermes (boot ordering)

Add to the **Hermes** systemd unit so Hermes never starts before its memory
is up:

```ini
[Unit]
After=jaswolf-mcp.service
Wants=jaswolf-mcp.service
```

For an even stricter gate, have Hermes' `ExecStartPre` block until healthy:

```ini
ExecStartPre=/home/jaswolf/.hermes/repos/jaswolf/.venv/bin/jaswolf mcp-health
```

## What happens if JASWOLF crashes or hangs?  (honest failure semantics)

JASWOLF cannot make Hermes "fall back to local memory" — fallback is Hermes'
behavior, and since you moved off Mem0 there is no second store to fall back
to. So robustness = (1) rarely down, (2) fail fast not hung, (3) observable:

- **Crash / clean exit** → `Restart=always` brings it back in ~3 s, prewarmed.
  During that gap, memory calls fail.
- **Hang** (process alive, unresponsive) → `Restart=always` would NOT catch
  it; the health timer does: `mcp-health` fails after its timeout and restarts
  the unit within ~2 min.
- **During any gap, what Hermes does is Hermes' choice.** Make it safe:
  - Give the Hermes→MCP client a **short timeout** so a slow/hung server fails
    fast instead of blocking the turn.
  - Configure Hermes to **degrade, not crash**: on a memory error, answer from
    the current conversation without the long-term block. A missing memory
    block should never abort a turn.
  - There is **no automatic Mem0 fallback** by design. If you want one, the
    only route is to keep Mem0 frozen and use a Hermes provider-fallback chain
    — that contradicts "move off Mem0", so the recommended posture is
    degrade-without-memory + fast auto-restart.

Net: a JASWOLF outage means "this turn has no long-term memory," self-healing in
seconds (crash) or ~2 min (hang) — never a corrupted store (WAL + the
fingerprint/integrity guards), never silent (below).

## How to find out if JASWOLF has a problem

```bash
systemctl status jaswolf-mcp                 # running? last restart? exit code?
journalctl -u jaswolf-mcp -n 100 --no-pager  # logs (prewarm, errors, restarts)
jaswolf mcp-health                           # 0=ok, 1=down/degraded; prints health JSON
curl -s localhost:8765/healthz | jq       # 200 ok / 503 starting|degraded|error
```

- `/healthz` and `mcp-health` go **503/exit-1** on: still starting, storage
  unreachable, **integrity (quick_check) failure**, or **embedder fallback**
  (silently degraded retrieval) — the conditions you'd want to know about.
- Keep the deterministic gate on a cron for quality drift:
  `jaswolf eval-shadow --probes … --user-id … --db … --embedding-provider local`
  (exit code + `GO_PILOT/CONTINUE_SHADOW/NO_GO`).
- For push alerts, wrap `mcp-health`/`eval-shadow` exit codes in whatever
  notifier Hermes already uses (e.g. the Telegram path).

## Notes

- `/healthz` exists on the **HTTP** transport (`--transport http`). stdio
  servers are supervised directly by whatever launches them.
- Running rootless instead? Use `systemctl --user` + `loginctl enable-linger
  jaswolf` so the user service starts at boot without a login session.
