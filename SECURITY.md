# JASWOLF Security

Agent memory is intimate data: preferences, relationships, operational
details, corrections. Treat the store like a credentials vault, not a cache.

## Authentication & tenancy

* The HTTP API **refuses to start** without `JASWOLF_API_KEYS` unless
  `JASWOLF_DEV_OPEN_MODE=true` is set explicitly (local development only).
* Keys map to tenants (`key:tenant,…`); every query is scoped by tenant_id.
  Keys are compared in constant time. One tenant per trust boundary.
* Per-key rate limiting (`JASWOLF_RATE_LIMIT_PER_MINUTE`, default 600).
* Embedded mode has no network surface and is unaffected by API auth.

## Network exposure

* `docker-compose.yml` binds JASWOLF (8400), Prometheus (9090), and Grafana
  (3000) to **127.0.0.1 only**. For remote access, front with a reverse
  proxy (TLS) or a private network (Tailscale/WireGuard) — do not rebind to
  0.0.0.0 on a public VPS.
* CORS is **disabled by default** (`JASWOLF_CORS_ORIGINS` empty). Set explicit
  origins if a browser dashboard needs access; never `*` in production.

## Secrets

* Compose requires `POSTGRES_PASSWORD` (and `GRAFANA_PASSWORD` for the
  monitoring profile) to be set in `.env` — there are no default passwords.
* `.env` is gitignored. `jaswolf diagnose` redacts credentials in URLs; never
  paste raw `.env` contents into issues or chats.
* Audit trail: every memory access is logged to `memory_access_logs`.

## Data protection

* Soft-deleted and superseded memories remain in the database for
  auditability. For true data-removal requests use hard delete
  (`DELETE /v1/memories/{id}?hard=true`) and note that content may persist
  in `memory_versions` — purge those rows too for full erasure.
* Back up `jaswolf.db` / Postgres dumps with the same care as the live store;
  encrypt backups at rest if the disk isn't encrypted.

## Reporting a vulnerability

Open a private GitHub issue on `iamalice5555/jaswolf` with the `security`
label, or follow [docs/BUG_REPORTS.md](docs/BUG_REPORTS.md) marking impact
as `blocking`. Include the minimal repro, never live memory contents.
