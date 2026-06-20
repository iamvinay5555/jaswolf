# JASWOLF REST API Reference

Base URL: `http://host:8400`. All `/v1/*` endpoints require
`Authorization: Bearer <key>` (or `X-API-Key: <key>`) unless the server runs
in open mode. Interactive docs: `/docs` (Swagger UI).

## POST /v1/memories — store one memory

```json
{
  "user_id": "alice",
  "content": "User prefers Python for backend work",
  "memory_type": "preference",      // working|episodic|semantic|preference|procedural|goal|relationship
  "importance": 0.9,                 // optional; auto-scored if omitted
  "namespace": "default",
  "session_id": "sess-42",           // optional
  "metadata": {"source": "chat"},
  "ttl_hours": 24                    // optional; working memories default to 24h
}
```

`201` →

```json
{"memory": {"id": "…", "content": "…", "importance": 0.9, …}, "created": true}
```

`created: false` means the content duplicated an existing memory, which was
**reinforced** instead (its id is returned).

## POST /v1/memories/extract — extract & store from conversation

```json
{"user_id": "alice", "text": "I love Python. Sarah is my cofounder."}
// or: {"user_id": "alice", "messages": [{"role": "user", "content": "…"}]}
```

`200` → `{"extracted": 2, "results": [{"memory": …, "created": true}, …]}`

## GET /v1/memories/{id} — fetch (add `?include_embedding=true` for the vector)
## GET /v1/memories/{id}/versions — content history (updates, consolidations)
## PATCH /v1/memories/{id} — partial update

```json
{"content": "…", "importance": 0.7, "state": "active", "metadata": {…}}
```

Content changes are versioned and re-embedded automatically.

## DELETE /v1/memories/{id} — soft delete (`?hard=true` to purge the row)

## POST /v1/memories/search

```json
{
  "user_id": "alice",
  "query": "docker deployment",
  "mode": "hybrid",            // semantic|keyword|hybrid|recency|importance
  "top_k": 8,
  "memory_types": ["semantic", "procedural"],   // optional filter
  "min_importance": 0.3,                          // optional
  "min_score": 0.4,                               // optional final-score floor
  "record_access": true
}
```

`200` →

```json
{
  "results": [
    {"memory": {…}, "relevance": 0.91, "recency": 0.84, "frequency": 0.2, "final_score": 0.78}
  ],
  "count": 1,
  "latency_ms": 6.3
}
```

`relevance` is normalized within the result set (ranking signal, not an
absolute similarity).

## POST /v1/memories/context — build an LLM-ready memory block

```json
{
  "user_id": "alice",
  "messages": [{"role": "user", "content": "help me plan the launch"}],
  "token_budget": 1500,
  "format": "markdown",        // or "xml"
  "include_ids": false
}
```

`200` →

```json
{
  "text": "# What I remember about this user\n\n## Preferences\n- …",
  "token_estimate": 412,
  "token_budget": 1500,
  "truncated": false,
  "sections": [{"title": "Preferences", "memory_type": "preference", "count": 3, "tokens": 88}],
  "memory_ids": ["…"],
  "latency_ms": 21.0
}
```

`text` is `""` when nothing relevant is stored — skip injection then.

## POST /v1/memories/consolidate — merge near-duplicates

```json
{"user_id": "alice", "dry_run": true}
```

`200` → `{"examined": 132, "clusters_found": 3, "memories_merged": 4, "merges": [...], "dry_run": true}`

## POST /v1/maintenance/sweep — run one lifecycle sweep
## GET /v1/stats — counts by state/type (`?user_id=` to narrow)
## GET /health — engine health (no auth)
## GET /metrics — Prometheus exposition (no auth)

## Errors

`401` missing/invalid key · `404` memory not found (or other tenant) ·
`422` validation · `429` rate limited · `500` logged server error.
Body: `{"detail": "…"}`.
