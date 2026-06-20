import httpx
import pytest

from jaswolf.api.app import create_app
from jaswolf.service import MemoryService


@pytest.fixture
async def client(settings, service):
    app = create_app(settings=settings, service=service)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_crud_flow(client):
    created = await client.post(
        "/v1/memories",
        json={"user_id": "alice", "content": "User prefers Python", "memory_type": "preference"},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["created"] is True
    memory_id = body["memory"]["id"]
    assert body["memory"]["importance"] > 0.5
    assert "embedding" not in body["memory"] or body["memory"]["embedding"] is None

    got = await client.get(f"/v1/memories/{memory_id}")
    assert got.status_code == 200
    assert got.json()["content"] == "User prefers Python"

    patched = await client.patch(f"/v1/memories/{memory_id}", json={"importance": 0.99})
    assert patched.status_code == 200
    assert patched.json()["importance"] == pytest.approx(0.99)

    deleted = await client.delete(f"/v1/memories/{memory_id}")
    assert deleted.status_code == 204

    versions = await client.get(f"/v1/memories/{memory_id}/versions")
    assert versions.status_code == 200  # soft-deleted rows remain addressable

    missing = await client.get("/v1/memories/00000000-0000-0000-0000-000000000000")
    assert missing.status_code == 404


async def test_duplicate_create_reports_reinforcement(client):
    payload = {"user_id": "alice", "content": "User runs Hermes on a VPS"}
    first = await client.post("/v1/memories", json=payload)
    second = await client.post("/v1/memories", json=payload)
    assert first.json()["created"] is True
    assert second.json()["created"] is False
    assert second.json()["memory"]["id"] == first.json()["memory"]["id"]


async def test_extract_endpoint(client):
    resp = await client.post(
        "/v1/memories/extract",
        json={"user_id": "alice", "text": "I love Python. Sarah is my cofounder."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["extracted"] == 2

    empty = await client.post("/v1/memories/extract", json={"user_id": "alice"})
    assert empty.status_code == 422


async def test_search_endpoint(client):
    await client.post(
        "/v1/memories",
        json={"user_id": "alice", "content": "User deploys with Docker Compose on a Hetzner VPS"},
    )
    await client.post(
        "/v1/memories", json={"user_id": "alice", "content": "User drinks oolong tea"}
    )
    resp = await client.post(
        "/v1/memories/search",
        json={"user_id": "alice", "query": "docker deployment", "top_k": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1
    assert body["results"][0]["memory"]["content"].startswith("User deploys")
    assert body["results"][0]["final_score"] > 0
    assert body["latency_ms"] >= 0


async def test_context_endpoint(client):
    await client.post(
        "/v1/memories",
        json={
            "user_id": "alice",
            "content": "User prefers concise answers",
            "memory_type": "preference",
            "importance": 0.9,  # identity-grade => force-pinned (tiered pinning)
        },
    )
    resp = await client.post(
        "/v1/memories/context",
        json={"user_id": "alice", "query": "how should I reply?", "token_budget": 500},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "concise answers" in body["text"]
    assert body["token_estimate"] <= 500
    assert body["memory_ids"]


async def test_consolidate_endpoint(client):
    resp = await client.post("/v1/memories/consolidate", json={"user_id": "alice", "dry_run": True})
    assert resp.status_code == 200
    assert resp.json()["dry_run"] is True


async def test_health_and_metrics(client):
    health = await client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    metrics = await client.get("/metrics")
    assert metrics.status_code == 200
    stats = await client.get("/v1/stats")
    assert stats.status_code == 200
    sweep = await client.post("/v1/maintenance/sweep")
    assert sweep.status_code == 200


# ---- auth & tenancy ----------------------------------------------------------


@pytest.fixture
async def secured(settings, tmp_path):
    settings = settings.model_copy(
        update={
            "api_keys": "key-alpha:tenant_a,key-beta:tenant_b",
            "rate_limit_per_minute": 0,
            "database_url": f"sqlite:///{tmp_path}/secured.db",
        }
    )
    service = await MemoryService.create(settings)
    app = create_app(settings=settings, service=service)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await service.close()


async def test_auth_required(secured):
    resp = await secured.post("/v1/memories", json={"user_id": "x", "content": "y"})
    assert resp.status_code == 401
    resp = await secured.post(
        "/v1/memories",
        json={"user_id": "x", "content": "y"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401


async def test_tenant_isolation(secured):
    created = await secured.post(
        "/v1/memories",
        json={"user_id": "x", "content": "tenant A's secret"},
        headers={"Authorization": "Bearer key-alpha"},
    )
    assert created.status_code == 201
    memory_id = created.json()["memory"]["id"]

    own = await secured.get(
        f"/v1/memories/{memory_id}", headers={"Authorization": "Bearer key-alpha"}
    )
    assert own.status_code == 200
    other = await secured.get(
        f"/v1/memories/{memory_id}", headers={"Authorization": "Bearer key-beta"}
    )
    assert other.status_code == 404

    search = await secured.post(
        "/v1/memories/search",
        json={"user_id": "x", "query": "secret"},
        headers={"X-API-Key": "key-beta"},
    )
    assert search.json()["count"] == 0


async def test_rate_limit(settings, tmp_path):
    settings = settings.model_copy(
        update={
            "api_keys": "limited-key",
            "rate_limit_per_minute": 3,
            "database_url": f"sqlite:///{tmp_path}/ratelimit.db",
        }
    )
    service = await MemoryService.create(settings)
    app = create_app(settings=settings, service=service)
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": "Bearer limited-key"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        statuses = [(await c.get("/v1/stats", headers=headers)).status_code for _ in range(5)]
    await service.close()
    assert statuses[:3] == [200, 200, 200]
    assert 429 in statuses[3:]
