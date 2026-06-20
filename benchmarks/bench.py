"""JASWOLF benchmark: insert throughput, search/context latency percentiles.

    python benchmarks/bench.py --n 5000 --queries 100

Uses the hash embedder by default so results measure the engine, not the
embedding model. Pass --provider local/openai to include real embedding cost.
"""

import argparse
import asyncio
import random
import time

from jaswolf import JaswolfSettings, MemoryService
from jaswolf.models import ContextRequest, MemoryCreate, MemoryType, SearchQuery

TOPICS = [
    "deploys Hermes with Docker on a {} VPS",
    "prefers {} for backend development",
    "is planning to launch a {} product",
    "uses {} for the data pipeline",
    "debugged a {} outage yesterday",
    "wants to migrate the stack to {}",
    "runs PostgreSQL with {} extensions",
    "favorite tool this month is {}",
]
NOUNS = [
    "Python", "Rust", "Kubernetes", "Hetzner", "pgvector", "Redis", "FastAPI",
    "LangGraph", "vLLM", "Grafana", "Prometheus", "SQLite", "Postgres", "Ollama",
]
TYPES = list(MemoryType)


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, int(len(values) * p))
    return values[idx]


async def run(n: int, queries: int, provider: str, db: str) -> None:
    settings = JaswolfSettings(
        database_url=db,
        embedding_provider=provider,
        log_level="WARNING",
        dedup_threshold=1.01,  # measure raw insert path, not dedup short-circuits
    )
    service = await MemoryService.create(settings)
    rng = random.Random(42)

    print(f"backend={service.storage.name} embedder={service.embedder.name} n={n}")

    # -- inserts ----------------------------------------------------------
    payloads = [
        MemoryCreate(
            user_id=f"user{i % 20}",
            content=f"User {rng.choice(TOPICS).format(rng.choice(NOUNS))} #{i}",
            memory_type=rng.choice(TYPES),
        )
        for i in range(n)
    ]
    insert_times: list[float] = []
    start = time.perf_counter()
    for payload in payloads:
        t0 = time.perf_counter()
        await service.add(payload)
        insert_times.append((time.perf_counter() - t0) * 1000)
    wall = time.perf_counter() - start
    print(
        f"insert: {n / wall:8.0f} ops/s   "
        f"p50 {pct(insert_times, 0.50):6.2f} ms   p95 {pct(insert_times, 0.95):6.2f} ms   "
        f"p99 {pct(insert_times, 0.99):6.2f} ms"
    )

    # -- search ------------------------------------------------------------
    search_times: list[float] = []
    for i in range(queries):
        query = SearchQuery(
            user_id=f"user{i % 20}",
            query=f"{rng.choice(NOUNS)} {rng.choice(['deployment', 'preference', 'plans'])}",
            top_k=8,
            record_access=False,
        )
        t0 = time.perf_counter()
        await service.search(query)
        search_times.append((time.perf_counter() - t0) * 1000)
    print(
        f"search: {' ' * 8}           "
        f"p50 {pct(search_times, 0.50):6.2f} ms   p95 {pct(search_times, 0.95):6.2f} ms   "
        f"p99 {pct(search_times, 0.99):6.2f} ms"
    )

    # -- context build -------------------------------------------------------
    context_times: list[float] = []
    for i in range(max(10, queries // 2)):
        request = ContextRequest(
            user_id=f"user{i % 20}",
            query=f"help me with {rng.choice(NOUNS)}",
            token_budget=1500,
        )
        t0 = time.perf_counter()
        await service.build_context(request)
        context_times.append((time.perf_counter() - t0) * 1000)
    print(
        f"context:{' ' * 8}           "
        f"p50 {pct(context_times, 0.50):6.2f} ms   p95 {pct(context_times, 0.95):6.2f} ms   "
        f"p99 {pct(context_times, 0.99):6.2f} ms"
    )

    stats = await service.stats()
    print(f"memories stored: {stats['total']}")
    await service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--provider", default="hash")
    parser.add_argument("--db", default="sqlite:///./bench.db")
    args = parser.parse_args()
    asyncio.run(run(args.n, args.queries, args.provider, args.db))
