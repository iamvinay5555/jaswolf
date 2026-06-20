"""Retrieval & context quality evaluation harness.

Measures the quality metrics from jasmine_feedback.md against a labeled
golden dataset: Recall@1, Recall@5, MRR, irrelevant-injection rate,
superseded-fact injection, and context precision/budget compliance.

    python benchmarks/eval_retrieval.py                     # hash embedder (deterministic smoke)
    python benchmarks/eval_retrieval.py --provider local    # bge-small (realistic quality)
    python benchmarks/eval_retrieval.py --min-recall5 0.8   # exit 1 below threshold (CI gate)

Numbers with the hash embedder are a lower bound for engine sanity only —
judge retrieval quality with --provider local or openai. Extend GOLDEN_*
with real (sanitized) workload examples over time; the harness scales with
the dataset.
"""

import argparse
import asyncio
import sys

from jaswolf import JaswolfSettings, MemoryService
from jaswolf.models import ContextRequest, MemoryCreate, SearchQuery

USER = "eval-user"

# (key, content, memory_type, importance)
GOLDEN_MEMORIES = [
    ("py",       "User prefers Python for backend development", "preference", 0.9),
    ("concise",  "User prefers concise answers without filler", "preference", 0.85),
    ("dark",     "User prefers dark mode in every editor", "preference", 0.7),
    ("saas",     "User wants to launch a SaaS product by December", "goal", 0.85),
    ("fitness",  "User wants to run a half marathon next year", "goal", 0.7),
    ("sarah",    "Sarah is user's cofounder", "relationship", 0.7),
    ("k8s",      "User's company runs Kubernetes on Hetzner", "semantic", 0.6),
    ("vps",      "User's Hermes agent runs on a 4GB VPS in Singapore", "semantic", 0.6),
    ("tg",       "User's Hermes agent is connected to Telegram for notifications", "semantic", 0.6),
    ("pg",       "User stores agent memory in PostgreSQL with pgvector", "semantic", 0.6),
    ("deploy",   "To deploy Hermes: build the Docker image, push to registry, run docker compose up on the VPS", "procedural", 0.7),
    ("backup",   "To back up the memory DB: run pg_dump nightly and sync to object storage", "procedural", 0.7),
    ("cricket",  "User follows cricket and checks scores during India matches", "preference", 0.6),
    ("toto",     "User runs a TOTO and football prediction system called JASX", "semantic", 0.7),
    ("mrt",      "User commutes on the MRT East West line", "semantic", 0.5),
    ("coffee0",  "User prefers tea in the morning", "preference", 0.7),  # superseded below
]

# query -> keys of memories that SHOULD be retrieved
GOLDEN_QUERIES = [
    ("what language should we use for the backend service?", ["py"]),
    ("how should I phrase replies to the user?", ["concise"]),
    ("what is the user trying to build this year?", ["saas"]),
    ("who works with the user on the startup?", ["sarah"]),
    ("what infrastructure does the user's company run?", ["k8s"]),
    ("where does the Hermes agent run?", ["vps"]),
    ("how do I deploy the Hermes agent?", ["deploy"]),
    ("how do we back up the memory database?", ["backup"]),
    ("which database stores agent memories?", ["pg"]),
    ("what sports does the user follow?", ["cricket"]),
    ("what is JASX?", ["toto"]),
    ("what does the user drink in the morning?", ["coffee1"]),  # post-supersession truth
]

# queries with NO relevant memory: anything returned above min_score is noise
NEGATIVE_QUERIES = [
    "what is the user's favorite renaissance painter?",
    "which scuba diving certification does the user hold?",
]

SUPERSESSION_INPUT = "Actually I prefer coffee in the morning now."  # archives coffee0


def mrr(ranks: list[int | None]) -> float:
    return sum(1.0 / r for r in ranks if r) / max(1, len(ranks))


async def run(provider: str, min_recall5: float | None, budget: int) -> int:
    settings = JaswolfSettings(
        database_url="sqlite://",  # in-memory
        embedding_provider=provider,
        log_level="ERROR",
    )
    service = await MemoryService.create(settings)
    key_to_id: dict[str, str] = {}
    try:
        for key, content, mtype, importance in GOLDEN_MEMORIES:
            memory, _ = await service.add(
                MemoryCreate(user_id=USER, content=content, memory_type=mtype, importance=importance)
            )
            key_to_id[key] = memory.id

        # apply the supersession correction; current truth becomes coffee1
        results = await service.ingest_text(USER, SUPERSESSION_INPUT)
        key_to_id["coffee1"] = results[0][0].id
        superseded_id = key_to_id["coffee0"]

        ranks: list[int | None] = []
        hits1 = hits5 = superseded_hits = irrelevant_at5 = considered_at5 = 0
        for query, relevant_keys in GOLDEN_QUERIES:
            relevant_ids = {key_to_id[k] for k in relevant_keys}
            scored = await service.search(
                SearchQuery(user_id=USER, query=query, top_k=5, record_access=False)
            )
            ids = [s.memory.id for s in scored]
            rank = next((i + 1 for i, mid in enumerate(ids) if mid in relevant_ids), None)
            ranks.append(rank)
            hits1 += 1 if rank == 1 else 0
            hits5 += 1 if rank is not None else 0
            superseded_hits += sum(1 for mid in ids if mid == superseded_id)
            irrelevant_at5 += sum(1 for mid in ids if mid not in relevant_ids)
            considered_at5 += len(ids)

        negative_noise = 0
        for query in NEGATIVE_QUERIES:
            scored = await service.search(
                SearchQuery(user_id=USER, query=query, top_k=5, min_score=0.55, record_access=False)
            )
            negative_noise += len(scored)

        budget_violations = 0
        context_relevant = context_total = 0
        for query, relevant_keys in GOLDEN_QUERIES:
            relevant_ids = {key_to_id[k] for k in relevant_keys}
            ctx = await service.build_context(
                ContextRequest(user_id=USER, query=query, token_budget=budget)
            )
            if ctx.token_estimate > budget:
                budget_violations += 1
            superseded_hits += sum(1 for s in ctx.memories if s.memory.id == superseded_id)
            context_total += len(ctx.memories)
            context_relevant += sum(1 for s in ctx.memories if s.memory.id in relevant_ids)

        n = len(GOLDEN_QUERIES)
        recall1, recall5 = hits1 / n, hits5 / n
        print(f"provider={service.embedder.name}  memories={len(key_to_id)}  queries={n}")
        print(f"Recall@1                  {recall1:6.2f}")
        print(f"Recall@5                  {recall5:6.2f}")
        print(f"MRR                       {mrr(ranks):6.2f}")
        print(f"irrelevant@5 rate         {irrelevant_at5 / max(1, considered_at5):6.2f}")
        print(f"superseded injections     {superseded_hits:6d}   (gate: must be 0)")
        print(f"negative-query noise      {negative_noise:6d}   (results above min_score on unrelated queries)")
        print(f"context precision         {context_relevant / max(1, context_total):6.2f}   (pinned identity memories count as 'extra')")
        print(f"budget violations         {budget_violations:6d}   (gate: must be 0)")
        if provider == "hash":
            print("note: hash embedder = engine sanity lower bound; use --provider local for quality")

        failed = superseded_hits > 0 or budget_violations > 0
        if min_recall5 is not None and recall5 < min_recall5:
            print(f"FAIL: Recall@5 {recall5:.2f} < required {min_recall5:.2f}")
            failed = True
        return 1 if failed else 0
    finally:
        await service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="hash", choices=["hash", "local", "openai", "auto"])
    parser.add_argument("--min-recall5", type=float, default=None)
    parser.add_argument("--budget", type=int, default=900)
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.provider, args.min_recall5, args.budget)))
