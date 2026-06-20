"""REST API routes (v1)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from ..models import (
    ChatMessage,
    ConsolidationReport,
    ContextRequest,
    MemoryCreate,
    MemoryNotFound,
    MemoryUpdate,
    SearchQuery,
    SweepReport,
)
from ..service import MemoryService
from . import metrics
from .auth import authenticate
from .schemas import (
    ConsolidateIn,
    ContextIn,
    ContextResponse,
    CreateMemoryResponse,
    ExtractIn,
    ExtractResponse,
    MemoryIn,
    MemoryOut,
    MemoryPatch,
    ScoredMemoryOut,
    SearchIn,
    SearchResponse,
)

router = APIRouter(prefix="/v1")


def get_service(request: Request) -> MemoryService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="service not initialized")
    return service


@router.post("/memories", response_model=CreateMemoryResponse, status_code=201)
async def create_memory(
    body: MemoryIn,
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    memory, created = await service.add(MemoryCreate(**body.model_dump()), tenant_id=tenant)
    if created:
        metrics.MEMORIES_CREATED.labels(memory_type=memory.memory_type.value).inc()
    else:
        metrics.MEMORIES_REINFORCED.inc()
    return CreateMemoryResponse(memory=MemoryOut.from_memory(memory), created=created)


@router.post("/memories/extract", response_model=ExtractResponse)
async def extract_memories(
    body: ExtractIn,
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    if not body.text and not body.messages:
        raise HTTPException(status_code=422, detail="provide text or messages")
    messages = body.messages or [ChatMessage(role="user", content=body.text or "")]
    results = await service.ingest_messages(
        user_id=body.user_id,
        messages=messages,
        agent_id=body.agent_id,
        session_id=body.session_id,
        namespace=body.namespace,
        tenant_id=tenant,
    )
    for memory, created in results:
        if created:
            metrics.MEMORIES_CREATED.labels(memory_type=memory.memory_type.value).inc()
        else:
            metrics.MEMORIES_REINFORCED.inc()
    return ExtractResponse(
        extracted=len(results),
        results=[
            CreateMemoryResponse(memory=MemoryOut.from_memory(m), created=c) for m, c in results
        ],
    )


@router.get("/memories/{memory_id}", response_model=MemoryOut)
async def get_memory(
    memory_id: str,
    include_embedding: bool = Query(default=False),
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    try:
        memory = await service.get(memory_id, tenant_id=tenant)
    except MemoryNotFound:
        raise HTTPException(status_code=404, detail="memory not found")
    return MemoryOut.from_memory(memory, include_embedding=include_embedding)


@router.get("/memories/{memory_id}/versions")
async def get_memory_versions(
    memory_id: str,
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    try:
        return {"memory_id": memory_id, "versions": await service.get_versions(memory_id, tenant)}
    except MemoryNotFound:
        raise HTTPException(status_code=404, detail="memory not found")


@router.patch("/memories/{memory_id}", response_model=MemoryOut)
async def update_memory(
    memory_id: str,
    body: MemoryPatch,
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    try:
        memory = await service.update(
            memory_id, MemoryUpdate(**body.model_dump(exclude_unset=True)), tenant_id=tenant
        )
    except MemoryNotFound:
        raise HTTPException(status_code=404, detail="memory not found")
    return MemoryOut.from_memory(memory)


@router.delete("/memories/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: str,
    hard: bool = Query(default=False),
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    try:
        await service.delete(memory_id, tenant_id=tenant, hard=hard)
    except MemoryNotFound:
        raise HTTPException(status_code=404, detail="memory not found")
    return Response(status_code=204)


@router.post("/memories/search", response_model=SearchResponse)
async def search_memories(
    body: SearchIn,
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    query = SearchQuery(**body.model_dump(exclude={"include_embeddings"}))
    results = await service.search(query, tenant_id=tenant)
    latency = service.retrieval.last_latency_ms
    metrics.SEARCH_LATENCY.observe(latency / 1000)
    return SearchResponse(
        results=[ScoredMemoryOut.from_scored(s, body.include_embeddings) for s in results],
        count=len(results),
        latency_ms=round(latency, 2),
    )


@router.post("/memories/context", response_model=ContextResponse)
async def build_context(
    body: ContextIn,
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    result = await service.build_context(ContextRequest(**body.model_dump()), tenant_id=tenant)
    latency = service.context.last_latency_ms
    metrics.CONTEXT_LATENCY.observe(latency / 1000)
    return ContextResponse(
        text=result.text,
        token_estimate=result.token_estimate,
        token_budget=result.token_budget,
        truncated=result.truncated,
        sections=result.sections,
        memory_ids=[s.memory.id for s in result.memories],
        latency_ms=round(latency, 2),
    )


@router.post("/memories/consolidate", response_model=ConsolidationReport)
async def consolidate_memories(
    body: ConsolidateIn,
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    return await service.consolidate(
        user_id=body.user_id,
        tenant_id=tenant,
        namespace=body.namespace,
        memory_types=body.memory_types,
        dry_run=body.dry_run,
    )


@router.post("/maintenance/sweep", response_model=SweepReport)
async def run_sweep(
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    return await service.sweep()


@router.get("/stats")
async def get_stats(
    user_id: str | None = Query(default=None),
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    return await service.stats(tenant_id=tenant, user_id=user_id)
