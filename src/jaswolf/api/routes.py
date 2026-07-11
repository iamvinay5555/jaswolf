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
    ConversationMessageOut,
    ConversationSearchIn,
    ConversationSearchResponse,
    CreateMemoryResponse,
    ExplainResponse,
    ExtractIn,
    ExtractResponse,
    MemoryIn,
    MemoryOut,
    MemoryPatch,
    PersonaIn,
    PersonaResponse,
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
    try:
        memory, created = await service.add(MemoryCreate(**body.model_dump()), tenant_id=tenant)
    except ValueError as exc:  # e.g. taste capture missing explicit_signal/why_useful
        raise HTTPException(status_code=422, detail=str(exc))
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


@router.get("/memories/{memory_id}/explain", response_model=ExplainResponse)
async def explain_memory(
    memory_id: str,
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    """Provenance drill-down: memory + versions + relationships + source turns."""
    try:
        explanation = await service.explain(memory_id, tenant_id=tenant)
    except MemoryNotFound:
        raise HTTPException(status_code=404, detail="memory not found")
    return ExplainResponse(
        memory=MemoryOut.from_memory(explanation.memory),
        versions=explanation.versions,
        relationships=explanation.relationships,
        sources=[
            ConversationMessageOut(
                id=m.id, role=m.role, content=m.content, session_id=m.session_id,
                agent_id=m.agent_id, namespace=m.namespace, created_at=m.created_at,
                score=1.0,
            )
            for m in explanation.sources
        ],
    )


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
    except ValueError as exc:  # e.g. a PATCH that would hollow out a taste memory
        raise HTTPException(status_code=422, detail=str(exc))
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
    try:
        context_request = ContextRequest(**body.model_dump())
    except ValueError as exc:  # e.g. unknown task_type
        raise HTTPException(status_code=422, detail=str(exc))
    result = await service.build_context(context_request, tenant_id=tenant)
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


@router.post("/conversations/search", response_model=ConversationSearchResponse)
async def search_conversations(
    body: ConversationSearchIn,
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    """Full-text search over the raw L0 conversation archive."""
    hits = await service.search_conversations(
        user_id=body.user_id,
        query=body.query,
        tenant_id=tenant,
        namespace=body.namespace,
        namespaces=body.namespaces,
        session_id=body.session_id,
        top_k=body.top_k,
        since_days=body.since_days,
    )
    return ConversationSearchResponse(
        results=[
            ConversationMessageOut(
                id=h.message.id, role=h.message.role, content=h.message.content,
                session_id=h.message.session_id, agent_id=h.message.agent_id,
                namespace=h.message.namespace, created_at=h.message.created_at,
                score=h.score,
            )
            for h in hits
        ],
        count=len(hits),
    )


@router.post("/persona", response_model=PersonaResponse)
async def get_persona(
    body: PersonaIn,
    tenant: str = Depends(authenticate),
    service: MemoryService = Depends(get_service),
):
    """Compile the deterministic L3 persona document for a user."""
    doc = await service.compile_persona(
        user_id=body.user_id,
        tenant_id=tenant,
        namespace=body.namespace,
        namespaces=body.namespaces,
        include_ids=body.include_ids,
        token_budget=body.token_budget,
    )
    return PersonaResponse(
        text=doc.text,
        memory_ids=doc.memory_ids,
        token_estimate=doc.token_estimate,
        compiled_at=doc.compiled_at,
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
