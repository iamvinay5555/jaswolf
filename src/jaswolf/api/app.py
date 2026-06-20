"""FastAPI application factory.

create_app() with no arguments builds everything from environment settings
and manages the service lifecycle (including the background lifecycle
sweeper). Passing a pre-built service skips lifespan setup — used by tests
and by embedded deployments that share one MemoryService.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import JaswolfSettings
from ..service import MemoryService
from . import metrics
from .routes import router

logger = logging.getLogger("jaswolf.api")


async def _sweeper_loop(service: MemoryService, interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            await service.sweep()
        except Exception as exc:
            logger.error("lifecycle sweep failed: %s", exc)


def create_app(
    settings: JaswolfSettings | None = None, service: MemoryService | None = None
) -> FastAPI:
    settings = settings or (service.settings if service else JaswolfSettings())

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_service = app.state.service is None
        if owns_service:
            app.state.service = await MemoryService.create(settings)
        sweeper = asyncio.create_task(
            _sweeper_loop(app.state.service, settings.sweep_interval_seconds)
        )
        try:
            yield
        finally:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper
            if owns_service:
                await app.state.service.close()

    if not settings.api_key_map() and not settings.dev_open_mode:
        raise RuntimeError(
            "Refusing to start the HTTP API without authentication: set "
            "JASWOLF_API_KEYS, or set JASWOLF_DEV_OPEN_MODE=true for local development."
        )

    app = FastAPI(
        title="JASWOLF Memory Engine",
        version="0.2.0",
        description="Long-term memory for autonomous agents: semantic retrieval, "
        "context generation, and memory evolution.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.api_key_map = settings.api_key_map()
    app.state.service = service  # may be None until lifespan runs

    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if origins:  # no CORS middleware at all unless explicitly configured
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        metrics.REQUESTS.labels(
            method=request.method, route=route_path, status=response.status_code
        ).inc()
        metrics.REQUEST_LATENCY.labels(route=route_path).observe(elapsed)
        response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.1f}"
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    app.include_router(router)

    @app.get("/health")
    async def health(request: Request):
        if request.app.state.service is None:
            return JSONResponse(status_code=503, content={"status": "starting"})
        report = await request.app.state.service.health()
        # 200 only when fully ok; 503 on degraded so a watchdog / load balancer
        # acts on it without parsing the body (consistent with MCP /healthz)
        code = 200 if report.get("status") == "ok" else 503
        return JSONResponse(status_code=code, content=report)

    @app.get("/metrics")
    async def prometheus_metrics():
        payload, content_type = metrics.render_latest()
        return Response(content=payload, media_type=content_type)

    return app
