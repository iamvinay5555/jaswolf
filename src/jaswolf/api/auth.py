"""API-key authentication, tenant resolution, and rate limiting.

Keys map to tenants (JASWOLF_API_KEYS="key1:tenantA,key2:tenantB"); every
storage query is scoped by the resolved tenant_id, which is the isolation
boundary. With no keys configured the API runs in open mode (dev only) and
everything lands in the "default" tenant.
"""

from __future__ import annotations

import hmac
import logging
import time

from fastapi import HTTPException, Request

logger = logging.getLogger("jaswolf.auth")
_warned_open_mode = False


def _extract_key(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-api-key")


async def authenticate(request: Request) -> str:
    """FastAPI dependency: resolves the tenant for this request, enforcing
    the API key and rate limit. Returns tenant_id."""
    global _warned_open_mode
    settings = request.app.state.settings
    key_map = request.app.state.api_key_map

    if not key_map:
        # create_app refuses to start in this state unless dev_open_mode is
        # set; this check is defense in depth for embedded/custom mounting.
        if not settings.dev_open_mode:
            raise HTTPException(status_code=401, detail="authentication not configured")
        if not _warned_open_mode:
            logger.warning("JASWOLF_DEV_OPEN_MODE — API running without auth (dev only)")
            _warned_open_mode = True
        return "default"

    provided = _extract_key(request)
    if not provided:
        raise HTTPException(status_code=401, detail="missing API key")
    tenant = None
    for key, mapped_tenant in key_map.items():
        if hmac.compare_digest(provided, key):
            tenant = mapped_tenant
            break
    if tenant is None:
        raise HTTPException(status_code=401, detail="invalid API key")

    limit = settings.rate_limit_per_minute
    if limit > 0:
        bucket = f"rl:{provided[:16]}:{int(time.time() // 60)}"
        count = await request.app.state.service.cache.incr(bucket, ttl=120)
        if count > limit:
            raise HTTPException(status_code=429, detail="rate limit exceeded")

    return tenant
