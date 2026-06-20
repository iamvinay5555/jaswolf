"""Embedding provider interface, caching wrapper, and factory."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Protocol, runtime_checkable

from ..cache.base import CacheBackend
from ..config import JaswolfSettings

logger = logging.getLogger("jaswolf.embeddings")


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def close(self) -> None: ...


class CachedEmbedder:
    """Wraps any provider with a content-hash cache.

    Repeated ingestion of the same text (very common for agents that re-state
    preferences) never recomputes an embedding.
    """

    def __init__(
        self,
        inner: EmbeddingProvider,
        cache: CacheBackend,
        ttl: int = 7 * 24 * 3600,
        is_fallback: bool = False,
    ):
        self.inner = inner
        self.cache = cache
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        # True when "auto" degraded to the hash embedder because nothing
        # better was available — surfaced as degraded health in production
        self.is_fallback = is_fallback

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def dim(self) -> int:
        return self.inner.dim

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"emb:{self.inner.name}:{self.inner.dim}:{digest}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for i, text in enumerate(texts):
            cached = await self.cache.get(self._key(text))
            if cached is not None:
                try:
                    results[i] = json.loads(cached)
                    self.hits += 1
                    continue
                except Exception:
                    pass
            missing.append(i)
        if missing:
            self.misses += len(missing)
            fresh = await self.inner.embed([texts[i] for i in missing])
            for slot, vec in zip(missing, fresh):
                results[slot] = vec
                await self.cache.set(self._key(texts[slot]), json.dumps(vec).encode(), ttl=self.ttl)
        return [r for r in results if r is not None]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

    async def close(self) -> None:
        await self.inner.close()


def create_embedder(settings: JaswolfSettings, cache: CacheBackend) -> CachedEmbedder:
    """Build the configured provider; "auto" degrades gracefully:
    sentence-transformers -> OpenAI-compatible -> hashing fallback."""
    provider = settings.embedding_provider.lower()
    inner: EmbeddingProvider | None = None

    if provider in ("auto", "local"):
        try:
            from .local import SentenceTransformerEmbedder

            inner = SentenceTransformerEmbedder(settings.embedding_model)
        except ImportError:
            if provider == "local":
                raise RuntimeError(
                    "embedding_provider=local requires `pip install jaswolf[local-embeddings]`"
                )

    if inner is None and provider in ("auto", "openai"):
        if settings.openai_api_key:
            from .openai_api import OpenAICompatibleEmbedder

            inner = OpenAICompatibleEmbedder(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
                dim=settings.embedding_dim,
            )
        elif provider == "openai":
            raise RuntimeError("embedding_provider=openai requires JASWOLF_OPENAI_API_KEY")

    is_fallback = False
    if inner is None:
        from .hashing import HashingEmbedder

        inner = HashingEmbedder(dim=settings.embedding_dim)
        if provider == "auto":
            is_fallback = True
            logger.warning(
                "No embedding model available — using deterministic hashing embedder. "
                "NOT suitable for production retrieval quality; health will report "
                "degraded. Install jaswolf[local-embeddings] or configure an "
                "OpenAI-compatible endpoint."
            )

    logger.info("embedding provider: %s (dim=%d)", inner.name, inner.dim)
    return CachedEmbedder(inner, cache, ttl=settings.embed_cache_ttl, is_fallback=is_fallback)
