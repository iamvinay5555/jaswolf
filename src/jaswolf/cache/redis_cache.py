"""Redis cache backend (optional: pip install jaswolf[redis])."""

from __future__ import annotations


class RedisCache:
    name = "redis"

    def __init__(self, url: str):
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise RuntimeError("redis_url configured but redis is not installed: pip install jaswolf[redis]") from exc
        self._redis = aioredis.from_url(url)

    async def get(self, key: str) -> bytes | None:
        return await self._redis.get(key)

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        await self._redis.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def incr(self, key: str, ttl: int | None = None) -> int:
        value = await self._redis.incr(key)
        if value == 1 and ttl:
            await self._redis.expire(key, ttl)
        return int(value)

    async def close(self) -> None:
        await self._redis.aclose()


def create_cache(redis_url: str | None, max_entries: int = 10_000):
    if redis_url:
        return RedisCache(redis_url)
    from .memory_cache import InMemoryCache

    return InMemoryCache(max_entries=max_entries)
