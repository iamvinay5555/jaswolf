"""In-process LRU cache with TTL. Default backend — zero dependencies.

Single-event-loop safe (no locks needed: methods don't await between reads
and writes of the dict).
"""

from __future__ import annotations

import time
from collections import OrderedDict


class InMemoryCache:
    name = "memory"

    def __init__(self, max_entries: int = 10_000):
        self.max_entries = max_entries
        self._data: OrderedDict[str, tuple[float | None, bytes]] = OrderedDict()

    def _evict(self) -> None:
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    async def get(self, key: str) -> bytes | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if expires is not None and time.monotonic() > expires:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        expires = time.monotonic() + ttl if ttl else None
        self._data[key] = (expires, value)
        self._data.move_to_end(key)
        self._evict()

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def incr(self, key: str, ttl: int | None = None) -> int:
        current = await self.get(key)
        value = int(current) + 1 if current else 1
        existing = self._data.get(key)
        # preserve the original expiry window for counters (rate-limit buckets)
        if existing and existing[0] is not None:
            self._data[key] = (existing[0], str(value).encode())
        else:
            await self.set(key, str(value).encode(), ttl=ttl)
        return value

    async def close(self) -> None:
        self._data.clear()
