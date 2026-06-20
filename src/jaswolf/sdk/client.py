"""JASWOLF Python SDK.

    from jaswolf import JaswolfClient

    client = JaswolfClient("http://localhost:8400", api_key="...")
    client.add_memory(user_id="alice", content="User prefers Python", memory_type="preference")
    hits = client.search(user_id="alice", query="language preference")
    block = client.build_context(user_id="alice", query="what stack should we use?")

Both sync (JaswolfClient) and async (AsyncJaswolfClient) variants share the same
request construction, so behavior is identical.
"""

from __future__ import annotations

from typing import Any

import httpx


class JaswolfError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"JASWOLF API error {status_code}: {detail}")


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _check(response: httpx.Response) -> Any:
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise JaswolfError(response.status_code, str(detail))
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


class _Requests:
    """Pure request builders: (method, path, json|params)."""

    @staticmethod
    def add_memory(user_id: str, content: str, **kwargs: Any):
        return "POST", "/v1/memories", {"json": {"user_id": user_id, "content": content, **kwargs}}

    @staticmethod
    def extract(user_id: str, text: str | None = None, messages: list[dict] | None = None, **kwargs: Any):
        payload = {"user_id": user_id, "text": text, "messages": messages, **kwargs}
        return "POST", "/v1/memories/extract", {"json": payload}

    @staticmethod
    def get_memory(memory_id: str, include_embedding: bool = False):
        return "GET", f"/v1/memories/{memory_id}", {
            "params": {"include_embedding": include_embedding}
        }

    @staticmethod
    def get_versions(memory_id: str):
        return "GET", f"/v1/memories/{memory_id}/versions", {}

    @staticmethod
    def update_memory(memory_id: str, **fields: Any):
        return "PATCH", f"/v1/memories/{memory_id}", {"json": fields}

    @staticmethod
    def delete_memory(memory_id: str, hard: bool = False):
        return "DELETE", f"/v1/memories/{memory_id}", {"params": {"hard": hard}}

    @staticmethod
    def search(user_id: str, query: str = "", **kwargs: Any):
        return "POST", "/v1/memories/search", {"json": {"user_id": user_id, "query": query, **kwargs}}

    @staticmethod
    def build_context(user_id: str, **kwargs: Any):
        return "POST", "/v1/memories/context", {"json": {"user_id": user_id, **kwargs}}

    @staticmethod
    def consolidate(user_id: str, **kwargs: Any):
        return "POST", "/v1/memories/consolidate", {"json": {"user_id": user_id, **kwargs}}

    @staticmethod
    def stats(user_id: str | None = None):
        params = {"user_id": user_id} if user_id else {}
        return "GET", "/v1/stats", {"params": params}

    @staticmethod
    def sweep():
        return "POST", "/v1/maintenance/sweep", {}

    @staticmethod
    def health():
        return "GET", "/health", {}


def _make_method(name: str, is_async: bool):
    builder = getattr(_Requests, name)
    if is_async:
        async def amethod(self, *args: Any, **kwargs: Any):
            method, path, opts = builder(*args, **kwargs)
            return _check(await self._client.request(method, path, **opts))
        amethod.__name__ = name
        return amethod

    def method(self, *args: Any, **kwargs: Any):
        method_, path, opts = builder(*args, **kwargs)
        return _check(self._client.request(method_, path, **opts))
    method.__name__ = name
    return method


_METHOD_NAMES = [
    "add_memory", "extract", "get_memory", "get_versions", "update_memory",
    "delete_memory", "search", "build_context", "consolidate", "stats", "sweep", "health",
]


class JaswolfClient:
    """Synchronous client."""

    def __init__(
        self,
        base_url: str = "http://localhost:8400",
        api_key: str | None = None,
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self._client = httpx.Client(
            base_url=base_url, headers=_headers(api_key), timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "JaswolfClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncJaswolfClient:
    """Asynchronous client."""

    def __init__(
        self,
        base_url: str = "http://localhost:8400",
        api_key: str | None = None,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._client = httpx.AsyncClient(
            base_url=base_url, headers=_headers(api_key), timeout=timeout, transport=transport
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncJaswolfClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


for _name in _METHOD_NAMES:
    setattr(JaswolfClient, _name, _make_method(_name, is_async=False))
    setattr(AsyncJaswolfClient, _name, _make_method(_name, is_async=True))
