"""OpenAI-compatible embeddings over HTTP (works with OpenAI, Voyage-style
proxies, vLLM, Ollama, LiteLLM — anything exposing /embeddings)."""

from __future__ import annotations

import math

import httpx


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.name = f"openai:{model}"
        self._client = client or httpx.AsyncClient(
            timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict = {"model": self.model, "input": texts}
        # text-embedding-3-* supports server-side dimensionality reduction
        if "text-embedding-3" in self.model:
            payload["dimensions"] = self.dim
        resp = await self._client.post(f"{self.base_url}/embeddings", json=payload)
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [_normalize(d["embedding"]) for d in data]

    async def close(self) -> None:
        await self._client.aclose()


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec
