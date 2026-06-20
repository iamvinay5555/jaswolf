"""Local sentence-transformers embedder (default: BAAI/bge-small-en-v1.5)."""

from __future__ import annotations

import asyncio


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # raises ImportError if absent

        self._model_cls = SentenceTransformer
        self.model_name = model_name
        self.name = f"st:{model_name}"
        self._model = None
        self._lock = asyncio.Lock()
        self.dim = 384  # corrected on first load

    def _load(self):
        if self._model is None:
            self._model = self._model_cls(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with self._lock:  # the model itself is not concurrency-safe
            model = await asyncio.to_thread(self._load)
            vectors = await asyncio.to_thread(
                model.encode, texts, normalize_embeddings=True, show_progress_bar=False
            )
        return [v.tolist() for v in vectors]

    async def close(self) -> None:
        return None
