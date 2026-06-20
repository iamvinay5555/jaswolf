"""Deterministic feature-hashing embedder.

Zero external model dependencies. Projects word unigrams/bigrams and char
trigrams into a fixed-dimension space via blake2b hashing. Not a semantic
model — but similar texts share features, vectors are deterministic, and it
keeps the entire system functional (and testable) with no downloads.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN.findall(text.lower())
        features: list[tuple[str, float]] = [(t, 2.0) for t in tokens]
        features += [(f"{a}_{b}", 2.0) for a, b in zip(tokens, tokens[1:])]
        joined = " ".join(tokens)
        features += [(joined[i : i + 3], 1.0) for i in range(max(0, len(joined) - 2))]
        for feat, weight in features:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign * weight
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    async def close(self) -> None:
        return None
