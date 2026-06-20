"""Per-query similarity-gate calibration.

The context boundary must answer one question per candidate: is this cosine
*evidence*, or just where this embedding model parks unrelated text? bge-small
is strongly anisotropic — arbitrary English pairs sit around 0.4-0.65, not
near 0 — so no absolute threshold transfers across models, and (Jasmine,
2026-06-13) fixed *anchor* sentences underestimate the floor: exotic anchors
(basalt, falconry) score ~0.39 against a query while the user's own unrelated
memories score ~0.63, so off-topic hits cleared an anchor-based gate.

So calibrate against the **actual corpus**: sample the user's own memory
embeddings and measure where this query lands against them. An off-topic
query is unrelated to ~everything, so its top hit is just the upper tail of
that background distribution; a real hit stands out in units of the corpus's
own spread. The gate is therefore `mean + noise_z·std` of the query's
similarity to a background sample — anisotropy-invariant, because it measures
separation relative to the corpus, not an absolute cosine.

The fixed anchors remain only as a fallback for corpora too small to estimate
a background distribution.
"""

from __future__ import annotations

from statistics import median

import numpy as np

ANCHOR_TEXTS = (
    "The chemical composition of basalt rock formations varies with cooling rate.",
    "Medieval falconry required years of patient training before the first hunt.",
    "The offside rule in ice hockey differs from the one used in association football.",
    "Folding an origami crane takes about twenty steps from a single square sheet.",
    "Tides in the estuary rearrange the sandbanks twice every day.",
    "A sourdough starter needs regular feeding with equal parts flour and water.",
)


async def context_similarity_threshold(
    embedder,
    query_vec: list[float],
    background: np.ndarray | None,
    *,
    noise_z: float,
    margin: float,
    min_background: int,
) -> float:
    """Raw-cosine gate a non-pinned candidate must clear to enter the prompt.

    `background` is an (N, dim) matrix of L2-normalized corpus embeddings.
    With enough of them, the gate is `mean + noise_z·std` of the query's
    cosine to the sample. Otherwise it falls back to the fixed-anchor median
    plus `margin` (the pre-v0.4.1 behavior), appropriate for tiny corpora.
    """
    q = np.asarray(query_vec, dtype=np.float32)
    if background is not None and background.shape[0] >= min_background:
        sims = background @ q
        return float(sims.mean() + noise_z * sims.std())
    anchor_vecs = await embedder.embed(list(ANCHOR_TEXTS))
    anchors = np.asarray(anchor_vecs, dtype=np.float32)
    return float(median((anchors @ q).tolist())) + margin
