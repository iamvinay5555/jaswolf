"""Corpus-aware similarity calibration (Jasmine v0.4.0 feedback, 2026-06-13).

The v0.4.0 anchor-median floor under-measured the false-positive ceiling on
the real BGE DB: gate landed ~0.49 while unrelated memories scored ~0.64, so
off-topic hits leaked. v0.4.1 calibrates against the corpus instead.
"""

import numpy as np

from jaswolf.calibration import context_similarity_threshold


class _StubEmbedder:
    """Anchor embeddings engineered so the *anchor* path returns ~0.49 for the
    test query — reproducing the exact v0.4.0 failure number."""

    async def embed(self, texts):
        # every anchor sits at cosine 0.49 to the unit query [1, 0, ...]
        return [[0.49, float(np.sqrt(1 - 0.49**2))] for _ in texts]


async def test_anchor_fallback_reproduces_v040_floor():
    # small corpus -> fall back to anchors -> the old ~0.49 gate (the bug)
    gate = await context_similarity_threshold(
        _StubEmbedder(), [1.0, 0.0], background=None,
        noise_z=3.5, margin=0.0, min_background=24,
    )
    assert abs(gate - 0.49) < 1e-3
    # an off-topic memory at 0.64 would clear this floor — the v0.4.0 leak
    assert 0.642 > gate


async def test_corpus_calibration_lifts_gate_above_offtopic_hits():
    # background: 256 memories whose cosine to the query is anisotropic-high
    # (mean 0.50, sd 0.05) — bge-small's real shape, not the exotic anchors
    rng = np.random.default_rng(0)
    sims = rng.normal(0.50, 0.05, size=256).astype(np.float32)
    # encode as 2D unit vectors [s, sqrt(1-s^2)] so background @ q == s for q=[1,0]
    background = np.stack([sims, np.sqrt(1 - sims**2)], axis=1)

    gate = await context_similarity_threshold(
        _StubEmbedder(), [1.0, 0.0], background=background,
        noise_z=3.5, margin=0.08, min_background=24,
    )
    # gate = mean + 3.5*sd ≈ 0.50 + 0.175 = 0.675 -> now ABOVE the 0.642 leak
    assert gate > 0.642
    # and a genuine on-topic hit at 0.78 still clears it
    assert 0.78 > gate


async def test_corpus_path_ignores_anchor_margin():
    # with a real background the anchor margin is irrelevant — proves we took
    # the corpus branch, not the fallback
    background = np.stack(
        [np.full(64, 0.50, np.float32), np.full(64, np.sqrt(1 - 0.25), np.float32)], axis=1
    )
    gate = await context_similarity_threshold(
        _StubEmbedder(), [1.0, 0.0], background=background,
        noise_z=2.0, margin=99.0, min_background=24,
    )
    assert gate < 1.0  # margin=99 ignored; sd=0 so gate == mean 0.50
    assert abs(gate - 0.50) < 1e-3
