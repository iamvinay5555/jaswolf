"""Token estimation for context budgeting.

Uses tiktoken when installed; otherwise a chars/4 heuristic, which slightly
overestimates for English prose — safe for budgeting purposes.
"""

from __future__ import annotations

from typing import Callable

_encoder: Callable[[str], int] | None = None


def _build_encoder() -> Callable[[str], int]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
    except Exception:
        return lambda text: max(1, len(text) // 4)


def estimate_tokens(text: str) -> int:
    global _encoder
    if _encoder is None:
        _encoder = _build_encoder()
    if not text:
        return 0
    return _encoder(text)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate at a sentence boundary that fits within max_tokens."""
    if estimate_tokens(text) <= max_tokens:
        return text
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    used = 0
    for sentence in sentences:
        cost = estimate_tokens(sentence) + 1
        if used + cost > max_tokens:
            break
        out.append(sentence)
        used += cost
    if out:
        return " ".join(out).rstrip() + " …"
    # No sentence fits: hard character cut as last resort.
    return text[: max(8, max_tokens * 4 - 2)].rstrip() + " …"
