"""Shared keyword tokenization and discriminative-term selection.

A keyword/FTS hit is only *evidence* of relevance if the matched term is
discriminative — not a stopword, long enough to mean something, and rare
enough in the corpus to distinguish memories. Jasmine, 2026-06-13: the
off-topic query "weather forecast Lisbon next week" produced 19 keyword
hits because generic tokens like "next"/"week" matched unrelated memories
under a blanket `OR`, and v0.4.0 exempted any keyword hit from the context
gate — so noise reached the prompt.

Both storage backends route through `discriminative_tokens` so the rule
can't drift between them; each supplies its own document-frequency lookup.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# Classic English function words + a few grammatical temporal determiners
# that are almost always noise. Generic *content* nouns (week, day, best,
# practice, ...) are deliberately NOT here — corpus document-frequency is the
# right, non-hand-curated tool for "common in this corpus", and a word that is
# genuinely rare in the user's memories is fair lexical evidence.
STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from
get got had has have how i if in into is it its just like make me my no
not of on or our out over should so than that the their them then there
these they this to too under up us was we were what when where which who
why will with would you your
this that these those next last first now then today tomorrow yesterday
""".split())

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def candidate_tokens(text: str, min_len: int = 3) -> list[str]:
    """Unique tokens surviving the cheap (corpus-free) filters: not a
    stopword, long enough. These are the only tokens worth a DF lookup.
    Backends that count DF asynchronously can pre-batch over this list."""
    out: list[str] = []
    seen: set[str] = set()
    for token in tokenize(text):
        if token in seen or len(token) < min_len or token in STOPWORDS:
            continue
        seen.add(token)
        out.append(token)
    return out


def discriminative_tokens(
    text: str,
    df_lookup: Callable[[str], int],
    total_docs: int,
    *,
    min_len: int = 3,
    max_df_ratio: float = 0.10,
    min_df: int = 3,
) -> list[str]:
    """Query tokens that carry real lexical evidence.

    Drops stopwords, short tokens, and any token that is *common* in the
    corpus — appearing in more than `max_df_ratio` of memories (an IDF-style
    cut) AND in at least `min_df` of them. The absolute `min_df` floor keeps
    the ratio from misfiring on tiny corpora, where every term trivially
    exceeds any percentage. `df_lookup` is only called for tokens that survive
    the cheap filters, so the backend pays at most one count per meaningful
    term.
    """
    if total_docs <= 0 or max_df_ratio >= 1.0:
        return candidate_tokens(text, min_len)
    df_cap = max_df_ratio * total_docs
    out: list[str] = []
    for token in candidate_tokens(text, min_len):
        df = df_lookup(token)
        if df >= min_df and df > df_cap:
            continue
        out.append(token)
    return out
