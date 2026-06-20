"""Discriminative-keyword gating (Jasmine v0.4.0 feedback, 2026-06-13).

v0.4.0 exempted any FTS hit from the context gate, so generic tokens like
"next"/"week" in an off-topic query matched unrelated memories and leaked 25
of them into the prompt. A keyword hit is evidence only if the matched term
is discriminative: non-stopword, long enough, and rare in the corpus.
"""

from jaswolf.keywords import candidate_tokens, discriminative_tokens
from jaswolf.models import ContextRequest, MemoryCreate, MemoryType, SearchMode, SearchQuery


# ---- unit: token selection -----------------------------------------------------


def test_stopwords_and_short_tokens_dropped():
    assert candidate_tokens("what is the next week like") == ["week"]  # 'next' is generic
    assert "harborfront" in candidate_tokens("HarborFront interchange")


def test_common_tokens_filtered_by_document_frequency():
    # "week" appears in 40% of a 10-doc corpus -> not discriminative;
    # "harborfront" in 1 -> discriminative
    df = {"week": 4, "harborfront": 1, "interchange": 1}
    toks = discriminative_tokens(
        "HarborFront interchange this week", df.get, total_docs=10, max_df_ratio=0.10
    )
    assert "harborfront" in toks and "interchange" in toks
    assert "week" not in toks


# ---- integration against the real SQLite FTS path ------------------------------


async def _seed_corpus(service):
    # one discriminative target...
    await service.add(MemoryCreate(
        user_id="alice",
        content="Return route home goes via HarborFront interchange",
        memory_type=MemoryType.SEMANTIC,
    ))
    # ...and many memories sharing the generic token "week" so its DF is high
    for i in range(12):
        await service.add(MemoryCreate(
            user_id="alice",
            content=f"Project status note number {i} updated this week",
            memory_type=MemoryType.SEMANTIC,
        ))


async def test_generic_token_does_not_produce_keyword_evidence(service):
    await _seed_corpus(service)
    # "week" is common -> not discriminative -> no keyword hits despite 12 matches
    hits = await service.search(SearchQuery(
        user_id="alice", query="what happened next week",
        mode=SearchMode.KEYWORD, record_access=False,
    ))
    assert all(not h.keyword_match for h in hits) or hits == []


async def test_discriminative_token_still_matches(service):
    await _seed_corpus(service)
    hits = await service.search(SearchQuery(
        user_id="alice", query="HarborFront interchange",
        mode=SearchMode.KEYWORD, record_access=False,
    ))
    assert any("HarborFront" in h.memory.content and h.keyword_match for h in hits)


async def test_off_topic_generic_query_grants_no_keyword_exemption(service):
    # the exact v0.4.0 leak: an off-topic query whose only corpus overlap is a
    # generic token ("week") must not exempt those 12 memories from the gate.
    # (The hash test embedder can't exercise the vector gate end-to-end — that
    # path is covered with realistic numbers in test_context_gate.py — so here
    # we assert the keyword exemption itself stays empty, which is the fix.)
    await _seed_corpus(service)
    result = await service.build_context(
        ContextRequest(user_id="alice", query="weather forecast Lisbon next week")
    )
    keyword_exempt = [
        m for m in result.memories
        if m.keyword_match and m.memory.memory_type not in (MemoryType.PREFERENCE, MemoryType.GOAL)
    ]
    assert keyword_exempt == []
