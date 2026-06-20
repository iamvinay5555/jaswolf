import pytest

from jaswolf.extraction import LLMExtractor, RuleExtractor, apply_durability_gate
from jaswolf.models import ExtractedItem, MemoryType


@pytest.fixture
def extractor() -> RuleExtractor:
    return RuleExtractor()


def _types(items):
    return {i.memory_type for i in items}


def _contents(items):
    return [i.content for i in items]


def test_preference_extraction(extractor):
    items = extractor.extract("I love Python. Also I prefer dark mode for editors.")
    contents = _contents(items)
    assert "User loves Python" in contents
    assert any(c.startswith("User prefers dark mode") for c in contents)
    assert _types(items) == {MemoryType.PREFERENCE}


def test_goal_extraction(extractor):
    items = extractor.extract("I want to launch a SaaS by December.")
    assert items[0].memory_type == MemoryType.GOAL
    assert items[0].content.startswith("User wants to launch a SaaS")


def test_relationship_extraction(extractor):
    items = extractor.extract("Sarah is my cofounder.")
    assert items[0].memory_type == MemoryType.RELATIONSHIP
    assert items[0].content == "Sarah is user's cofounder"


def test_favorite_is_preference_not_relationship(extractor):
    items = extractor.extract("Python is my favorite language.")
    assert all(i.memory_type != MemoryType.RELATIONSHIP for i in items)


def test_fact_extraction(extractor):
    items = extractor.extract("My company uses Kubernetes. I'm a backend engineer.")
    contents = _contents(items)
    assert "User's company uses Kubernetes" in contents
    assert "User is a backend engineer" in contents
    assert _types(items) == {MemoryType.SEMANTIC}


def test_procedural_extraction(extractor):
    text = (
        "Here is how I deploy Hermes:\n"
        "1. Build the docker image\n"
        "2. Push to the registry\n"
        "3. Run docker compose up on the VPS"
    )
    items = extractor.extract(text)
    assert MemoryType.PROCEDURAL in _types(items)


def _proc_item(content: str) -> ExtractedItem:
    return ExtractedItem(
        content=content, memory_type=MemoryType.PROCEDURAL,
        importance=0.5, confidence=0.7, source="rules",
    )


def test_gate_drops_copied_transcript_as_procedure():
    # 2026-06-20 hygiene follow-up: the greedy whole-message procedure capture
    # stored 1500-char chat transcripts (assistant turn-markers, model-switch
    # notes, "honey" chatter) as durable `procedural` memories.
    blob = (
        "[Note: model was just switched from MiniMax to gpt-5.5.]\n"
        "honey, this morning u helped get the audit report:\n"
        "1. ran the audit\n2. sent it to claude\n"
        "🧠 OpenAI Codex: Done honey — both parts complete."
    )
    assert apply_durability_gate(_proc_item(blob)) is None


def test_gate_keeps_a_real_procedure():
    proc = (
        "How to deploy Hermes:\n1. Build the docker image\n"
        "2. Push to the registry\n3. Run docker compose up on the VPS"
    )
    kept = apply_durability_gate(_proc_item(proc))
    assert kept is not None and kept.memory_type is MemoryType.PROCEDURAL


def test_questions_are_skipped(extractor):
    assert extractor.extract("Do I love Python?") == []


def test_compound_sentence_splits_into_clauses(extractor):
    items = extractor.extract("I love Python and I prefer dark mode in every editor.")
    contents = _contents(items)
    assert "User loves Python" in contents
    assert "User prefers dark mode in every editor" in contents
    assert len(items) == 2


def test_first_person_possessives_rewritten(extractor):
    items = extractor.extract("We plan to launch our SaaS by December.")
    assert items[0].content == "User's team plans to launch their SaaS by December"
    items = extractor.extract("I prefer dark mode in my editor.")
    assert items[0].content == "User prefers dark mode in user's editor"


def test_batch_dedupe(extractor):
    items = extractor.extract("I love Python. I love Python. I love python!")
    assert len([i for i in items if "loves Python" in i.content or "loves python" in i.content]) == 1


def test_llm_parse_valid_json():
    items = LLMExtractor._parse(
        'Here you go: [{"content": "User prefers Rust", "type": "preference", '
        '"importance": 0.9, "confidence": 0.85}]'
    )
    assert len(items) == 1
    assert items[0].memory_type == MemoryType.PREFERENCE
    assert items[0].importance == 0.9
    assert items[0].source == "llm"


def test_llm_parse_garbage():
    assert LLMExtractor._parse("I could not find anything.") == []
    assert LLMExtractor._parse("[not json}") == []
    assert LLMExtractor._parse('[{"type": "preference"}]') == []  # missing content
