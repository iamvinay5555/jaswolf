"""L0 conversation archive: capture, provenance, search, explain, retention.

The semantic-pyramid evidence layer (v0.3.0): raw turns are archived when
conversation_capture is on, extracted memories carry source_message_ids, and
explain() walks a memory back to the exact turns it came from.
"""

from datetime import timedelta

import pytest

from jaswolf.config import JaswolfSettings
from jaswolf.models import ChatMessage, ConversationMessage, MemoryNotFound, utcnow
from jaswolf.service import MemoryService
from jaswolf.storage.base import QueryScope


@pytest.fixture
def capture_settings(tmp_path) -> JaswolfSettings:
    return JaswolfSettings(
        database_url=f"sqlite:///{tmp_path}/jaswolf_conv.db",
        embedding_provider="hash",
        embedding_dim=384,
        dev_open_mode=True,
        conversation_capture=True,
        conversation_retention_days=30,
        sweep_interval_seconds=3600,
        log_level="WARNING",
    )


@pytest.fixture
async def capture_service(capture_settings) -> MemoryService:
    svc = await MemoryService.create(capture_settings)
    yield svc
    await svc.close()


TURNS = [
    ChatMessage(role="user", content="My office is in Changi Business Park"),
    ChatMessage(role="assistant", content="Noted — Changi it is."),
]


async def test_capture_stores_l0_and_provenance(capture_service):
    results = await capture_service.ingest_messages(
        user_id="alice", messages=TURNS, session_id="s1"
    )
    assert results, "extraction should produce at least one memory"
    memory = results[0][0]
    source_ids = memory.metadata.get("source_message_ids")
    assert source_ids, "extracted memory must carry L0 provenance"

    rows = await capture_service.storage.get_conversation_messages(source_ids, "default")
    assert len(rows) == len(TURNS)
    assert rows[0].content == TURNS[0].content
    assert rows[0].role == "user"
    assert rows[0].session_id == "s1"


async def test_capture_off_stores_nothing(service):
    # the default conftest service has conversation_capture=False
    results = await service.ingest_messages(user_id="alice", messages=TURNS)
    assert results
    memory = results[0][0]
    assert "source_message_ids" not in memory.metadata
    hits = await service.search_conversations(user_id="alice", query="Changi")
    assert hits == []


async def test_search_conversations_fts(capture_service):
    await capture_service.ingest_messages(user_id="alice", messages=TURNS, session_id="s1")
    await capture_service.ingest_messages(
        user_id="alice",
        messages=[ChatMessage(role="user", content="thinking about the Hakone ryokan for the trip")],
        session_id="s2",
    )

    hits = await capture_service.search_conversations(user_id="alice", query="Hakone ryokan")
    assert hits
    assert "Hakone" in hits[0].message.content
    # scoped: other user sees nothing
    assert await capture_service.search_conversations(user_id="other", query="Hakone") == []


async def test_search_conversations_empty_query_returns_recent(capture_service):
    await capture_service.ingest_messages(user_id="alice", messages=TURNS, session_id="s1")
    hits = await capture_service.search_conversations(user_id="alice", query="")
    assert len(hits) == len(TURNS)


async def test_explain_walks_back_to_sources(capture_service):
    results = await capture_service.ingest_messages(
        user_id="alice", messages=TURNS, session_id="s1"
    )
    memory = results[0][0]
    explanation = await capture_service.explain(memory.id)
    assert explanation.memory.id == memory.id
    assert explanation.sources, "explain must surface the L0 turns"
    assert any("Changi" in s.content for s in explanation.sources)


async def test_explain_includes_supersession_edge(capture_service):
    from jaswolf.models import MemoryCreate, MemoryType

    old, _ = await capture_service.add(
        MemoryCreate(user_id="alice", content="User's office is Buona Vista",
                     memory_type=MemoryType.SEMANTIC)
    )
    new, _ = await capture_service.add(
        MemoryCreate(user_id="alice", content="Actually user's office is Changi now",
                     memory_type=MemoryType.SEMANTIC)
    )
    explanation = await capture_service.explain(new.id)
    edges = [e for e in explanation.relationships if e["relation"] == "supersedes"]
    assert edges and edges[0]["other_id"] == old.id
    assert edges[0]["other_content"] == "User's office is Buona Vista"
    # and the archived side shows the incoming edge
    explanation_old = await capture_service.explain(old.id)
    incoming = [e for e in explanation_old.relationships if e["direction"] == "incoming"]
    assert incoming and incoming[0]["other_id"] == new.id


async def test_explain_unknown_id_raises(capture_service):
    with pytest.raises(MemoryNotFound):
        await capture_service.explain("no-such-id")


async def test_retention_prunes_old_turns_only(capture_service):
    await capture_service.ingest_messages(user_id="alice", messages=TURNS, session_id="s1")
    # plant an old turn directly
    old_row = ConversationMessage(
        user_id="alice", role="user", content="ancient chatter about kopi",
        created_at=utcnow() - timedelta(days=90),
    )
    await capture_service.storage.add_conversation_messages([old_row])

    report = await capture_service.sweep()
    assert report.pruned_conversations == 1
    hits = await capture_service.search_conversations(user_id="alice", query="kopi")
    assert hits == []
    # recent turns survive
    hits = await capture_service.search_conversations(user_id="alice", query="Changi")
    assert hits


async def test_prune_disabled_when_retention_zero(tmp_path):
    settings = JaswolfSettings(
        database_url=f"sqlite:///{tmp_path}/jaswolf_forever.db",
        embedding_provider="hash",
        dev_open_mode=True,
        conversation_capture=True,
        conversation_retention_days=0,
        log_level="WARNING",
    )
    svc = await MemoryService.create(settings)
    try:
        old_row = ConversationMessage(
            user_id="alice", role="user", content="keep me forever",
            created_at=utcnow() - timedelta(days=999),
        )
        await svc.storage.add_conversation_messages([old_row])
        report = await svc.sweep()
        assert report.pruned_conversations == 0
        assert await svc.search_conversations(user_id="alice", query="forever")
    finally:
        await svc.close()


async def test_namespace_isolation_in_conversation_search(capture_service):
    await capture_service.ingest_messages(
        user_id="alice", messages=TURNS, namespace="jasmine"
    )
    scope_hits = await capture_service.search_conversations(
        user_id="alice", query="Changi", namespace="freya"
    )
    assert scope_hits == []
    scope_hits = await capture_service.search_conversations(
        user_id="alice", query="Changi", namespaces=["freya", "jasmine"]
    )
    assert scope_hits


async def test_storage_conv_scope_where_tenant_isolation(capture_service):
    row = ConversationMessage(
        tenant_id="tenant-b", user_id="alice", role="user", content="secret other tenant"
    )
    await capture_service.storage.add_conversation_messages([row])
    hits = await capture_service.storage.search_conversations(
        QueryScope(tenant_id="default", user_id="alice"), "secret", k=5
    )
    assert hits == []
    hits = await capture_service.storage.search_conversations(
        QueryScope(tenant_id="tenant-b", user_id="alice"), "secret", k=5
    )
    assert len(hits) == 1
