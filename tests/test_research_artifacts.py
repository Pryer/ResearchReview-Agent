"""研究资料存储的会话隔离与上下文快照测试。"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base
from app.database.repositories import ResearchArtifactCorruptionError
from app.services.research_artifact_service import ResearchArtifactService


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def test_artifact_resolve_is_scoped_to_session(db):
    service = ResearchArtifactService(db)
    ref = service.persist_payload(
        session_id="session-a",
        artifact_type="evidence_bundle",
        payload={"paper_ids": ["p1"]},
        provenance={"source": "test"},
    )

    assert service.resolve("session-a", ref)["payload"] == {"paper_ids": ["p1"]}
    with pytest.raises(PermissionError):
        service.resolve("session-b", ref)
    assert len(ref.removeprefix("artifact://")) <= 64


def test_persist_main_context_keeps_metadata_outside_five_field_view(db):
    state = {
        "state_schema_version": "2",
        "user_query": "研究主题",
        "topic": "研究主题",
        "core_deliverables": [],
    }
    service = ResearchArtifactService(db)

    ref = service.persist_main_context("session-a", state)
    stored = service.resolve("session-a", ref)

    assert set(stored["payload"]["context"]) == {
        "goal", "state", "key_evidence", "decisions", "open_questions",
    }
    assert stored["payload"]["source_fingerprint"]
    assert state["main_context_artifact_ref"] == ref


def test_main_context_versions_are_immutable(db):
    service = ResearchArtifactService(db)
    state = {"state_schema_version": "2", "user_query": "主题"}

    first = service.persist_main_context("session-a", state)
    second = service.persist_main_context("session-a", state)

    assert first != second
    assert service.resolve("session-a", first)["version"] == 1
    assert service.resolve("session-a", second)["version"] == 2


def test_corrupt_artifact_is_reported_instead_of_returning_empty_payload(db):
    service = ResearchArtifactService(db)
    ref = service.persist_payload(
        session_id="session-a", artifact_type="evidence_bundle", payload={"ok": True}
    )
    artifact_id = ref.removeprefix("artifact://")
    row = service.repo.db.get(__import__(
        "app.database.models", fromlist=["ResearchArtifactModel"]
    ).ResearchArtifactModel, artifact_id)
    row.payload_json = "{broken"
    db.flush()

    with pytest.raises(ResearchArtifactCorruptionError):
        service.resolve("session-a", ref)


def test_legacy_state_refs_are_read_only_and_typed(db):
    service = ResearchArtifactService(db)
    state = {"paper_cards": [{"paper_id": "p1", "title": "T"}]}

    resolved = service.resolve("session-a", "state://paper-card/p1", state=state)

    assert resolved == {
        "artifact_type": "paper_card",
        "payload": {"paper_id": "p1", "title": "T"},
    }
