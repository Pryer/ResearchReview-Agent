"""追加式研究记忆的回放、会话隔离与问题失效。"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base
from app.services.research_memory_service import ResearchMemoryService


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def test_memory_rebuilds_from_events_and_resolves_questions(db):
    memory = ResearchMemoryService(db)
    memory.append_event(
        session_id="s1",
        event_type="analysis",
        payload={
            "findings": [{"finding": "F", "source_refs": ["artifact://e1"]}],
            "decisions": [{"decision_id": "d1", "decision": "D"}],
            "open_questions": [{"question_id": "q1", "question": "Q"}],
            "citations": ["artifact://e1"],
        },
    )
    memory.append_event(
        session_id="s1",
        event_type="resolution",
        payload={"resolved_question_ids": ["q1"]},
    )

    ref = memory.rebuild_summary("s1")
    summary = memory.artifacts.resolve("s1", ref)["payload"]

    assert summary["cursor"] == 2
    assert summary["findings"][0]["finding"] == "F"
    assert summary["decisions"][0]["decision_id"] == "d1"
    assert summary["open_questions"] == []
    with pytest.raises(PermissionError):
        memory.artifacts.resolve("s2", ref)


def test_legacy_history_import_is_idempotent_and_marks_unknown_loss(db):
    memory = ResearchMemoryService(db)
    state = {"conversation_history": [{
        "role": "user", "type": "research_request", "content": "old request",
    }]}

    first = memory.import_legacy_history("s1", state)
    second = memory.import_legacy_history("s1", state)

    assert len(first) == 1
    assert second == []
    assert state["research_memory_migrated"] is True
    assert state["research_memory_legacy_loss_unknown"] is True


def test_summary_only_reads_events_after_saved_cursor(db):
    from app.database.models import ResearchEventModel
    from sqlalchemy import select
    memory = ResearchMemoryService(db)
    memory.append_event(session_id="s", event_type="analysis", payload={"findings": [{"finding": "first"}]})
    memory.rebuild_summary("s")
    # 历史 payload 若被意外重放就会报错；恢复必须只处理游标之后的新事件。
    old = db.scalar(select(ResearchEventModel))
    old.payload_json = "invalid history must not be replayed"
    memory.append_event(session_id="s", event_type="analysis", payload={"findings": [{"finding": "second"}]})
    ref = memory.rebuild_summary("s")
    assert [x["finding"] for x in memory.artifacts.resolve("s", ref)["payload"]["findings"]] == ["first", "second"]


def test_history_is_archived_before_window_truncation_and_not_duplicated(db):
    from app.database.models import ResearchEventModel
    from sqlalchemy import select, func
    memory = ResearchMemoryService(db)
    history = [{"role": "user", "content": str(i)} for i in range(70)]
    retained = memory.archive_history("s", history)
    assert len(retained) == 50
    assert db.scalar(select(func.count()).select_from(ResearchEventModel)) == 70
    memory.archive_history("s", retained)
    assert db.scalar(select(func.count()).select_from(ResearchEventModel)) == 70
    retained.append({"role": "user", "content": "69"})
    memory.archive_history("s", retained)
    assert db.scalar(select(func.count()).select_from(ResearchEventModel)) == 71


def test_resolved_and_invalidated_items_do_not_return_after_checkpoint(db):
    memory = ResearchMemoryService(db)
    memory.append_event(session_id="s", event_type="decision", payload={
        "decisions": [{"decision_id": "d", "decision": "old", "source": "user"}],
        "open_questions": [{"question_id": "q", "question": "old", "blocking": True}]})
    memory.rebuild_summary("s")
    memory.append_event(session_id="s", event_type="correction", payload={
        "invalidated_decision_ids": ["d"], "resolved_question_ids": ["q"]})
    ref = memory.rebuild_summary("s")
    payload = memory.artifacts.resolve("s", ref)["payload"]
    assert payload["decisions"] == [] and payload["open_questions"] == []


def test_database_marker_governs_legacy_import_across_reloaded_state(db):
    memory = ResearchMemoryService(db)
    # 旧 artifact 版本已写过布尔标记，仍须迁到新事件表。
    old = {"research_memory_migrated": True, "conversation_history": [
        {"role": "user", "content": "legacy"}]}
    assert len(memory.import_legacy_history("s", old)) == 1
    assert memory.import_legacy_history("s", {"conversation_history": [
        {"role": "user", "content": "legacy"}]}) == []


def test_public_history_does_not_expose_archive_keys(db):
    from app.database.repositories import ResearchSessionRepository
    from app.api.routes_review import get_research_session_api
    ResearchSessionRepository(db).save("s", "created", "test", {
        "conversation_history": [{"role": "user", "content": "test"}]})
    db.commit()
    result = get_research_session_api("s", db)
    assert result.data["conversation_history"] == [{"role": "user", "content": "test"}]
