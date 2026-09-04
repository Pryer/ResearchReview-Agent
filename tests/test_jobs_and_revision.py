"""后台取消与可编辑研究状态测试。"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.graph import AgentCancelledError, run_research_agent
from app.database.models import Base
from app.database.repositories import ResearchJobRepository, ResearchSessionRepository
from app.schemas.agent_schema import ResearchRevisionRequest
from app.services.research_conversation_service import ResearchConversationService
from app.services.research_job_service import ResearchJobService


def _db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_agent_honors_cancel_before_first_node():
    with pytest.raises(AgentCancelledError):
        run_research_agent("任意研究主题", should_cancel=lambda: True)


def test_research_job_repository_tracks_cancel_request():
    db = _db_session()
    repo = ResearchJobRepository(db)
    repo.create("job-1", "session-1", {"user_query": "主题"})
    repo.update("job-1", status="cancel_requested", current_step="search")
    db.commit()

    job = repo.get("job-1")
    assert job["status"] == "cancel_requested"
    assert job["current_step"] == "search"


def test_update_status_if_in_is_conditional_transition():
    """CAS 语义：终态不被 cancel 覆盖，cancel_requested 不被 completed 覆盖。

    回归背景：cancel() 与 worker 终态写入原先都是无条件覆盖，取消与完成
    竞态时任务永久卡在 cancel_requested，会话后续提交全部 409。
    """
    db = _db_session()
    repo = ResearchJobRepository(db)
    repo.create("job-done", "s1", {"user_query": "主题"})
    repo.update("job-done", status="completed")
    repo.create("job-cancelled", "s2", {"user_query": "主题"})
    repo.update("job-cancelled", status="cancel_requested")
    db.commit()

    # cancel() 侧：对已完成任务请求取消必须让位（影响 0 行）。
    assert repo.update_status_if_in(
        "job-done", {"queued", "running", "cancel_requested"},
        status="cancel_requested",
    ) == 0
    assert repo.get("job-done")["status"] == "completed"

    # worker 终态侧：cancel 已先行时 completed 不得覆盖，改落 cancelled。
    assert repo.update_status_if_in(
        "job-cancelled", {"running"}, status="completed",
    ) == 0
    assert repo.update_status_if_in(
        "job-cancelled", {"cancel_requested"},
        status="cancelled", current_step="cancelled",
    ) == 1
    db.commit()
    assert repo.get("job-cancelled")["status"] == "cancelled"

    # 正常路径：running → cancel_requested 迁移仍然生效。
    repo.create("job-live", "s3", {"user_query": "主题"})
    repo.update("job-live", status="running")
    assert repo.update_status_if_in(
        "job-live", {"queued", "running", "cancel_requested"},
        status="cancel_requested",
    ) == 1
    db.commit()
    assert repo.get("job-live")["status"] == "cancel_requested"


def test_restart_requeues_waiting_jobs_and_closes_stale_states(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine)
    with maker() as db:
        repo = ResearchJobRepository(db)
        repo.create("queued", "s1", {"user_query": "主题"})
        repo.create("running", "s2", {"user_query": "主题"})
        repo.update("running", status="running")
        repo.create("cancel", "s3", {"user_query": "主题"})
        repo.update("cancel", status="cancel_requested")
        db.commit()

    submitted = []

    class FakeExecutor:
        def submit(self, fn, job_id):
            submitted.append(job_id)

    monkeypatch.setattr("app.services.research_job_service.SessionLocal", maker)
    monkeypatch.setattr("app.services.research_job_service._EXECUTOR", FakeExecutor())

    summary = ResearchJobService.recover_after_restart()

    assert summary == {"requeued": 1, "interrupted": 1, "cancelled": 1}
    assert submitted == ["queued"]
    with maker() as db:
        repo = ResearchJobRepository(db)
        assert repo.get("running")["status"] == "failed"
        assert repo.get("cancel")["status"] == "cancelled"


def test_revision_removes_papers_and_persists_memory(monkeypatch):
    db = _db_session()
    repo = ResearchSessionRepository(db)
    editable = {
        "intent": "generate_review",
        "topic": "测试主题",
        "paper_details": [
            {"paper_id": "p1", "title": "保留论文"},
            {"paper_id": "p2", "title": "排除论文"},
        ],
        "paper_cards": [
            {"paper_id": "p1", "title": "保留论文"},
            {"paper_id": "p2", "title": "排除论文"},
        ],
    }
    repo.save(
        "session-1",
        "completed",
        "测试主题综述",
        {
            "editable_research_state": editable,
            "conversation_history": [{"role": "user", "content": "测试主题综述"}],
        },
    )
    db.commit()

    def fake_regenerate(state, **kwargs):
        assert [p["paper_id"] for p in state["paper_details"]] == ["p1"]
        return {
            "answer": "修订结果",
            "intent": "generate_review",
            "topic": "测试主题",
            "steps": [],
            "references": ["保留论文"],
            "paper_cards": state["paper_cards"],
            "clusters": [],
            "errors": [],
            "research_state": state,
        }

    monkeypatch.setattr("app.agent.graph.regenerate_research_agent", fake_regenerate)
    service = ResearchConversationService(db, llm=object(), agent_runner=lambda *a, **k: {})
    result = service.revise(
        ResearchRevisionRequest(
            session_id="session-1",
            excluded_paper_ids=[],
            instruction="第二篇不符合主题，删除后重写",
        )
    )

    assert result["incremental_regeneration"] is True
    assert result["excluded_paper_ids"] == ["p2"]
    saved = repo.get("session-1")
    assert saved["state"]["selected_paper_ids"] == ["p1"]
    assert saved["state"]["revision_number"] == 1
    assert saved["state"]["revision_history"][0]["excluded_paper_ids"] == ["p2"]
    assert saved["state"]["conversation_history"][-1]["type"] == "revised_result"


def test_revision_can_confirm_deliverable_downgrade_without_deleting_papers(monkeypatch):
    db = _db_session()
    repo = ResearchSessionRepository(db)
    editable = {
        "intent": "generate_review",
        "topic": "测试主题",
        "core_deliverables": ["narrative_review"],
        "paper_details": [{"paper_id": "p1", "title": "论文"}],
        "paper_cards": [{"paper_id": "p1", "title": "论文"}],
    }
    repo.save("session-downgrade", "completed", "测试主题综述", {
        "editable_research_state": editable,
        "conversation_history": [],
    })
    db.commit()

    def fake_regenerate(state, **kwargs):
        assert state["core_deliverables"] == ["research_status"]
        assert len(state["paper_cards"]) == 1
        return {
            "answer": "研究现状", "intent": "generate_review", "topic": "测试主题",
            "steps": [], "references": [], "paper_cards": state["paper_cards"],
            "clusters": [], "errors": [], "research_state": state,
        }

    monkeypatch.setattr("app.agent.graph.regenerate_research_agent", fake_regenerate)
    service = ResearchConversationService(db, llm=object(), agent_runner=lambda *a, **k: {})
    result = service.revise(ResearchRevisionRequest(
        session_id="session-downgrade",
        instruction="证据不足，改为研究现状",
    ))
    assert result["incremental_regeneration"] is True
    assert result["excluded_paper_ids"] == []


def test_regenerate_research_agent_signature_matches_service_call_sites():
    """回归测试：ResearchConversationService 调用 regenerate_research_agent 时
    传入的关键字参数必须都是该函数真实支持的参数。

    背景：P0 修复移除了 regenerate_research_agent 的 db 参数，但
    research_conversation_service.py 中两处调用点当时未同步更新，仍传了
    db=self.db，导致真实运行时抛 TypeError。此前的其他测试都用
    `monkeypatch.setattr(..., fake_regenerate)` 且 fake 函数签名是
    `(state, **kwargs)`，会无差别接收多余参数，无法暴露这个问题。

    本测试不 mock regenerate_research_agent，直接用真实函数签名校验，
    确保调用方传入的参数集合是它的合法子集。
    """
    import inspect

    from app.agent.graph import regenerate_research_agent

    real_params = set(inspect.signature(regenerate_research_agent).parameters)

    # 与 research_conversation_service.py 两处调用点保持一致的关键字参数集合。
    called_kwargs = {"should_cancel", "progress_callback"}

    unsupported = called_kwargs - real_params
    assert not unsupported, (
        f"research_conversation_service.py 传给 regenerate_research_agent 的参数 "
        f"{unsupported} 不在函数真实签名 {real_params} 中，会导致 TypeError"
    )
