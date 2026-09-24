"""真实 SQLite 事务故障注入：并发、崩溃窗口、取消和恢复。"""
import copy
import json
import time
from types import SimpleNamespace as NS
import pytest
from sqlalchemy import create_engine, select, update, func
from sqlalchemy.orm import Session
from app.database.models import (Base, ResearchRuntimeModel, ResearchTaskModel,
    ResearchAttemptModel, ResearchCheckpointModel, ResearchEventModel)
from app.database.runtime_repository import ResearchRuntimeRepository, RuntimeConflict, cancel_runtime
from app.schemas.agent_task_schema import AgentTask, AgentTaskResult, AgentRole, AgentTaskStatus
from app.services.durable_execution_service import durable_scope
from app.agent.execution_budget import budget_scope, budgeted_create, AgentExecutionCancelled
from app.agent.controller import AgentController


@pytest.fixture
def db(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "runtime.db"), connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def state():
    return {"session_id": "s", "user_query": "test", "topic": "test", "steps": [], "errors": [],
            "research_plan": {"task_graph": []}, "agent_orchestration_mode": "autonomous"}


def task(key="one"):
    return AgentTask(role=AgentRole.SEARCH, operation="search_and_rank", objective="search",
        source_state_version="1", source_state_fingerprint="fp", idempotency_key=key)


def result(t):
    return AgentTaskResult(task_id=t.task_id, role=t.role, operation=t.operation,
        status=AgentTaskStatus.COMPLETED, input_state_fingerprint="fp")


def test_two_workers_cannot_acquire_same_live_session(db):
    a = ResearchRuntimeRepository(db, "s")
    a.acquire({}, {})
    with Session(db.bind) as other:
        b = ResearchRuntimeRepository(other, "s")
        with pytest.raises(RuntimeConflict):
            b.acquire({}, {})


def test_preselected_action_and_dequeue_commit_in_one_checkpoint(db):
    current = state()
    current["agent_mandatory_actions"] = [{"action": "search_and_rank"}]
    with budget_scope(current):
        with durable_scope(db, "s", current, current["agent_execution_budget"]) as runtime:
            AgentController().execute(
                role=AgentRole.SEARCH,
                operation="search_and_rank",
                objective="履行已选检索",
                state=current,
                handler=lambda working: working.update(candidate_papers=[{"paper_id": "p1"}]),
                consume_mandatory_action=True,
            )
            checkpoint = runtime.restore()
            assert checkpoint["candidate_papers"] == [{"paper_id": "p1"}]
            assert not checkpoint.get("agent_mandatory_actions")
            assert not current.get("agent_mandatory_actions")


def test_simultaneous_first_acquire_has_exactly_one_owner(db):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(2)
    def acquire():
        with Session(db.bind) as worker:
            repository = ResearchRuntimeRepository(worker, "first-race")
            barrier.wait(timeout=5)
            try:
                repository.acquire({}, {})
                return "owner"
            except RuntimeConflict:
                return "rejected"
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(acquire) for _ in range(2)]
        assert sorted(item.result(timeout=10) for item in futures) == ["owner", "rejected"]


def test_cas_rejects_other_writer_without_partial_task_or_artifact_commit(db):
    r = ResearchRuntimeRepository(db, "s")
    r.acquire({}, {})
    t = task()
    r.begin_task(t, {})
    with Session(db.bind) as other:
        other.execute(update(ResearchRuntimeModel).where(ResearchRuntimeModel.session_id == "s").values(version=1))
        other.commit()
    with pytest.raises(RuntimeConflict):
        r.snapshot({"paper_cards": [{"paper_id": "old"}]}, task=t, result=result(t))
    assert db.get(ResearchTaskModel, t.task_id).status == "running"
    assert db.scalar(select(func.count()).select_from(ResearchCheckpointModel)) == 0


def test_cancel_before_commit_rejects_late_result_but_retains_paid_usage(db):
    current = state()
    with budget_scope(current):
        ledger = current["agent_execution_budget"]
        with durable_scope(db, "s", current, ledger):
            def handler(working):
                response = NS(usage=NS(prompt_tokens=10, completion_tokens=5))
                client = NS(chat=NS(completions=NS(create=lambda **kw: response)))
                budgeted_create(client, messages=[], max_tokens=10)
                working["candidate_papers"] = [{"paper_id": "late"}]
                with Session(db.bind) as other:
                    cancel_runtime(other, "s")
                    other.commit()
            with pytest.raises(AgentExecutionCancelled):
                AgentController().execute(role=AgentRole.SEARCH, operation="search_and_rank",
                    objective="search", state=current, handler=handler)
            assert "candidate_papers" not in current
            assert ledger["llm_tokens"] == 15
    assert db.scalar(select(ResearchTaskModel)).status == "cancelled"
    assert json.loads(db.get(ResearchRuntimeModel, "s").budget_json)["llm_tokens"] == 15


def test_budget_refusal_marks_task_blocked_instead_of_user_cancelled(db):
    from app.agent.execution_budget import AgentBudgetExceeded
    current = state()
    current["agent_execution_budget"] = {"token_limit": 1}
    with budget_scope(current):
        with durable_scope(db, "s", current, current["agent_execution_budget"]):
            def handler(working):
                budgeted_create(NS(), messages=[], max_tokens=20)
            with pytest.raises(AgentBudgetExceeded):
                AgentController().execute(role=AgentRole.SEARCH, operation="search_and_rank",
                    objective="search", state=current, handler=handler)
    assert db.scalar(select(ResearchTaskModel)).status == "blocked"
    assert current["agent_execution_status"] == "blocked"


def test_crash_with_unsettled_request_marks_uncertain_and_never_repeats_same_task(db):
    a = ResearchRuntimeRepository(db, "s", ttl=100)
    ledger = {"llm_tokens": 0, "tokens_reserved": 400}
    a.acquire({}, ledger)
    t = task()
    a.begin_task(t, ledger)
    a.reserve_attempt(t.task_id, 400, ledger)
    db.execute(update(ResearchRuntimeModel).values(lease_until=time.time() - 1))
    db.commit()
    with Session(db.bind) as other:
        b = ResearchRuntimeRepository(other, "s")
        restored_ledger = {}
        b.acquire({}, restored_ledger)
        assert restored_ledger["llm_tokens"] == 400
        assert restored_ledger["tokens_reserved"] == 0
        with pytest.raises(RuntimeConflict, match="interrupted"):
            b.begin_task(t, restored_ledger)
        assert other.scalar(select(ResearchAttemptModel)).status == "uncertain"


def test_atomic_snapshot_restores_committed_evidence_after_restart(db):
    current = state()
    with budget_scope(current):
        with durable_scope(db, "s", current, current["agent_execution_budget"]):
            AgentController().execute(role=AgentRole.SEARCH, operation="search_and_rank",
                objective="search", state=current,
                handler=lambda working: working.update(candidate_papers=[{"paper_id": "p1"}]))
    with Session(db.bind) as other:
        runtime = ResearchRuntimeRepository(other, "s")
        runtime.acquire({}, {})
        restored = runtime.restore()
        assert restored["candidate_papers"] == [{"paper_id": "p1"}]
        assert restored["agent_task_results"][0]["status"] == "completed"
        assert other.scalar(select(ResearchTaskModel)).status == "completed"
        assert other.scalar(select(func.count()).select_from(ResearchCheckpointModel)) == 1


def test_commit_failure_rolls_back_state_task_and_snapshot(db, monkeypatch):
    from app.services.research_artifact_service import ResearchArtifactService
    runtime = ResearchRuntimeRepository(db, "s")
    runtime.acquire({}, {})
    t = task()
    runtime.begin_task(t, {})
    def fail(*args, **kwargs):
        raise RuntimeError("disk fault")
    monkeypatch.setattr(ResearchArtifactService, "externalize_state", fail)
    with pytest.raises(RuntimeError, match="disk fault"):
        runtime.snapshot({}, task=t, result=result(t))
    assert db.get(ResearchRuntimeModel, "s").version == 0
    assert db.get(ResearchTaskModel, t.task_id).status == "running"


def test_cancelled_response_is_still_counted_in_metrics(db):
    from app.services.llm_service import LLMService
    from app.core.metrics import get_metrics_collector
    service = LLMService()
    service.api_key, service.backup_enabled = "test", False
    def create(**kw):
        with Session(db.bind) as other:
            cancel_runtime(other, "s")
            other.commit()
        return NS(usage=NS(prompt_tokens=10, completion_tokens=5),
            choices=[NS(message=NS(content="paid", reasoning_content=None), finish_reason="stop")])
    service._client = NS(chat=NS(completions=NS(create=create)))
    get_metrics_collector().reset()
    current = state()
    with budget_scope(current):
        with durable_scope(db, "s", current, current["agent_execution_budget"]):
            with pytest.raises(AgentExecutionCancelled):
                service.complete("test", max_tokens=20)
    assert get_metrics_collector().get_token_report()["total_tokens"] == 15


def test_late_old_owner_cannot_settle_over_recovered_budget(db):
    old = ResearchRuntimeRepository(db, "s")
    ledger = {"llm_tokens": 0, "tokens_reserved": 400}
    old.acquire({}, ledger)
    attempt = old.reserve_attempt(None, 400, ledger)
    db.execute(update(ResearchRuntimeModel).values(lease_until=time.time() - 1))
    db.commit()
    with Session(db.bind) as other:
        new = ResearchRuntimeRepository(other, "s")
        recovered = {}
        new.acquire({}, recovered)
        old.settle_attempt(attempt, {"total_tokens": 10}, {"llm_tokens": 10}, estimated=False)
        assert json.loads(other.get(ResearchRuntimeModel, "s", populate_existing=True).budget_json)["llm_tokens"] == 400


def test_repeated_terminal_calls_do_not_double_settle_or_duplicate_events(db):
    r = ResearchRuntimeRepository(db, "s")
    r.acquire({}, {})
    t = task()
    r.begin_task(t, {})
    attempt = r.reserve_attempt(t.task_id, 400, {})
    r.settle_attempt(attempt, {"total_tokens": 10}, {"llm_tokens": 10}, estimated=False)
    r.settle_attempt(attempt, {"total_tokens": 10}, {"llm_tokens": 20}, estimated=False)
    assert json.loads(db.get(ResearchRuntimeModel, "s").budget_json)["llm_tokens"] == 10
    r.fail_task(t.task_id, "failed", {"llm_tokens": 10})
    r.fail_task(t.task_id, "cancelled", {"llm_tokens": 10})
    assert db.get(ResearchTaskModel, t.task_id).status == "failed"


def test_public_session_save_cannot_publish_after_cancel(db):
    from app.database.repositories import ResearchSessionRepository
    current = state()
    with budget_scope(current):
        with durable_scope(db, "s", current, current["agent_execution_budget"]):
            with Session(db.bind) as other:
                cancel_runtime(other, "s")
                other.commit()
            with pytest.raises(RuntimeConflict):
                ResearchSessionRepository(db).save("s", "completed", "test", {"result_snapshot": {"answer": "late"}})
            db.rollback()


def test_session_read_restores_newer_checkpoint_with_structured_diagnostics(db):
    from app.database.repositories import ResearchSessionRepository
    from app.schemas.paper_schema import SourceDiagnostic
    repo = ResearchSessionRepository(db)
    repo.save("s", "running", "test", {"editable_research_state": {"topic": "old"}})
    db.commit()
    runtime = ResearchRuntimeRepository(db, "s")
    runtime.acquire({}, {})
    runtime.snapshot({"topic": "new", "source_diagnostics": [SourceDiagnostic(source="cnki", status="success")],
                      "paper_cards": [{"paper_id": "p1"}]}, ledger={"llm_tokens": 12})
    runtime.release({"llm_tokens": 12})
    with Session(db.bind) as fresh:
        restored = ResearchSessionRepository(fresh).get("s")["state"]["editable_research_state"]
        assert restored["topic"] == "new"
        assert restored["paper_cards"] == [{"paper_id": "p1"}]
        assert restored["source_diagnostics"][0]["source"] == "cnki"
        assert restored["agent_execution_budget"]["llm_tokens"] == 12


def test_large_task_patch_is_fragmented_and_restored_without_dropping_evidence(db):
    from app.services.research_artifact_service import ResearchArtifactService
    service = ResearchArtifactService(db)
    results = [{"state_patch": {"candidate_papers": [
        {"paper_id": str(i), "abstract": "evidence " * 1000} for i in range(45)]}}]
    stored = service.externalize_state("s", {"agent_task_results": results})
    assert len(stored["artifact_manifest"]["agent_task_results"]["items"]) > 1
    assert service.hydrate_state("s", stored)["agent_task_results"] == results


def test_additive_runtime_migration_is_idempotent_and_preserves_existing_session(tmp_path):
    from app.database.models import ResearchSessionModel
    from scripts.migrate_agent_runtime import migrate
    engine = create_engine("sqlite:///" + str(tmp_path / "old.db"))
    ResearchSessionModel.__table__.create(engine)
    with Session(engine) as db:
        db.add(ResearchSessionModel(session_id="old", original_query="preserve"))
        db.commit()
    migrate(engine)
    migrate(engine)
    with Session(engine) as db:
        assert db.get(ResearchSessionModel, "old").original_query == "preserve"
        assert db.scalar(select(func.count()).select_from(ResearchRuntimeModel)) == 0
    engine.dispose()


def test_explicit_resume_uses_committed_state_and_keeps_budget_and_constraints(db, monkeypatch):
    from app.database.repositories import ResearchSessionRepository
    from app.services.research_conversation_service import ResearchConversationService
    from app.schemas.agent_schema import AgentRequest
    repo = ResearchSessionRepository(db)
    repo.save("s", "failed", "original topic", {"editable_research_state": {"topic": "old"}})
    db.commit()
    runtime = ResearchRuntimeRepository(db, "s")
    ledger = {"llm_tokens": 456, "actions_reserved": 3}
    runtime.acquire({}, ledger)
    runtime.snapshot({**state(), "required_reference_count": 40, "start_year": 2024, "end_year": 2026,
                      "paper_cards": [{"paper_id": "p1"}], "user_operation_sequence": 2}, ledger=ledger)
    runtime.release(ledger)
    seen = {}
    def run(current, **kwargs):
        from app.agent.execution_budget import active_budget
        seen.update(copy.deepcopy(current))
        seen["budget"] = dict(active_budget()["ledger"])
        return {"status": "blocked", "research_state": current}
    monkeypatch.setattr("app.agent.graph._run_autonomous_pipeline", run)
    service = ResearchConversationService(db, llm=object())
    monkeypatch.setattr(service, "_persist_or_pause_result", lambda sid, query, saved, result: result)
    service.handle(AgentRequest(session_id="s", user_query="继续", resume_from_checkpoint=True))
    assert seen["paper_cards"] == [{"paper_id": "p1"}]
    assert seen["required_reference_count"] == 40
    assert (seen["start_year"], seen["end_year"]) == (2024, 2026)
    assert seen["budget"]["llm_tokens"] == 456
    assert seen["user_operation_sequence"] == 3


def test_losing_job_cancel_race_does_not_cancel_live_runtime(db, monkeypatch):
    from app.database.repositories import ResearchJobRepository
    from app.services.research_job_service import ResearchJobService
    ResearchJobRepository(db).create("job", "s", {"user_query": "test"})
    db.commit()
    runtime = ResearchRuntimeRepository(db, "s")
    runtime.acquire({}, {})
    service = ResearchJobService(db)
    monkeypatch.setattr(service.repo, "update_status_if_in", lambda *args, **kwargs: 0)
    service.cancel("job")
    assert not db.get(ResearchRuntimeModel, "s", populate_existing=True).cancelled


def test_duplicate_job_delivery_never_executes_completed_job_again(db, monkeypatch):
    from sqlalchemy.orm import sessionmaker
    from app.database.repositories import ResearchJobRepository
    from app.services.research_job_service import ResearchJobService
    from app.services.research_conversation_service import ResearchConversationService
    ResearchJobRepository(db).create("job", "s", {"session_id": "s", "user_query": "test"})
    db.commit()
    calls = []
    monkeypatch.setattr("app.services.research_job_service.SessionLocal", sessionmaker(bind=db.bind))
    monkeypatch.setattr(ResearchConversationService, "handle", lambda *args: calls.append(1) or {"status": "blocked"})
    ResearchJobService._run_job("job")
    ResearchJobService._run_job("job")
    assert calls == [1]


def test_explicit_budget_update_preserves_usage_and_survives_restore(db):
    from app.database.repositories import ResearchSessionRepository
    runtime = ResearchRuntimeRepository(db, "budget-update")
    ledger = {"token_limit": 100000, "llm_tokens": 84601, "retrieval_used": 1}
    runtime.acquire({}, ledger)
    runtime.snapshot({"topic": "研究", "required_reference_count": 40}, ledger=ledger)
    runtime.release(ledger)
    sessions = ResearchSessionRepository(db)
    sessions.save(session_id="budget-update", status="blocked", original_query="研究",
        state={"runtime_checkpoint_version": runtime.version, "agent_execution_budget": dict(ledger),
               "editable_research_state": {"agent_execution_budget": dict(ledger)}})
    db.commit()
    updated = runtime.update_token_limit(expected_limit=100000, new_limit=1000000)
    assert updated == {**ledger, "token_limit": 1000000}
    restored = runtime.restore()
    assert restored["agent_execution_budget"] == updated
    assert restored["required_reference_count"] == 40
    public_state = sessions.get("budget-update")["state"]
    assert public_state["agent_execution_budget"] == updated
    assert public_state["editable_research_state"]["agent_execution_budget"] == updated
    event = db.scalar(select(ResearchEventModel).where(ResearchEventModel.event_type == "budget_limit_updated"))
    assert json.loads(event.payload_json)["consumed_tokens"] == 84601
    with pytest.raises(RuntimeConflict):
        runtime.update_token_limit(expected_limit=100000, new_limit=2000000)
    with pytest.raises(ValueError):
        runtime.update_token_limit(expected_limit=1000000, new_limit=10)


def test_budget_update_rejects_owned_runtime_and_queued_job(db):
    from app.database.models import ResearchJobModel
    runtime = ResearchRuntimeRepository(db, "busy-budget")
    ledger = {"token_limit": 100000, "llm_tokens": 123}
    runtime.acquire({}, ledger)
    with pytest.raises(RuntimeConflict):
        runtime.update_token_limit(expected_limit=100000, new_limit=1000000)
    runtime.release(ledger)
    db.add(ResearchJobModel(job_id="budget-job", session_id="busy-budget", status="queued"))
    db.commit()
    with pytest.raises(RuntimeConflict):
        runtime.update_token_limit(expected_limit=100000, new_limit=1000000)
    assert json.loads(db.get(ResearchRuntimeModel, "busy-budget").budget_json) == ledger
