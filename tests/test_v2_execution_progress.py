"""实际Owner范围内的Job/API进度与离线报告，不混入同会话后续运行。"""
from sqlalchemy import select
import pytest
from app.agent.execution_budget import budget_scope
from app.core.config import get_settings
from app.database.models import ResearchJobModel, ResearchEventModel
from app.database.runtime_repository import RuntimeConflict
from app.services.durable_execution_service import durable_scope, active_runtime
from app.services.execution_progress import bind_job, job_progress
from app.services.execution_report_capture import Capture
from app.services.execution_report_service import freeze_source, build_report, write_report
from app.services.research_delegation_service import run_delegation
from app.services.tool_execution import execute_tool
from tests.test_v2_delegation_runtime import db, state
from tests.test_v2_search_runner import handlers


def add_job(db, job_id):
    db.add(ResearchJobModel(job_id=job_id, session_id='v2', status='running'))
    db.commit()
    return {'job_id': job_id, 'session_id': 'v2', 'status': 'running'}


def run_search(db, current, job):
    bind_job(active_runtime(), job['job_id'])
    operations = handlers([])
    original = operations['search_and_rank']
    operations['search_and_rank'] = lambda w, a: execute_tool('paper_search', lambda: original(w, a))
    result = run_delegation(current, {'agent': 'search', 'profile': 'initial_research', 'goal': 'research'},
                            llm=object(), handlers=operations)
    assert result.outcome == 'satisfied'


def test_job_progress_and_report_share_real_v2_execution(db, tmp_path, monkeypatch):
    from app.services.research_job_service import ResearchJobService
    from app.frontend.progress_labels import describe_execution_progress
    monkeypatch.setattr(get_settings(), 'execution_report_enabled', False)
    current, job = state(), add_job(db, 'first')
    cap = Capture('v2', 'research')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        cap.start_version, cap.owner = active_runtime().version, active_runtime().owner
        run_search(db, current, job)
        progress = ResearchJobService(db).get('first')['execution_progress']
        assert progress['completed_actions'] == 2 and progress['delegations'] == 1
        assert progress['tool_calls'] == progress['completed_tools'] == 1
        assert '已完成 2 个研究动作' in describe_execution_progress(progress)
        assert not any(k in progress for k in ('owner', 'task_id', 'delegation_id', 'path'))
        cap.end_version = active_runtime().version
        cap.source = freeze_source(db, cap)
    report = build_report(cap, {'status': 'completed'})
    assert len(report['calls']) == 2 and len(report['delegation_runs']) == 1
    assert report['summary']['tool_calls'] == 1
    assert report['calls'][0]['returned_to_main']['outcome'] == 'satisfied'
    assert not any('paper_resources' in event['payload'] for event in report['events'])
    write_report(report, tmp_path / 'report')
    assert '任务委派与工具回执' in (tmp_path / 'report' / 'report.html').read_text(encoding='utf-8')


def test_same_session_later_execution_does_not_change_old_job_counts(db):
    current, first = state(), add_job(db, 'first')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        run_search(db, current, first)
        old = job_progress(db, first)
    second = add_job(db, 'second')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        run_search(db, current, second)
        assert job_progress(db, first) == old
        assert job_progress(db, second)['completed_actions'] == 2


def test_runtime_cannot_bind_a_job_from_another_session(db):
    current = state()
    db.add(ResearchJobModel(job_id='foreign', session_id='other', status='running'))
    db.commit()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        with pytest.raises(RuntimeConflict):
            bind_job(active_runtime(), 'foreign')
        assert db.scalar(select(ResearchEventModel).where(ResearchEventModel.event_type == 'job_execution_bound')) is None


def test_failed_action_before_checkpoint_is_reported_without_fake_commit(db):
    current, job = state(), add_job(db, 'failed')
    cap = Capture('v2', 'research')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        cap.start_version, cap.owner = active_runtime().version, active_runtime().owner
        bind_job(active_runtime(), job['job_id'])
        def fail(w, args):
            raise ValueError('controlled upstream failure')
        with pytest.raises(ValueError):
            run_delegation(current, {'agent': 'search', 'profile': 'initial_research', 'goal': 'research'},
                           llm=object(), handlers={'search_and_rank': fail})
        cap.end_version = active_runtime().version
        cap.source = freeze_source(db, cap)
    report = build_report(cap, {'status': 'failed'})
    assert report['status'] == 'failed' and len(report['calls']) == 1
    assert report['calls'][0]['result']['status'] == 'failed'
    assert report['calls'][0]['commit_checkpoint'] is None


@pytest.mark.parametrize('architecture', ['v1', 'v2'])
def test_job_before_first_delegation_retains_frozen_architecture(db, monkeypatch, architecture):
    from app.frontend.progress_labels import describe_execution_progress
    current, job = state(), add_job(db, 'planning')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        bind_job(active_runtime(), job['job_id'])
        if architecture == 'v1':
            import json
            binding = db.scalar(select(ResearchEventModel).where(ResearchEventModel.event_type == 'job_execution_bound'))
            payload = json.loads(binding.payload_json)
            payload.pop('architecture_version')
            binding.payload_json = json.dumps(payload)
            db.commit()
        # 已冻结的任务不受进程配置随后变化影响。
        monkeypatch.setattr(get_settings(), 'agent_architecture_version', 'v1' if architecture == 'v2' else 'v2')
        job['status'] = 'needs_clarification'
        progress = job_progress(db, job)
        if architecture == 'v1':
            assert progress is None
        else:
            assert progress['architecture_version'] == 2 and progress['action_attempts'] == 0
            assert '等待补充研究信息' in describe_execution_progress(progress)


def test_takeover_unknown_receipt_stays_with_old_job_and_report(db, monkeypatch):
    from app.frontend.progress_labels import describe_execution_progress
    current, first = state(), add_job(db, 'interrupted')
    cap = Capture('v2', 'research')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        runtime = active_runtime()
        cap.start_version, cap.owner = runtime.version, runtime.owner
        # 模拟durable start后未收到回执，接管使用真实acquire的uncertain结算逻辑。
        monkeypatch.setattr(runtime, 'finish_tool_call', lambda *args: False)
        run_search(db, current, first)
        assert job_progress(db, first)['pending_tools'] == 1
        cap.end_version = runtime.version
    first['status'] = 'failed'
    second = add_job(db, 'resumed')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        run_search(db, current, second)
        old, new = job_progress(db, first), job_progress(db, second)
        assert old['uncertain_tools'] == 1 and old['pending_tools'] == old['completed_tools'] == 0
        assert old['current_role'] is None and '结果未知 1 次' in describe_execution_progress(old)
        assert new['completed_tools'] == 1 and new['uncertain_tools'] == 0
        cap.source = freeze_source(db, cap)
    report = build_report(cap, {'status': 'failed'})
    assert len(report['tool_calls']) == 1 and report['tool_calls'][0]['status'] == 'uncertain'
    assert len(report['calls']) == 2


def test_terminal_job_does_not_advertise_crashed_action_as_active(db):
    from app.frontend.progress_labels import describe_execution_progress
    current, job = state(), add_job(db, 'crashed')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        bind_job(active_runtime(), job['job_id'])
        def crash(w, args):
            raise SystemExit('process interrupted')
        with pytest.raises(SystemExit):
            run_delegation(current, {'agent': 'search', 'profile': 'initial_research', 'goal': 'research'},
                           llm=object(), handlers={'search_and_rank': crash})
        assert job_progress(db, job)['current_role'] == 'search'
        job['status'] = 'failed'
        progress = job_progress(db, job)
        assert progress['current_role'] is progress['current_action'] is progress['current_profile'] is None
        assert '执行已失败' in describe_execution_progress(progress)


def test_failed_tools_are_visible_in_progress_and_report(db):
    from app.frontend.progress_labels import describe_execution_progress
    current, job = state(), add_job(db, 'upstream')
    cap = Capture('v2', 'research')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        cap.start_version, cap.owner = active_runtime().version, active_runtime().owner
        bind_job(active_runtime(), job['job_id'])
        def fail(w, args):
            return execute_tool('paper_search', lambda: (_ for _ in ()).throw(ValueError('upstream')))
        with pytest.raises(ValueError):
            run_delegation(current, {'agent': 'search', 'profile': 'initial_research', 'goal': 'research'},
                           llm=object(), handlers={'search_and_rank': fail})
        progress = job_progress(db, job)
        assert progress['failed_tools'] == 1 and '失败 1 次' in describe_execution_progress(progress)
        assert progress['pending_tools'] == progress['uncertain_tools'] == 0
        cap.end_version = active_runtime().version
        cap.source = freeze_source(db, cap)
    report = build_report(cap, {'status': 'failed'})
    assert report['tool_calls'][0]['status'] == 'failed'
    assert len(report['calls']) == 1 and report['calls'][0]['commit_checkpoint'] is None
