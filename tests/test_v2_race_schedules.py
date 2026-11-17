"""固定种子与屏障交错，使用真实SQLite和Action提交内核；不随机sleep。"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import create_engine, event, select, update
from sqlalchemy.orm import Session

from app.agent.context_builder import context_source_fingerprint
from app.agent.controller import AgentController
from app.agent.execution_budget import budget_scope, budgeted_create, submit_with_context, is_control_exception
from app.agent.policy.authorization import AuthorizationPolicy
from app.agent.runtime.action_executor import ActionExecutor
from app.database.delegation_repository import DelegationRepository
from app.database.models import (Base, ResearchRuntimeModel as Runtime, ResearchInvocationModel as Invocation,
    ResearchAttemptModel as Attempt, ResearchWorkVersionModel as Work, ResearchEventModel as Event)
from app.database.runtime_repository import ResearchRuntimeRepository as Repository, cancel_runtime
from app.schemas.delegation_schema import DelegationRequest, ActionInvocation
from app.services.durable_execution_service import durable_scope, active_runtime
from tests.test_v2_delegation_runtime import state

MANIFEST = Path(__file__).resolve().parents[1] / 'examples/architecture-v2/race-schedules.json'
SCHEDULES = json.loads(MANIFEST.read_text(encoding='utf-8'))['schedules']


@pytest.fixture(scope='module')
def race_engine(tmp_path_factory):
    engine = create_engine('sqlite:///' + str(tmp_path_factory.mktemp('race-matrix') / 'races.db'),
        connect_args={'check_same_thread': False, 'timeout': 5})
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def test_race_manifest_has_unique_predeclared_schedules():
    for family in ('cancel_commit', 'takeover_commit', 'ack_interrupt'):
        rows = [row for row in SCHEDULES if row['family'] == family]
        assert len(rows) == 100 and len({row['seed'] for row in rows}) == 100
        assert len({json.dumps({k: v for k, v in row.items() if k != 'seed'}, sort_keys=True) for row in rows}) == 100
        assert {row['multi'] for row in rows} == {False, True}
        assert {row['worker'] for row in rows} == {False, True}
        expected = {'before_tool_ack', 'after_tool_ack', 'after_reservation', 'provider_inflight', 'after_settlement'} if family == 'ack_interrupt' else {'after_paid', 'before_cas', 'cas_locked', 'after_commit'}
        assert {row['phase'] for row in rows} == expected


@pytest.mark.parametrize('schedule', SCHEDULES,
    ids=lambda row: f"{row['family']}-{row['seed']}-{row['phase']}")
def test_seeded_commit_and_ack_interleavings(race_engine, monkeypatch, schedule):
    monkeypatch.setattr(Repository, '_start_heartbeat', lambda self: None)
    sid = f"race-{schedule['family']}-{schedule['seed']}"
    gate, resume, interrupt_sql, interrupt_done = (threading.Event() for _ in range(4))
    local, shared, actual_calls = threading.local(), {}, []
    family, phase = schedule['family'], schedule['phase']

    def pause(name):
        if name == phase and getattr(local, 'target', False):
            gate.set()
            assert resume.wait(10), f'scheduler did not release {schedule}'

    def before_sql(conn, cursor, sql, params, context, many):
        if sql.startswith('UPDATE research_runtime'):
            if getattr(local, 'actor', '') == 'interrupt':
                interrupt_sql.set()
            elif getattr(local, 'snapshot', False) and 'SET version=(research_runtime.version +' in sql:
                pause('before_cas')

    def after_sql(conn, cursor, sql, params, context, many):
        if getattr(local, 'snapshot', False) and sql.startswith('UPDATE research_runtime SET version=(research_runtime.version +'):
            pause('cas_locked')

    snapshot, begin, reserve, settle = Repository.snapshot, Repository.begin_tool_call, Repository.reserve_attempt, Repository.settle_attempt
    def snapshot_hook(repo, *args, **kwargs):
        target = repo.session_id == sid and kwargs.get('task') is not None and getattr(local, 'target', False)
        local.snapshot = target
        try:
            result = snapshot(repo, *args, **kwargs)
        finally:
            local.snapshot = False
        if target:
            pause('after_commit')
        return result
    def begin_hook(repo, *args, **kwargs):
        if repo.session_id == sid:
            pause('before_tool_ack')
        result = begin(repo, *args, **kwargs)
        if repo.session_id == sid:
            pause('after_tool_ack')
        return result
    def reserve_hook(repo, *args, **kwargs):
        result = reserve(repo, *args, **kwargs)
        if repo.session_id == sid:
            pause('after_reservation')
        return result
    def settle_hook(repo, attempt_id, *args, **kwargs):
        result = settle(repo, attempt_id, *args, **kwargs)
        if repo.session_id == sid:
            before = repo.db.get(Runtime, sid, populate_existing=True).budget_json
            for _ in range(schedule['duplicates']):
                settle(repo, attempt_id, {'total_tokens': 99999}, {'llm_tokens': 99999}, estimated=False)
                assert repo.db.get(Runtime, sid, populate_existing=True).budget_json == before
            pause('after_settlement')
        return result
    monkeypatch.setattr(Repository, 'snapshot', snapshot_hook)
    monkeypatch.setattr(Repository, 'begin_tool_call', begin_hook)
    monkeypatch.setattr(Repository, 'reserve_attempt', reserve_hook)
    monkeypatch.setattr(Repository, 'settle_attempt', settle_hook)

    def action_thread():
        local.actor, local.target = 'action', False
        current = state()
        current['session_id'] = sid
        paper = {'paper_id': f"synthetic-{schedule['seed']}", 'title': 'Offline race fixture'}
        with Session(race_engine) as db, budget_scope(current), durable_scope(db, sid, current, current['agent_execution_budget']):
            runtime, executor = active_runtime(), ActionExecutor(AgentController())
            repo = DelegationRepository(runtime)
            grant = AuthorizationPolicy().issue(DelegationRequest(agent='search', profile='initial_research', goal='race fixture'), current)
            repo.create(grant)
            runtime.snapshot(current, ledger=current['agent_execution_budget'])
            def product(w, args):
                if schedule['multi']:
                    w['paper_details'] = [paper]
                else:
                    w.update(candidate_papers=[paper], ranked_papers=[paper])
            if schedule['multi']:
                predecessor = ActionInvocation(delegation_id=grant.delegation_id, step_index=0, action='search_and_rank',
                    expected_version=0, source_fingerprint=context_source_fingerprint(current))
                executor.execute(predecessor, current, {'search_and_rank': lambda w, a: w.update(candidate_papers=[paper], ranked_papers=[paper])})
            operation = 'fetch_metadata' if schedule['multi'] else 'search_and_rank'
            invocation = ActionInvocation(delegation_id=grant.delegation_id, step_index=int(schedule['multi']),
                action=operation, expected_version=repo.version(), source_fingerprint=context_source_fingerprint(current))
            shared.update(invocation=invocation, owner=runtime.owner, product=product)
            def work(w, args):
                local.target = True
                for _ in range(schedule['refreshes']):
                    db.get(Runtime, sid)  # 保留旧ORM视图，执行端仍必须用SQL条件裁决。
                def paid(index):
                    local.actor, local.target = 'action', index == schedule['paid_calls'] - 1
                    def provider(**kwargs):
                        actual_calls.append(index)
                        pause('provider_inflight')
                        return NS(usage=NS(total_tokens=7, prompt_tokens=4, completion_tokens=3))
                    client = NS(chat=NS(completions=NS(create=provider)))
                    budgeted_create(client, messages=[{'role': 'user', 'content': 'offline fixture'}], max_tokens=8)
                if schedule['worker']:
                    with ThreadPoolExecutor(max_workers=1) as worker:
                        for index in range(schedule['paid_calls']):
                            submit_with_context(worker, paid, index).result(timeout=15)
                else:
                    for index in range(schedule['paid_calls']):
                        paid(index)
                local.target = True
                product(w, args)
                pause('after_paid')
            try:
                executor.execute(invocation, current, {operation: work})
                shared['action_completed'] = True
            except Exception as exc:
                assert is_control_exception(exc), f'{schedule}: unexpected {exc!r}'
                shared['action_completed'] = False

    def interrupt_thread():
        local.actor = 'interrupt'
        try:
            with Session(race_engine) as db:
                if schedule['interrupt'] == 'cancel':
                    cancel_runtime(db, sid)
                    db.commit()
                else:
                    # WHY: 测试调度器显式模拟原Owner租约结束，生产领取仍调用真实acquire CAS。
                    db.execute(update(Runtime).where(Runtime.session_id == sid).values(lease_until=0))
                    db.commit()
                    successor = Repository(db, sid)
                    successor.acquire({}, {})
                    shared['successor_owner'] = successor.owner
        finally:
            interrupt_done.set()

    event.listen(race_engine, 'before_cursor_execute', before_sql)
    event.listen(race_engine, 'after_cursor_execute', after_sql)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(action_thread)
            try:
                assert gate.wait(10), f'gate not reached: {schedule}'
                second = pool.submit(interrupt_thread)
                if phase == 'cas_locked':
                    assert interrupt_sql.wait(5), f'writer did not enter SQL: {schedule}'
                else:
                    assert interrupt_done.wait(5), f'interrupt did not finish: {schedule}'
            finally:
                resume.set()
            first.result(timeout=15)
            second.result(timeout=15)
    finally:
        resume.set()
        event.remove(race_engine, 'before_cursor_execute', before_sql)
        event.remove(race_engine, 'after_cursor_execute', after_sql)

    committed = family != 'ack_interrupt' and phase in {'cas_locked', 'after_commit'}
    before_rpc = family == 'ack_interrupt' and phase in {'before_tool_ack', 'after_tool_ack', 'after_reservation'}
    expected_calls = schedule['paid_calls'] - int(before_rpc)
    assert len(actual_calls) == expected_calls, schedule
    with Session(race_engine) as db:
        invocation = shared['invocation']
        row = db.get(Runtime, sid, populate_existing=True)
        ledger = json.loads(row.budget_json)
        attempts = list(db.scalars(select(Attempt).where(Attempt.session_id == sid)))
        uncertain = [attempt for attempt in attempts if attempt.status == 'uncertain']
        handoff_unknown = family == 'ack_interrupt' and schedule['interrupt'] == 'takeover' and phase in {'after_reservation', 'provider_inflight'}
        expected_cost = 7 * (schedule['paid_calls'] - 1) + uncertain[0].reserved_tokens if handoff_unknown else 7 * expected_calls
        assert len(uncertain) == int(handoff_unknown), schedule
        assert ledger.get('llm_tokens', 0) == expected_cost and ledger.get('tokens_reserved', 0) == 0, schedule
        expected_requests = schedule['paid_calls'] if handoff_unknown else expected_calls
        assert ledger.get('llm_requests', 0) == expected_requests, schedule
        assert ledger.get('estimated_tokens', 0) == (uncertain[0].reserved_tokens if handoff_unknown else 0), schedule
        assert db.get(Work, sid).version == int(schedule['multi']) + int(committed), schedule
        saved = db.get(Invocation, invocation.invocation_id)
        assert (saved.status == 'committed') == committed == shared['action_completed'], schedule
        rows = list(db.scalars(select(Event).where(Event.session_id == sid)))
        settled = [e.event_key for e in rows if e.event_type == 'attempt_settled']
        assert len(settled) == len(set(settled)), schedule
        terminals = [e.event_key for e in rows if e.event_type in {'tool_finished', 'tool_interrupted'}]
        assert len(terminals) == len(set(terminals)), schedule
        starts = [e for e in rows if e.event_type == 'tool_started']
        assert len(starts) == len(terminals) == schedule['paid_calls'] - int(family == 'ack_interrupt' and phase == 'before_tool_ack'), schedule
        # 模拟接管者随后退出；不能让测试恢复偷取仍有效的租约。
        db.execute(update(Runtime).where(Runtime.session_id == sid).values(lease_until=0))
        db.commit()
        restored = Repository(db, sid).restore()
        product_field = 'paper_details' if schedule['multi'] else 'candidate_papers'
        assert bool(restored.get(product_field)) == committed, schedule
        with budget_scope(restored), durable_scope(db, sid, restored, restored['agent_execution_budget']):
            def replay(w, args):
                assert not committed, f'committed action replayed: {schedule}'
                shared['product'](w, args)
            executor = ActionExecutor(AgentController())
            executor.execute(invocation, restored, {invocation.action: replay})
            executor.execute(invocation, restored, {invocation.action: lambda *a: pytest.fail('duplicate replay')})
            assert DelegationRepository(active_runtime()).version() == int(schedule['multi']) + 1
            assert restored['agent_execution_budget'].get('llm_tokens', 0) == expected_cost
            assert len(actual_calls) == expected_calls
