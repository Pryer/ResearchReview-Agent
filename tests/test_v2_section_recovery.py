"""章节缓存仅复用计算；取消、恢复、期限与正式候选提交分离。"""
import copy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import pytest
from sqlalchemy import select

from app.agent.execution_budget import (budget_scope, check_execution, active_budget, action_time_budget,
    AgentActionTimeout, AgentBudgetExceeded, AgentExecutionCancelled, budgeted_create, submit_with_context)
from app.agent.controller import AgentController
from app.agent.runtime.action_executor import ActionExecutor
from app.database.delegation_repository import DelegationRepository
from app.database.runtime_repository import RuntimeConflict
from app.database.models import ResearchArtifactModel
from app.services.durable_execution_service import durable_scope, active_runtime
from app.services.section_recovery_cache import cached_section, descriptor
from app.core.config import get_settings
from app.schemas.deliverable_schema import WritingPlan, WritingSection, CoreDeliverableType
from tests.test_tool_execution import invocation, events
from tests.test_v2_delegation_runtime import db
from tests.test_v2_artifact_resources import writer_state as base_writer_state


def writer_state():
    current = base_writer_state()
    current['claim_plans'] = [{'route_name': '问题背景', 'claims': [
        {'claim_id': 'allowed', 'claim_text': '原文支持研究问题', 'evidence_ids': ['p0:e1'], 'support_level': 'single'}]}]
    return current


def plan():
    return WritingPlan(deliverable_type=CoreDeliverableType.RESEARCH_BACKGROUND, purpose='章节恢复',
        organizing_strategy='evidence_driven', sections=[
            WritingSection(id='s1', title='问题背景', purpose='背景', supporting_paper_ids=['p0']),
            WritingSection(id='s2', title='研究方法', purpose='方法', supporting_paper_ids=['p1'])])


def response(section):
    title, pid = ('问题背景', 'p0') if section == 's1' else ('研究方法', 'p1')
    return f'## {title}\n\n该研究通过分层检索组织输入材料，并利用原文内容建立可回查的论据链条，使分析过程能够对应具体研究设置[{pid}]。'


class Writer:
    def __init__(self, calls, *, interrupted=False):
        self.calls, self.interrupted = calls, interrupted
    def complete(self, prompt, **kwargs):
        sid = kwargs['operation'].split(':')[2]
        self.calls.append(sid)
        if sid == 's2' and self.interrupted:
            raise AgentExecutionCancelled('stop after first section')
        client = NS(chat=NS(completions=NS(create=lambda **kw: NS(
            usage=NS(total_tokens=3, prompt_tokens=2, completion_tokens=1)))))
        budgeted_create(client, messages=[{'role': 'user', 'content': 'controlled section call'}], max_tokens=8)
        return response(sid)


def handler(writer):
    from app.deliverables.renderers.base_renderer import _write_sections_in_chinese
    def write(w, args):
        draft = response('s1') + '\n\n' + response('s2')
        w['review'] = _write_sections_in_chinese(draft, plan(), w,
            [{'paper_id': 'p0'}, {'paper_id': 'p1'}], writer)
        w['pending_result_validation'] = True
    return write


def test_cancelled_writer_recovers_durable_section_without_exposing_partial_candidate(db, monkeypatch):
    monkeypatch.setattr(get_settings(), 'writer_section_max_workers', 1)
    current, calls = writer_state(), []
    current['writing_plans'] = [plan().model_dump(mode='json')]
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        active_runtime().snapshot(current, ledger=current['agent_execution_budget'])
        action = invocation(current, operation='generate_deliverables', role='writing', profile='draft_generation')
        with pytest.raises(AgentExecutionCancelled):
            ActionExecutor(AgentController()).execute(action, current,
                {'generate_deliverables': handler(Writer(calls, interrupted=True))})
        assert len(events(db, 'section_calculation_cached')) == 1
        assert not current.get('review') and not current.get('pending_result_validation')
        assert DelegationRepository(active_runtime()).version() == 0
        assert current['agent_execution_budget']['llm_tokens'] == 3
        restored = active_runtime().restore()
    with budget_scope(restored), durable_scope(db, 'v2', restored, restored['agent_execution_budget']):
        ActionExecutor(AgentController()).execute(action, restored,
            {'generate_deliverables': handler(Writer(calls))})
        assert calls == ['s1', 's2', 's2']
        assert restored['agent_execution_budget']['llm_tokens'] == 6
        assert restored['pending_result_validation'] and not restored.get('check_report')
        assert all(title in restored['review'] for title in ('问题背景', '研究方法'))
        assert DelegationRepository(active_runtime()).version() == 1
        assert len(events(db, 'section_calculation_cached')) == 2


@pytest.mark.parametrize('field', ['review', 'claim_plans', 'paper_cards', 'prompt', 'writer', 'years'])
def test_cache_fingerprint_tracks_base_authorization_evidence_prompt_and_model(field):
    current, p = writer_state(), plan()
    before = descriptor(p, p.sections[0], current, object(), 'prompt')
    modified = copy.deepcopy(current)
    prompt, model = 'prompt', object()
    if field == 'review': modified['review'] = 'changed base'
    elif field == 'claim_plans': modified['claim_plans'] = [{'claims': [{'claim_text': 'changed', 'evidence_ids': ['p0:e1']}]}]
    elif field == 'paper_cards': modified['paper_cards'][0]['evidence_spans'][0]['text'] = 'changed original evidence'
    elif field == 'prompt': prompt = 'changed prompt'
    elif field == 'years': modified['end_year'] = 2024
    else: model = NS(model='changed-writer')
    assert descriptor(p, p.sections[0], modified, model, prompt) != before


def test_cache_is_immutable_and_corruption_stops_reuse(db):
    current = writer_state()
    key = {'deliverable': 'research_background', 'section_id': 's1', 'fingerprint': 'a' * 64}
    def work(w, args):
        assert cached_section(key) is None
        assert cached_section(key, text=response('s1')) == response('s1')
        assert cached_section(key, text='later output') == response('s1')
        assert cached_section({**key, 'fingerprint': 'b' * 64}) is None
        row = db.scalar(select(ResearchArtifactModel).where(ResearchArtifactModel.artifact_type == 'writer_candidate'))
        row.payload_json = '{}'
        db.commit()
        with pytest.raises(RuntimeConflict, match='cache artifact'):
            cached_section(key)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current, operation='generate_deliverables', role='writing', profile='draft_generation')
        ActionExecutor(AgentController()).execute(action, current, {'generate_deliverables': work})


def test_search_cannot_write_section_cache(db):
    from tests.test_v2_delegation_runtime import state
    current = state()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        with pytest.raises(RuntimeConflict, match='capability'):
            ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': lambda w, a:
                cached_section({'deliverable': 'background', 'section_id': 's1', 'fingerprint': 'a' * 64}, text='text')})


def test_action_deadline_is_inherited_by_workers_and_restored_after_scope(monkeypatch):
    from tests.test_v2_delegation_runtime import state
    clock = [100.0]
    monkeypatch.setattr('app.agent.execution_budget.time.monotonic', lambda: clock[0])
    with budget_scope(state()):
        parent = active_budget()['deadline']
        with action_time_budget(10):
            assert active_budget()['deadline'] == 110
            with ThreadPoolExecutor(max_workers=1) as pool:
                clock[0] = 111
                with pytest.raises(AgentActionTimeout):
                    submit_with_context(pool, check_execution).result(timeout=5)
        assert active_budget()['deadline'] == parent
        check_execution()


def test_global_deadline_remains_stricter_than_action_deadline(monkeypatch):
    from tests.test_v2_delegation_runtime import state
    clock = [100.0]
    monkeypatch.setattr('app.agent.execution_budget.time.monotonic', lambda: clock[0])
    with budget_scope(state()):
        active_budget()['deadline'] = 105
        with action_time_budget(10):
            clock[0] = 106
            with pytest.raises(AgentBudgetExceeded, match='research execution') as caught:
                check_execution()
            assert not isinstance(caught.value, AgentActionTimeout)


def revision(current):
    from app.agent.policy.authorization import AuthorizationPolicy
    from app.schemas.delegation_schema import DelegationRequest, ActionInvocation
    from app.agent.context_builder import context_source_fingerprint
    repo = DelegationRepository(active_runtime())
    grant = AuthorizationPolicy().issue(DelegationRequest(agent='writing', profile='section_revision', goal='revise',
        scope={'section_ids': ['s1']}), current)
    repo.create(grant)
    return ActionInvocation(delegation_id=grant.delegation_id, step_index=0, action='rewrite_sections',
        arguments={'section_ids': ['s1']}, expected_version=repo.version(), source_fingerprint=context_source_fingerprint(current))


def test_expired_revision_cannot_commit_candidate_and_restores_global_deadline(db, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('app.agent.execution_budget.time.monotonic', lambda: clock[0])
    monkeypatch.setattr(get_settings(), 'agent_section_revision_timeout_seconds', 10)
    current = writer_state()
    current.update(review=response('s1') + response('s2'), writing_plans=[plan().model_dump(mode='json')])
    original = current['review']
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        parent = active_budget()['deadline']
        action = revision(current)
        def expired(w, args):
            w.update(review='expired candidate', pending_result_validation=True)
            clock[0] += 11
        with pytest.raises(AgentActionTimeout):
            ActionExecutor(AgentController()).execute(action, current, {'rewrite_sections': expired})
        assert current['review'] == original and not current.get('pending_result_validation')
        assert DelegationRepository(active_runtime()).version() == 0
        assert active_budget()['deadline'] == parent
        check_execution()


def test_revision_scope_cannot_read_or_write_other_section_cache(db):
    current = writer_state()
    current.update(review=response('s1') + response('s2'), writing_plans=[plan().model_dump(mode='json')])
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        def work(w, args):
            outside = {'deliverable': 'research_background', 'section_id': 's2', 'fingerprint': 'a' * 64}
            with pytest.raises(RuntimeConflict, match='revision scope'):
                cached_section(outside)
            with pytest.raises(RuntimeConflict, match='revision scope'):
                cached_section(outside, text=response('s2'))
        ActionExecutor(AgentController()).execute(revision(current), current, {'rewrite_sections': work})


def test_request_timeout_uses_remaining_action_deadline():
    from tests.test_v2_delegation_runtime import state
    seen = []
    client = NS(chat=NS(completions=NS(create=lambda **kw: seen.append(kw['timeout']) or NS(
        usage=NS(total_tokens=3, prompt_tokens=2, completion_tokens=1)))))
    with budget_scope(state()), action_time_budget(10):
        budgeted_create(client, messages=[], max_tokens=8, timeout=60)
    assert 0 < seen[0] <= 10


def test_sidecar_consumes_same_revision_deadline_before_handler(db, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('app.agent.execution_budget.time.monotonic', lambda: clock[0])
    settings = get_settings()
    monkeypatch.setattr(settings, 'agent_section_revision_timeout_seconds', 10)
    monkeypatch.setattr(settings, 'agent_sidecar_enabled', True)
    monkeypatch.setattr(settings, 'agent_sidecar_review_actions', 'rewrite_sections')
    current = writer_state()
    current.update(review=response('s1') + response('s2'), writing_plans=[plan().model_dump(mode='json')])
    class Review:
        def review(self, **kwargs):
            clock[0] += 11
            return NS(verdict='allow', reason='review completed')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        with pytest.raises(AgentActionTimeout):
            ActionExecutor(AgentController(), sidecar=Review()).execute(revision(current), current,
                {'rewrite_sections': lambda *args: pytest.fail('review cannot reset the action deadline')})
        assert DelegationRepository(active_runtime()).version() == 0


def test_real_process_exit_recovers_first_section_before_action_commit(tmp_path):
    import subprocess
    import sys
    from sqlalchemy import create_engine, update
    from sqlalchemy.orm import Session
    from app.database.models import ResearchRuntimeModel, ResearchInvocationModel
    from app.database.runtime_repository import ResearchRuntimeRepository
    from app.schemas.delegation_schema import ActionInvocation
    path = tmp_path / 'section-crash.db'
    script = r'''
import os,sys
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.database.models import Base
from tests.test_v2_section_recovery import *
class CrashWriter(Writer):
    def complete(self,prompt,**kwargs):
        if ':s2:' in kwargs['operation']: os._exit(17)
        return super().complete(prompt,**kwargs)
get_settings().writer_section_max_workers=1
engine=create_engine('sqlite:///'+sys.argv[1],connect_args={'check_same_thread':False})
Base.metadata.create_all(engine)
current=writer_state()
current['writing_plans']=[plan().model_dump(mode='json')]
with Session(engine) as db,budget_scope(current),durable_scope(db,'v2',current,current['agent_execution_budget']):
    active_runtime().snapshot(current,ledger=current['agent_execution_budget'])
    action=invocation(current,operation='generate_deliverables',role='writing',profile='draft_generation')
    ActionExecutor(AgentController()).execute(action,current,{'generate_deliverables':handler(CrashWriter([]))})
'''
    child = subprocess.run([sys.executable, '-c', script, str(path)], capture_output=True, text=True, timeout=30)
    assert child.returncode == 17, child.stderr
    engine = create_engine('sqlite:///' + str(path), connect_args={'check_same_thread': False})
    try:
        with Session(engine) as session:
            assert len(events(session, 'section_calculation_cached')) == 1
            saved = session.scalar(select(ResearchInvocationModel))
            assert saved.status == 'running'
            action = ActionInvocation.model_validate_json(saved.invocation_json)
            session.execute(update(ResearchRuntimeModel).values(lease_until=0))
            session.commit()
            restored = ResearchRuntimeRepository(session, 'v2').restore()
            calls = []
            with budget_scope(restored), durable_scope(session, 'v2', restored, restored['agent_execution_budget']):
                ActionExecutor(AgentController()).execute(action, restored, {'generate_deliverables': handler(Writer(calls))})
                assert calls == ['s2']
                assert restored['agent_execution_budget']['llm_tokens'] == 6
                assert restored['pending_result_validation'] and not restored.get('check_report')
                assert DelegationRepository(active_runtime()).version() == 1
    finally:
        engine.dispose()
