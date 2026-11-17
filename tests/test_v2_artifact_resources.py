"""真实Artifact/Controller/SQLite：冻结输入、派生产物、候选和线程归属。"""
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import select

from app.agent.context_builder import context_source_fingerprint
from app.agent.controller import AgentController
from app.agent.execution_budget import budget_scope, submit_with_context
from app.agent.runtime.action_executor import ActionExecutor
from app.database.delegation_repository import DelegationRepository
from app.database.models import ResearchArtifactModel, ResearchEventModel as Event
from app.database.runtime_repository import RuntimeConflict
from app.schemas.delegation_schema import ActionInvocation
from app.services.durable_execution_service import durable_scope, active_runtime
from app.services.research_artifact_service import ResearchArtifactService
from app.services.research_execution_service import _STORE
from app.services.tool_execution import ToolAuthorizationError
from app.services.writer_support import record_candidate, read_candidate
from tests.test_tool_execution import invocation, events
from tests.test_v2_delegation_runtime import db, state


def seed_input(store, current):
    current['paper_details'] = [{'paper_id': 'p', 'title': 'Study', 'year': 2024}]
    frozen = store.externalize_state('v2', current)
    current['artifact_manifest'] = frozen['artifact_manifest']
    store.repo.db.commit()
    return frozen['artifact_manifest']['paper_details']['items'][0]['ref']


def writer_state():
    from tests.test_main_quality_loop import prepared
    current = prepared()
    current['session_id'] = 'v2'
    return current


def test_artifact_read_uses_frozen_input_and_preserves_source(db):
    current, store = state(), ResearchArtifactService(db)
    ref = seed_input(store, current)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        def work(w, args):
            value = store.resolve('v2', ref, role='service', allowed_types={'paper_metadata'})
            assert value['payload']['value']['paper_id'] == 'p'
        ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': work})
        assert [e['tool'] for e in events(db, 'tool_started')] == ['artifact_read']
        assert len(events(db, 'action_resources_bound')) == 1
        assert DelegationRepository(active_runtime()).version() == 1


@pytest.mark.parametrize('kind', ['unbound', 'cross_session', 'wrong_version', 'wrong_type'])
def test_artifact_read_rejects_guessed_scope_and_role_escalation(db, kind):
    current, store = state(), ResearchArtifactService(db)
    ref = seed_input(store, current)
    other = store.persist_payload(session_id='other' if kind == 'cross_session' else 'v2',
        artifact_type='paper_metadata', payload={'value': {'title': 'not current input'}})
    db.commit()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        def work(w, args):
            store.resolve('v2', other if kind in {'unbound', 'cross_session'} else ref,
                role='service', expected_version=2 if kind == 'wrong_version' else None,
                allowed_types={'paper_card'} if kind == 'wrong_type' else None,
                state={'session_id': 'v2', 'artifact_manifest': {'fake': {'ref': other}}})
        with pytest.raises(RuntimeConflict):
            ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': work})
        assert DelegationRepository(active_runtime()).version() == 0


def test_committed_search_artifacts_become_next_action_inputs(db):
    current, store = state(), ResearchArtifactService(db)
    paper = {'paper_id': 'p', 'title': 'Study', 'year': 2024}
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        first = invocation(current)
        executor = ActionExecutor(AgentController())
        executor.execute(first, current, {'search_and_rank': lambda w, a:
            w.update(candidate_papers=[paper], ranked_papers=[paper])})
        lineage = events(db, 'action_artifacts_committed')[0]
        ref = lineage['outputs']['ranked_papers']['items'][0]['ref']
        assert lineage['business_version'] == 1
        second = ActionInvocation(delegation_id=first.delegation_id, step_index=1,
            action='fetch_metadata', expected_version=1, source_fingerprint=context_source_fingerprint(current))
        def fetch(w, args):
            value = store.resolve('v2', ref, role='service')
            w['paper_details'] = [value['payload']['value']]
        executor.execute(second, current, {'fetch_metadata': fetch})
        assert current['paper_details'][0]['paper_id'] == 'p'
        assert len(events(db, 'action_artifacts_committed')) == 2
        assert DelegationRepository(active_runtime()).version() == 2


def test_check_cannot_write_candidates_or_evidence_authorizations(db):
    current, store = writer_state(), ResearchArtifactService(db)
    current.update(review='Existing candidate', pending_result_validation=True)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current, operation='validate_result', role='check', profile='independent_check')
        with pytest.raises(RuntimeConflict):
            ActionExecutor(AgentController()).execute(action, current, {'validate_result': lambda w, a:
                store.persist_payload(session_id='v2', artifact_type='claim_plan', payload={'value': 'forged'})})
        assert not events(db, 'artifact_staged')
        with pytest.raises(ToolAuthorizationError):
            ActionExecutor(AgentController()).execute(action, current, {'validate_result': lambda w, a:
                record_candidate(w, NS(model_dump=lambda **k: {}), 'bad', 'forged', 'forged', {})})
        assert current['review'] == 'Existing candidate'


@pytest.mark.parametrize('workers', [1, 4])
def test_writer_candidate_round_trip_uses_thread_owned_storage(db, monkeypatch, workers):
    current, store = writer_state(), ResearchArtifactService(db)
    main_thread = threading.get_ident()
    def guarded(fn):
        def run(*args, **kwargs):
            assert threading.get_ident() == main_thread, 'worker accessed parent Session'
            return fn(*args, **kwargs)
        return run
    for method in ('execute', 'get', 'flush'):
        monkeypatch.setattr(db, method, guarded(getattr(db, method)))
    token = _STORE.set(store)
    try:
        with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
            action = invocation(current, operation='generate_deliverables', role='writing', profile='draft_generation')
            def work(w, args):
                def section(index):
                    local = copy.deepcopy(w)
                    raw = 'Current authorized draft text ' * 30 + str(index)
                    cid = record_candidate(local, NS(model_dump=lambda **kw: {'id': 'background'}),
                                           'draft', raw, raw, {'valid': True})
                    restored = read_candidate(local, cid, role='service')
                    assert restored['raw_text'] == raw
                    return local['writer_candidates'][-1]
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    records = [f.result(timeout=20) for f in [submit_with_context(pool, section, i) for i in range(8)]]
                w.setdefault('writer_candidates', []).extend(records)
                w.update(review='Current authorized draft text', pending_result_validation=True)
            ActionExecutor(AgentController()).execute(action, current, {'generate_deliverables': work})
            assert len(current['writer_candidates']) == 8
            assert len(events(db, 'artifact_staged')) == 8
            assert current['pending_result_validation']
            staged = events(db, 'artifact_staged')[0]
            raw = ResearchArtifactService(db).resolve('v2', staged['ref'], role='service')
            assert raw['provenance']['v2_lineage']['task_id'] == staged['task_id']
    finally:
        _STORE.reset(token)


def test_staged_artifact_does_not_authorize_new_attempt_after_crash(db):
    current, store, written = state(), ResearchArtifactService(db), []
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        def crash(w, args):
            written.append(store.persist_payload(session_id='v2', artifact_type='raw_search_result', payload={'value': 'staged only'}))
            raise SystemExit('after durable artifact, before action commit')
        with pytest.raises(SystemExit):
            ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': crash})
        assert len(events(db, 'artifact_staged')) == 1 and not events(db, 'action_artifacts_committed')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        with pytest.raises(RuntimeConflict, match='resource scope'):
            ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': lambda w, a:
                store.resolve('v2', written[0], role='service')})
        assert DelegationRepository(active_runtime()).version() == 0


def test_unbound_parent_reference_cannot_be_laundered_into_lineage(db):
    current, store = state(), ResearchArtifactService(db)
    other = store.persist_payload(session_id='v2', artifact_type='raw_search_result', payload={'value': 'old'})
    db.commit()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        with pytest.raises(RuntimeConflict):
            ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': lambda w, a:
                store.persist_payload(session_id='v2', artifact_type='raw_search_result', payload={'value': 'new'},
                    provenance={'source_refs': [other], 'v2_lineage': {'task_id': 'invented'}})})
        assert not events(db, 'artifact_staged')


def test_writer_artifact_survives_checkpoint_and_is_read_by_independent_check(db):
    from app.services.research_delegation_service import run_delegation
    from tests.test_check_agent import checked
    current, store, candidate_ids = writer_state(), ResearchArtifactService(db), []
    token = _STORE.set(store)
    try:
        with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
            def write(w, args):
                candidate_ids.append(record_candidate(w, NS(model_dump=lambda **kw: {}), 'draft',
                    'Text grounded in the available evidence.', 'Text grounded in the available evidence.', {'valid': True}))
                w.update(review='Text grounded in the available evidence.', pending_result_validation=True)
            result = run_delegation(current, {'agent': 'writing', 'profile': 'draft_generation', 'goal': 'write'},
                                    llm=object(), handlers={'generate_deliverables': write})
            assert result.outcome == 'satisfied'
            restored = active_runtime().restore()
        with budget_scope(restored), durable_scope(db, 'v2', restored, restored['agent_execution_budget']):
            def check(w, args):
                candidate = read_candidate(w, candidate_ids[0], role='service')
                assert candidate['raw_text'] == w['review']
                checked(w)
            result = run_delegation(restored, {'agent': 'check', 'profile': 'independent_check', 'goal': 'check'},
                                    llm=object(), handlers={'validate_result': check})
            assert result.outcome == 'satisfied' and not restored['pending_result_validation']
            assert len(events(db, 'action_artifacts_committed')) == 2
    finally:
        _STORE.reset(token)


def test_business_commit_failure_rolls_back_lineage_and_version_together(db, monkeypatch):
    current = state()
    original = ResearchArtifactService.validate_payload
    trigger = []
    def validate(payload, *, max_bytes=262144):
        if trigger and max_bytes == 8 * 1024 * 1024:
            raise ValueError('injected final snapshot failure')
        return original(payload, max_bytes=max_bytes)
    monkeypatch.setattr(ResearchArtifactService, 'validate_payload', staticmethod(validate))
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        def work(w, args):
            w.update(candidate_papers=[{'paper_id': 'p', 'title': 'Study'}])
            trigger.append(True)
        with pytest.raises(ValueError, match='final snapshot'):
            ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': work})
        assert not events(db, 'action_artifacts_committed')
        assert DelegationRepository(active_runtime()).version() == 0
        assert not current.get('candidate_papers')


def test_grant_revoked_after_artifact_receipt_prevents_storage_write(db, monkeypatch):
    current, store = state(), ResearchArtifactService(db)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        runtime = active_runtime()
        original = runtime.begin_tool_call
        def revoke(payload):
            original(payload)
            if payload['tool'] == 'artifact_write':
                DelegationRepository(runtime).revoke(action.delegation_id)
        monkeypatch.setattr(runtime, 'begin_tool_call', revoke)
        with pytest.raises(RuntimeConflict):
            ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': lambda w, a:
                store.persist_payload(session_id='v2', artifact_type='raw_search_result', payload={'value': 'forbidden'})})
        assert not events(db, 'artifact_staged')
        assert not db.scalar(select(ResearchArtifactModel).where(ResearchArtifactModel.artifact_type == 'raw_search_result'))


def test_old_work_candidate_is_not_in_frozen_resource_scope(db):
    current, store = writer_state(), ResearchArtifactService(db)
    current['writer_candidates'] = [{'candidate_id': 'old', 'task_identity': 'different-work', 'raw_text': 'old audit'}]
    token = _STORE.set(store)
    try:
        with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
            action = invocation(current, operation='generate_deliverables', role='writing', profile='draft_generation')
            def write(w, args):
                packet = events(db, 'action_resources_bound')[-1]
                roots = [entry for entry in packet['items'] if entry['type'] == 'writer_candidate']
                assert not roots
                w.update(review='Current work', pending_result_validation=True)
            ActionExecutor(AgentController()).execute(action, current, {'generate_deliverables': write})
    finally:
        _STORE.reset(token)
