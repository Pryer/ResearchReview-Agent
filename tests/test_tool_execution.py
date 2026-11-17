"""真实Action/SQLite边界上的工具能力、持久回执和并发记账。"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import threading
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.agent.context_builder import context_source_fingerprint
from app.agent.controller import AgentController
from app.agent.execution_budget import budget_scope, budgeted_create, submit_with_context, AgentExecutionCancelled
from app.agent.policy.authorization import AuthorizationPolicy
from app.agent.runtime.action_executor import ActionExecutor
from app.database.delegation_repository import DelegationRepository
from app.database.models import ResearchEventModel as Event, ResearchAttemptModel, ResearchRuntimeModel
from app.database.runtime_repository import RuntimeConflict
from app.schemas.delegation_schema import DelegationRequest, ActionInvocation
from app.services.durable_execution_service import durable_scope, active_runtime
from app.services.tool_execution import execute_tool, ToolAuthorizationError
from tests.test_v2_delegation_runtime import db, state


def invocation(current, *, operation='search_and_rank', role='search', profile='initial_research'):
    repo = DelegationRepository(active_runtime())
    grant = AuthorizationPolicy().issue(DelegationRequest(agent=role, profile=profile, goal='test'), current)
    repo.create(grant)
    return ActionInvocation(delegation_id=grant.delegation_id, step_index=0, action=operation,
        expected_version=repo.version(), source_fingerprint=context_source_fingerprint(current))


def events(db, kind):
    return [json.loads(row.payload_json) for row in db.scalars(select(Event).where(Event.event_type == kind))]


def test_tool_ack_precedes_provider_and_cost_is_settled_once(db):
    current = state()
    def provider(**kwargs):
        with Session(db.bind) as observer:
            assert len(events(observer, 'tool_started')) == 1
            assert observer.scalar(select(ResearchAttemptModel)) is not None
        return NS(usage=NS(total_tokens=7, prompt_tokens=4, completion_tokens=3))
    client = NS(chat=NS(completions=NS(create=provider)))
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': lambda w, a:
            budgeted_create(client, messages=[{'role': 'user', 'content': 'private prompt'}], max_tokens=8)})
        assert current['agent_execution_budget']['llm_tokens'] == 7
        assert current['agent_execution_budget']['llm_requests'] == 1
        assert DelegationRepository(active_runtime()).version() == 1
        starts, ends = events(db, 'tool_started'), events(db, 'tool_finished')
        assert len(starts) == len(ends) == 1 and ends[0]['status'] == 'completed'
        assert starts[0]['invocation_id'] == action.invocation_id
        assert starts[0]['boundary'] == 'app.agent.execution_budget.budgeted_create'
        assert events(db, 'attempt_started')[0]['tool_call_id'] == starts[0]['call_id']
        assert 'private prompt' not in json.dumps(starts + ends)


def test_analysis_cannot_call_metadata_client(db, monkeypatch):
    from app.clients import crossref_client
    monkeypatch.setattr(crossref_client, '_request_crossref', lambda *a, **k: pytest.fail('unauthorized request'))
    current = state()
    current['paper_details'] = [{'paper_id': 'p', 'title': 'Study'}]
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current, operation='extract_paper_cards', role='analysis', profile='evidence_build')
        with pytest.raises(ToolAuthorizationError):
            ActionExecutor(AgentController()).execute(action, current, {'extract_paper_cards': lambda w, a:
                crossref_client.get_crossref_detail('10.1/test')})
        assert not events(db, 'tool_started')
        assert DelegationRepository(active_runtime()).version() == 0


def test_live_grant_is_rechecked_before_each_tool(db):
    current, calls = state(), []
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        def work(w, args):
            DelegationRepository(active_runtime()).revoke(action.delegation_id)
            execute_tool('paper_search', lambda: calls.append('network'))
        with pytest.raises(RuntimeConflict):
            ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': work})
        assert not calls and not events(db, 'tool_started')


def test_business_version_change_rejects_tool_before_remote_call(db):
    from app.database.models import ResearchWorkVersionModel
    current, calls = state(), []
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        def work(w, args):
            with Session(db.bind) as other:
                other.execute(update(ResearchWorkVersionModel).values(version=1))
                other.commit()
            execute_tool('paper_search', lambda: calls.append('network'))
        with pytest.raises(RuntimeConflict):
            ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': work})
        assert not calls and not events(db, 'tool_started')


def test_cancellation_after_tool_ack_prevents_external_call(db, monkeypatch):
    current, calls = state(), []
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        runtime = active_runtime()
        action = invocation(current)
        begin = runtime.begin_tool_call
        def cancel(payload):
            begin(payload)
            with Session(db.bind) as other:
                other.execute(update(ResearchRuntimeModel).values(cancelled=True))
                other.commit()
        monkeypatch.setattr(runtime, 'begin_tool_call', cancel)
        with pytest.raises(AgentExecutionCancelled):
            ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': lambda w, a:
                execute_tool('paper_search', lambda: calls.append('network'))})
        assert not calls and events(db, 'tool_finished')[0]['status'] == 'failed'


@pytest.mark.parametrize('workers', [1, 4, 8])
def test_worker_receipts_do_not_touch_parent_session_or_duplicate_cost(db, monkeypatch, workers):
    current = state()
    count = 100
    owner_thread = threading.get_ident()
    def guarded(fn):
        def call(*args, **kwargs):
            assert threading.get_ident() == owner_thread
            return fn(*args, **kwargs)
        return call
    monkeypatch.setattr(db, 'get', guarded(db.get))
    monkeypatch.setattr(db, 'execute', guarded(db.execute))
    client = NS(chat=NS(completions=NS(create=lambda **kwargs: NS(
        usage=NS(total_tokens=7, prompt_tokens=4, completion_tokens=3)))))
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        def work(w, args):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [submit_with_context(pool, partial(budgeted_create, client, messages=[], max_tokens=8))
                           for _ in range(count)]
                for future in futures:
                    future.result(timeout=20)
        ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': work})
        assert len(events(db, 'tool_started')) == len(events(db, 'tool_finished')) == count
        assert current['agent_execution_budget']['llm_requests'] == count
        assert current['agent_execution_budget']['llm_tokens'] == count * 7
        assert len({r['invocation_attempt_id'] for r in events(db, 'tool_started')}) == 1


def test_receipt_failure_preserves_original_control_exception_and_recovers_unknown(db, monkeypatch):
    current = state()
    error = AgentExecutionCancelled('original cancellation')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        runtime = active_runtime()
        def fail_receipt(*args):
            raise OSError('receipt storage failed')
        monkeypatch.setattr(runtime, 'finish_tool_call', fail_receipt)
        def stop():
            raise error
        with pytest.raises(AgentExecutionCancelled) as caught:
            ActionExecutor(AgentController()).execute(action, current,
                {'search_and_rank': lambda w, a: execute_tool('paper_search', stop)})
        assert caught.value is error and not events(db, 'tool_finished')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        assert len(events(db, 'tool_interrupted')) == 1
        assert events(db, 'tool_interrupted')[0]['status'] == 'uncertain'
        assert DelegationRepository(active_runtime()).version() == 0


def test_metadata_fallback_does_not_swallow_control_error(monkeypatch):
    from app.tools.fetch_metadata import fetch_batch_details
    error = ToolAuthorizationError('denied')
    def denied(*args, **kwargs):
        raise error
    monkeypatch.setattr('app.clients.crossref_client.get_crossref_detail', denied)
    with pytest.raises(ToolAuthorizationError) as caught:
        fetch_batch_details([{'paper_id': 'p', 'doi': '10.1/p', 'source': 'openalex'}], max_workers=2)
    assert caught.value is error


@pytest.mark.parametrize('kind', ['embedding', 'rerank'])
def test_retrieval_http_has_tool_receipt_without_duplicate_request_charge(db, kind):
    from app.clients.retrieval_http import post_json
    current, requests = state(), []
    def post(*args, **kwargs):
        requests.append('http')
        return NS(status_code=200, json=lambda: {'output': {}}, headers={})
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': lambda w, a:
            post_json('https://model.example.test', 'private-key', {'input': 'private text'},
                timeout=1, retries=0, session=NS(post=post), kind=kind)})
        assert requests == ['http']
        assert current['agent_execution_budget']['retrieval_model_requests'] == 1
        starts = events(db, 'tool_started')
        assert len(starts) == 1 and starts[0]['tool'] == kind + '_request'
        assert 'private' not in json.dumps(starts + events(db, 'tool_finished'))


def test_check_cannot_migrate_or_delete_vector_store(db, monkeypatch):
    from app.storage.pgvector_store import PgVectorStore
    from tests.test_main_quality_loop import prepared
    current = prepared()
    current.update(session_id='v2', review='Candidate [p0]', pending_result_validation=True)
    store = PgVectorStore(NS(card_vector_database_url='postgresql://unused.test/research'))
    monkeypatch.setattr(store, '_db', lambda *a, **k: pytest.fail('administrative vector operation ran'))
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current, operation='validate_result', role='check', profile='independent_check')
        with pytest.raises(ToolAuthorizationError):
            ActionExecutor(AgentController()).execute(action, current,
                {'validate_result': lambda w, a: store.delete('tenant', 'p0')})
        assert not events(db, 'tool_started')


def test_receipt_settlement_is_idempotent_and_does_not_advance_business_version(db):
    current = state()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current)
        def work(w, args):
            execute_tool('paper_search', lambda: None)
            call_id = events(db, 'tool_started')[0]['call_id']
            assert not active_runtime().finish_tool_call(call_id, {'status': 'failed'})
            assert DelegationRepository(active_runtime()).version() == 0
        ActionExecutor(AgentController()).execute(action, current, {'search_and_rank': work})
        assert len(events(db, 'tool_finished')) == 1
        assert events(db, 'tool_finished')[0]['status'] == 'completed'


@pytest.mark.parametrize('function', ['download_pdf_node', 'parse_pdf_node'])
def test_pdf_nodes_propagate_tool_control_failure(monkeypatch, function):
    from app.agent.nodes import extraction
    error = ToolAuthorizationError('denied')
    def denied(*args, **kwargs):
        raise error
    target = 'app.tools.download_pdf.batch_download_pdfs' if function == 'download_pdf_node' else 'app.tools.parse_pdf.batch_parse_pdfs'
    monkeypatch.setattr(target, denied)
    with pytest.raises(ToolAuthorizationError) as caught:
        getattr(extraction, function)({'paper_details': [{'paper_id': 'p'}], 'pdf_paths': {'p': 'fake.pdf'}})
    assert caught.value is error


def test_boundaries_and_registry_dependencies_cannot_disappear_silently(monkeypatch):
    import ast
    from pathlib import Path
    from dataclasses import replace
    from app.agent.registry.tool_registry import validate_tool_registry
    from app.agent.action_registry import ACTION_REGISTRY
    boundaries = {
        'app/agent/execution_budget.py': {'budgeted_create'},
        'app/clients/retrieval_http.py': {'post_json'},
        'app/clients/arxiv_client.py': {'search_arxiv'},
        'app/clients/crossref_client.py': {'search_crossref', 'get_crossref_detail'},
        'app/clients/openalex_client.py': {'search_openalex', 'get_openalex_detail'},
        'app/clients/semantic_scholar_client.py': {'search_semantic_scholar', 'get_semantic_scholar_detail'},
        'app/clients/cnki_client.py': {'search_cnki'},
        'app/tools/download_pdf.py': {'download_open_access_pdf'},
        'app/tools/parse_pdf.py': {'parse_pdf'},
        'app/storage/pgvector_store.py': {'migrate', 'delete', 'register_source', 'claim', 'renew', 'write', 'activate', 'load', 'search'},
    }
    for path, expected in boundaries.items():
        tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
        guarded = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                   and any(isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                           and d.func.id == 'research_tool' for d in node.decorator_list)}
        assert expected <= guarded, path
    # WHY: Handler组合根只能组合业务能力，禁止绕回裸HTTP/Provider客户端。
    graph = ast.parse(Path('app/agent/graph.py').read_text(encoding='utf-8-sig'))
    build = next(node for node in graph.body if isinstance(node, ast.FunctionDef) and node.name == 'build_research_actions')
    forbidden = {'requests', 'httpx', 'urllib', 'openai'}
    for node in ast.walk(build):
        if isinstance(node, ast.Import):
            assert not any(alias.name.split('.')[0] in forbidden for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or '').split('.')[0] not in forbidden
    validate_tool_registry()
    monkeypatch.setitem(ACTION_REGISTRY, 'plan_claims', replace(ACTION_REGISTRY['plan_claims'], tool_dependencies=('missing_tool',)))
    with pytest.raises(ValueError, match='unregistered tool'):
        validate_tool_registry()
