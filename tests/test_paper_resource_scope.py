"""论文、文件和向量参数绑定真实Action输入；不请求真实Provider。"""
import hashlib
from types import SimpleNamespace as NS
import pytest

from app.agent.controller import AgentController
from app.agent.execution_budget import budget_scope
from app.agent.runtime.action_executor import ActionExecutor
from app.core.config import get_settings
from app.database.runtime_repository import RuntimeConflict
from app.services.durable_execution_service import durable_scope
from app.services.tool_execution import execute_tool
from app.services.document_ingestion import source_fingerprint
from app.schemas.vector_schema import IndexSpec
from tests.test_tool_execution import invocation, events
from tests.test_v2_delegation_runtime import db, state


def setup_pdf(current, tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), 'pdf_save_dir', str(tmp_path))
    monkeypatch.setattr(get_settings(), 'parsed_save_dir', str(tmp_path / 'parsed'))
    paper = {'paper_id': 'p', 'doi': '10.1/p', 'source': 'crossref', 'pdf_url': 'https://papers.example.test/p.pdf',
             'is_open_access': True, 'title': 'Study'}
    current['ranked_papers'] = [paper]
    path = tmp_path / 'p.pdf'
    path.write_bytes(b'%PDF-current-source')
    return paper, path


@pytest.mark.parametrize('change', ['paper', 'url', 'oa', 'directory'])
def test_download_rejects_unapproved_paper_transport_before_call(db, tmp_path, monkeypatch, change):
    current, calls = state(), []
    paper, path = setup_pdf(current, tmp_path, monkeypatch)
    proposed = dict(paper)
    if change == 'paper': proposed['paper_id'] = 'other'
    if change == 'url': proposed['pdf_url'] = 'https://papers.example.test/other.pdf'
    if change == 'oa': proposed['is_open_access'] = False
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current, operation='fetch_metadata')
        with pytest.raises(RuntimeConflict):
            ActionExecutor(AgentController()).execute(action, current, {'fetch_metadata': lambda w, a:
                execute_tool('pdf_download', lambda: calls.append('network'), arguments={
                    'paper': proposed, 'save_dir': str(tmp_path.parent) if change == 'directory' else None})})
        assert not calls


def test_verified_metadata_download_parse_and_vector_expand_same_attempt(db, tmp_path, monkeypatch):
    current = state()
    paper, path = setup_pdf(current, tmp_path, monkeypatch)
    current['ranked_papers'] = [{k: v for k, v in paper.items() if k not in {'pdf_url', 'is_open_access'}}]
    parsed = {'paper_id': 'p', 'full_text': 'The paper source.', 'pages': [{'page': 1, 'text': 'The paper source.'}],
              'source_pdf_hash': hashlib.sha256(path.read_bytes()).hexdigest()}
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current, operation='fetch_metadata')
        def work(w, args):
            result = execute_tool('metadata_read', lambda: [paper], arguments={'papers': w['ranked_papers']})
            downloaded = execute_tool('pdf_download', lambda: str(path), arguments={'paper': result[0]})
            document = execute_tool('pdf_parse', lambda: parsed, arguments={'paper_id': 'p', 'pdf_path': downloaded})
            spec = IndexSpec(get_settings().card_vector_tenant, 'p', source_fingerprint(document), 'processing', 'embedding', 3, 1, 1)
            execute_tool('vector_write', lambda: None, arguments={'spec': spec})
            w.update(paper_details=[paper], pdf_paths={'p': str(path)}, parsed_papers={'p': document})
        ActionExecutor(AgentController()).execute(action, current, {'fetch_metadata': work})
        assert len(events(db, 'action_resources_extended')) == 3
        assert current['parsed_papers']['p']['source_pdf_hash'] == parsed['source_pdf_hash']


@pytest.mark.parametrize('changed', [False, True])
def test_parse_rejects_unknown_or_changed_file_before_read(db, tmp_path, monkeypatch, changed):
    current, calls = state(), []
    paper, path = setup_pdf(current, tmp_path, monkeypatch)
    current['pdf_paths'] = {'p': str(path)}
    other = tmp_path / 'other.pdf'
    other.write_bytes(b'%PDF-other')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current, operation='fetch_metadata')
        def work(w, args):
            if changed: path.write_bytes(b'%PDF-changed')
            execute_tool('pdf_parse', lambda: calls.append('read'), arguments={
                'paper_id': 'p', 'pdf_path': str(path if changed else other)})
        with pytest.raises(RuntimeConflict):
            ActionExecutor(AgentController()).execute(action, current, {'fetch_metadata': work})
        assert not calls


@pytest.mark.parametrize('change', ['tenant', 'paper', 'source'])
def test_vector_access_rejects_unbound_namespace_and_source(db, tmp_path, monkeypatch, change):
    current, calls = state(), []
    paper, path = setup_pdf(current, tmp_path, monkeypatch)
    current['paper_details'] = [paper]
    parsed = {'paper_id': 'p', 'full_text': 'Current source', 'source_pdf_hash': 'pdf-hash'}
    current['parsed_papers'] = {'p': parsed}
    spec = IndexSpec('other' if change == 'tenant' else get_settings().card_vector_tenant,
        'other' if change == 'paper' else 'p', 'other' if change == 'source' else source_fingerprint(parsed), 'a', 'b', 3, 1, 1)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current, operation='extract_paper_cards', role='analysis', profile='evidence_build')
        with pytest.raises(RuntimeConflict):
            ActionExecutor(AgentController()).execute(action, current, {'extract_paper_cards': lambda w, a:
                execute_tool('vector_read', lambda: calls.append('database'), arguments={'spec': spec})})
        assert not calls


def test_metadata_identifier_is_confined_to_known_aliases(db):
    current, calls = state(), []
    current['ranked_papers'] = [{'paper_id': 'p', 'doi': '10.1/p', 'title': 'Study'}]
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current, operation='fetch_metadata')
        with pytest.raises(RuntimeConflict):
            ActionExecutor(AgentController()).execute(action, current, {'fetch_metadata': lambda w, a:
                execute_tool('metadata_read', lambda: calls.append('network'), arguments={'doi': '10.1/other'})})
        assert not calls


def test_new_download_invalidates_old_parsed_source_until_reparse(db, tmp_path, monkeypatch):
    current, calls = state(), []
    paper, path = setup_pdf(current, tmp_path, monkeypatch)
    parsed = {'paper_id': 'p', 'full_text': 'old', 'source_pdf_hash': 'old-hash'}
    current['parsed_papers'] = {'p': parsed}
    spec = IndexSpec(get_settings().card_vector_tenant, 'p', source_fingerprint(parsed), 'a', 'b', 3, 1, 1)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        action = invocation(current, operation='fetch_metadata')
        def work(w, args):
            execute_tool('pdf_download', lambda: str(path), arguments={'paper': paper})
            execute_tool('vector_read', lambda: calls.append('old source'), arguments={'spec': spec})
        with pytest.raises(RuntimeConflict):
            ActionExecutor(AgentController()).execute(action, current, {'fetch_metadata': work})
        assert not calls
