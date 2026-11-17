"""实际发布快照→HTTP轮询→Streamlit聊天与下载控件，外部模型不参与。"""
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app.agent.controller import AgentController
from app.agent.execution_budget import budget_scope
from app.core.config import get_settings
from app.database.db import get_db
from app.database.repositories import ResearchJobRepository
from app.main import app
from app.services.durable_execution_service import active_runtime, durable_scope
from app.services.execution_progress import bind_job
from tests.test_draft_delivery_policy import draft_state
from tests.test_v2_delegation_runtime import db


@pytest.mark.parametrize('status', ['cancelled', 'failed', 'needs_clarification'])
def test_published_delivery_reaches_terminal_chat_and_download(db, monkeypatch, status):
    monkeypatch.setattr(get_settings(), 'agent_architecture_version', 'v2')
    monkeypatch.setattr(get_settings(), 'execution_report_enabled', False)
    current = draft_state()
    current['session_id'] = 'v2'
    jobs = ResearchJobRepository(db)
    jobs.create('product-job', 'v2', {'user_query': '研究请求'})
    jobs.update('product-job', status='running')
    db.commit()
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        bind_job(active_runtime(), 'product-job')
        published = AgentController().publish_draft(current)
        current.update(review='UNVERIFIED REPLACEMENT', writing_version=2, pending_result_validation=True)
        active_runtime().snapshot(current, ledger=current['agent_execution_budget'])
    jobs.update('product-job', status=status, result={'status': status,
        'clarification': {'needed': True, 'kind': 'main_agent', 'question': '需要哪个主题？'}
        if status == 'needs_clarification' else None})
    db.commit()

    def same_db():
        yield db
    monkeypatch.setitem(app.dependency_overrides, get_db, same_db)
    headers = {'X-API-Key': get_settings().app_api_key} if get_settings().app_api_key else {}
    response = TestClient(app).get('/api/reviews/jobs/product-job', headers=headers)
    assert response.status_code == 200
    job = response.json()['data']
    assert job['status'] == status and job['execution_progress']['architecture_version'] == 2
    assert job['execution_progress']['current_role'] is None
    assert job['available_delivery']['body'] == published['body']
    assert 'UNVERIFIED REPLACEMENT' not in job['available_delivery']['body']

    ui = AppTest.from_string('''
from app.frontend.chat_app import init_session_state, _finish_job, get_messages, render_assistant_message
import streamlit as st
init_session_state()
st.session_state.active_job = {'session_id': 'v2', 'job_id': 'product-job'}
_finish_job(st.session_state.job)
message = get_messages('v2')[-1]
render_assistant_message(message['content'], message['metadata'], allow_actions=True)
''')
    ui.session_state['job'] = job
    ui.run(timeout=20)
    assert not ui.exception
    message = ui.session_state['conversations']['v2']['messages'][-1]
    assert ui.session_state['active_job'] is None
    assert message['metadata']['body'] == published['body']
    assert message['metadata']['status'] == status
    assert message['metadata']['verification_status'] == published['verification_status']
    assert not message['metadata']['requirements_satisfied']
    assert any(button.label == '下载带限制说明的草稿' for button in ui.get('download_button'))
    if status == 'needs_clarification':
        assert '需要哪个主题？' in message['content']
        assert any('输入框' in caption.value for caption in ui.caption)
    else:
        assert ('取消' if status == 'cancelled' else '失败') in message['content']
