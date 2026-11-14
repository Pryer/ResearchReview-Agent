"""LLM和检索模型的统一预算；禁止真实网络请求。"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app.agent.execution_budget import AgentBudgetExceeded, budget_scope, budgeted_create, record_retrieval_request, submit_with_context
from app.core.config import get_settings


def client():
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=1))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def test_combined_llm_embedding_and_reranker_requests_share_one_hard_cap(monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_model_request_limit", 3)
    state={};api,calls=client()
    with budget_scope(state):
        record_retrieval_request("embedding")
        budgeted_create(api,messages=[],max_tokens=1)
        record_retrieval_request("rerank")
        with pytest.raises(AgentBudgetExceeded,match="total model request"):
            budgeted_create(api,messages=[],max_tokens=1)
    ledger=state['agent_execution_budget']
    assert ledger['llm_requests']==1 and ledger['retrieval_model_requests']==2
    assert len(calls)==1 and ledger['model_request_limit']==3


def test_parallel_retrieval_cannot_oversubscribe_total_request_cap(monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_model_request_limit", 5)
    state={'agent_execution_budget':{'llm_requests':2}}
    def reserve():
        try:
            record_retrieval_request('embedding')
            return True
        except AgentBudgetExceeded:
            return False
    with budget_scope(state),ThreadPoolExecutor(max_workers=8) as pool:
        futures=[submit_with_context(pool,reserve) for _ in range(20)]
        outcomes=[]
        for f in futures:
            try:outcomes.append(f.result())
            except AgentBudgetExceeded:outcomes.append(False)
    ledger=state['agent_execution_budget']
    assert sum(outcomes)==3
    assert ledger['llm_requests']+ledger['retrieval_model_requests']==5


def test_restored_request_usage_and_stricter_saved_limit_are_preserved(monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_model_request_limit", 10)
    state={'agent_execution_budget':{'model_request_limit':2,'llm_requests':1,'retrieval_model_requests':1}}
    api,calls=client()
    with budget_scope(state):
        with pytest.raises(AgentBudgetExceeded,match="total model request"):
            budgeted_create(api,messages=[],max_tokens=1)
    assert not calls and state['agent_execution_budget']['model_request_limit']==2


def test_zero_keeps_existing_application_behavior_without_extra_total_limit(monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_model_request_limit", 0)
    state={'agent_execution_budget':{'llm_requests':128}}
    with budget_scope(state):
        record_retrieval_request('embedding')
    assert state['agent_execution_budget']['retrieval_model_requests']==1


def test_main_sees_remaining_combined_requests_from_restored_ledger(monkeypatch):
    from app.agent.context_builder import build_main_agent_context
    from tests.test_subagent_react import state
    monkeypatch.setattr(get_settings(), 'agent_model_request_limit', 10)
    current=state()
    current['agent_execution_budget']={'model_request_limit':5,'llm_requests':2,'retrieval_model_requests':1}
    assert build_main_agent_context(current).state.budget_remaining['model_requests']==2
    current['agent_execution_budget']['model_request_limit']=0
    assert 'model_requests' not in build_main_agent_context(current).state.budget_remaining
