"""真实请求边界和公开停止原因回归，外部服务完全替身。"""

from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from app.agent.execution_budget import budget_scope, budgeted_create, AgentBudgetExceeded
from app.agent.graph import _build_output


def client_with_usage(usage):
    create = Mock(return_value=NS(usage=usage))
    return NS(chat=NS(completions=NS(create=create))), create


@pytest.mark.parametrize("cache_ratio", [0, 0.8, 1])
def test_full_output_and_cached_input_cannot_evade_total_token_limit(cache_ratio):
    state = {"agent_execution_budget": {"token_limit": 800, "cache_hit_ratio": cache_ratio}}
    client, create = client_with_usage(None)
    with budget_scope(state), pytest.raises(AgentBudgetExceeded):
        budgeted_create(client, messages=[{"role": "user", "content": "hello"}], max_tokens=1000)
    create.assert_not_called()
    assert state["agent_execution_budget"].get("llm_tokens", 0) == 0


def test_unexpected_actual_overrun_settles_before_rejecting_response():
    state = {"agent_execution_budget": {"token_limit": 800}}
    client, create = client_with_usage(NS(total_tokens=1050, prompt_tokens=1040, completion_tokens=10))
    reported = Mock()
    with budget_scope(state):
        with pytest.raises(AgentBudgetExceeded, match="actual token usage"):
            budgeted_create(client, messages=[], max_tokens=10, on_response=reported)
        with pytest.raises(AgentBudgetExceeded):
            budgeted_create(client, messages=[], max_tokens=10)
    assert create.call_count == reported.call_count == 1
    assert state["agent_execution_budget"]["llm_tokens"] == 1050
    assert state["agent_execution_budget"]["tokens_reserved"] == 0


def test_cache_telemetry_does_not_change_missing_usage_estimate():
    costs = []
    for ratio in (0, 1):
        state = {"agent_execution_budget": {"token_limit": 10000, "cache_hit_ratio": ratio}}
        client, _ = client_with_usage(None)
        with budget_scope(state):
            budgeted_create(client, messages=[{"role": "user", "content": "中文证据"}], max_tokens=1000)
        ledger = state["agent_execution_budget"]
        assert ledger["usage_estimated"]
        assert ledger["estimated_tokens"] >= 1000
        costs.append(ledger["estimated_tokens"])
    assert costs[0] == costs[1]


@pytest.mark.parametrize("error,expected", [
    ("search: private diagnostic", "查看服务端诊断"),
    ({"code": "LLMInvocationError", "message": "SECRET https://private.example/path"}, "模型服务调用失败"),
    ({"code": "unknown", "message": "SECRET C:\\private\\prompt.txt"}, "内部错误"),
    ({"code": "main_agent_blocked", "message": "缺少用户提供的论文方法说明"}, "缺少用户提供的论文方法说明"),
    ({"code": "main_agent_blocked", "message": "缺少 SECRET https://private.example"}, "必要输入不足"),
    ({"code": "AgentBudgetExceeded", "message": "research execution deadline exceeded"}, "执行已超过时限"),
    ({"code": "AgentBudgetExceeded", "message": "retrieval budget exhausted"}, "检索次数预算"),
    ({"code": "AgentBudgetExceeded", "message": "token reservation exceeds remaining budget"}, "剩余 token 预算不足"),
])
def test_public_stop_reason_is_safe_and_specific(error, expected):
    output = _build_output({"agent_orchestration_mode": "autonomous", "result_status": "blocked",
                            "generation_blocked": True, "errors": [error]})
    assert expected in output["answer"]
    assert "SECRET" not in output["answer"]
    assert "private" not in output["answer"]
    assert output["body"] == ""


def test_latest_stop_reason_takes_precedence_over_recovered_error():
    output = _build_output({"agent_orchestration_mode": "autonomous", "result_status": "blocked",
        "errors": ["search: recovered", {"code": "agent_no_progress", "message": "no progress"}]})
    assert "实质研究进展" in output["answer"]
    assert "search" not in output["answer"]
