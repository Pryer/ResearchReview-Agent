"""操作脚本与公开请求/任务生命周期契约保持一致，不访问外部服务。"""

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.schemas.agent_schema import AgentRequest


ROOT = Path(__file__).resolve().parents[1]


def test_submit_example_is_accepted_by_public_request_schema(monkeypatch):
    import scripts.submit_research as script

    monkeypatch.setattr(script, "get_api_key", lambda: "")
    post = Mock(return_value=SimpleNamespace(
        status_code=200, json=lambda: {"code": 0, "data": {"job_id": "example-job"}}
    ))
    monkeypatch.setattr(script.requests, "post", post)
    assert script.submit_task() == "example-job"
    request = AgentRequest.model_validate(post.call_args.kwargs["json"])
    assert request.state["required_reference_count"] == 40
    assert "自动识别" in request.user_query


@pytest.mark.parametrize("status", ["partial", "blocked", "needs_clarification"])
@pytest.mark.parametrize("entry", ["submit", "monitor"])
def test_monitors_stop_after_non_success_terminal_response(monkeypatch, capsys, status, entry):
    import scripts.submit_research as script

    data = {"status": status, "result": {
        "answer": "研究要求仍未满足", "clarification": {"question": "请选择范围"}
    }}
    get = Mock(return_value=SimpleNamespace(
        status_code=200, json=lambda: {"code": 0, "data": data}
    ))
    monkeypatch.setattr(script.requests, "get", get)
    monkeypatch.setattr(script, "get_api_key", lambda: "")
    # WHY: 用 BaseException 中止意外轮询，避免脚本的网络错误重试吞掉测试失败。
    class UnexpectedPoll(BaseException):
        pass

    sleep = Mock(side_effect=UnexpectedPoll)
    monkeypatch.setattr(script.time, "sleep", sleep)
    if entry == "submit":
        script.monitor_job("example-job")
    else:
        monkeypatch.setattr("sys.argv", ["monitor_job.py", "example-job"])
        runpy.run_path(str(ROOT / "scripts" / "monitor_job.py"), run_name="__main__")
    assert get.call_count == 1
    sleep.assert_not_called()
    output = capsys.readouterr().out
    assert status in output
    assert "质量状态  : ✅ 通过" not in output


def test_monitor_blocked_without_answer_shows_real_error_reason(monkeypatch, capsys):
    """blocked 且 answer 为空时展示 result.errors 的真实阻断原因，不回退到"质量缺口"话术。"""
    import scripts.submit_research as script

    data = {"status": "blocked", "result": {
        "answer": "",
        "errors": [{"code": "agent_budget_exhausted", "message": "主 Agent 执行预算已耗尽"}],
    }}
    get = Mock(return_value=SimpleNamespace(
        status_code=200, json=lambda: {"code": 0, "data": data}
    ))
    monkeypatch.setattr(script.requests, "get", get)
    monkeypatch.setattr(script, "get_api_key", lambda: "")

    class UnexpectedPoll(BaseException):
        pass

    monkeypatch.setattr(script.time, "sleep", Mock(side_effect=UnexpectedPoll))
    monkeypatch.setattr("sys.argv", ["monitor_job.py", "example-job"])
    runpy.run_path(str(ROOT / "scripts" / "monitor_job.py"), run_name="__main__")
    assert get.call_count == 1
    output = capsys.readouterr().out
    assert "主 Agent 执行预算已耗尽" in output
    assert "质量缺口" not in output
