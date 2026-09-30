"""LLM 调用边界测试。"""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.core.exceptions import LLMInvocationError
from app.services.llm_service import LLMService


# ---------- 测试辅助 ----------

def _resp(content: str, finish_reason: str = "stop"):
    """构造一个最小可用的 LLM 响应对象。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, reasoning_content=None),
            finish_reason=finish_reason,
        )]
    )


def _tool_resp(name: str, arguments: dict):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
            )],
        ))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
    )


def _fake_client(create_fn):
    """构造一个假 OpenAI 客户端，chat.completions.create 调用 create_fn。"""
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_fn)))


def _raise(exc):
    """返回一个调用即抛 exc 的假 create 函数。"""
    return lambda **kw: (_ for _ in ()).throw(exc)


def _req() -> httpx.Request:
    return httpx.Request("POST", "http://test")


def _timeout_exc() -> APITimeoutError:
    return APITimeoutError(_req())


def _conn_exc() -> APIConnectionError:
    return APIConnectionError(message="conn err", request=_req())


def _status_exc(code: int) -> APIStatusError:
    return APIStatusError(
        f"status {code}",
        response=httpx.Response(code, request=_req()),
        body=None,
    )


def _new_service() -> LLMService:
    """构造一个主备都已就绪、client 可被替换的 LLMService。"""
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = True
    service.backup_model = "backup-model"
    return service


# ---------- 既有测试（修正 lambda 签名以兼容 client 关键字参数）----------

def test_control_plane_can_disable_empty_content_retry(monkeypatch):
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False  # 只测主用，避免构造备用 client
    service._client = _fake_client(lambda **kw: _resp(""))
    response = _resp("", finish_reason="length")
    response.choices[0].message.reasoning_content = "reasoning"
    monkeypatch.setattr(
        service,
        "_call_with_possible_fallback",
        lambda kwargs, response_format, client=None, record=None, operation="": response,
    )
    monkeypatch.setattr(
        service,
        "_retry_if_content_empty",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not retry")),
    )

    assert service.complete("test", retry_empty=False) == ""


def test_role_bound_llm_uses_stable_system_prefix_and_structured_messages():
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    captured = []
    service._client = _fake_client(
        lambda **kwargs: (captured.append(kwargs) or _resp("ok"))
    )
    bound = service.for_agent("search")

    assert bound.complete("query one", retry_empty=False) == "ok"
    assert bound.complete("query two", retry_empty=False) == "ok"

    assert captured[0]["messages"][0]["role"] == "system"
    assert captured[0]["messages"][0] == captured[1]["messages"][0]
    assert captured[0]["messages"][1] == {"role": "user", "content": "query one"}
    assert captured[1]["messages"][1] == {"role": "user", "content": "query two"}


def test_role_bound_native_tool_call_injects_prefix_and_returns_one_call():
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    captured = {}
    service._client = _fake_client(
        lambda **kwargs: (captured.update(kwargs) or _tool_resp("search_and_rank", {}))
    )

    result = service.for_agent("main").complete_tool_call(
        [{"role": "user", "content": '{"goal":{}}'}],
        tools=[{"type": "function", "function": {
            "name": "search_and_rank", "description": "search",
            "parameters": {"type": "object", "properties": {}},
        }}],
    )

    assert result["name"] == "search_and_rank"
    assert result["arguments"] == {}
    assert result["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
    }
    assert captured["tool_choice"] == "required"
    assert captured["messages"][0]["role"] == "system"


def test_auto_reasoning_enables_tool_decision_without_required_tool_choice():
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    service.provider = "deepseek"
    service.reasoning_mode = "auto"
    captured = {}
    service._client = _fake_client(
        lambda **kwargs: (captured.update(kwargs) or _tool_resp("search_and_rank", {}))
    )
    result = service.complete_tool_call(
        [{"role": "user", "content": '{"goal":{}}'}],
        tools=[{"type": "function", "function": {
            "name": "search_and_rank", "description": "search",
            "parameters": {"type": "object", "properties": {}},
        }}],
        operation="main_agent_decision",
    )
    assert result["name"] == "search_and_rank"
    assert "tool_choice" not in captured
    assert "temperature" not in captured
    assert captured["extra_body"] == {
        "thinking": {"type": "enabled"}, "reasoning_effort": "low",
    }


def test_auto_reasoning_keeps_simple_keyword_call_off():
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    service.provider = "deepseek"
    service.reasoning_mode = "auto"
    captured = {}
    service._client = _fake_client(
        lambda **kwargs: (captured.update(kwargs) or _resp("ok"))
    )
    service.complete("keywords", operation="generate_search_keywords")
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_reasoning_policy_respects_static_switch_and_upgrades_auto_retry():
    from app.services.reasoning_policy import choose_reasoning

    base = {"default_enabled": False, "default_effort": "low"}
    assert choose_reasoning("write_section:background:attempt_1", mode="static", **base) == (False, "low")
    assert choose_reasoning("write_section:background:attempt_1", mode="auto", **base) == (True, "low")
    assert choose_reasoning("write_section:background:attempt_2", mode="auto", **base) == (True, "high")
    assert choose_reasoning("generate_search_keywords", mode="auto", **base) == (False, "low")
    assert choose_reasoning("write_section:background:attempt_2", mode="auto",
                            requested_enabled=False, **base) == (False, "low")
    with pytest.raises(ValueError, match="reasoning_effort"):
        choose_reasoning("write_section", mode="auto", requested_effort="ultra", **base)


@pytest.mark.parametrize("native", [False, True])
def test_balance_failure_stops_current_execution_and_retries_in_new_execution(native):
    from app.agent.execution_budget import budget_scope, is_control_exception
    from app.core.exceptions import LLMProviderUnavailableError
    service = _new_service()
    service.backup_enabled = False
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise _status_exc(402)
    service._client = _fake_client(fail)
    def call():
        if native:
            return service.complete_tool_call([{"role": "user", "content": "test"}], tools=[{"type": "function"}])
        return service.complete("test")
    with budget_scope({}):
        for _ in range(2):
            with pytest.raises(LLMProviderUnavailableError) as raised:
                call()
            assert is_control_exception(raised.value)
    assert len(calls) == 1
    with budget_scope({}):
        with pytest.raises(LLMProviderUnavailableError):
            call()
    assert len(calls) == 2


def test_primary_balance_failure_uses_backup_without_retrying_primary_in_execution():
    from app.agent.execution_budget import budget_scope
    service = _new_service()
    calls = []
    def primary(**kwargs):
        calls.append('primary')
        raise _status_exc(402)
    service._client = _fake_client(primary)
    service._backup_client = _fake_client(lambda **kwargs: (calls.append('backup') or _resp('ok')))
    with budget_scope({}):
        assert service.complete('test') == 'ok'
        assert service.complete('test') == 'ok'
    assert calls == ['primary', 'backup', 'backup']


def test_static_writing_does_not_expand_token_limit_when_thinking_disabled():
    service = _new_service()
    service.backup_enabled = False
    service.reasoning_mode = 'static'
    service.thinking_enabled = False
    service.provider = 'deepseek'
    captured = {}
    service._client = _fake_client(lambda **kwargs: (captured.update(kwargs) or _resp('ok')))
    service.complete('test', operation='write_section:background:attempt_1')
    assert captured['max_tokens'] == service.max_tokens
    assert captured['extra_body']['thinking']['type'] == 'disabled'


def test_native_tool_decision_records_usage_and_cache_tokens():
    """主 Agent 原生工具决策轮次必须进入指标，并带缓存命中明细。"""
    from app.core.metrics import get_metrics_collector

    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(name="search_and_rank", arguments="{}"),
            )],
        ))],
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=20,
            prompt_cache_hit_tokens=700,
            prompt_cache_miss_tokens=300,
        ),
    )
    service._client = _fake_client(lambda **kwargs: response)
    collector = get_metrics_collector()
    collector.reset()

    result = service.complete_tool_call(
        [{"role": "user", "content": '{"goal":{}}'}],
        tools=[{"type": "function", "function": {
            "name": "search_and_rank", "description": "search",
            "parameters": {"type": "object", "properties": {}},
        }}],
        operation="main_decision",
    )

    assert result["usage"]["prompt_cache_hit_tokens"] == 700
    report = collector.get_token_report()
    assert report["total_calls"] == 1
    assert report["total_cache_hit_tokens"] == 700
    assert report["total_cache_miss_tokens"] == 300
    assert report["by_operation"]["main_decision"]["prompt_tokens"] == 1000


# ---------- 主用 → 备用 切换测试 ----------

def test_primary_succeeds_no_backup_called():
    """主用成功时不应触碰备用。"""
    service = _new_service()
    service._client = _fake_client(lambda **kw: _resp("primary-content"))
    service._backup_client = _fake_client(
        _raise(AssertionError("backup must not be called when primary succeeds")),
    )
    assert service.complete("hi", retry_empty=False) == "primary-content"


def test_deepseek_v4_explicitly_disables_thinking():
    """DeepSeek V4 必须显式关闭默认开启的 thinking 模式。"""
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    service.provider = "deepseek"
    service.thinking_enabled = False
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _resp("content")

    service._client = _fake_client(create)
    assert service.complete("hi", retry_empty=False) == "content"
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_non_deepseek_provider_does_not_receive_thinking_extension():
    """其他 OpenAI 兼容提供商不应收到 DeepSeek 私有参数。"""
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    service.provider = "openai"
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _resp("content")

    service._client = _fake_client(create)
    assert service.complete("hi", retry_empty=False) == "content"
    assert "extra_body" not in captured


def test_call_can_enable_thinking_without_changing_global_default():
    """最终写作可单次开启 thinking，后续控制面调用仍保持关闭。"""
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    service.provider = "deepseek"
    service.thinking_enabled = False
    captured = []

    def create(**kwargs):
        captured.append(kwargs)
        return _resp("content")

    service._client = _fake_client(create)
    assert service.complete(
        "final writing",
        retry_empty=False,
        thinking_enabled=True,
    ) == "content"
    assert service.complete("control plane", retry_empty=False) == "content"
    assert captured[0]["extra_body"] == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }
    assert captured[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "temperature" not in captured[0]
    assert "temperature" in captured[1]
    assert captured[0]["max_tokens"] == 32768
    assert captured[1]["max_tokens"] == 8192


def test_control_plane_budget_stays_bounded_even_when_thinking_is_enabled():
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    service.provider = "deepseek"
    service.control_plane_max_tokens = 2048
    captured = {}
    service._client = _fake_client(
        lambda **kwargs: (captured.update(kwargs) or _resp("content"))
    )

    assert service.complete(
        "verify",
        retry_empty=False,
        operation="verify_claim_entailment",
        thinking_enabled=True,
    ) == "content"

    assert captured["max_tokens"] == 2048
    assert captured["extra_body"]["thinking"] == {"type": "enabled"}


def test_explicit_max_tokens_overrides_operation_budget():
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    captured = {}
    service._client = _fake_client(
        lambda **kwargs: (captured.update(kwargs) or _resp("content"))
    )

    service.complete(
        "verify", retry_empty=False, operation="verify_claim_entailment",
        max_tokens=777,
    )

    assert captured["max_tokens"] == 777


def test_openai_sdk_serializes_deepseek_thinking_in_request_body():
    """验证 extra_body 最终会被 OpenAI SDK 合并进 DeepSeek HTTP 请求体。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-v4-flash",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = OpenAI(
        api_key="test",
        base_url="https://api.deepseek.com",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "test"}],
        extra_body={"thinking": {"type": "enabled"}},
    )
    assert captured["thinking"] == {"type": "enabled"}


def test_backup_switch_when_primary_times_out():
    """主用超时 → 切备用 → 返回备用内容。"""
    service = _new_service()
    service._client = _fake_client(_raise(_timeout_exc()))
    service._backup_client = _fake_client(lambda **kw: _resp("backup-content"))
    assert service.complete("hi", retry_empty=False, operation="test_op") == "backup-content"


def test_failover_attempts_share_one_total_deadline():
    service = _new_service()
    service.request_timeout = 120
    service.failover_total_timeout = 30
    timeouts = []

    def primary(**kwargs):
        timeouts.append(kwargs["timeout"])
        raise _timeout_exc()

    def backup(**kwargs):
        timeouts.append(kwargs["timeout"])
        return _resp("backup-content")

    service._client = _fake_client(primary)
    service._backup_client = _fake_client(backup)

    assert service.complete("hi", retry_empty=False) == "backup-content"
    assert len(timeouts) == 2
    assert all(0 < value <= 30 for value in timeouts)


def test_backup_switch_on_connection_error():
    """主用连接错误 → 切备用。"""
    service = _new_service()
    service._client = _fake_client(_raise(_conn_exc()))
    service._backup_client = _fake_client(lambda **kw: _resp("backup-content"))
    assert service.complete("hi", retry_empty=False) == "backup-content"


def test_backup_switch_on_429():
    """主用 429 限流 → 切备用。"""
    service = _new_service()
    service._client = _fake_client(_raise(_status_exc(429)))
    service._backup_client = _fake_client(lambda **kw: _resp("backup-content"))
    assert service.complete("hi", retry_empty=False) == "backup-content"


def test_backup_switch_on_provider_quota_exhaustion_402():
    """兼容接口用 402 表示 Token 额度不足时也应切到备用。"""
    service = _new_service()
    service._client = _fake_client(_raise(_status_exc(402)))
    service._backup_client = _fake_client(lambda **kw: _resp("backup-content"))
    assert service.complete("hi", retry_empty=False) == "backup-content"


def test_all_providers_fail_raises_llm_invocation_error():
    """主备都失败 → 抛 LLMInvocationError（调用方兜底应能接住）。"""
    service = _new_service()
    service._client = _fake_client(_raise(_timeout_exc()))
    service._backup_client = _fake_client(_raise(_timeout_exc()))
    with pytest.raises(LLMInvocationError):
        service.complete("hi", retry_empty=False)


def test_non_retryable_status_bubbles_up():
    """401 等非可重试状态码：不切备用，直接抛 LLMInvocationError。"""
    service = _new_service()
    service._client = _fake_client(_raise(_status_exc(401)))
    service._backup_client = _fake_client(
        _raise(AssertionError("backup must not be called on non-retryable status")),
    )
    with pytest.raises(LLMInvocationError):
        service.complete("hi", retry_empty=False)


# ---------- 空 content 切备用测试 ----------

def test_backup_switch_when_primary_returns_empty():
    """主用返回空 content → 切备用 → 返回备用内容。"""
    service = _new_service()
    service._client = _fake_client(lambda **kw: _resp(""))  # 主用返回空字符串
    service._backup_client = _fake_client(lambda **kw: _resp("backup-content"))
    assert service.complete("hi", retry_empty=False, operation="test_empty") == "backup-content"
    # 这里的日志断言需要 caplog，简化测试只断言返回值，日志可选手动验证


def test_both_providers_empty_is_a_failed_invocation():
    """主备都返回空 content 时不得记录成功或把空结果交给下游。"""
    service = _new_service()
    service._client = _fake_client(lambda **kw: _resp(""))
    service._backup_client = _fake_client(lambda **kw: _resp(""))
    with pytest.raises(LLMInvocationError):
        service.complete("hi", retry_empty=False)


# ---------- 空 choices 防护与指标记录 ----------

def test_empty_choices_does_not_crash_and_switches_to_backup():
    """提供商返回空 choices 时视为空内容，切备用而不是 IndexError。"""
    service = _new_service()
    service._client = _fake_client(lambda **kw: SimpleNamespace(choices=[]))
    service._backup_client = _fake_client(lambda **kw: _resp("backup-content"))
    assert service.complete("hi", retry_empty=False) == "backup-content"


def test_empty_choices_without_backup_returns_empty_not_crash():
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    service._client = _fake_client(lambda **kw: SimpleNamespace(choices=[]))
    assert service.complete("hi", retry_empty=False) == ""


def test_raw_string_response_with_retry_empty_switches_to_backup():
    """提供商返回裸字符串（无 choices）且 retry_empty=True 时，
    _retry_if_content_empty 不得因 str.choices 崩溃，应回退到空内容分支
    并切换到备用提供商。复现自实测日志中的
    "'str' object has no attribute 'choices'"。
    """
    service = _new_service()
    service._client = _fake_client(lambda **kw: "内容过滤时返回的原始字符串")
    service._backup_client = _fake_client(lambda **kw: _resp("backup-content"))
    assert service.complete("hi", retry_empty=True, operation="claim_thesis_clustering") == "backup-content"


def test_raw_string_response_without_backup_returns_empty_not_crash():
    """无备用时裸字符串响应也不应崩溃，按空内容返回交由调用方兜底。"""
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    service._client = _fake_client(lambda **kw: "原始字符串")
    assert service.complete("hi", retry_empty=True) == ""


def test_metrics_record_real_operation_tokens_and_duration(monkeypatch):
    """usage 指标必须带真实 operation 与恰好一次记录（此前恒默认值/双计）。"""
    from app.core import metrics as metrics_module

    recorded = []

    class FakeCollector:
        def record_llm_call(self, **kw):
            recorded.append(kw)

    monkeypatch.setattr(metrics_module, "get_metrics_collector", lambda: FakeCollector())

    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="ok", reasoning_content=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )
    service._client = _fake_client(lambda **kw: resp)

    assert service.complete("hi", retry_empty=False, operation="plan_op") == "ok"
    assert len(recorded) == 1
    assert recorded[0]["operation"] == "plan_op"
    assert recorded[0]["provider"] == service.provider
    assert recorded[0]["prompt_tokens"] == 11
    assert recorded[0]["completion_tokens"] == 7
    assert recorded[0]["duration_ms"] >= 0


def test_no_usage_response_skips_metrics_recording(monkeypatch):
    from app.core import metrics as metrics_module

    recorded = []

    class FakeCollector:
        def record_llm_call(self, **kw):
            recorded.append(kw)

    monkeypatch.setattr(metrics_module, "get_metrics_collector", lambda: FakeCollector())

    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    service._client = _fake_client(lambda **kw: _resp("no-usage"))
    assert service.complete("hi", retry_empty=False) == "no-usage"
    assert recorded == []
