"""测试链路可观测性与 Token 消耗监控。"""

from app.core.metrics import MetricsCollector, get_metrics_collector
from app.agent.nodes.base import append_step

def test_metrics_collector_records_llm_tokens():
    collector = MetricsCollector()
    collector.record_llm_call(
        model="deepseek-v4-flash",
        prompt_tokens=150,
        completion_tokens=50,
        duration_ms=450,
        operation="planning",
    )
    collector.record_llm_call(
        model="deepseek-v4-flash",
        prompt_tokens=300,
        completion_tokens=100,
        duration_ms=600,
        operation="synthesis",
    )

    report = collector.get_token_report()
    assert report["total_calls"] == 2
    assert report["total_prompt_tokens"] == 450
    assert report["total_completion_tokens"] == 150
    assert report["total_tokens"] == 600

    by_model = report["by_model"]
    assert "deepseek-v4-flash" in by_model
    assert by_model["deepseek-v4-flash"]["calls"] == 2
    assert by_model["deepseek-v4-flash"]["total_tokens"] == 600

    by_op = report["by_operation"]
    assert "planning" in by_op
    assert "synthesis" in by_op
    assert by_op["planning"]["total_tokens"] == 200
    assert by_op["synthesis"]["total_tokens"] == 400


def test_cache_hit_and_miss_tokens_are_reported_per_model_and_operation():
    collector = MetricsCollector()
    collector.record_llm_call(
        model="deepseek-v4-flash",
        prompt_tokens=1000,
        completion_tokens=200,
        operation="write_section:research_status:theme_T1",
        prompt_cache_hit_tokens=800,
        prompt_cache_miss_tokens=200,
    )
    collector.record_llm_call(
        model="deepseek-v4-flash",
        prompt_tokens=500,
        completion_tokens=100,
        operation="write_section:research_status:theme_T2",
        prompt_cache_hit_tokens=100,
        prompt_cache_miss_tokens=400,
    )

    report = collector.get_token_report()
    assert report["total_cache_hit_tokens"] == 900
    assert report["total_cache_miss_tokens"] == 600
    assert report["cache_hit_rate"] == 0.6
    assert report["by_model"]["deepseek-v4-flash"]["cache_hit_tokens"] == 900

    by_op = report["by_operation"]
    t1 = by_op["write_section:research_status:theme_T1"]
    t2 = by_op["write_section:research_status:theme_T2"]
    assert t1["cache_hit_tokens"] == 800
    assert t1["cache_miss_tokens"] == 200
    assert t2["cache_hit_tokens"] == 100
    assert t2["cache_miss_tokens"] == 400


def test_cache_hit_rate_is_none_when_provider_reports_no_cache_fields():
    """未上报缓存字段时必须是 None，不能用 0 冒充 0% 命中率。"""
    collector = MetricsCollector()
    collector.record_llm_call(
        model="deepseek-v4-flash", prompt_tokens=100, completion_tokens=10
    )

    report = collector.get_token_report()
    assert report["cache_hit_rate"] is None
    assert report["total_cache_hit_tokens"] == 0
    assert report["total_cache_miss_tokens"] == 0


def test_reset_clears_cache_token_totals():
    """reset 必须一并清空缓存累计，否则全局单例会跨测试/跨会话串味。"""
    collector = MetricsCollector()
    collector.record_llm_call(
        model="deepseek-v4-flash",
        prompt_tokens=100,
        completion_tokens=10,
        prompt_cache_hit_tokens=80,
        prompt_cache_miss_tokens=20,
    )

    collector.reset()

    report = collector.get_token_report()
    assert report["total_calls"] == 0
    assert report["total_cache_hit_tokens"] == 0
    assert report["total_cache_miss_tokens"] == 0
    assert report["cache_hit_rate"] is None
    assert report["by_model"] == {}
    assert report["by_operation"] == {}

def test_append_step_attaches_step_metrics():
    collector = get_metrics_collector()
    collector.reset()
    collector.record_llm_call("mock-model", 100, 50, operation="search")

    state = {}
    append_step(state, "search_node", "success", duration_ms=123)

    assert "step_metrics" in state
    metrics = state["step_metrics"]
    assert metrics["last_step"] == "search_node"
    assert metrics["last_duration_ms"] == 123
    assert metrics["total_tokens"] == 150
    assert metrics["total_llm_calls"] == 1
