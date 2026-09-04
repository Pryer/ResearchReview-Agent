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
