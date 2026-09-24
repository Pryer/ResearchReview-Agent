"""每个响应计一次费；缓存优化不丢提示规则。"""
import json
from types import SimpleNamespace as NS
import pytest
from app.services.llm_service import LLMService
from app.core.metrics import get_metrics_collector
from app.core.exceptions import LLMInvocationError


def response(content="ok", *, invalid=False, reasoning=False):
    return NS(usage=NS(prompt_tokens=100, completion_tokens=20,
                       prompt_cache_hit_tokens=80, prompt_cache_miss_tokens=20),
              choices=[NS(finish_reason="length" if reasoning else "stop", message=NS(
                  content=content, reasoning_content="reason" if reasoning else None,
                  tool_calls=[NS(id="call", function=NS(name="read_summary", arguments="{" if invalid else "{}"))]))])


def service_with(*responses):
    service = LLMService()
    service.api_key = "test"
    service.backup_enabled = False
    service.native_tools_enabled = True
    items = iter(responses)
    service._client = NS(chat=NS(completions=NS(create=lambda **kw: next(items))))
    get_metrics_collector().reset()
    return service


def test_invalid_tool_response_is_charged():
    service = service_with(response(invalid=True))
    with pytest.raises(LLMInvocationError):
        service.complete_tool_call([{"role": "user", "content": "test"}], tools=[{}])
    report = get_metrics_collector().get_token_report()
    assert report["total_calls"] == 1
    assert report["total_tokens"] == 120
    assert report["total_cache_hit_tokens"] == 80


def test_empty_reasoning_retry_records_both_responses_once():
    service = service_with(response("", reasoning=True), response())
    assert service.complete("test", retry_empty=True, max_tokens=10) == "ok"
    report = get_metrics_collector().get_token_report()
    assert report["total_calls"] == 2
    assert report["total_tokens"] == 240
    assert report["total_cache_hit_tokens"] == 160


def test_malformed_primary_and_valid_backup_are_both_charged():
    service = service_with(response(invalid=True))
    service.backup_enabled = True
    service._backup_client = NS(chat=NS(completions=NS(create=lambda **kw: response())))
    assert service.complete_tool_call([{"role": "user", "content": "test"}], tools=[{}])["name"] == "read_summary"
    assert get_metrics_collector().get_token_report()["total_tokens"] == 240


def test_retry_failure_does_not_erase_first_paid_response():
    service = service_with(response("", reasoning=True))
    with pytest.raises(LLMInvocationError):
        service.complete("test", retry_empty=True, max_tokens=10)
    assert get_metrics_collector().get_token_report()["total_tokens"] == 120


def test_native_prefix_has_no_duplicate_catalog_but_json_keeps_catalog():
    service = service_with()
    captured = []
    service._client = NS(chat=NS(completions=NS(create=lambda **kw: (captured.append(kw) or response()))))
    role = service.for_agent("main")
    role.complete_tool_call([{"role": "user", "content": "test"}], tools=[{}])
    role.complete_messages([{"role": "user", "content": "test"}])
    assert "核心工具目录" not in captured[0]["messages"][0]["content"]
    assert "核心工具目录" in captured[1]["messages"][0]["content"]


def test_paper_prompt_static_schema_precedes_dynamic_evidence():
    from app.prompt.paper_card import PAPER_CARD_EXTRACTION_PROCTION_PROMPT as template
    rendered = template.format(paper_id="paper-x", title="TITLE-X", evidence_label="LABEL-X",
                               evidence_source="abstract", full_text_or_json="EVIDENCE-X")
    assert rendered.index("返回 JSON 结构") < rendered.index("TITLE-X")
    assert rendered.index("results 中的数字") < rendered.index("EVIDENCE-X")


def test_section_prompt_keeps_rules_before_changing_input():
    from app.prompt.writing.section import _section_rewrite_prompt
    from app.schemas.deliverable_schema import CoreDeliverableType
    prompt = _section_rewrite_prompt(deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        section_id="theme_1", title="TITLE-X", topic="TOPIC-X", original="EVIDENCE-X", required_ids=["p1"])
    assert prompt.index("证据强度决定语言强度") < prompt.index("TOPIC-X")
    assert "## TITLE-X" in prompt
    assert "EVIDENCE-X" in prompt
