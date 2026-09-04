"""架构硬化后的公开契约测试。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.agent.graph import derive_result_status
from app.agent.research_semantic_parser import parse_research_semantics
from app.agent.topic_disambiguation import analyze_topic_ambiguity
from app.clients.crossref_client import parse_crossref_response
from app.core.config import get_settings
from app.core.json_utils import parse_json_object
from app.core.security import validate_deployment_security
from app.schemas.agent_schema import AgentRequest
from app.services.library_service import LibraryService


class MarineSemanticsLLM:
    def complete(self, prompt: str, **kwargs) -> str:
        return json.dumps({
            "canonical_topic": "marine corrosion monitoring",
            "application_domains": [{
                "id": "marine_engineering", "label": "marine engineering",
                "surface_text": "海洋工程", "explicit": True, "confidence": 0.96,
            }],
            "research_objects": [{
                "id": "steel_corrosion", "label": "steel corrosion",
                "surface_text": "钢结构腐蚀", "explicit": True, "confidence": 0.96,
            }],
            "methods": [{
                "id": "electrochemical_impedance_spectroscopy",
                "label": "electrochemical impedance spectroscopy",
                "surface_text": "电化学阻抗谱监测方法", "category": "measurement",
                "explicit": True, "confidence": 0.96,
            }],
            "research_actions": [],
            "analysis_targets": [],
            "terminal_goal": {"type": "domain_monitoring", "target": "steel_corrosion"},
            "task_chain": ["measure_impedance", "estimate_corrosion_state"],
            "required_focuses": ["电化学阻抗谱监测"],
            "evidence_requirements": [{
                "requirement_id": "measurement:eis",
                "label": "电化学阻抗谱监测证据",
                "evidence_role": "measurement_method",
                "aliases": ["电化学阻抗谱", "electrochemical impedance spectroscopy"],
                "source_ids": ["electrochemical_impedance_spectroscopy"],
            }],
            "confidence": {"overall": 0.94},
        }, ensure_ascii=False)


def test_semantics_are_generated_for_unseen_domain_by_llm():
    frame = parse_research_semantics(
        "研究海洋工程钢结构腐蚀的电化学阻抗谱监测方法",
        "海洋工程腐蚀监测",
        llm=MarineSemanticsLLM(),
    )

    assert frame.application_domains[0].id == "marine_engineering"
    assert frame.methods[0].id == "electrochemical_impedance_spectroscopy"
    assert "removed_ungrounded_method:electrochemical_impedance_spectroscopy" not in frame.validation_issues
    assert frame.task_chain == ["measure_impedance", "estimate_corrosion_state"]
    assert frame.evidence_requirements[0].aliases[-1] == "electrochemical impedance spectroscopy"


class ClearMarineSemanticsLLM(MarineSemanticsLLM):
    def __init__(self):
        self.operations = []

    def complete(self, prompt: str, **kwargs) -> str:
        operation = kwargs.get("operation")
        self.operations.append(operation)
        if operation == "topic_disambiguation":
            raise AssertionError("明确主题不应再次调用消歧模型")
        data = json.loads(super().complete(prompt, **kwargs))
        data["clarification_needed"] = True
        data["clarification_question"] = "是否还要限定具体服役海域？"
        data["scope_ambiguities"] = ["具体服役海域未限定"]
        return json.dumps(data, ensure_ascii=False)


def test_clear_object_method_and_goal_skip_second_disambiguation_call():
    llm = ClearMarineSemanticsLLM()
    result = analyze_topic_ambiguity(
        "调研近五年海洋工程钢结构腐蚀的电化学阻抗谱监测方法，并生成研究现状",
        llm=llm,
        current_year=2026,
    )

    assert result["needs_clarification"] is False
    assert result["ambiguity"]["recommended_strategy"] == "single_scope"
    assert result["research_request"]["semantic_frame"]["clarification_needed"] is False
    assert llm.operations == ["research_semantic_parsing"]


def test_no_llm_does_not_inject_a_historical_domain_template():
    frame = parse_research_semantics("量子材料缺陷研究", "量子材料缺陷研究", llm=None)
    serialized = frame.model_dump_json()
    assert frame.methods == []
    assert frame.application_domains == []
    assert "classroom" not in serialized and "education" not in serialized


def test_disambiguation_without_llm_does_not_fabricate_fixed_scopes():
    result = analyze_topic_ambiguity("调研量子材料缺陷研究", llm=None, current_year=2026)
    assert result["ambiguity"]["scopes"] == []


def test_client_state_rejects_internal_control_fields():
    with pytest.raises(ValidationError):
        AgentRequest(user_query="研究主题", state={"generation_blocked": False})


def test_status_derivation_treats_blocked_and_failed_as_non_success():
    assert derive_result_status({"generation_blocked": True}) == "blocked"
    assert derive_result_status({"planning_failed": True}) == "failed"
    # 只有显式获准发布的降级草稿才是 partial；未授权失败草稿仍是 blocked。
    assert derive_result_status({
        "quality_gate": {"passed": False, "partial_success": True, "draft_released": True}
    }) == "partial"
    assert derive_result_status({
        "quality_gate": {"passed": False, "partial_success": False, "draft_released": False}
    }) == "blocked"


def test_json_object_parser_never_returns_a_list():
    assert parse_json_object('[{"not": "an object"}]') == {}


def test_json_comment_stripping_preserves_urls_inside_strings():
    """字符串内的 ``//``（URL）不得被当注释截断，字符串外的注释仍被去除。"""
    text = (
        '{"url": "https://arxiv.org/abs/2401.12345", '
        '"path": "a//b", '
        '"note": "keep /* this */ too", '  # 字符串内块注释样式
        '// 行注释应被去除\n'
        '"n": 1 /* 块注释 */}'
    )
    result = parse_json_object(text)
    assert result.get("url") == "https://arxiv.org/abs/2401.12345"
    assert result.get("path") == "a//b"
    assert result.get("note") == "keep /* this */ too"
    assert result.get("n") == 1


def test_concurrent_first_get_logger_does_not_duplicate_handlers(monkeypatch):
    """多线程首次并发调用 get_logger 时根日志器只配置一次。"""
    import logging as _logging
    from concurrent.futures import ThreadPoolExecutor

    from app.core import logger as logger_module

    root = _logging.getLogger("research_review_agent")
    monkeypatch.setattr(logger_module, "_root_initialized", False)
    monkeypatch.setattr(root, "handlers", [], raising=False)

    def _get(_):
        return logger_module.get_logger(f"race.{_}")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_get, range(32)))

    assert len(root.handlers) == 2  # 控制台 + 文件，各一个
    for handler in root.handlers:
        handler.close()
        root.removeHandler(handler)


def test_crossref_pdf_without_open_license_is_not_marked_open_access():
    response = {"message": {"items": [{
        "DOI": "10.1/example",
        "title": ["Example"],
        "link": [{"content-type": "application/pdf", "URL": "https://publisher.test/paper.pdf"}],
        "is-referenced-by-count": 10,
    }]}}
    paper = parse_crossref_response(response)[0]
    assert paper.pdf_url
    assert paper.is_open_access is False


def test_external_bind_requires_api_key(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "app_host", "0.0.0.0")
    monkeypatch.setattr(settings, "app_api_key", "")
    with pytest.raises(RuntimeError):
        validate_deployment_security()


def test_library_import_rejects_path_outside_configured_inbox(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(get_settings(), "library_import_dir", str(inbox))

    assert LibraryService(None).import_pdf(str(outside)) is None


def test_session_id_columns_fit_schema_allowed_length():
    """L6 回归：session_id 列宽必须不小于 schema 允许的 128 字符。"""
    from app.database.models import ResearchJobModel, ResearchSessionModel

    for model in (ResearchSessionModel, ResearchJobModel):
        column = model.__table__.columns["session_id"]
        assert column.type.length >= 128, (
            f"{model.__tablename__}.session_id 列宽 {column.type.length} < 128，"
            "长会话 ID 入库会截断报错"
        )
