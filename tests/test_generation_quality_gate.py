"""写作前与写作后质量门禁测试。"""

from __future__ import annotations

import json
import re

from app.agent.deliverable_router import (
    check_deliverable_readiness,
    check_generation_readiness,
)
from app.agent.nodes import (
    _apply_final_quality_gate,
    _assemble_answer,
    _citation_allocation_budget,
    _global_deliverable_citation_quotas,
    _plan_citation_allocation,
    _remove_unsupported_claims,
    citation_check_node,
    final_answer_node,
    generate_deliverables_node,
)
from app.agent.writing_plan import build_writing_plan
from app.schemas.deliverable_schema import (
    CoreDeliverableType,
    WritingPlan,
    WritingSection,
)
from app.tools.synthesize_themes import synthesize_themes
from app.tools.validate_deliverable import (
    _english_sentences,
    validate_deliverable,
    validate_final_review_integrity,
)
from app.deliverables.renderers import (
    _merge_failed_or_missing_sections,
    _write_sections_in_chinese,
)
from app.tools.write_deliverable import write_deliverable


def _card(index: int) -> dict:
    paper_id = f"p{index}"
    claim = f"论文{index}明确研究课堂行为分析。"
    return {
        "paper_id": paper_id,
        "title": f"Paper {index}",
        "year": 2024,
        "quality_status": "partial",
        "evidence_source": "abstract",
        "evidence_state": {"access_level": "abstract"},
        "field_claims": {
            "research_problem": [{
                "claim": claim,
                "source_text": claim,
                "source_section": "abstract",
                "evidence_id": f"{paper_id}:e001",
                "evidence_level": "abstract",
                "explicitly_reported": True,
            }]
        },
        "unsupported_fields": ["limitations"],
    }


def test_explicit_minimum_reference_count_is_a_pre_generation_hard_gate():
    state = {
        "max_papers_explicit": True,
        "required_reference_count": 40,
        "core_deliverables": ["research_background"],
        "paper_cards": [_card(index) for index in range(1, 29)],
    }

    result = check_generation_readiness(state)

    assert result.ready is False
    assert result.usable_reference_count == 28
    assert result.blocking_issues[0]["code"] == "minimum_references_not_met"
    assert result.blocking_issues[0]["requested"] == 40


def _detail(index: int, decision: str) -> dict:
    return {"paper_id": f"p{index}", "title": f"Paper {index}", "_screening_decision": decision}


def test_rule_screened_reserve_papers_do_not_count_toward_reference_requirement():
    """规则回填、未经 LLM 语义确认的论文不得冒充达标证据。"""
    reserve_indexes = list(range(29, 41))
    state = {
        "max_papers_explicit": True,
        "required_reference_count": 40,
        "core_deliverables": ["research_background"],
        "paper_cards": [_card(index) for index in range(1, 41)],
        "paper_details": (
            [_detail(index, "include") for index in range(1, 29)]
            + [_detail(index, "rule_screened_reserve") for index in reserve_indexes]
        ),
    }

    result = check_generation_readiness(state)

    assert result.ready is False
    assert result.usable_reference_count == 28
    assert result.blocking_issues[0]["code"] == "minimum_references_not_met"
    assert result.blocking_issues[0]["requested"] == 40
    assert result.blocking_issues[0]["available"] == 28
    eligible = set(state["citation_eligible_paper_ids"])
    assert not {f"p{index}" for index in reserve_indexes} & eligible


def test_unconfirmed_reserve_stays_excluded_through_writing_and_final_count():
    """前置门禁、写作计划和最终计数必须共用同一份未经确认论文排除集。"""
    state = {
        "intent": "generate_review",
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "max_papers_explicit": True,
        "required_reference_count": 3,
        "max_papers": 3,
        "core_deliverables": ["research_background"],
        "paper_cards": [_card(index) for index in range(1, 4)],
        "paper_details": [
            {
                **_detail(index, "rule_screened_reserve" if index == 3 else "include"),
                "authors": [f"Author {index}"],
                "year": 2024,
                "venue": "Test Venue",
            }
            for index in range(1, 4)
        ],
        "steps": [],
        "errors": [],
    }

    readiness = check_generation_readiness(state)
    assert readiness.ready is False
    assert readiness.usable_reference_count == 2
    assert state["citation_eligible_paper_ids"] == ["p1", "p2"]

    plan = build_writing_plan("research_background", state)
    planned_ids = {
        paper_id
        for section in plan.sections
        for paper_id in section.supporting_paper_ids
    }
    assert "p3" not in state["citation_eligible_paper_ids"]
    assert "p3" not in planned_ids

    state["writing_plans"] = [plan.model_dump(mode="json")]
    state["review"] = "课堂行为分析已有多项研究[p1][p2]，另有待确认工作[p3]。"
    state["citation_style"] = "gbt7714"
    citation_check_node(state)

    assert state["unique_cited_paper_count"] == 2
    assert state["final_requirement_met"] is False
    assert "p3" in state["citation_validation"]["missing_citations"]

    _apply_final_quality_gate(state)
    shortfall = next(
        issue for issue in state["quality_gate"]["blocking_issues"]
        if issue["code"] == "minimum_cited_references_not_met"
    )
    assert shortfall["requested"] == 3
    assert shortfall["actual"] == 2


def test_llm_screened_uncertain_papers_still_count_toward_reference_requirement():
    """uncertain 是 LLM 实际给出的判定，与规则回填不同，不得被一并排除。"""
    state = {
        "max_papers_explicit": True,
        "required_reference_count": 40,
        "core_deliverables": ["research_background"],
        "paper_cards": [_card(index) for index in range(1, 41)],
        "paper_details": (
            [_detail(index, "include") for index in range(1, 29)]
            + [_detail(index, "uncertain") for index in range(29, 41)]
        ),
    }

    result = check_generation_readiness(state)

    assert result.usable_reference_count == 40
    assert not any(
        issue["code"] == "minimum_references_not_met"
        for issue in result.blocking_issues
    )


def test_required_focus_evidence_is_a_pre_generation_hard_gate():
    cards = [_card(1), _card(2)]
    state = {
        "core_deliverables": ["research_status"],
        "paper_cards": cards,
        "research_semantic_frame": {
            "required_focuses": ["S-T分析法", "滞后序列分析法"],
        },
    }

    result = check_generation_readiness(state)

    assert result.ready is False
    issue = next(item for item in result.blocking_issues if item["code"] == "required_focus_evidence_not_met")
    assert set(issue["missing_focuses"]) == {"S-T分析法", "滞后序列分析法"}


def test_best_effort_generation_can_proceed_with_unvalidated_taxonomy(monkeypatch):
    cards = [_card(1), _card(2), _card(3), _card(4)]
    state = {
        "intent": "generate_review",
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "core_deliverables": ["research_status"],
        "requested_sections": ["research_status"],
        "paper_details": cards,
        "paper_cards": cards,
        "dynamic_taxonomy": {},
        "taxonomy_validation": {"valid": False, "status": "invalid"},
        "required_reference_count": 4,
        "max_papers": 4,
        "max_papers_explicit": True,
        "best_effort_generation": True,
        "allow_unvalidated_taxonomy": True,
        "steps": [],
        "errors": [],
    }

    monkeypatch.setattr(
        "app.tools.write_deliverable.write_deliverable",
        lambda plan, state, llm=None: "## 研究现状\n\n当前证据形成初步研究进展[p1][p2][p3][p4]。",
    )
    generate_deliverables_node(state, llm=None)

    assert state["generation_blocked"] is False
    assert state["writing_plans"]
    assert "研究现状" in state["review"]


def test_best_effort_final_gate_replaces_stale_pre_generation_gate():
    state = {
        "intent": "generate_review",
        "best_effort_generation": True,
        "forced_generation_issues": [{"code": "taxonomy_not_ready"}],
        "taxonomy_validation": {"valid": False},
        "quality_gate": {
            "passed": False,
            "phase": "pre_generation",
            "blocking_issues": [{"code": "taxonomy_not_ready"}],
        },
        "review": "## 研究现状\n\n当前证据支持形成最佳可用综合[p1]。",
        "paper_cards": [_card(1)],
        "deliverable_validation": [],
    }

    _apply_final_quality_gate(state)

    assert state["quality_gate"]["phase"] == "post_generation"
    assert state["quality_gate"]["partial_success"] is True
    assert any(
        issue["code"] == "user_accepted_best_effort_generation"
        for issue in state["quality_gate"]["blocking_issues"]
    )


def test_best_effort_generation_still_reports_cited_reference_shortfall():
    """尽力生成是降级标记，不能把"引用 2 篇"当成"达到 3 篇要求"。"""
    state = {
        "intent": "generate_review",
        "best_effort_generation": True,
        "taxonomy_validation": {"valid": True},
        "required_reference_count": 3,
        "max_papers": 3,
        "max_papers_explicit": True,
        "unique_cited_paper_count": 2,
        "citation_validation": {
            "valid": True,
            "metadata_quality_valid": True,
            "missing_citations": [],
            "incomplete_metadata": [],
            "unverified_metadata": [],
            "duplicate_dois": [],
            "suggestions": [],
        },
        "review": "## 研究现状\n\n当前证据支持形成最佳可用综合[p1]。",
        "paper_cards": [_card(1), _card(2)],
        "deliverable_validation": [],
    }

    _apply_final_quality_gate(state)

    codes = {issue["code"] for issue in state["quality_gate"]["blocking_issues"]}
    assert "user_accepted_best_effort_generation" in codes
    assert "minimum_cited_references_not_met" in codes
    shortfall = next(
        issue for issue in state["quality_gate"]["blocking_issues"]
        if issue["code"] == "minimum_cited_references_not_met"
    )
    assert shortfall["requested"] == 3
    assert shortfall["actual"] == 2


def test_final_gate_blocks_incomplete_citation_metadata():
    state = {
        "intent": "generate_review",
        "review": "## 研究背景\n\n课堂分析已有相关研究[1]。",
        "citation_validation": {
            "valid": False,
            "metadata_quality_valid": False,
            "missing_citations": [],
            "incomplete_metadata": ["p1"],
            "unverified_metadata": [],
            "duplicate_dois": [],
            "suggestions": ["缺少作者"],
        },
        "deliverable_validation": [],
    }
    _apply_final_quality_gate(state)
    codes = {item["code"] for item in state["quality_gate"]["blocking_issues"]}
    assert "citation_metadata_not_verified" in codes


def test_final_gate_blocks_failed_section_diagnostics():
    state = {
        "intent": "generate_review",
        "review": "## 研究现状\n\n保守降级文本。",
        "writer_section_diagnostics": [{
            "deliverable_type": "research_status",
            "sections": [{"section_id": "theme_T1", "status": "evidence_limited"}],
        }],
        "deliverable_validation": [],
    }
    _apply_final_quality_gate(state)
    codes = {item["code"] for item in state["quality_gate"]["blocking_issues"]}
    assert "section_generation_failed" in codes


def test_citation_ids_are_removed_before_english_sentence_detection():
    text = "模型准确率达到84%[cnki:BsQQ9aL8NZtLVk5Knr-d76HKfJ68wFe-zukU6hgus-gjsS1ASUi2RDhbU9u24YY4]。"
    assert _english_sentences(text) == []


def test_failed_route_is_removed_and_its_citations_are_locally_reassigned():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="研究现状",
        organizing_strategy="themes",
        sections=[
            WritingSection(id="status_overview", title="研究现状", purpose="总体", supporting_paper_ids=["p1"]),
            WritingSection(id="theme_T1", title="路线一", purpose="识别", supporting_paper_ids=["p1"], heading_level=3),
            WritingSection(id="theme_T2", title="路线二", purpose="编码", supporting_paper_ids=["p2"], heading_level=3),
        ],
    )
    cards = [_card(1), _card(2)]
    state = {"citation_allocation_plan": {"sections": [
        {"section_id": "theme_T1", "paper_ids": ["p1"]},
        {"section_id": "theme_T2", "paper_ids": ["p2"]},
    ]}}
    sections = {
        "status_overview": "## 研究现状\n\n总体进展[p1]。",
        "theme_T1": "### 路线一\n\n已有研究形成识别路线[p1]。",
        "theme_T2": "### 路线二\n\n编码研究[p2]。",
    }
    completed = {
        "status_overview": sections["status_overview"],
        "theme_T1": sections["theme_T1"],
        "theme_T2": "",
    }
    diagnostics = [
        {"section_id": "theme_T1", "status": "success"},
        {"section_id": "theme_T2", "status": "evidence_limited"},
    ]

    def fake_rewrite(section):
        return section.id, "### 路线一\n\n已有研究形成识别与编码两类路线[p1][p2]。", {"section_id": section.id, "status": "success"}

    _merge_failed_or_missing_sections(
        plan=plan, state=state, cards=cards, sections=sections,
        completed=completed, diagnostics=diagnostics, rewrite=fake_rewrite,
    )

    assert "theme_T2" not in {section.id for section in plan.sections}
    assert "p2" in completed["theme_T1"]
    target = next(item for item in state["citation_allocation_plan"]["sections"] if item["section_id"] == "theme_T1")
    assert target["paper_ids"] == ["p1", "p2"]


def test_unsupported_claims_are_removed_before_citation_generation():
    text = (
        "## 路线一\n\n有证据的中文事实[p1]。\n\n"
        "## 路线二\n\n未经证据支持的事实[p2]。"
    )
    claims = [{
        "claim_id": "c2",
        "sentence": "未经证据支持的事实[p2]。",
        "factual": True,
        "support_status": "unsupported",
    }]

    repaired, removed = _remove_unsupported_claims(text, claims)

    assert len(removed) == 1
    assert "未经证据支持的事实" not in repaired
    assert "当前可访问证据不足" not in repaired
    assert "有证据的中文事实[p1]" in repaired


def test_pre_generation_gate_does_not_call_writer():
    class ForbiddenLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            raise AssertionError("硬约束失败后不得调用 Writer")

    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "core_deliverables": ["research_background"],
        "max_papers_explicit": True,
        "required_reference_count": 3,
        "paper_cards": [_card(1), _card(2)],
        "steps": [],
        "errors": [],
    }

    generate_deliverables_node(state, llm=ForbiddenLLM())

    assert state["generation_blocked"] is True
    assert state["writing_plans"] == []
    assert "正文生成已阻止" in state["review"]


def test_writer_normalizes_evidence_id_to_paper_id():
    card = _card(1)
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": [card],
        "theme_synthesis": [],
        "search_report": {},
        "evidence_quality_report": {},
    }
    plan = build_writing_plan("research_background", state)

    class EvidenceIdLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return "\n\n".join(
                f"## {section.title}\n\n论文明确研究课堂行为分析。[p1:e001]"
                for section in plan.sections
            )

    text = write_deliverable(plan, state, llm=EvidenceIdLLM())

    assert "[p1:e001]" not in text
    assert "[p1]" in text


def test_validator_rejects_invented_user_research_objective():
    state = {
        "topic": "课堂行为分析",
        "paper_cards": [_card(1), _card(2)],
        "user_paper_profile": {},
    }
    plan = build_writing_plan("research_background", state)
    text = "\n\n".join(
        f"## {section.title}\n\n论文明确研究课堂行为分析。[p1]"
        for section in plan.sections
    ) + "\n\n因此，本研究旨在开发课堂行为分析系统。"

    validation = validate_deliverable(text, plan, state)

    assert validation["valid"] is False
    assert any("用户未提供研究目标" in error for error in validation["errors"])


def test_route_evidence_deficit_is_reported_as_warning_not_blocking():
    """路线补检索未达目标只作 warning：正文仍可交付，缺口如实报告。"""
    state = {
        "intent": "generate_review",
        "review": "## 研究现状\n\n课堂行为分析已有可核验证据[p1]。",
        "references": ["Author A. Paper One[J]. CVPR, 2025."],
        "citation_validation": {"valid": True, "cited_ids": ["p1"], "metadata_quality_valid": True},
        "claim_verification": {"valid": True},
        "generation_quality": {"passed": True, "support_rate": 0.95},
        "deliverable_validation": [{"valid": True, "errors": []}],
        "route_evidence_deficits": [{
            "route_id": "R2",
            "route_name": "师生互动分析",
            "core_evidence_count": 1,
            "target_core_evidence": 4,
            "core_evidence_deficit": 3,
            "recovery_attempts": 2,
        }],
        "citation_allocation_plans": [{
            "section_floor_deficits": [{
                "section_id": "theme_T2",
                "assigned": 1,
                "required": 2,
            }],
        }],
        "steps": [],
        "errors": [],
    }

    final_answer_node(state)

    gate = state["quality_gate"]
    assert gate["passed"] is True
    warning = next(
        item for item in gate["warnings"]
        if item["code"] == "route_evidence_target_not_met"
    )
    assert "师生互动分析" in warning["message"]
    assert warning["routes"][0]["core_evidence_deficit"] == 3
    assert warning["sections"][0]["section_id"] == "theme_T2"
    # 正文照常交付，不因未达目标而阻断。
    assert "课堂行为分析已有可核验证据" in state["body"]


def test_post_generation_gate_quarantines_failed_draft_without_authorization():
    """普通门禁失败必须真正隔离草稿，不能以 partial 形式发布正式样式正文。"""
    state = {
        "intent": "generate_review",
        "review": "## 研究现状\n\n这是一份未经验证的正文。[p1]",
        "max_papers_explicit": False,
        "claim_verification": {"valid": False},
        "generation_quality": {
            "passed": False,
            "support_rate": 0.039,
            "unsupported_claims": 49,
        },
        "deliverable_validation": [],
        "steps": [],
        "errors": [],
    }

    final_answer_node(state)

    assert "这是一份未经验证的正文" not in state["answer"]
    assert "正式正文已被质量门禁阻止" in state["answer"]
    assert state["body"] == ""
    assert state["quality_gate"]["draft_available"] is True
    assert state["quality_gate"]["draft_released"] is False
    assert state["quality_gate"]["draft_disposition"] == "quarantined"
    assert state["generation_quality"]["support_rate"] == 0.039
    assert state["quarantined_draft"].startswith("## 研究现状")
    assert state["steps"][-1]["status"] == "blocked"


def test_user_accepted_best_effort_draft_is_released_with_warning():
    """用户明确接受降级交付时，仍可发布带警告的草稿并保持 partial。"""
    state = {
        "intent": "generate_review",
        "review": "## 研究现状\n\n这是一份最佳努力草稿。[p1]",
        "best_effort_generation": True,
        "max_papers_explicit": False,
        "claim_verification": {"valid": False},
        "generation_quality": {
            "passed": False,
            "support_rate": 0.42,
            "unsupported_claims": 12,
        },
        "deliverable_validation": [],
        "steps": [],
        "errors": [],
    }

    final_answer_node(state)

    assert "这是一份最佳努力草稿" in state["answer"]
    assert "质量门禁提示" in state["answer"]
    assert state["body"] == state["review"]
    assert state["quality_gate"]["draft_released"] is True
    assert state["quality_gate"]["draft_disposition"] == "released_best_effort"
    assert state["steps"][-1]["status"] == "partial"


def test_abstract_only_research_status_uses_summary_level_evidence():
    cards = [_card(index) for index in range(1, 5)]
    taxonomy = {
        "themes": [
            {"theme_id": "T1", "name": "课堂观察"},
            {"theme_id": "T2", "name": "自动识别"},
        ],
        "assignments": [
            {"paper_id": "p1", "primary_theme_id": "T1"},
            {"paper_id": "p2", "primary_theme_id": "T1"},
            {"paper_id": "p3", "primary_theme_id": "T2"},
            {"paper_id": "p4", "primary_theme_id": "T2"},
        ],
    }
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "core_deliverables": ["research_status"],
        "paper_details": cards,
        "paper_cards": cards,
        "dynamic_taxonomy": taxonomy,
        "theme_synthesis": synthesize_themes(cards, taxonomy),
        "taxonomy_validation": {"valid": True, "requires_revision": False},
        "steps": [],
        "errors": [],
    }

    generate_deliverables_node(state, llm=None)

    assert state["writing_plans"][0]["deliverable_type"] == "research_status"
    assert state["deliverable_downgrades"] == []
    assert "证据等级降级" not in state["review"]
    assert "均为摘要级证据" not in state["review"]
    assert "PDF 下载或解析失败不阻断研究现状生成" not in state["review"]
    assert state["evidence_quality_report"]["limitations"]


def test_validator_rejects_english_and_repeated_sentences():
    cards = [_card(1), _card(2)]
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
    }
    plan = build_writing_plan("research_background", state)
    repeated = "课堂行为分析需要结合多源证据理解复杂互动过程[p1]。"
    section = plan.sections[0]
    text = f"## {section.title}\n\n{repeated}\n\n{repeated}"
    text += "\n\nThis complete English sentence must be translated into Chinese before delivery."

    validation = validate_deliverable(text, plan, state)

    assert validation["valid"] is False
    assert any("完整英文句子" in error for error in validation["errors"])
    assert any("重复或高度相似句子" in error for error in validation["errors"])
    assert validation["metrics"]["english_sentence_count"] == 1
    assert validation["metrics"]["duplicate_sentence_count"] >= 1


def test_research_status_rejects_geographic_comparison_without_cited_geo_metadata():
    cards = [_card(1), _card(2), _card(3), _card(4)]
    state = {"topic": "课堂行为分析", "canonical_topic": "课堂行为分析", "paper_cards": cards}
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="研究现状",
        organizing_strategy="themes",
        sections=[
            WritingSection(id="status_overview", title="国内外研究现状", purpose="总体"),
            WritingSection(id="theme_T1", title="（一）行为识别", purpose="路线", heading_level=3),
        ],
    )
    text = (
        "## 国内外研究现状\n\n国内研究重视课堂应用，国外研究关注通用模型[p1][p2]。\n\n"
        "### （一）行为识别\n\n多项研究形成视觉识别路线[p1][p2]。综合来看，不同方法具有互补性[p3][p4]。"
    )

    validation = validate_deliverable(text, plan, state)

    assert any("可靠地域元数据" in error for error in validation["errors"])


def test_background_rejects_excessive_experimental_metrics():
    cards = [_card(1), _card(2)]
    state = {"topic": "课堂行为分析", "canonical_topic": "课堂行为分析", "paper_cards": cards}
    plan = build_writing_plan("research_background", state)
    title = plan.sections[0].title
    text = (
        f"## {title}\n\n课堂行为分析具有现实需求[p1]。\n\n"
        "模型准确率达到91%，召回率达到90%，mAP达到88%[p1]。\n\n"
        "另一模型F1达到87%，精确率达到89%[p2]。"
    )

    validation = validate_deliverable(text, plan, state)

    assert any("过多模型指标" in error for error in validation["errors"])


def test_repeated_template_conclusions_are_rejected():
    cards = [_card(1), _card(2)]
    state = {"topic": "课堂行为分析", "canonical_topic": "课堂行为分析", "paper_cards": cards}
    plan = build_writing_plan("research_background", state)
    title = plan.sections[0].title
    text = (
        f"## {title}\n\n这种差异反映的是研究目标与数据条件不同[p1]。\n\n"
        "课堂观察与自动识别分别服务于不同教学问题[p1][p2]。\n\n"
        "这些差异反映的是研究目标与数据条件不同[p2]。"
    )

    validation = validate_deliverable(text, plan, state)

    assert any("模板化总结短语" in error for error in validation["errors"])


def test_fallback_writer_does_not_append_citation_filler():
    cards = [_card(index) for index in range(1, 5)]
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "theme_synthesis": [],
        "search_report": {},
        "evidence_quality_report": {},
    }
    plan = build_writing_plan("research_background", state)

    text = write_deliverable(plan, state, llm=None)

    assert "论文明确报告" not in text
    assert "另见文献" not in text


def test_deliverable_gate_rejects_doubled_sentence_punctuation():
    cards = [_card(1), _card(2)]
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "theme_synthesis": [],
    }
    plan = build_writing_plan("research_background", state)
    text = "\n\n".join(
        f"## {section.title}\n\n课堂行为分析涉及多类教学场景[p1][p2]。。"
        for section in plan.sections
    )

    validation = validate_deliverable(text, plan, state)

    assert validation["valid"] is False
    assert any("异常标点" in error for error in validation["errors"])
    assert validation["metrics"]["abnormal_punctuation_count"] == 1


def test_deliverable_gate_rejects_truncated_sentence_and_editorial_note():
    cards = [_card(1), _card(2)]
    state = {"topic": "课堂行为分析", "paper_cards": cards, "theme_synthesis": []}
    plan = build_writing_plan("research_background", state)
    section = plan.sections[0]
    body = (
        "课堂行为分析已有多项研究支持[p1][p2]。"
        "〔注：此处需改为半角方括号，见说明〕\n\n已有证据表明课堂"
    )
    validation = validate_deliverable(
        f"## {section.title}\n\n{body}", plan, state
    )

    assert validation["valid"] is False
    assert any("编辑提示" in error for error in validation["errors"])
    assert any("截断" in error for error in validation["errors"])


def test_final_integrity_rechecks_text_after_claim_repair():
    state = {
        "required_reference_count": 40,
        "writing_plans": [{
            "sections": [{"id": "importance", "title": "研究价值与必要性"}],
        }],
    }
    report = validate_final_review_integrity(
        "## 研究价值与必要性\n\n已有证据使课堂",
        state,
    )

    assert report["valid"] is False
    assert report["metrics"]["incomplete_fragment_count"] == 1


def test_final_integrity_rejects_duplicate_sentences_from_claim_rewrite():
    """主张改写发生在写作期校验之后，终审必须自己查重句。"""
    duplicated = (
        "编码体系为课堂行为分析提供了可复用的观察框架，"
        "使不同研究之间的结果可以相互比较。"
    )
    text = (
        "## 编码应用与实证研究\n\n"
        f"{duplicated}后续研究进一步扩展了样本范围与学段覆盖。\n\n{duplicated}"
    )
    plans = [{"sections": [{"id": "theme_a", "title": "编码应用与实证研究"}]}]

    report = validate_final_review_integrity(
        text, {"required_reference_count": 5, "writing_plans": plans}
    )

    assert report["valid"] is False
    assert any("重复" in error for error in report["errors"]), report["errors"]
    assert report["metrics"]["duplicate_sentence_count"] == 1

    gate_state = {
        "intent": "generate_review",
        "review": text,
        "writing_plans": plans,
        "required_reference_count": 5,
        "paper_cards": [_card(1)],
        "deliverable_validation": [],
    }
    _apply_final_quality_gate(gate_state)
    codes = {issue["code"] for issue in gate_state["quality_gate"]["blocking_issues"]}
    assert "final_text_integrity_not_met" in codes


def test_claim_citation_mismatch_below_threshold_blocks_delivery():
    """一致性指标此前只写日志，实测 9/43 句错配既不阻断也不降级。"""
    state = {
        "intent": "generate_review",
        "review": "## 研究现状\n\n当前证据支持形成综合结论[p1]。",
        "paper_cards": [_card(1)],
        "deliverable_validation": [],
        "claim_citation_consistency": {
            "consistent_sentences": 34,
            "inconsistent_sentences": 9,
            "consistency_rate": 34 / 43,
            "unmapped_papers": [],
            "inconsistent_samples": [{"sentence": "示例句", "unauthorized_in_claim": ["p9"]}],
        },
    }

    _apply_final_quality_gate(state)

    codes = {issue["code"] for issue in state["quality_gate"]["blocking_issues"]}
    assert "claim_citation_consistency_not_met" in codes
    assert state["generation_blocked"] is True


def test_any_claim_citation_mismatch_blocks_delivery():
    state = {
        "intent": "generate_review",
        "review": "## 研究现状\n\n当前证据支持形成综合结论[p1]。",
        "paper_cards": [_card(1)],
        "deliverable_validation": [],
        "claim_citation_consistency": {
            "consistent_sentences": 40,
            "inconsistent_sentences": 4,
            "consistency_rate": 40 / 44,
            "unmapped_papers": [],
            "inconsistent_samples": [],
        },
    }

    _apply_final_quality_gate(state)

    codes = {issue["code"] for issue in state["quality_gate"]["blocking_issues"]}
    assert "claim_citation_consistency_not_met" in codes
    assert state["generation_blocked"] is True


def test_final_gate_enforces_language_coverage_on_actual_citations():
    state = {
        "intent": "generate_review",
        "review": "## 研究现状\n\n当前证据支持形成综合结论[p1]。",
        "paper_cards": [_card(1)],
        "reference_papers": [{
            **_card(1),
            "title": "课堂互动行为分析",
            "abstract": "课堂教学互动研究",
            "source": "cnki",
        }],
        "deliverable_validation": [],
        "language_coverage_target": {
            "enabled": True,
            "minimum_zh": 1,
            "minimum_en": 1,
            "required_total": 2,
        },
    }

    _apply_final_quality_gate(state)

    codes = {issue["code"] for issue in state["quality_gate"]["blocking_issues"]}
    assert "language_coverage_not_met" in codes
    assert state["language_coverage"]["cited_en"] == 0
    assert state["generation_blocked"] is True


def test_unknown_publication_status_is_explicit_warning():
    state = {
        "intent": "generate_review",
        "review": "## 研究现状\n\n当前证据支持形成综合结论[p1]。",
        "paper_cards": [_card(1)],
        "deliverable_validation": [],
        "citation_validation": {
            "valid": True,
            "metadata_quality_valid": True,
            "unknown_publication_status": ["p1"],
        },
    }

    _apply_final_quality_gate(state)

    warnings = {item["code"] for item in state["quality_gate"]["warnings"]}
    assert "publication_status_unknown" in warnings


def test_final_gate_rechecks_actual_references_against_confirmed_scope():
    state = {
        "intent": "generate_review",
        "review": "## 研究现状\n\n当前证据支持形成综合结论[p1]。",
        "paper_cards": [_card(1)],
        "reference_papers": [{
            **_card(1),
            "title": "Visual Student Action Recognition with YOLO",
            "abstract": "Computer vision detection of classroom gestures.",
        }],
        "selected_scope": {
            "include_terms": ["classroom interaction analysis"],
            "exclude_terms": ["computer vision", "action recognition"],
        },
        "deliverable_validation": [],
    }

    _apply_final_quality_gate(state)

    codes = {issue["code"] for issue in state["quality_gate"]["blocking_issues"]}
    assert "citation_scope_not_met" in codes


def test_reference_minimum_counts_only_claim_authorized_citations():
    state = {
        "intent": "generate_review",
        "review": "## 研究现状\n\n当前证据支持形成综合结论[p1]。",
        "max_papers_explicit": True,
        "required_reference_count": 40,
        "unique_cited_paper_count": 40,
        "unique_valid_cited_paper_count": 33,
        "citation_validation": {"valid": True},
        "claim_citation_consistency": {
            "consistent_sentences": 33,
            "inconsistent_sentences": 7,
            "consistency_rate": 0.825,
        },
        "paper_cards": [_card(1)],
        "deliverable_validation": [],
    }

    _apply_final_quality_gate(state)

    issue = next(
        item for item in state["quality_gate"]["blocking_issues"]
        if item["code"] == "minimum_cited_references_not_met"
    )
    assert issue["actual"] == 33


def test_strip_section_ordinal_handles_common_prefixes():
    from app.tools.validate_deliverable import _strip_section_ordinal

    assert _strip_section_ordinal("（二）元学习与优化策略") == "元学习与优化策略"
    assert _strip_section_ordinal("(3)度量学习") == "度量学习"
    assert _strip_section_ordinal("二、研究现状") == "研究现状"
    assert _strip_section_ordinal("研究背景") == "研究背景"


def test_final_integrity_ignores_renumbered_theme_sections():
    """主题章节序号被重排后，完整性检查不得误报缺少计划章节。"""
    state = {
        "required_reference_count": 60,
        "writing_plans": [{
            "sections": [
                {"id": "theme_a", "title": "（二）元学习与优化策略"},
                {"id": "theme_b", "title": "（三）度量学习与原型网络"},
            ],
        }],
    }
    text = (
        "## 二、研究现状\n\n"
        "### （三）元学习与优化策略\n\n"
        "已有充分证据表明该路线在多个基准上持续演进，多项工作报告了稳定的性能提升，"
        "相关结论可复核，且跨域设定下的适配效率同样得到了系统性验证[p1][p2]。\n\n"
        "### （四）度量学习与原型网络\n\n"
        "该路线同样积累了充分证据，原型构造与度量设计的改良在多个数据集上得到了"
        "一致性验证，判别性与泛化性的权衡也被多项工作反复确认[p3][p4]。"
    )
    report = validate_final_review_integrity(text, state)

    assert not any("缺少计划章节" in error for error in report["errors"])


def test_explicit_reference_minimum_has_no_eighty_percent_shortcut():
    state = {
        "intent": "generate_review",
        "review": "## 研究现状\n\n课堂行为分析已有可核验证据[p1]。",
        "max_papers_explicit": True,
        "required_reference_count": 40,
        "unique_cited_paper_count": 34,
        "citation_validation": {"valid": True},
        "generation_quality": {"passed": True, "support_rate": 1.0, "unsupported_claims": 0},
        "deliverable_validation": [],
        "writing_plans": [],
        "steps": [],
        "errors": [],
    }

    final_answer_node(state)

    issue = next(
        item for item in state["quality_gate"]["blocking_issues"]
        if item["code"] == "minimum_cited_references_not_met"
    )
    assert issue["requested"] == 40
    assert issue["actual"] == 34
    assert state["quality_gate"]["passed"] is False


def test_background_plan_keeps_full_requested_reference_target():
    cards = [_card(index) for index in range(1, 49)]
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "required_reference_count": 40,
    }

    plan = build_writing_plan("research_background", state)

    assert plan.citation_policy["minimum_unique_references"] == 40


def test_research_status_plan_generically_merges_overfragmented_themes():
    cards = [_card(index) for index in range(1, 13)]
    state = {
        "topic": "课堂行为分析",
        "selected_scope": {
            "scope_id": "technology_assisted_domain_analysis",
            "description": "先自动识别和行为编码，再进行教育学分析",
        },
        "paper_cards": cards,
        "theme_synthesis": [
            {"theme_id": "a", "theme_name": "目标检测与细粒度行为识别", "paper_ids": ["p1", "p2"]},
            {"theme_id": "b", "theme_name": "通用自动识别架构", "paper_ids": ["p3", "p4"]},
            {"theme_id": "c", "theme_name": "课堂行为自动编码与时序建模", "paper_ids": ["p5", "p6"]},
            {"theme_id": "d", "theme_name": "视觉、姿态与语音融合识别", "paper_ids": ["p7", "p8"]},
            {"theme_id": "e", "theme_name": "S-T分析与滞后序列分析", "paper_ids": ["p9", "p10"]},
            {"theme_id": "f", "theme_name": "教学反馈、干预与实践应用", "paper_ids": ["p11", "p12"]},
        ],
    }

    plan = build_writing_plan("research_status", state)
    titles = [section.title for section in plan.sections]

    route_titles = [title for title in titles if title.startswith("（")]
    # spec 子节预算为 6，但 12 篇证据按 _MIN_PAPERS_PER_SUBSECTION 自适应
    # 缩放到 4 节：证据不足时不把预算用满，避免两篇一节的碎片小节。
    assert 1 <= len(route_titles) <= 4
    assert all(
        title.startswith(f"（{numeral}）")
        for title, numeral in zip(route_titles, "一二三四五六七八九十")
    )
    assert all(section.heading_level == 3 for section in plan.sections if section.id.startswith("theme_"))
    assert {
        paper_id
        for section in plan.sections if section.id.startswith("theme_")
        for paper_id in section.supporting_paper_ids
    } == {f"p{index}" for index in range(1, 13)}


def test_two_part_request_has_exact_heading_hierarchy_and_no_diagnostics():
    cards = [_card(index) for index in range(1, 13)]
    taxonomy = {
        "organizing_principle": "自动识别到教育分析的任务链",
        "themes": [
            {"theme_id": "T1", "name": "目标检测与课堂行为识别"},
            {"theme_id": "T2", "name": "自动编码与时序建模"},
            {"theme_id": "T3", "name": "S-T与滞后序列分析"},
            {"theme_id": "T4", "name": "人工智能与教学评价融合"},
        ],
        "assignments": [
            {
                "paper_id": card["paper_id"],
                "primary_theme_id": f"T{min(4, (index // 3) + 1)}",
            }
            for index, card in enumerate(cards)
        ],
    }
    state = {
        "user_query": "先自动识别和编码课堂行为，再用S-T或滞后分析法进行教育分析",
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "requested_sections": ["background", "research_status"],
        "core_deliverables": ["research_background", "research_status"],
        "paper_details": cards,
        "paper_cards": cards,
        "ranked_papers": cards,
        "dynamic_taxonomy": taxonomy,
        "taxonomy_validation": {"valid": True, "status": "valid"},
        "theme_synthesis": synthesize_themes(cards, taxonomy),
        "selected_scope": {
            "scope_id": "technology_assisted_domain_analysis",
            "description": "自动识别与自动编码后进行教育学分析",
        },
        "required_reference_count": 8,
        "steps": [],
        "errors": [],
    }

    generate_deliverables_node(state, llm=None)

    h2_titles = re.findall(r"^##\s+(.+?)$", state["review"], re.M)
    h3_titles = re.findall(r"^###\s+(.+?)$", state["review"], re.M)
    # 测试卡片没有可靠地域元数据，标题不能暗示国内外比较。
    assert h2_titles == ["一、研究背景", "二、研究现状"]
    assert 3 <= len(h3_titles) <= 4
    assert all(title.startswith("（") for title in h3_titles)
    for forbidden in (
        "研究问题与场景", "研究价值与必要性", "进一步研究的必要性",
        "研究范围说明", "证据范围说明", "研究路线比较与共性问题", "研究空白",
    ):
        assert not re.search(rf"^#+\s+{re.escape(forbidden)}$", state["review"], re.M)

    state.update({
        "quality_gate": {
            "passed": False,
            "draft_available": True,
            "draft_released": True,
            "draft_disposition": "released_best_effort",
            "partial_success": True,
            "blocking_issues": [{"message": "机器内部质量提示"}],
        },
        "quarantined_draft": state["review"],
        "references": [f"Reference {index}" for index in range(1, 9)],
    })
    answer = _assemble_answer(state)
    assert "质量门禁提示" in answer
    assert "未完全达标草稿" in answer
    assert "证据范围说明" not in answer
    assert "## 参考文献" in answer


def test_global_reference_target_is_split_across_background_and_status():
    quotas = _global_deliverable_citation_quotas(
        ["research_background", "research_status"],
        40,
    )

    assert quotas == {
        "research_background": 10,
        "research_status": 30,
    }
    assert sum(quotas.values()) == 40


def test_citation_allocation_budget_extends_beyond_floor_within_generation_limit():
    # 可用证据在 [下限, generation_limit] 区间：预算用满可用证据
    assert _citation_allocation_budget(required=60, usable=76, generation_limit=120) == 76
    # 可用证据超过生成预算：截断到 generation_limit
    assert _citation_allocation_budget(required=60, usable=200, generation_limit=120) == 120
    # 可用证据不足下限：退化为可用量（降级路径与旧行为一致）
    assert _citation_allocation_budget(required=60, usable=40, generation_limit=120) == 40
    # 未提供 generation_limit 时预算不越过用户下限
    assert _citation_allocation_budget(required=60, usable=90, generation_limit=0) == 60


def test_allocation_selection_target_extends_pool_above_required_floor():
    cards = [_card(index) for index in range(1, 51)]
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "ranked_papers": cards,
        "required_reference_count": 40,
    }
    plan = build_writing_plan("research_status", state)

    allocation = _plan_citation_allocation(
        state=state,
        llm=None,
        ranked_papers=cards,
        required_count=30,
        writing_plan=plan,
        selection_target=40,
    )
    assigned = {
        paper_id
        for section in allocation["sections"]
        for paper_id in section["paper_ids"]
    }

    # 分配层按预算多选（40），不再把“不少于 30 篇”精确执行成 30 篇；
    # 验收下限仍由写作计划的 citation_policy 承载，不受分配预算影响。
    assert len(assigned) == 40


def test_allocation_without_selection_target_keeps_legacy_exact_floor():
    cards = [_card(index) for index in range(1, 41)]
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "ranked_papers": cards,
        "required_reference_count": 25,
    }
    plan = build_writing_plan("research_background", state)

    allocation = _plan_citation_allocation(
        state=state,
        llm=None,
        ranked_papers=cards,
        required_count=25,
        writing_plan=plan,
    )
    assigned = {
        paper_id
        for section in allocation["sections"]
        for paper_id in section["paper_ids"]
    }

    assert len(assigned) == 25


def test_allocation_budget_not_capped_by_route_paper_union():
    """回归 2026-08 会话：路线论文并集不得成为引用分配天花板。

    真实场景：84 篇合格证据、主题综合只挂 46 篇进路线，预算 76 被
    写作计划支撑集合钳到 46，"不少于 60 篇"必然失败。
    """
    cards = [_card(index) for index in range(1, 31)]  # 30 篇合格证据
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "ranked_papers": cards,
        "required_reference_count": 25,
        "theme_synthesis": [{
            # 主题综合层窄覆盖：路线只挂 6 篇
            "theme_id": "theme_A",
            "theme_name": "行为编码与观察路线",
            "paper_ids": [f"p{index}" for index in range(1, 7)],
            "reported_problems": [{"claim": "课堂行为编码成本高", "paper_id": "p1"}],
            "reported_methods": [{"claim": "自动编码流水线", "paper_id": "p2"}],
            "reported_findings": [{"claim": "编码一致性提升", "paper_id": "p3"}],
        }],
        "research_semantic_frame": {
            "canonical_topic": "课堂行为分析",
            "evidence_requirements": [{
                "requirement_id": "req_coding",
                "label": "行为编码相关证据",
                "route_required": True,
                "route_group": "coding",
            }],
        },
    }
    plan = build_writing_plan("research_status", state)

    allocation = _plan_citation_allocation(
        state=state,
        llm=None,
        ranked_papers=cards,
        required_count=20,
        writing_plan=plan,
        selection_target=28,  # 预算高于路线并集(6)，低于合格池(30)
    )
    assigned = {
        paper_id
        for section in allocation["sections"]
        for paper_id in section["paper_ids"]
    }

    # 旧缺陷：assigned 被钳到路线并集 6；修复后按预算兑现 28
    assert len(assigned) == 28


def test_second_deliverable_prefers_papers_not_used_by_first_deliverable():
    cards = [_card(index) for index in range(1, 51)]
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "ranked_papers": cards,
        "required_reference_count": 40,
    }
    plan = build_writing_plan("research_status", state)

    allocation = _plan_citation_allocation(
        state=state,
        llm=None,
        ranked_papers=cards,
        required_count=30,
        writing_plan=plan,
        excluded_ids={f"p{index}" for index in range(1, 11)},
    )
    assigned = {
        paper_id
        for section in allocation["sections"]
        for paper_id in section["paper_ids"]
    }

    assert len(assigned) == 30
    assert assigned.isdisjoint({f"p{index}" for index in range(1, 11)})


def test_sparse_citation_plan_is_completed_and_executed_by_fallback():
    cards = [_card(index) for index in range(1, 49)]
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "ranked_papers": cards,
        "required_reference_count": 40,
        "theme_synthesis": [],
        "search_report": {},
        "evidence_quality_report": {},
    }
    plan = build_writing_plan("research_background", state)

    class SparsePlanner:
        def complete(self, prompt: str, **kwargs) -> str:
            return (
                '{"sections":[{"section_id":"background_body",'
                '"paper_ids":["p1","p2"]}]}'
            )

    allocation = _plan_citation_allocation(
        state=state,
        llm=SparsePlanner(),
        ranked_papers=cards,
        required_count=40,
        writing_plan=plan,
    )
    state["citation_allocation_plan"] = allocation
    text = write_deliverable(plan, state, llm=None)
    validation = validate_deliverable(text, plan, state)

    assigned = {
        paper_id
        for section in allocation["sections"]
        for paper_id in section["paper_ids"]
    }
    assigned_occurrences = [
        paper_id
        for section in allocation["sections"]
        for paper_id in section["paper_ids"]
    ]
    assert len(assigned) == 40
    assert len(assigned_occurrences) == len(assigned)
    assert allocation["assigned_unique_references"] == 40
    assert validation["metrics"]["unique_cited_papers"] >= 40


def test_citation_selection_keeps_minority_research_route():
    cards = [_card(index) for index in range(1, 43)]
    taxonomy = {
        "themes": [
            {"theme_id": "T1", "name": "人工智能行为识别"},
            {"theme_id": "T2", "name": "教育观察与质性编码"},
        ],
        "assignments": [
            {
                "paper_id": f"p{index}",
                "primary_theme_id": "T1" if index <= 40 else "T2",
            }
            for index in range(1, 43)
        ],
    }
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "ranked_papers": cards,
        "required_reference_count": 40,
        "dynamic_taxonomy": taxonomy,
    }
    plan = build_writing_plan("research_background", state)

    allocation = _plan_citation_allocation(
        state=state,
        llm=None,
        ranked_papers=cards,
        required_count=40,
        writing_plan=plan,
    )
    assigned = {
        paper_id
        for section in allocation["sections"]
        for paper_id in section["paper_ids"]
    }

    assert {"p41", "p42"}.issubset(assigned)


def test_citation_selection_excludes_fallback_theme_when_main_routes_are_sufficient():
    cards = [_card(index) for index in range(1, 51)]
    taxonomy = {
        "themes": [
            {"theme_id": "T1", "name": "人工智能行为识别"},
            {"theme_id": "T2", "name": "教育观察与质性编码"},
            {"theme_id": "T3", "name": "其他相关研究"},
        ],
        "assignments": [
            {
                "paper_id": f"p{index}",
                "primary_theme_id": (
                    "T3" if index > 45 else "T1" if index % 2 else "T2"
                ),
            }
            for index in range(1, 51)
        ],
    }
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "ranked_papers": cards,
        "required_reference_count": 40,
        "dynamic_taxonomy": taxonomy,
    }
    plan = build_writing_plan("research_background", state)

    allocation = _plan_citation_allocation(
        state=state,
        llm=None,
        ranked_papers=cards,
        required_count=40,
        writing_plan=plan,
    )
    assigned = {
        paper_id
        for section in allocation["sections"]
        for paper_id in section["paper_ids"]
    }

    assert len(assigned) == 40
    assert not assigned.intersection({f"p{index}" for index in range(46, 51)})


def test_large_reference_writer_uses_sectionwise_chinese_synthesis():
    cards = []
    for index in range(1, 21):
        card = _card(index)
        english = (
            f"This study investigates classroom behavior analysis route {index} "
            "with evidence reported in the abstract."
        )
        card["field_claims"]["research_problem"][0].update({
            "claim": english,
            "source_text": english,
        })
        cards.append(card)
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "ranked_papers": cards,
        "required_reference_count": 20,
        "theme_synthesis": [],
        "search_report": {},
        "evidence_quality_report": {},
    }
    plan = build_writing_plan("research_background", state)
    state["citation_allocation_plan"] = _plan_citation_allocation(
        state=state,
        llm=None,
        ranked_papers=cards,
        required_count=20,
        writing_plan=plan,
    )

    class SectionWriter:
        operations: list[str] = []
        prompts: list[str] = []

        def complete(self, prompt: str, **kwargs) -> str:
            self.operations.append(str(kwargs.get("operation") or ""))
            self.prompts.append(prompt)
            title = re.search(r"章节标题：(.+)", prompt).group(1).strip()
            raw_ids = re.search(
                r"必须原样保留且每个至少出现一次的引用编号：\s*(\[[^\n]*\])",
                prompt,
            ).group(1)
            paper_ids = json.loads(raw_ids)
            citations = "".join(f"[{paper_id}]" for paper_id in paper_ids)
            return (
                f"## {title}\n\n围绕课堂行为分析，现有证据呈现相互关联的问题设定{citations}。\n\n"
                "相关工作也从方法路径解释了这一研究对象。\n\n"
                "综合来看，这些证据共同构成后续研究的背景基础。"
            )

    writer = SectionWriter()
    text = write_deliverable(plan, state, llm=writer)
    validation = validate_deliverable(text, plan, state)

    assert validation["valid"] is True
    assert validation["metrics"]["unique_cited_papers"] == 20
    assert all(operation.startswith("write_section:") for operation in writer.operations)
    assert all("脱敏写作少样本" in prompt for prompt in writer.prompts)
    assert all("真实证据草稿" in prompt for prompt in writer.prompts)
    assert state["writer_diagnostics"][-1]["strategy"] == "sectionwise_chinese_synthesis"


def test_small_fallback_theme_does_not_downgrade_research_status():
    cards = [_card(index) for index in range(1, 61)]
    cards[0]["evidence_state"] = {"access_level": "full_text"}
    themes = [
        {"theme_id": f"T{index}", "name": f"研究路线{index}"}
        for index in range(1, 7)
    ] + [{"theme_id": "T7", "name": "其他计算方法"}]
    assignments = [
        {
            "paper_id": f"p{index}",
            "primary_theme_id": (
                "T7" if index > 58 else f"T{((index - 1) % 6) + 1}"
            ),
        }
        for index in range(1, 61)
    ]
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "dynamic_taxonomy": {
            "themes": themes,
            "assignments": assignments,
        },
        "taxonomy_validation": {"valid": True, "requires_revision": False},
    }

    readiness = check_deliverable_readiness(
        "research_status", state, phase="post_evidence"
    )

    assert readiness.ready is True
    assert readiness.downgrade_suggestion is None


def test_large_fallback_theme_does_not_hide_four_valid_research_routes():
    """复现41篇中9篇长尾证据的真实场景：四条正式路线仍应正常生成。"""
    cards = [_card(index) for index in range(1, 42)]
    themes = [
        {"theme_id": f"T{index}", "name": f"研究路线{index}"}
        for index in range(1, 5)
    ] + [{"theme_id": "T5", "name": "其他相关研究"}]
    assignments = [
        {
            "paper_id": f"p{index}",
            "primary_theme_id": (
                "T5" if index > 32 else f"T{((index - 1) % 4) + 1}"
            ),
        }
        for index in range(1, 42)
    ]
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "dynamic_taxonomy": {
            "themes": themes,
            "assignments": assignments,
        },
        "taxonomy_validation": {"valid": True, "status": "valid"},
    }

    readiness = check_deliverable_readiness(
        "research_status", state, phase="post_evidence"
    )

    assert readiness.ready is True
    assert readiness.downgrade_suggestion is None


def test_41_paper_two_part_generation_keeps_status_and_splits_40_citations():
    cards = [_card(index) for index in range(1, 42)]
    themes = [
        {"theme_id": f"T{index}", "name": f"研究路线{index}"}
        for index in range(1, 5)
    ] + [{"theme_id": "T5", "name": "其他相关研究"}]
    assignments = [
        {
            "paper_id": f"p{index}",
            "primary_theme_id": (
                "T5" if index > 32 else f"T{((index - 1) % 4) + 1}"
            ),
        }
        for index in range(1, 42)
    ]
    taxonomy = {"themes": themes, "assignments": assignments}
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "core_deliverables": ["research_background", "research_status"],
        "paper_details": cards,
        "paper_cards": cards,
        "ranked_papers": cards,
        "dynamic_taxonomy": taxonomy,
        "theme_synthesis": synthesize_themes(cards, taxonomy),
        "taxonomy_validation": {"valid": True, "status": "valid"},
        "required_reference_count": 40,
        "max_papers_explicit": True,
        "steps": [],
        "errors": [],
    }

    generate_deliverables_node(state, llm=None)

    assert [
        item["deliverable_type"] for item in state["writing_plans"]
    ] == ["research_background", "research_status"]
    assert state["deliverable_downgrades"] == []
    quotas = {
        item["deliverable_type"]: item["minimum_unique_references"]
        for item in state["citation_allocation_plans"]
    }
    assert quotas == {"research_background": 10, "research_status": 30}
    assigned = {
        paper_id
        for plan in state["citation_allocation_plans"]
        for section in plan["sections"]
        for paper_id in section["paper_ids"]
    }
    assert len(assigned) == 40


def test_valid_with_warning_taxonomy_does_not_downgrade_research_status():
    """碎片主题警告可由写作计划忽略，不能吞掉整个研究现状。"""
    cards = [_card(index) for index in range(1, 9)]
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "dynamic_taxonomy": {
            "themes": [
                {"theme_id": "T1", "name": "自动识别"},
                {"theme_id": "T2", "name": "教育解释"},
                {"theme_id": "T3", "name": "碎片主题"},
            ],
            "assignments": [
                {
                    "paper_id": f"p{index}",
                    "primary_theme_id": (
                        "T3" if index == 8 else "T1" if index <= 4 else "T2"
                    ),
                }
                for index in range(1, 9)
            ],
        },
        "taxonomy_validation": {
            "valid": True,
            "requires_revision": True,
            "status": "valid_with_warning",
            "warnings": ["存在仅含一篇论文的碎片主题"],
        },
    }

    readiness = check_deliverable_readiness(
        "research_status", state, phase="post_evidence"
    )

    assert readiness.ready is True
    assert readiness.downgrade_suggestion is None


def test_reference_quota_is_not_collapsed_into_background_after_downgrade():
    """研究现状若未生成，不能把整份40篇配额强压进研究背景。"""
    cards = [_card(index) for index in range(1, 41)]
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "core_deliverables": ["research_background", "research_status"],
        "paper_details": cards,
        "paper_cards": cards,
        "ranked_papers": cards,
        "dynamic_taxonomy": {"themes": [], "assignments": []},
        "taxonomy_validation": {
            "valid": False,
            "requires_revision": True,
            "status": "invalid",
        },
        "required_reference_count": 40,
        "max_papers_explicit": True,
        "steps": [],
        "errors": [],
    }

    generate_deliverables_node(state, llm=None)

    assert len(state["writing_plans"]) == 1
    assert state["writing_plans"][0]["deliverable_type"] == "research_background"
    assert state["writing_plans"][0]["citation_policy"]["minimum_unique_references"] == 10
    assert state["citation_allocation_plans"][0]["minimum_unique_references"] == 10
    assert "研究现状未单独生成" not in state["review"]
    assert state["deliverable_downgrades"][0]["requested_type"] == "research_status"


def test_english_residue_is_locally_repaired_without_discarding_section():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="测试",
        organizing_strategy="evidence_driven",
        sections=[WritingSection(
            id="theme_T1",
            title="自动识别",
            purpose="综合研究路线",
            supporting_paper_ids=["p1", "p2"],
        )],
        citation_policy={"minimum_unique_references": 2},
    )
    state = {"topic": "课堂行为分析"}

    class LocalRepairLLM:
        operations: list[str] = []

        def complete(self, prompt: str, **kwargs) -> str:
            operation = str(kwargs.get("operation") or "")
            self.operations.append(operation)
            if operation.startswith("repair_english_fragments:"):
                return (
                    "## 自动识别\n\n现有研究围绕课堂行为形成自动识别路线[p1]。"
                    "相关方法进一步支持行为编码[p2]。"
                )
            return (
                "## 自动识别\n\nThis study investigates classroom behavior "
                "recognition with a multimodal neural network[p1][p2]."
            )

    llm = LocalRepairLLM()
    result = _write_sections_in_chinese(
        "## 自动识别\n\n英文证据[p1][p2]。",
        plan,
        state,
        [],
        llm,
    )

    assert "This study" not in result
    assert "[p1]" in result and "[p2]" in result
    assert any(item.startswith("repair_english_fragments:") for item in llm.operations)
    diagnostic = state["writer_section_diagnostics"][-1]["sections"][0]
    assert diagnostic["status"] == "success_after_english_repair"


def test_unrepaired_english_residue_keeps_citation_complete_best_draft():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="测试",
        organizing_strategy="evidence_driven",
        sections=[WritingSection(
            id="theme_T1",
            title="自动识别",
            purpose="综合研究路线",
            supporting_paper_ids=["p1", "p2"],
        )],
        citation_policy={"minimum_unique_references": 2},
    )
    state = {"topic": "课堂行为分析"}

    class StubbornLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return (
                "## 自动识别\n\nThis study investigates classroom behavior "
                "recognition with a multimodal neural network[p1][p2]."
            )

    result = _write_sections_in_chinese(
        "## 自动识别\n\n英文证据[p1][p2]。",
        plan,
        state,
        [],
        StubbornLLM(),
    )

    assert "仅保留证据边界" not in result
    assert "[p1]" in result and "[p2]" in result
    diagnostic = state["writer_section_diagnostics"][-1]["sections"][0]
    assert diagnostic["status"] == "partial_english_residue"


def test_english_fallback_can_be_polished_without_losing_citations():
    cards = []
    for index in range(1, 3):
        paper_id = f"p{index}"
        claim = (
            f"This study examines classroom behavior analysis route {index} "
            "using explicitly reported evidence."
        )
        cards.append({
            "paper_id": paper_id,
            "title": f"Paper {index}",
            "year": 2024,
            "quality_status": "partial",
            "evidence_source": "abstract",
            "evidence_state": {"access_level": "abstract"},
            "field_claims": {
                "research_problem": [{
                    "claim": claim,
                    "source_text": claim,
                    "source_section": "abstract",
                    "evidence_id": f"{paper_id}:e1",
                    "evidence_level": "abstract",
                    "explicitly_reported": True,
                }]
            },
            "unsupported_fields": ["limitations"],
        })
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "ranked_papers": cards,
        "theme_synthesis": [],
        "search_report": {},
        "evidence_quality_report": {},
        "citation_allocation_plan": {
            "minimum_unique_references": 2,
            "sections": [
                {
                    "section_id": "background_body",
                    "section": "一、研究背景",
                    "paper_ids": ["p1", "p2"],
                },
            ],
        },
    }
    plan = build_writing_plan("research_background", state)

    class PolishLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            if str(kwargs.get("operation") or "").startswith("polish_fallback"):
                return (
                    "## 研究背景\n\n"
                    "课堂行为分析需要明确行为编码对象与教学场景[p1]。\n\n"
                    "行为证据可为理解学生课堂参与提供基础[p2]。\n\n"
                    "相关研究由此形成问题界定与行为测量两条相互联系的线索[p1][p2]。"
                )
            return "\n\n".join(
                f"## {section.title}\n\nThis is an invalid English draft."
                for section in plan.sections
            )

    text = write_deliverable(plan, state, llm=PolishLLM())
    validation = validate_deliverable(text, plan, state)

    assert validation["valid"] is True
    assert validation["metrics"]["unique_cited_papers"] == 2
    assert "invalid English draft" not in text


def test_research_status_rejects_third_deliverable_title():
    cards = [_card(1), _card(2)]
    taxonomy = {
        "themes": [
            {"theme_id": "T1", "name": "行为观察"},
            {"theme_id": "T2", "name": "自动识别"},
        ],
        "assignments": [
            {"paper_id": "p1", "primary_theme_id": "T1"},
            {"paper_id": "p2", "primary_theme_id": "T2"},
        ],
    }
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "dynamic_taxonomy": taxonomy,
        "theme_synthesis": [
            {"theme_id": "T1", "theme_name": "行为观察", "paper_ids": ["p1"]},
            {"theme_id": "T2", "theme_name": "自动识别", "paper_ids": ["p2"]},
        ],
    }
    plan = build_writing_plan("research_status", state)
    valid_body = "\n\n".join(
        f"## {section.title}\n\n课堂行为分析的当前证据见[p1]。"
        for section in plan.sections
    )
    text = "# 课堂行为分析研究现状：叙述性综述初稿\n\n" + valid_body

    validation = validate_deliverable(text, plan, state)

    assert validation["valid"] is False
    assert any("计划外总标题" in error for error in validation["errors"])
    assert any("错误包装" in error for error in validation["errors"])


def test_status_structure_validation_reads_subsection_budget_from_spec():
    """结构校验的子节上限必须取自 spec，不能写死。

    回归：spec 的 max_subsections 放宽到 6 后，硬编码的 <=4 会让每份
    5-6 节的研究现状都被判"结构非法"（2026-08-22 会话实测）。
    """
    from app.deliverables.registry import get_deliverable_spec
    from app.schemas.deliverable_schema import (
        CoreDeliverableType, WritingPlan, WritingSection,
    )
    from app.tools.validate_deliverable import validate_deliverable

    spec = get_deliverable_spec(CoreDeliverableType.RESEARCH_STATUS)
    budget = int(spec.structure.max_subsections)
    assert budget >= 5, "本用例针对放宽后的子节预算"

    numerals = "一二三四五六七八九十"

    def _plan_and_text(subsection_count: int):
        sections = [WritingSection(
            id="status_overview", title="二、研究现状",
            purpose="总体", heading_level=2, supporting_paper_ids=["p1"],
        )]
        body = ["## 二、研究现状", "总体进展综合如下[p1]。"]
        for index in range(subsection_count):
            title = f"（{numerals[index]}）路线{index + 1}"
            sections.append(WritingSection(
                id=f"theme_r{index}", title=title, purpose="综合",
                heading_level=3, supporting_paper_ids=["p1"],
            ))
            body.append(f"### {title}")
            tail = (
                "综合各路线的共同进展与差异，证据支持的不足集中于此[p1]。"
                if index == subsection_count - 1
                else "该路线的代表机制见相关工作[p1]。"
            )
            body.append(tail)
        plan = WritingPlan(
            deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
            purpose="综合研究现状",
            organizing_strategy="evidence_driven",
            sections=sections,
        )
        return plan, "\n\n".join(body)

    structure_error = "个动态三级研究路线"
    # 达到 spec 上限时结构不再被判非法
    plan, text = _plan_and_text(budget)
    errors = validate_deliverable(text, plan, {"paper_cards": []}).get("errors") or []
    assert not [item for item in errors if structure_error in item]

    # 实际写作计划的路线数量优先于规格上限，超过默认预算也应按计划校验
    plan, text = _plan_and_text(budget + 1)
    errors = validate_deliverable(text, plan, {"paper_cards": []}).get("errors") or []
    assert not [item for item in errors if structure_error in item]


def test_doi_citation_is_not_split_into_a_duplicate_sentence():
    """DOI 里的小数点不得被当成句末标点。

    回归 2026-08-29 实测缺陷：``[doi:10.1142/s0219843625400080]`` 在 "10." 处
    被切开，后半截 "1142/s0219843625400080]。" 成为独立"句子"；同一篇论文
    被引两次就产生两个相同残句，交付物结构检查恒报"正文存在重复或高度
    相似句子"。
    """
    from app.core.text_quality import content_sentences
    from app.tools.validate_deliverable import _find_duplicate_sentences

    text = (
        "另有研究提出基于深度学习的自动化学习行为分析框架，"
        "以增强课堂教学评价的客观性[doi:10.1142/s0219843625400080]。"
        "还有研究从深度学习框架出发，将自动化学习行为分析作为"
        "增强课堂教与学评价的手段[doi:10.1142/s0219843625400080]。"
    )

    sentences = content_sentences(text)

    assert len(sentences) == 2
    assert all(sentence.endswith("]。") for sentence in sentences)
    assert _find_duplicate_sentences(text) == []


def test_decimal_numbers_do_not_end_a_sentence():
    """句末标点后紧跟数字属于小数或版本号，不是句子边界。"""
    from app.core.text_quality import content_sentences

    sentences = content_sentences("准确率提升 1.77%。模型基于 YOLOv8.2 实现。")

    assert sentences == ["准确率提升 1.77%。", "模型基于 YOLOv8.2 实现。"]


def test_genuine_duplicate_sentences_are_still_detected():
    """去掉误报后，真正的重复长句仍必须被检出。"""
    from app.tools.validate_deliverable import _find_duplicate_sentences

    text = (
        "课堂行为分析为教学改进提供了客观的数据依据[p1]。"
        "课堂行为分析为教学改进提供了客观的数据依据[p2]。"
    )

    assert len(_find_duplicate_sentences(text)) == 1


# ============================================================
# 章节级证据密度（研究路线小节不得只用一篇论文）
# ============================================================
def _status_plan_with_section_floor() -> WritingPlan:
    return WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="梳理研究现状",
        organizing_strategy="按研究路线组织",
        sections=[
            WritingSection(
                id="status_overview", title="研究现状", purpose="总体进展",
                supporting_paper_ids=["p1", "p2", "p3", "p4"],
            ),
            WritingSection(
                id="theme_T1", title="（一）行为识别方法", purpose="路线一",
                supporting_paper_ids=["p1", "p2"], heading_level=3,
                minimum_unique_references=2,
            ),
            WritingSection(
                id="theme_T2", title="（二）互动编码分析", purpose="路线二",
                supporting_paper_ids=["p3", "p4"], heading_level=3,
                minimum_unique_references=2,
            ),
        ],
        citation_policy={"minimum_unique_references": 4},
    )


def _status_body(theme_two_citations: list[str]) -> str:
    """两节结构相同、只有第二节引用数不同的正文，用于隔离章节级判据。"""
    second = "".join(f"[{paper_id}]" for paper_id in theme_two_citations)
    return (
        "## 研究现状\n\n"
        "课堂行为分析在识别与编码两条路线上均有进展[p1][p3]。\n\n"
        "### （一）行为识别方法\n\n"
        "该路线以视觉模型识别课堂行为为共同问题设定，方法上以目标检测与"
        "姿态估计为主，并在真实课堂视频上完成了识别性能评估；从适用条件看，"
        "其效果依赖课堂视频采集质量与标注粒度，在遮挡与密集场景下仍受限"
        "[p1][p2]。\n\n"
        "### （二）互动编码分析\n\n"
        "该路线关注师生互动的结构化编码，方法上以课堂观察量表与序列分析"
        "为主，并把编码结果用于教学过程诊断；综合来看，两条路线在数据条件"
        "与分析粒度上存在差异，其结论的适用范围仍需在更多课堂情境中检验"
        f"{second}。\n"
    )


def test_route_section_with_single_citation_fails_both_validators():
    """授权充足但正文只引用一篇时，章节级判据必须在两处校验同时失败。

    回归 2026-08-30 实测缺陷：路线统计有十余篇证据，最终第五节仍写成
    "本节纳入 1 篇文献"。全局唯一引用达标不能替代章节级证据密度。
    """
    plan = _status_plan_with_section_floor()
    cards = [_card(index) for index in range(1, 5)]
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "required_reference_count": 40,
        "citation_allocation_plan": {"sections": [
            {"section_id": "theme_T1", "paper_ids": ["p1", "p2"]},
            {"section_id": "theme_T2", "paper_ids": ["p3", "p4"]},
        ]},
        "writing_plans": [plan.model_dump(mode="json")],
    }
    sparse_text = _status_body(["p3"])

    validation = validate_deliverable(sparse_text, plan, state)
    integrity = validate_final_review_integrity(sparse_text, state)

    assert validation["valid"] is False
    assert any(
        "研究路线章节证据过少或内容过短" in error and "（二）互动编码分析" in error
        for error in validation["errors"]
    ), validation["errors"]
    assert integrity["valid"] is False
    assert any(
        "研究路线章节证据或内容密度不足" in error for error in integrity["errors"]
    ), integrity["errors"]
    floors = {
        item["section_id"]: item
        for item in validation["metrics"]["section_evidence_floors"]
    }
    assert floors["theme_T2"]["status"] == "sparse_citations"
    assert floors["theme_T2"]["actual_unique_references"] == 1
    assert floors["theme_T1"]["status"] == "ok"


def test_route_section_floor_passes_with_two_authorized_citations():
    """同一结构下引用两篇授权论文即通过，避免章节级判据误报。"""
    plan = _status_plan_with_section_floor()
    cards = [_card(index) for index in range(1, 5)]
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "required_reference_count": 40,
        "citation_allocation_plan": {"sections": [
            {"section_id": "theme_T1", "paper_ids": ["p1", "p2"]},
            {"section_id": "theme_T2", "paper_ids": ["p3", "p4"]},
        ]},
        "writing_plans": [plan.model_dump(mode="json")],
    }
    text = _status_body(["p3", "p4"])

    validation = validate_deliverable(text, plan, state)
    integrity = validate_final_review_integrity(text, state)

    assert not [
        error for error in validation["errors"]
        if "研究路线章节证据过少或内容过短" in error
    ]
    assert integrity["valid"] is True
    assert validation["metrics"]["section_floor_failure_count"] == 0


def test_unauthorized_citations_do_not_satisfy_the_section_floor():
    """未授权给本节的论文不能算作该节达标，否则章节配额可被越权引用绕过。"""
    plan = _status_plan_with_section_floor()
    cards = [_card(index) for index in range(1, 5)]
    state = {
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "paper_cards": cards,
        "required_reference_count": 40,
        "citation_allocation_plan": {"sections": [
            {"section_id": "theme_T1", "paper_ids": ["p1", "p2"]},
            {"section_id": "theme_T2", "paper_ids": ["p3", "p4"]},
        ]},
        "writing_plans": [plan.model_dump(mode="json")],
    }
    # 第二节凑数引用了属于第一节的 p1：唯一引用数是 2，但授权范围内只有 1 篇
    text = _status_body(["p3", "p1"])

    validation = validate_deliverable(text, plan, state)

    floors = {
        item["section_id"]: item
        for item in validation["metrics"]["section_evidence_floors"]
    }
    assert floors["theme_T2"]["status"] == "sparse_citations"
    assert floors["theme_T2"]["actual_unique_references"] == 1


def test_rendered_numeric_citations_do_not_false_fail_the_section_floor():
    """成文渲染为顺序编码后，章节级判据不得把每一节都误判为零篇引用。

    回归 2026-08-30 实测缺陷：引用校验把正文改写成 [1][2] 形式，而计划授权
    始终是 paper_id，两者直接取交集使六个路线小节全部误报证据不足。
    """
    sections = [
        {
            "id": "theme_T1", "title": "（一）行为识别方法",
            "minimum_unique_references": 2,
            "supporting_paper_ids": ["p1", "p2"],
        },
        {
            "id": "theme_T2", "title": "（二）互动编码分析",
            "minimum_unique_references": 2,
            "supporting_paper_ids": ["p3", "p4"],
        },
    ]
    state = {
        "writing_plans": [{"sections": sections}],
        "required_reference_count": 40,
        "paper_cards": [_card(index) for index in range(1, 5)],
    }
    rendered = (
        "## 研究现状\n\n"
        "两条路线均有进展[1][3]。\n\n"
        "### （一）行为识别方法\n\n"
        "该路线以视觉模型识别课堂行为为共同问题设定，方法上以目标检测与"
        "姿态估计为主，并在真实课堂视频上完成了识别性能评估；从适用条件看，"
        "其效果依赖课堂视频采集质量与标注粒度[1][2]。\n\n"
        "### （二）互动编码分析\n\n"
        "该路线关注师生互动的结构化编码，方法上以课堂观察量表与序列分析"
        "为主；综合来看，两条路线在数据条件与分析粒度上存在差异，其结论的"
        "适用范围仍需在更多课堂情境中检验[3]。\n"
    )

    # 没有 citation_map 时按渲染空间的唯一引用计数：够两篇的节不误报
    without_map = validate_final_review_integrity(rendered, state)
    floors = {
        item["section_id"]: item
        for item in without_map["metrics"]["section_evidence_floors"]
    }
    assert floors["theme_T1"]["status"] == "ok"
    assert floors["theme_T1"]["identifier_space"] == "rendered"
    # 真正只引用一篇的小节仍然失败
    assert floors["theme_T2"]["status"] == "sparse_citations"

    # 有 citation_map 时回到 paper_id 空间，越权引用不算本节达标
    with_map = validate_final_review_integrity(
        rendered, {**state, "citation_map": {"p1": 1, "p2": 2, "p3": 3, "p4": 4}},
    )
    mapped = {
        item["section_id"]: item
        for item in with_map["metrics"]["section_evidence_floors"]
    }
    assert mapped["theme_T1"]["identifier_space"] == "paper_id"
    assert mapped["theme_T1"]["actual_unique_references"] == 2
    assert mapped["theme_T2"]["status"] == "sparse_citations"


def test_section_floor_tops_up_selection_from_the_same_route():
    """轮转分桶与计划小节不同构时，仍须为每条正式路线选够本节授权证据。

    回归 2026-08-30 实测缺陷：证据要求路线的 section_id 不在
    dynamic_taxonomy 里，轮转选择把名额全给了首个主题桶，某条路线只拿到
    一篇；独占归属又让它无法从别的小节借调，正文只能写成单篇罗列。
    """
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="梳理研究现状",
        organizing_strategy="按研究路线组织",
        sections=[
            WritingSection(
                id="theme_stage_coding", title="（一）行为编码方法", purpose="路线一",
                supporting_paper_ids=[f"p{index}" for index in range(1, 25)],
                heading_level=3, minimum_unique_references=2,
            ),
            WritingSection(
                id="theme_stage_management", title="（二）课堂管理行为", purpose="路线二",
                supporting_paper_ids=[f"p{index}" for index in range(25, 30)],
                heading_level=3, minimum_unique_references=2,
            ),
        ],
    )
    cards = [_card(index) for index in range(1, 30)]
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "ranked_papers": cards,
        "required_reference_count": 24,
        # 主题桶只有一个，且与计划小节 ID 不同构：轮转会先吃掉 p1..p24
        "dynamic_taxonomy": {
            "themes": [{"theme_id": "VR1", "name": "课堂行为编码"}],
            "assignments": [
                {"paper_id": f"p{index}", "primary_theme_id": "VR1"}
                for index in range(1, 30)
            ],
        },
    }

    allocation = _plan_citation_allocation(
        state=state,
        llm=None,
        ranked_papers=cards,
        required_count=24,
        writing_plan=plan,
    )

    by_section = {
        item["section_id"]: item["paper_ids"] for item in allocation["sections"]
    }
    management = by_section["theme_stage_management"]
    assert len(management) >= 2
    # 补选只用本节授权论文，不复用其他小节的证据
    assert set(management) <= {f"p{index}" for index in range(25, 30)}
    assert allocation["section_floor_deficits"] == []
    # 全局唯一引用不因章节补选而减少
    assigned = {
        paper_id
        for item in allocation["sections"]
        for paper_id in item["paper_ids"]
    }
    assert len(assigned) >= 24


def test_citation_allocation_reports_section_floor_deficit():
    """授权池不足两篇时，分配层必须留下确定性缺口诊断而不是静默通过。"""
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="梳理研究现状",
        organizing_strategy="按研究路线组织",
        sections=[
            WritingSection(
                id="theme_T1", title="（一）行为识别方法", purpose="路线一",
                supporting_paper_ids=[f"p{index}" for index in range(1, 25)],
                heading_level=3, minimum_unique_references=2,
            ),
            WritingSection(
                id="theme_T2", title="（二）综述与总体格局", purpose="路线二",
                supporting_paper_ids=["p25"], heading_level=3,
                minimum_unique_references=2,
            ),
        ],
    )
    cards = [_card(index) for index in range(1, 26)]
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "ranked_papers": cards,
        "required_reference_count": 25,
    }

    allocation = _plan_citation_allocation(
        state=state,
        llm=None,
        ranked_papers=cards,
        required_count=25,
        writing_plan=plan,
    )

    deficits = {
        item["section_id"]: item
        for item in allocation["section_floor_deficits"]
    }
    assert "theme_T2" in deficits
    assert deficits["theme_T2"]["required_unique_references"] == 2
    assert deficits["theme_T2"]["assigned_unique_references"] == 1
    assert deficits["theme_T2"]["authorized_paper_count"] == 1
    # 缺口不得靠越权引用补齐：分配仍只使用本节授权论文
    by_section = {
        item["section_id"]: item["paper_ids"] for item in allocation["sections"]
    }
    assert by_section["theme_T2"] == ["p25"]
