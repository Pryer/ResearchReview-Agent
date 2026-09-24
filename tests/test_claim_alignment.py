"""Claim Plan 输出契约与工作流防御性检查。"""

from __future__ import annotations

from app.agent.claim_plan import (
    _apply_access_limit,
    _paper_id_from_evidence,
    enforce_claim_evidence_gate,
    build_reference_coverage_plan,
    validate_claim_citation_consistency,
    validate_claim_support,
)
from app.agent.graph import _claim_alignment_check


def _claim_plans() -> list[dict]:
    return [{
        "route_id": "route_1",
        "route_name": "方法路线",
        "claims": [{
            "claim_id": "claim_1",
            "claim_text": "该方法在少样本视频动作识别任务中提升了分类性能",
            "claim_type": "finding",
            "support_level": "single",
            "allowed_language": "一项研究报告",
        }],
    }]


def test_single_source_claim_budget_drops_excess_claims() -> None:
    cards = [{
        "paper_id": "p1",
        "evidence_spans": [{"evidence_id": "p1:e1", "text": "研究提出一种方法。"}],
    }]
    claims = [{
        "claim_id": f"c{i}", "claim_text": f"主张{i}", "claim_type": "finding",
        "evidence_ids": ["p1:e1"], "support_level": "single",
    } for i in range(1, 4)]
    plans, report = enforce_claim_evidence_gate(
        [{"route_id": "r1", "claims": claims}], cards,
    )
    assert len(plans[0]["claims"]) == 2
    assert report["single_source_claims_dropped"] == 1
    assert report["single_source_claim_limit"] == 2


def test_multiple_spans_from_one_paper_do_not_meet_comparison_minimum() -> None:
    cards = [{"paper_id": "p1", "evidence_spans": [
        {"evidence_id": "p1:e1", "text": "方法甲的实验结果。"},
        {"evidence_id": "p1:e2", "text": "方法乙的实验结果。"},
    ]}]
    plans, report = enforce_claim_evidence_gate(
        [{"route_id": "r1", "claims": [{
            "claim_id": "compare", "claim_text": "两条方法路线的整体效果不同",
            "claim_type": "comparison", "evidence_ids": ["p1:e1", "p1:e2"],
            "support_level": "moderate",
        }]}], cards,
    )

    assert plans[0]["claims"] == []
    assert report["dropped_claims"] == 1
    assert "有效论文 1 篇" in report["gaps"][0]["reason"]


def test_pre_generation_entailment_rejects_claim_not_entailed() -> None:
    cards = [{
        "paper_id": "p1",
        "evidence_spans": [{"evidence_id": "p1:e1", "text": "证据仅说明模型被提出。"}],
    }]
    class LLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return '{"results":[{"claim_id":"c1","label":"insufficient","confidence":0.99,"reason":"证据不蕴含该结论"}]}'
    plans, report = enforce_claim_evidence_gate(
        [{"route_id": "r1", "claims": [{
            "claim_id": "c1", "claim_text": "模型显著提升准确率", "claim_type": "finding",
            "evidence_ids": ["p1:e1"], "support_level": "single",
        }]}], cards, llm=LLM(),
    )
    assert plans[0]["claims"] == []
    assert report["entailment_checked_claims"] == 1
    assert report["entailment_failed_claims"] == 1


def test_pre_generation_entailment_drops_claim_without_readable_evidence() -> None:
    cards = [{"paper_id": "p1", "field_claims": {
        "results": [{"evidence_id": "p1:result", "text": ""}],
    }}]

    class LLM:
        def complete(self, prompt: str, **kwargs) -> str:
            raise AssertionError("没有证据片段时不应请求模型判定")

    plans, report = enforce_claim_evidence_gate(
        [{"route_id": "r1", "claims": [{
            "claim_id": "c1", "claim_text": "模型显著提升准确率",
            "claim_type": "finding", "evidence_ids": ["p1:result"],
            "support_level": "single",
        }]}], cards, llm=LLM(),
    )

    assert plans[0]["claims"] == []
    assert report["passed"] is False
    assert report["entailment_checked_claims"] == 0
    assert report["entailment_failed_claims"] == 1


def test_claim_alignment_result_always_contains_overclaimed_samples() -> None:
    result = validate_claim_support(
        "该方法在少样本视频动作识别任务中提升了分类性能，"
        "这说明它已经具备跨数据集稳定泛化和实际部署能力[1]。",
        _claim_plans(),
    )

    assert "overclaimed_samples" in result
    assert result["partial_authorized_sentences"] == 1
    assert len(result["overclaimed_samples"]) == 1


def test_claim_alignment_accepts_legacy_result_without_optional_field(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.agent.claim_plan.validate_claim_support",
        lambda review, plans: {
            "unsupported_sentences": 0,
            "total_factual_sentences": 1,
            "support_rate": 1.0,
        },
    )
    state = {"review": "有证据支持的正文。", "claim_plans": _claim_plans()}

    _claim_alignment_check(state)

    assert state["claim_alignment"]["support_rate"] == 1.0


def test_claim_alignment_skips_generation_block_message(monkeypatch) -> None:
    def forbidden_validator(review, plans):
        raise AssertionError("生成被阻断时不应校验阻断说明")

    monkeypatch.setattr(
        "app.agent.claim_plan.validate_claim_support",
        forbidden_validator,
    )
    state = {
        "generation_blocked": True,
        "review": "## 正文生成已阻止\n\n可用论文不足。",
        "claim_plans": _claim_plans(),
    }

    _claim_alignment_check(state)

    assert "claim_alignment" not in state


def test_paper_id_from_evidence_handles_colon_prefixed_ids() -> None:
    """evidence_id 为 {paper_id}:eNNN，paper_id 自身含冒号，
    必须从最后一个冒号切分，否则卡片查询全部落空。"""
    assert _paper_id_from_evidence("s2:abc123:e001") == "s2:abc123"
    assert (
        _paper_id_from_evidence("doi:10.1109/icpr56361.2022.9956701:e004")
        == "doi:10.1109/icpr56361.2022.9956701"
    )
    assert _paper_id_from_evidence("cnki:-kV3hmJm4:e002") == "cnki:-kV3hmJm4"
    assert _paper_id_from_evidence("nocolon") == ""


def test_access_limit_only_flags_abstract_only_evidence() -> None:
    """摘要级限制只能命中真实卡片：全文论文不得被误伤。"""
    card_map = {
        "s2:aaa": {"evidence_state": {"access_level": "abstract"}},
        "s2:bbb": {"evidence_state": {"access_level": "full_text"}},
    }
    entry = {"claim_id": "c1", "allowed_language": "可陈述"}

    limited = _apply_access_limit(dict(entry), ["s2:aaa:e001"], card_map)
    assert limited["evidence_access_limit"] == "abstract_only"

    assert "evidence_access_limit" not in _apply_access_limit(
        dict(entry), ["s2:bbb:e001"], card_map
    )
    assert "evidence_access_limit" not in _apply_access_limit(
        dict(entry), ["s2:aaa:e001", "s2:bbb:e001"], card_map
    )


def test_claim_citation_consistency_uses_real_paper_ids() -> None:
    """授权论文集合必须用真实 paper_id，否则所有引用都被误判为不一致。"""
    plans = [{
        "route_id": "R1",
        "route_name": "路线一",
        "claims": [{
            "claim_id": "R1:c1",
            "claim_text": "少样本动作识别旨在用极少量标注样本识别新动作类别",
            "claim_type": "problem",
            "evidence_ids": ["s2:aaa:e001", "openalex:W1:e002"],
            "evidence_count": 2,
            "support_level": "moderate",
            "allowed_language": "可陈述",
        }],
    }]

    result = validate_claim_citation_consistency(
        "少样本动作识别旨在用极少量标注样本识别新动作类别[1][2]。",
        plans,
        citation_map={"s2:aaa": 1, "openalex:W1": 2},
    )
    assert result["inconsistent_sentences"] == 0
    assert result["unmapped_papers"] == []

    unauthorized = validate_claim_citation_consistency(
        "少样本动作识别旨在用极少量标注样本识别新动作类别[3]。",
        plans,
        citation_map={"s2:zzz": 3},
    )
    assert unauthorized["unmapped_papers"] == ["s2:zzz"]


def test_route_sibling_claim_evidence_does_not_authorize_a_citation() -> None:
    """同路线其它 claim 的证据不得为当前句子的引用背书。"""
    plans = [{
        "route_id": "R1",
        "route_name": "路线一",
        "claims": [
            {
                "claim_id": "R1:c1",
                "claim_text": "少样本动作识别旨在用极少量标注样本识别新动作类别",
                "claim_type": "problem",
                "evidence_ids": ["s2:aaa:e001"],
                "evidence_count": 1,
                "support_level": "weak",
                "allowed_language": "可陈述",
            },
            {
                "claim_id": "R1:c2",
                "claim_text": "多模态融合可提升课堂参与度估计的稳定性",
                "claim_type": "method",
                "evidence_ids": ["openalex:W1:e002"],
                "evidence_count": 1,
                "support_level": "weak",
                "allowed_language": "可陈述",
            },
        ],
    }]

    # 正文语义匹配 c1，却引用了只为 c2 授权的论文。
    result = validate_claim_citation_consistency(
        "少样本动作识别旨在用极少量标注样本识别新动作类别[2]。",
        plans,
        citation_map={"s2:aaa": 1, "openalex:W1": 2},
    )
    assert result["inconsistent_sentences"] == 1
    assert result["inconsistent_samples"][0]["best_claim"] == "R1:c1"
    assert result["inconsistent_samples"][0]["unauthorized_in_claim"] == ["openalex:W1"]
    assert result["validly_authorized_paper_ids"] == []
    assert result["unauthorized_cited_paper_ids"] == ["openalex:W1"]


def test_duplicate_claim_text_uses_cited_paper_as_tie_break():
    """多篇论文的同文主张并列时，引用论文只解决同分归属。"""
    plans = [{
        "route_id": "r1",
        "claims": [
            {"claim_id": "c1", "claim_text": "课堂行为编码可支持互动分析",
             "evidence_ids": ["p1:e1"]},
            {"claim_id": "c2", "claim_text": "课堂行为编码可支持互动分析",
             "evidence_ids": ["p2:e1"]},
        ],
    }]
    result = validate_claim_citation_consistency(
        "课堂行为编码可支持互动分析[2]。",
        plans,
        citation_map={"p1": 1, "p2": 2},
    )
    assert result["inconsistent_sentences"] == 0
    assert result["validly_authorized_paper_ids"] == ["p2"]


def test_reference_coverage_plan_adds_only_missing_evidence_backed_cards():
    """显式篇数不足时，补充计划只接纳有证据且在语义范围内的卡片。"""
    cards = [
        {"paper_id": "p1", "relation_type": "direct", "quality_status": "valid",
         "evidence_state": {"access_level": "abstract"},
         "research_problem": "已有路线问题", "field_evidence": {"research_problem": ["p1:e1"]}},
        {"paper_id": "p2", "relation_type": "direct", "quality_status": "partial",
         "evidence_state": {"access_level": "abstract"},
         "research_problem": "新增路线问题", "field_evidence": {"research_problem": ["p2:e1"]}},
        {"paper_id": "p3", "relation_type": "direct", "quality_status": "invalid",
         "evidence_state": {"access_level": "abstract"},
         "research_problem": "无效卡片", "field_evidence": {"research_problem": ["p3:e1"]}},
        {"paper_id": "p4", "relation_type": "direct", "quality_status": "valid",
         "evidence_state": {"access_level": "metadata_only"},
         "research_problem": "只有元数据", "field_evidence": {"research_problem": ["p4:e1"]}},
    ]
    plan = build_reference_coverage_plan(
        cards,
        {"canonical_topic": "课堂行为分析"},
        [{"claims": [{"evidence_ids": ["p1:e1"]}]}],
        required_reference_count=2,
    )
    assert plan is not None
    assert [claim["evidence_ids"] for claim in plan["claims"]] == [["p2:e1"]]
