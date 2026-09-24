"""章节检查点失效判定：只依赖本节真实输入，不被无关证据变化牵连。"""

from app.schemas.deliverable_schema import (
    CoreDeliverableType,
    WritingPlan,
    WritingSection,
)
from app.tools.write_deliverable import _section_input_fingerprint


def _plan(citation_policy=None):
    return WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="测试",
        organizing_strategy="evidence_driven",
        sections=[
            WritingSection(
                id="theme_T1",
                title="自动识别",
                purpose="测试",
                supporting_paper_ids=["p1"],
            )
        ],
        citation_policy=citation_policy or {"minimum_unique_references": 2},
    )


def _card(paper_id, role="method", evidence="原文报告的方法"):
    return {
        "paper_id": paper_id,
        "title": f"论文{paper_id}",
        "year": 2024,
        "evidence_role": role,
        "quality_status": "verified",
        "evidence_state": {"access_level": "full_text"},
        "field_evidence": {"method": evidence},
        "field_claims": {},
        "evidence_spans": [],
    }


def _claim_plan(claim_text="采用 CNN 识别课堂行为"):
    return [{
        "route_name": "自动识别",
        "claims": [{
            "claim_text": claim_text,
            "support_level": "single",
            "allowed_language": "确定",
            "evidence_ids": ["p1:e001"],
        }],
    }]


def _state(**overrides):
    state = {
        "topic": "课堂行为分析",
        "selected_scope": {"description": "中小学课堂"},
        "research_semantic_frame": {
            "required_focuses": ["行为识别"],
            "evidence_requirements": [{"label": "实证研究", "source": "user_explicit"}],
        },
        "paper_cards": [_card("p1")],
        "paper_details": [{"paper_id": "p1", "_screening_decision": "include"}],
        "validated_routes": [{"route_id": "R1", "core_paper_ids": ["p1"]}],
        "claim_plans": _claim_plan(),
    }
    state.update(overrides)
    return state


def _fingerprint(state):
    plan = _plan()
    return _section_input_fingerprint(plan, plan.sections[0], state)


def test_unrelated_paper_and_route_do_not_invalidate_section_checkpoint():
    """新增无关论文曾通过全局证据快照废掉所有章节，现在必须保持可复用。"""
    before = _fingerprint(_state())
    after = _fingerprint(_state(
        paper_cards=[_card("p1"), _card("p9")],
        paper_details=[
            {"paper_id": "p1", "_screening_decision": "include"},
            {"paper_id": "p9", "_screening_decision": "include"},
        ],
        validated_routes=[
            {"route_id": "R1", "core_paper_ids": ["p1"]},
            {"route_id": "R9", "core_paper_ids": ["p9"]},
        ],
        # 全局快照已不再参与指纹；显式改动它也不应影响本节复用。
        evidence_snapshot_fingerprint="globally-changed",
        evidence_snapshot_version=7,
    ))
    assert before == after


def test_claim_constraint_change_invalidates_section_checkpoint():
    """claim 约束直接进入提示词，旧的全局快照并不覆盖它。"""
    assert _fingerprint(_state()) != _fingerprint(
        _state(claim_plans=_claim_plan("采用 Transformer 识别课堂行为"))
    )


def test_topic_scope_and_focus_changes_invalidate_section_checkpoint():
    base = _fingerprint(_state())
    assert base != _fingerprint(_state(topic="职业教育课堂"))
    assert base != _fingerprint(_state(selected_scope={"description": "高中课堂"}))
    assert base != _fingerprint(_state(research_semantic_frame={
        "required_focuses": ["行为识别", "情感分析"],
        "evidence_requirements": [{"label": "实证研究"}],
    }))


def test_own_evidence_screening_and_route_changes_still_invalidate():
    """收窄范围不得削弱本节证据、筛选决定与路线归属的失效判定。"""
    base = _fingerprint(_state())
    assert base != _fingerprint(_state(paper_cards=[_card("p1", evidence="改写后的证据")]))
    assert base != _fingerprint(_state(
        paper_details=[{"paper_id": "p1", "_screening_decision": "exclude"}]
    ))
    assert base != _fingerprint(_state(
        validated_routes=[{"route_id": "R2", "core_paper_ids": ["p1"]}]
    ))


def test_survey_paper_list_is_a_shared_prompt_input_and_invalidates():
    """综述清单注入每一节提示词，是唯一必须保留的全局输入。"""
    base = _fingerprint(_state())
    assert base == _fingerprint(
        _state(paper_cards=[_card("p1"), _card("p2", role="survey")])
    )
    # 非综述论文不进入该清单，因此不影响本节。
    assert base == _fingerprint(
        _state(paper_cards=[_card("p1"), _card("p2", role="method")])
    )


def test_citation_policy_change_invalidates_section_checkpoint():
    assert _fingerprint(_state()) != _section_input_fingerprint(
        _plan({"minimum_unique_references": 5}),
        _plan({"minimum_unique_references": 5}).sections[0],
        _state(),
    )


def test_other_section_authorizing_survey_changes_actual_input():
    plan = _plan()
    state = _state(paper_cards=[_card("p1"), _card("p2", role="survey")])
    before = _section_input_fingerprint(plan, plan.sections[0], state)
    plan.sections.append(WritingSection(id="background", title="背景", purpose="背景", supporting_paper_ids=["p2"]))
    assert before != _section_input_fingerprint(plan, plan.sections[0], state)


def test_last_theme_obligation_is_fingerprinted():
    plan = _plan()
    before = _section_input_fingerprint(plan, plan.sections[0], _state())
    plan.sections.insert(0, WritingSection(id="theme_T0", title="其他路线", purpose="比较"))
    assert before != _section_input_fingerprint(plan, plan.sections[1], _state())


def test_actual_authorization_projection_and_draft_invalidate():
    from copy import deepcopy
    plan = _plan()
    state = _state()
    state["paper_cards"][0]["field_claims"] = {"method": [
        {"evidence_id": "p1:e1", "claim": "明确报告的方法", "explicitly_reported": True},
    ]}
    before = _section_input_fingerprint(plan, plan.sections[0], state)
    other = deepcopy(plan)
    other.sections.append(WritingSection(id="background", title="背景", purpose="背景", supporting_claim_ids=["p1:e2"]))
    assert before != _section_input_fingerprint(other, other.sections[0], state)
