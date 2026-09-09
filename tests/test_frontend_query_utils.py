"""前端查询处理测试。"""

from __future__ import annotations

from app.frontend.progress_labels import (
    can_offer_best_effort_draft,
    describe_job_step,
    recovery_progress_view,
)
from app.frontend.query_utils import build_agent_request_payload, normalize_review_query


def test_normalize_review_query_wraps_bare_topic():
    assert normalize_review_query("少样本动作识别") == (
        "帮我调研少样本动作识别相关论文，并生成研究现状"
    )


def test_normalize_review_query_keeps_explicit_request():
    query = "帮我生成少样本动作识别的综述"
    assert normalize_review_query(query) == query


def test_clarification_answer_is_not_rewritten_as_new_research_request():
    answer = "先自动识别和编码，再从教育学角度分析"
    payload = build_agent_request_payload(
        answer,
        "session-1",
        clarification_answer=answer,
    )
    assert payload == {
        "user_query": answer,
        "session_id": "session-1",
        "clarification_answer": answer,
    }


def test_request_payload_can_explicitly_keep_strict_gate_behavior():
    payload = build_agent_request_payload(
        "生成研究现状",
        "session-1",
        best_effort_on_failure=False,
    )

    assert payload["best_effort_on_failure"] is False


def test_quality_recovery_step_is_rendered_as_user_facing_progress():
    assert describe_job_step("quality_recovery:rewrite_sections") == (
        "正在重写未通过验证的章节"
    )
    assert describe_job_step("quality_recovery:final_best_effort_generation") == (
        "正在生成最终最佳可用草稿"
    )
    assert describe_job_step("custom_future_step") == "custom_future_step"


def test_recovery_progress_distinguishes_pending_and_verified():
    pending = recovery_progress_view(
        {"action": "REWRITE_SECTIONS"},
        [{"action": "REWRITE_SECTIONS", "outcome": "started"}],
        {"claim_authorized": 40, "planned": 40, "final_valid": 37},
        {"passed": False},
    )
    assert pending["action"] == "重写失败章节"
    assert pending["verification"] == "尚未重新验证"
    assert pending["final_valid"] == 37

    verified = recovery_progress_view(
        {"action": "REBUILD_CLAIMS"},
        [{"action": "REBUILD_CLAIMS", "outcome": "improved"}],
        {"final_valid": 40},
        {"passed": True},
    )
    assert verified["verification"] == "重新验证通过"


def test_recovery_progress_marks_uncomputed_counts_instead_of_zero():
    view = recovery_progress_view({}, [], {}, {"passed": False})

    assert view["claim_authorized"] == "尚未统计"
    assert view["planned"] == "尚未统计"
    assert view["final_valid"] == "尚未统计"


def test_blocked_evidence_result_offers_best_effort_action():
    assert can_offer_best_effort_draft({
        "status": "blocked",
        "paper_cards": [{"paper_id": "p1"}],
        "quality_gate": {
            "passed": False,
            "draft_released": False,
            "blocking_issues": [{"code": "required_focus_evidence_not_met"}],
        },
    }) is True
    assert can_offer_best_effort_draft({
        "status": "blocked",
        "paper_cards": [{"paper_id": "p1"}],
        "quality_gate": {
            "passed": False,
            "blocking_issues": [{"code": "deliverable_generation_failed"}],
        },
    }) is False
    assert can_offer_best_effort_draft({
        "status": "needs_clarification",
        "clarification": {"kind": "quality_decision"},
        "paper_cards": [{"paper_id": "p1"}],
        "quality_gate": {
            "passed": False,
            "blocking_issues": [{"code": "required_focus_evidence_not_met"}],
        },
    }) is True
