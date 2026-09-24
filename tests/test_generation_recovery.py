"""生成恢复闭环的确定性集成与契约回归。"""

from __future__ import annotations

from app.agent.deliverable_router import check_generation_readiness
from app.agent.graph import _build_output, _verify_generated_draft
from app.agent.generation_recovery import (
    active_focus_recovery_targets,
    candidate_is_not_worse,
    complete_recovery_action,
    decide_generation_recovery,
    input_fingerprint,
    quality_progress_vector,
    recovery_action_count,
)
from app.schemas.deliverable_schema import (
    CoreDeliverableType,
    WritingPlan,
    WritingSection,
)
from app.schemas.recovery_schema import RecoveryAction, RecoveryStatus
from app.agent.nodes.synthesis import final_answer_node, generate_deliverables_node
from app.tools.write_deliverable import (
    promote_section_checkpoints,
    _reuse_and_checkpoint_sections,
    _section_checkpoint_key,
    _section_input_fingerprint,
    write_deliverable,
)


def _card(index: int) -> dict:
    paper_id = f"p{index}"
    claim = f"研究{index}围绕课堂互动编码开展分析"
    return {
        "paper_id": paper_id,
        "title": f"课堂互动研究{index}",
        "authors": [f"作者{index}"],
        "year": 2024,
        "venue": "教育技术研究",
        "source": "openalex",
        "doi": f"10.1234/classroom.{index}",
        "publication_type": "journal_article",
        "publication_status": "published",
        "peer_review_status": "peer_reviewed",
        "quality_status": "partial",
        "evidence_source": "abstract",
        "evidence_state": {"access_level": "abstract"},
        "research_problem": claim,
        "field_evidence": {"research_problem": [f"{paper_id}:e1"]},
        "field_claims": {
            "research_problem": [{
                "claim": claim,
                "source_text": claim,
                "evidence_id": f"{paper_id}:e1",
                "explicitly_reported": True,
            }],
        },
    }


def _count_failure(*, eligible: int, authorized: int, allow_search: bool = True) -> dict:
    return {
        "required_reference_count": 4,
        "allow_evidence_expansion": allow_search,
        "reference_coverage_stats": {
            "evidence_backed": eligible,
            "claim_authorized": authorized,
            "final_valid": 2,
        },
        "quality_gate": {
            "passed": False,
            "phase": "post_generation",
            "blocking_issues": [{
                "code": "minimum_cited_references_not_met",
                "requested": 4,
                "actual": 2,
                "message": "正文有效引用不足",
            }],
        },
    }


def test_recovery_controller_rebuilds_claims_before_searching():
    state = _count_failure(eligible=6, authorized=2)

    decision = decide_generation_recovery(state, max_actions=6)

    assert decision.action == RecoveryAction.REBUILD_CLAIMS
    assert decision.requires_user_input is False


def test_recovery_progress_uses_persisted_final_valid_coverage():
    state = _count_failure(eligible=60, authorized=40)
    state["required_reference_count"] = 40
    state["reference_coverage_stats"]["final_valid"] = 35

    progress = quality_progress_vector(state)

    assert progress.valid_reference_shortfall == 5


def test_private_research_state_keeps_transactional_generation_products():
    state = {
        "quality_gate": {"passed": False, "blocking_issues": []},
        "answer": "正文被隔离",
        "review": "带引用的隔离草稿[p1]",
        "references": ["参考文献一"],
        "reference_papers": [{"paper_id": "p1"}],
        "citation_map": {"p1": 1},
        "citation_validation": {"valid": True},
        "claim_verification": {"unsupported": 1},
        "unique_cited_paper_count": 1,
        "unique_valid_cited_paper_count": 1,
        "final_requirement_met": False,
        "generation_blocked": True,
    }

    output = _build_output(state)
    private = output["research_state"]

    assert output["body"] == ""
    assert private["review"] == state["review"]
    assert private["references"] == state["references"]
    assert private["citation_map"] == state["citation_map"]
    assert private["claim_verification"] == state["claim_verification"]
    assert private["unique_valid_cited_paper_count"] == 1


def test_final_answer_treats_missing_valid_count_as_zero():
    state = {
        "max_papers_explicit": True,
        "required_reference_count": 3,
        "citation_validation": {"valid": True},
        "claim_citation_consistency": {"consistency_rate": 1.0},
        "quality_gate": {"passed": False, "phase": "post_generation"},
        "review": "隔离草稿[p1]",
        "deliverable_validation": [],
        "final_review_integrity": {},
    }

    final_answer_node(state)

    codes = {item["code"] for item in state["quality_gate"]["blocking_issues"]}
    assert "minimum_cited_references_not_met" in codes


def test_recovery_controller_switches_action_after_no_progress():
    state = _count_failure(eligible=6, authorized=2)
    fingerprint = input_fingerprint(state)
    state["quality_recovery_history"] = [{
        "action": RecoveryAction.REBUILD_CLAIMS.value,
        "input_fingerprint": fingerprint,
        "outcome": "no_progress",
    }]

    decision = decide_generation_recovery(state, max_actions=6)

    assert decision.action == RecoveryAction.TARGETED_SEARCH


def test_mixed_count_and_claim_failure_rewrites_after_reallocation():
    state = _count_failure(eligible=6, authorized=4)
    state["quality_gate"]["blocking_issues"].append({
        "code": "claim_evidence_quality_not_met",
        "message": "仍有未支持主张",
    })
    fingerprint = input_fingerprint(state)
    state["quality_recovery_history"] = [{
        "action": RecoveryAction.REALLOCATE_CITATIONS.value,
        "input_fingerprint": fingerprint,
        "outcome": "no_progress",
    }]

    decision = decide_generation_recovery(state, max_actions=6)

    assert decision.action == RecoveryAction.REWRITE_SECTIONS


def test_recovery_controller_respects_existing_evidence_only_permission():
    state = _count_failure(eligible=2, authorized=2, allow_search=False)

    decision = decide_generation_recovery(state, max_actions=6)

    assert decision.action == RecoveryAction.REQUEST_USER_INPUT
    assert decision.requires_user_input is True


def test_recovery_controller_degrades_when_evidence_budget_exhausted():
    state = _count_failure(eligible=3, authorized=3)
    # 证据级恢复预算已耗尽，此时 TARGETED_SEARCH 必然是空动作。
    state["recovery_round"] = 2
    state["evidence_recovery_status"] = "EXHAUSTED"

    decision = decide_generation_recovery(state, max_actions=6)

    # 必须落到 EXHAUSTED，调用方才能执行明确标注限制的最佳努力生成；
    # 若仍报 RECOVERABLE，主循环会重复同一诊断直到 no-progress 阻断。
    assert decision.action == RecoveryAction.DEGRADE
    assert decision.status == RecoveryStatus.EXHAUSTED
    assert decision.requires_user_input is False


def test_recovery_controller_still_searches_while_evidence_budget_remains():
    state = _count_failure(eligible=3, authorized=3)
    state["recovery_round"] = 0

    decision = decide_generation_recovery(state, max_actions=6)

    assert decision.action == RecoveryAction.TARGETED_SEARCH
    assert decision.status == RecoveryStatus.RECOVERABLE


def _focus_failure(*, allow_search: bool = True) -> dict:
    return {
        "required_reference_count": 4,
        "allow_evidence_expansion": allow_search,
        "reference_coverage_stats": {
            "evidence_backed": 4, "claim_authorized": 4, "final_valid": 4,
        },
        "quality_gate": {
            "passed": False,
            "phase": "pre_generation",
            "blocking_issues": [{
                "code": "required_focus_evidence_not_met",
                "message": "用户明确研究重点缺少足够直接证据：S-T分析法",
                "missing_focuses": ["S-T分析法"],
            }],
        },
    }


def test_focus_gap_advertises_only_the_reachable_action():
    decision = decide_generation_recovery(_focus_failure(), max_actions=6)

    assert decision.issues[0].category == "focus_coverage"
    assert decision.action == RecoveryAction.TARGETED_SEARCH
    # 重写章节无法为缺失的研究重点造出直接证据，此前却被广告为可用动作。
    assert decision.issues[0].available_actions == [RecoveryAction.TARGETED_SEARCH]


def test_focus_gap_degrades_when_evidence_budget_exhausted():
    state = _focus_failure()
    state["recovery_round"] = 2
    state["evidence_recovery_status"] = "EXHAUSTED"

    decision = decide_generation_recovery(state, max_actions=6)

    # 门禁与前端已把该码视为可降级（can_offer_best_effort_draft 返回 True、
    # 最佳努力草稿保留 gate 失败并释放 partial），阶梯不得停在询问用户。
    assert decision.action == RecoveryAction.DEGRADE
    assert decision.status == RecoveryStatus.EXHAUSTED


def test_focus_gap_asks_user_when_expansion_forbidden():
    decision = decide_generation_recovery(
        _focus_failure(allow_search=False), max_actions=6
    )

    assert decision.action == RecoveryAction.REQUEST_USER_INPUT
    assert decision.requires_user_input is True


def test_focus_recovery_target_and_progress_use_the_blocking_issue():
    state = _focus_failure()
    decision = decide_generation_recovery(state, max_actions=6)
    state["active_quality_recovery"] = decision.model_dump(mode="json")
    state["quality_recovery_history"] = [{
        "progress_before": decision.progress.model_dump(mode="json"),
        "outcome": "started",
    }]

    assert active_focus_recovery_targets(state) == ["S-T分析法"]
    state["quality_gate"]["blocking_issues"] = []
    complete_recovery_action(state)

    assert state["quality_recovery_history"][-1]["outcome"] == "improved"
    assert "active_quality_recovery" not in state


def test_count_gap_keeps_cheaper_in_evidence_remedy_before_focus_search():
    state = _focus_failure()
    state["reference_coverage_stats"] = {
        "evidence_backed": 6, "claim_authorized": 4, "planned": 2, "final_valid": 2,
    }
    state["quality_gate"]["blocking_issues"].append({
        "code": "minimum_planned_references_not_met",
        "message": "证据池有 6 篇可用论文，4 篇已获主张授权，但只有 2 篇进入分配计划",
        "requested": 4,
        "available": 2,
        "eligible": 6,
    })

    decision = decide_generation_recovery(state, max_actions=6)

    # 升级顺序必须保持"先证据内补救、后外部检索"：重分配引用比定向补检索便宜，
    # 篇数缺口解决后下一轮再由 focus 分支处理重点缺口。
    assert decision.action == RecoveryAction.REALLOCATE_CITATIONS


def test_recovery_controller_targets_failed_section():
    state = {
        "quality_gate": {
            "passed": False,
            "phase": "post_generation",
            "blocking_issues": [{
                "code": "section_generation_failed",
                "section_id": "theme_T2",
                "message": "第二节生成失败",
            }],
        },
    }

    decision = decide_generation_recovery(state, max_actions=6)

    assert decision.action == RecoveryAction.REWRITE_SECTIONS
    assert decision.target_section_ids == ["theme_T2"]


def test_recovery_controller_routes_metadata_failure_to_evidence_refresh():
    state = {
        "allow_evidence_expansion": True,
        "quality_gate": {
            "passed": False,
            "phase": "post_generation",
            "blocking_issues": [{
                "code": "reference_metadata_not_met",
                "message": "引用元数据未通过核验",
            }],
        },
    }

    decision = decide_generation_recovery(state, max_actions=6)

    assert decision.issues[0].category == "metadata"
    assert decision.action == RecoveryAction.REFRESH_EVIDENCE


def test_recovery_controller_stops_at_shared_budget():
    state = _count_failure(eligible=6, authorized=2)
    state["recovery_action_count"] = 6

    decision = decide_generation_recovery(state, max_actions=6)

    assert decision.action == RecoveryAction.DEGRADE
    assert decision.status == RecoveryStatus.EXHAUSTED


def test_shared_budget_migrates_both_legacy_counters():
    assert recovery_action_count({
        "recovery_action_count": 1,
        "quality_recovery_attempts": 2,
        "recovery_round": 3,
    }) == 5


def test_recovery_input_fingerprint_changes_when_evidence_content_changes():
    state = {
        "paper_cards": [_card(1)],
        "paper_details": [{"paper_id": "p1", "_screening_decision": "confirmed"}],
    }
    before = input_fingerprint(state)
    state["paper_cards"][0]["field_claims"]["research_problem"][0]["claim"] = (
        "同一论文的新证据版本"
    )

    assert input_fingerprint(state) != before


def test_generation_readiness_uses_claim_authorized_union():
    cards = [_card(1), _card(2)]
    state = {
        "paper_cards": cards,
        "max_papers_explicit": True,
        "required_reference_count": 2,
        "core_deliverables": ["research_background"],
        "research_semantic_frame": {},
        "claim_plans": [{
            "route_id": "r1",
            "claims": [{"claim_id": "c1", "evidence_ids": ["p1:e1"]}],
        }],
    }

    result = check_generation_readiness(state)

    assert result.ready is False
    assert result.authorized_reference_count == 1
    assert result.blocking_issues[0]["code"] == "minimum_planned_references_not_met"
    assert state["reference_coverage_stats"]["claim_authorized"] == 1


def test_reference_coverage_separates_in_scope_from_evidence_backed():
    cards = [_card(1), _card(2)]
    cards[1]["evidence_source"] = "metadata_only"
    cards[1]["evidence_state"] = {"access_level": "metadata_only"}
    state = {
        "paper_cards": cards,
        "core_deliverables": ["research_background"],
        "research_semantic_frame": {},
    }

    check_generation_readiness(state)

    assert state["reference_coverage_stats"]["confirmed_in_scope"] == 2
    assert state["reference_coverage_stats"]["evidence_backed"] == 1


def test_section_checkpoint_reuses_only_untargeted_matching_section():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="测试章节恢复",
        organizing_strategy="evidence_driven",
        sections=[
            WritingSection(id="s1", title="第一节", purpose="说明路线一", supporting_paper_ids=["p1"]),
            WritingSection(id="s2", title="第二节", purpose="说明路线二", supporting_paper_ids=["p2"]),
        ],
    )
    state = {
        "paper_cards": [_card(1), _card(2)],
        "writing_version": 2,
        "target_section_ids": ["s2"],
        "citation_allocation_plan": {"sections": [
            {"section_id": "s1", "paper_ids": ["p1"]},
            {"section_id": "s2", "paper_ids": ["p2"]},
        ]},
    }
    first_key = _section_checkpoint_key(plan, "s1")
    first = plan.sections[0]
    state["section_checkpoints"] = {first_key: {
        "text": "## 第一节\n\n这是已经通过章节检查并需要保留的研究路线描述[p1]。",
        "input_fingerprint": _section_input_fingerprint(plan, first, state),
        "status": "validated",
    }}
    candidate = (
        "## 第一节\n\n这是不应替换旧检查点的新文本[p1]。\n\n"
        "## 第二节\n\n这是本轮需要重写的新文本[p2]。"
    )

    merged = _reuse_and_checkpoint_sections(candidate, plan, state)

    assert "需要保留" in merged
    assert "不应替换" not in merged
    assert "本轮需要重写" in merged
    assert state["section_candidate_checkpoints"][first_key]["status"] == "reused"


def test_write_deliverable_skips_llm_for_reusable_checkpoint_section():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="测试真实局部写作",
        organizing_strategy="evidence_driven",
        sections=[
            WritingSection(id="s1", title="第一节", purpose="说明路线一", supporting_paper_ids=["p1"]),
            WritingSection(id="s2", title="第二节", purpose="说明路线二", supporting_paper_ids=["p2"]),
        ],
    )
    state = {
        "paper_cards": [_card(1), _card(2)],
        "target_section_ids": ["s2"],
        "citation_allocation_plan": {"sections": [
            {"section_id": "s1", "paper_ids": ["p1"]},
            {"section_id": "s2", "paper_ids": ["p2"]},
        ]},
        "claim_plans": [],
    }
    checkpoint_key = _section_checkpoint_key(plan, "s1")
    state["section_checkpoints"] = {checkpoint_key: {
        "text": "## 第一节\n\n这是已经验证并必须原样保留的章节内容[p1]。",
        "input_fingerprint": _section_input_fingerprint(plan, plan.sections[0], state),
        "status": "validated",
    }}

    class RecordingLLM:
        def __init__(self):
            self.operations: list[str] = []

        def complete(self, _prompt, **kwargs):
            operation = str(kwargs.get("operation") or "")
            self.operations.append(operation)
            return "## 第二节\n\n本节依据授权证据说明课堂互动编码的研究路径与适用边界[p2]。"

    llm = RecordingLLM()
    text = write_deliverable(plan, state, llm=llm)

    assert "必须原样保留" in text
    assert llm.operations
    assert all(":s1:" not in operation for operation in llm.operations)
    assert any(":s2:" in operation for operation in llm.operations)


def test_candidate_acceptance_rejects_new_hard_issue_despite_more_references():
    before = {
        "required_reference_count": 4,
        "unique_valid_cited_paper_count": 2,
        "quality_gate": {"passed": False, "blocking_issues": [{
            "code": "minimum_cited_references_not_met",
        }]},
    }
    after = {
        "required_reference_count": 4,
        "unique_valid_cited_paper_count": 4,
        "quality_gate": {"passed": False, "blocking_issues": [{
            "code": "final_text_integrity_not_met",
        }]},
    }

    assert candidate_is_not_worse(before, after) is False


def test_candidate_acceptance_rejects_regression_in_later_vector_dimension():
    before = {
        "required_reference_count": 4,
        "unique_valid_cited_paper_count": 2,
        "claim_verification": {"unsupported": 1},
        "quality_gate": {"passed": False, "blocking_issues": [{
            "code": "minimum_cited_references_not_met",
        }]},
    }
    after = {
        "required_reference_count": 4,
        "unique_valid_cited_paper_count": 3,
        "claim_verification": {"unsupported": 2},
        "quality_gate": {"passed": False, "blocking_issues": [{
            "code": "minimum_cited_references_not_met",
        }]},
    }

    assert candidate_is_not_worse(before, after) is False


def test_sufficient_fixture_runs_real_planning_writing_and_validation(monkeypatch):
    """只替换外部 LLM，真实运行规划、写作、引用检查和最终门禁。"""
    cards = [_card(index) for index in range(1, 7)]
    claim_plans = [{
        "route_id": "coverage",
        "route_name": "证据覆盖",
        "claims": [
            {
                "claim_id": f"c{index}",
                "claim_text": card["research_problem"],
                "claim_type": "background_fact",
                "evidence_ids": [f"p{index}:e1"],
                "evidence_count": 1,
                "independent_source_count": 1,
                "support_level": "single",
                "allowed_language": "一项研究报告",
                "allowed_verbs": ["报告"],
            }
            for index, card in enumerate(cards, 1)
        ],
    }]
    state = {
        "user_query": "调研学习分析并生成研究背景，引用不少于6篇",
        "intent": "generate_introduction",
        "topic": "学习分析",
        "canonical_topic": "学习分析",
        "core_deliverables": ["research_background"],
        "paper_details": cards,
        "paper_cards": cards,
        "ranked_papers": cards,
        "required_reference_count": 6,
        "max_papers": 6,
        "max_papers_explicit": True,
        "generation_limit": 12,
        "research_semantic_frame": {},
        "claim_plans": claim_plans,
        "steps": [],
        "errors": [],
    }
    monkeypatch.setattr("app.agent.graph._get_llm", lambda: None)

    generate_deliverables_node(state, llm=None)
    _verify_generated_draft(state)
    final_answer_node(state)

    assert state["writing_plans"]
    assert state["section_checkpoints"]
    assert state["reference_coverage_stats"]["planned"] >= 6
    assert state["unique_valid_cited_paper_count"] >= 6
    assert state["quality_gate"]["passed"] is True
    assert state["body"]


def test_other_domain_and_reference_target_use_same_recovery_policy():
    state = _count_failure(eligible=12, authorized=8)
    state.update({
        "topic": "检索增强生成",
        "required_reference_count": 10,
        "reference_coverage_stats": {
            "evidence_backed": 12,
            "claim_authorized": 8,
            "final_valid": 7,
        },
    })
    state["quality_gate"]["blocking_issues"][0].update({
        "requested": 10,
        "actual": 7,
    })

    decision = decide_generation_recovery(state, max_actions=6)

    assert decision.action == RecoveryAction.REBUILD_CLAIMS


def test_incomplete_semantic_verification_retries_without_rewriting():
    state = {
        "claim_verification": {"unsupported": 2, "unverified": 2},
        "generation_quality": {"unverified_claims": 2},
        "quality_gate": {
            "passed": False,
            "phase": "post_generation",
            "blocking_issues": [{
                "code": "claim_verification_incomplete",
                "details": {"claim_ids": ["c001", "c004"]},
            }],
        },
    }

    decision = decide_generation_recovery(state, max_actions=4)

    assert decision.action == RecoveryAction.REVERIFY_CLAIMS
    assert decision.target_claim_ids == ["c001", "c004"]
    assert decision.progress.unverified_claims == 2


def test_repeated_incomplete_verification_stops_without_search_or_rewrite():
    state = {
        "claim_verification": {"unsupported": 1, "unverified": 1},
        "quality_gate": {
            "passed": False,
            "blocking_issues": [{"code": "claim_verification_incomplete"}],
        },
    }
    fingerprint = input_fingerprint(state)
    state["quality_recovery_history"] = [{
        "action": RecoveryAction.REVERIFY_CLAIMS.value,
        "input_fingerprint": fingerprint,
        "outcome": "no_progress",
    }]

    decision = decide_generation_recovery(state, max_actions=4)

    assert decision.action == RecoveryAction.DEGRADE
    assert decision.status == RecoveryStatus.EXHAUSTED


def test_checkpoint_promotion_rejects_sparse_and_duplicate_sections():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="研究现状",
        organizing_strategy="theme",
        sections=[
            WritingSection(id="theme_a", title="路线A", purpose="梳理A", minimum_unique_references=2),
            WritingSection(id="theme_b", title="路线B", purpose="梳理B", minimum_unique_references=2),
        ],
    )
    state = {
        "section_candidate_checkpoints": {
            _section_checkpoint_key(plan, "theme_a"): {
                "section_id": "theme_a", "text": "## 路线A\n\n重复内容句。", "local_valid": True,
            },
            _section_checkpoint_key(plan, "theme_b"): {
                "section_id": "theme_b", "text": "## 路线B\n\n证据较少。", "local_valid": True,
            },
        },
    }
    validation = {
        "valid": False,
        "metrics": {
            "section_evidence_floors": [
                {"section_id": "theme_a", "status": "ok"},
                {"section_id": "theme_b", "status": "short_body"},
            ],
            "duplicate_sentence_samples": [
                {"first": "重复内容句。", "duplicate": "重复内容句。"},
            ],
        },
    }

    promote_section_checkpoints(state, plan, validation)

    assert state["section_checkpoints"] == {}
