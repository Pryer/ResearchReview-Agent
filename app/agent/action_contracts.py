"""操作级状态契约：字段来自现有节点读写依赖，新增字段默认不可见、不可提交。"""
from dataclasses import dataclass
from functools import lru_cache
from typing import get_type_hints
from pydantic import ConfigDict, Field, create_model
from app.agent.state import ResearchAgentState


def fields(value):
    return frozenset(value.split())

GOAL = fields("""user_query session_id user_operation_sequence agent_orchestration_mode agent_operation_mode
user_clarifications intent intent_context_role confidence topic canonical_topic start_year end_year
max_papers required_reference_count year_range_explicit strict_year_range max_papers_explicit
requested_sections core_deliverables user_paper_profile research_request research_plan research_semantic_frame
selected_scope language citation_style workflow our_work background existing_limitations verified_results target_length
state_schema_version state_revision evidence_snapshot_version evidence_snapshot_fingerprint writing_version""")
TRACE = fields("steps errors step_metrics contract_violations state_invariant_check")
SEARCH = fields("""keywords core_keywords expanded_keywords keyword_batches required_concepts topic_anchors
excluded_title_terms retrieval_target generation_limit evidence_pool_target evidence_yield screening_protocol
search_branches compiled_scope scope_search_queries scope_query_roles query_roles global_recall_queries
search_expanded search_failed retrieval_requirement_met language_branch_zh_ratio language_branch_zh_ratio_reason
language_coverage_target language_coverage searched_keywords searched_query_windows search_refinement_count
search_drift_diagnostics retrieval_stop_reason focus_coverage incremental_retrieval incremental_search_window
incremental_search_new_candidates incremental_new_paper_ids incremental_new_paper_keys incremental_required_new_evidence
candidate_papers ranked_papers source_diagnostics screening_report last_search_new_results retrieval_eligible_count
screening_report_low_pass_protection english_screening_recovery english_screening_recovery_attempted
cnki_anchor_query_used citation_shortfall_count""")
MATERIAL = fields("paper_details pdf_paths parsed_papers paper_cards evidence_quality_report")
ANALYSIS = fields("""provisional_framework validated_routes route_decisions route_validation_report
clusters dynamic_taxonomy taxonomy_validation taxonomy_remediation theme_synthesis claim_plans
claim_evidence_gate claim_alignment claim_citation_consistency global_evidence_gate evidence_gap_report
force_taxonomy_remediation route_merge_diagnostics search_report coverage""")
RECOVERY = fields("""quality_recovery_attempts recovery_action_count quality_recovery_history active_quality_recovery
quality_recovery_decision recovery_candidate_rejections allow_evidence_expansion refresh_existing_evidence
best_effort_generation best_effort_on_failure best_effort_policy_source automatic_best_effort_attempted
automatic_best_effort_generation allow_unvalidated_taxonomy forced_generation_issues recovery_decision recovery_round
route_recovery_attempts route_recovery_progress route_evidence_deficits scope_revision_count scope_revision_failed
recovery_history evidence_recovery_status recovery_statistics conservative_regeneration force_section_rewrite
user_accepted_best_effort_generation generation_readiness generation_blocked
citation_gap_repair_attempted citation_gap_repair_history citation_gap_repair_count
_citation_gap_repair_snapshot _citation_gap_repair_previous_cited""")
# WHY: 写作/改写内部也执行写后验证；对齐与引用授权诊断属于可提交产物，不只由分析动作写入。
WRITING = fields("""review body answer introduction introduction_data related_work related_work_data references
reference_papers citation_map citation_registry citation_validation claim_verification claim_verification_cache
claim_alignment claim_citation_consistency
claim_repairs generation_quality unique_cited_paper_count unique_valid_cited_paper_count final_requirement_met
writing_plans citation_allocation_plan citation_allocation_plans deliverable_readiness deliverable_downgrades
deliverable_validation writer_diagnostics writer_section_diagnostics final_review_integrity quality_gate quarantined_draft
citation_eligible_paper_ids reference_coverage_stats target_section_ids target_claim_ids section_checkpoints
section_candidate_checkpoints result_status autonomous_verified_fingerprint draft_available draft_released
 draft_disposition generation_readiness generation_blocked writing_version focus_coverage""")
PROTECTED = fields("""user_query research_request required_reference_count year_range_explicit max_papers_explicit
start_year end_year selected_scope core_deliverables session_id agent_orchestration_mode user_clarifications""")
# WHY: 这些是内层 Controller 可重建的视图；不属于专业动作提交的研究产物。
DERIVED = fields("""main_agent_context main_context_snapshot main_context_artifact_ref context_snapshot_version
agent_task_results agent_execution_status allowed_agent_actions state_revision artifact_manifest""")


@dataclass(frozen=True)
class ActionContract:
    inputs: frozenset[str]
    outputs: frozenset[str]


def contract(read, write):
    return ActionContract(GOAL | TRACE | read, (TRACE | write) - PROTECTED - DERIVED)


CONTRACTS = {
    # WHY: 重点质量恢复的检索目标来自本轮 active decision；检索任务需只读访问它，
    # 否则隔离后的专业任务会退化为普通路线检索，无法生成缺失重点的查询。
    "search_and_rank": contract(SEARCH | fields("paper_details paper_cards validated_routes recovery_decision active_quality_recovery _citation_gap_repair_previous_cited"),
        SEARCH | fields("generation_blocked research_semantic_frame")),
    "fetch_metadata": contract(SEARCH | MATERIAL | fields("validated_routes generation_readiness recovery_decision"),
        SEARCH | MATERIAL),
    "extract_paper_cards": contract(MATERIAL | fields("incremental_retrieval incremental_new_paper_ids"), fields("paper_cards")),
    "validate_routes": contract(MATERIAL | ANALYSIS | RECOVERY | fields("ranked_papers"), ANALYSIS | fields("paper_cards paper_details")),
    "cluster_papers": contract(MATERIAL | ANALYSIS | RECOVERY | fields("ranked_papers"), ANALYSIS | fields("paper_cards paper_details")),
    "plan_claims": contract(MATERIAL | ANALYSIS | RECOVERY | fields("ranked_papers"),
        fields("claim_plans claim_evidence_gate global_evidence_gate evidence_gap_report evidence_quality_report claim_alignment claim_citation_consistency")),
    "generate_deliverables": contract((MATERIAL - fields("pdf_paths parsed_papers")) | ANALYSIS | RECOVERY | WRITING | fields("ranked_papers language_coverage_target generation_limit required_concepts"),
        WRITING | RECOVERY | fields("evidence_quality_report evidence_yield route_merge_diagnostics language_coverage search_report theme_synthesis")),
    "rewrite_sections": contract((MATERIAL - fields("pdf_paths parsed_papers")) | ANALYSIS | RECOVERY | WRITING | fields("ranked_papers language_coverage_target generation_limit required_concepts"),
        WRITING | RECOVERY | fields("evidence_quality_report evidence_yield route_merge_diagnostics language_coverage search_report theme_synthesis")),
    "validate_result": contract(MATERIAL | ANALYSIS | RECOVERY | WRITING | fields("ranked_papers language_coverage_target required_concepts"),
        WRITING | fields("evidence_quality_report evidence_yield language_coverage search_report")),
    # WHY: 补检索是有界复合动作，既有实现包含提取、授权和引用修复，故显式组合依赖。
    "targeted_search": contract(SEARCH | MATERIAL | ANALYSIS | RECOVERY | WRITING,
        SEARCH | MATERIAL | ANALYSIS | RECOVERY | WRITING | fields("evidence_snapshot_version evidence_snapshot_fingerprint research_semantic_frame")),
}


@lru_cache(maxsize=None)
def schema_for(operation: str, output: bool = False):
    spec = CONTRACTS[operation]
    hints = get_type_hints(ResearchAgentState)
    names = spec.outputs if output else spec.inputs
    return create_model(operation + ("Output" if output else "Input"),
        __config__=ConfigDict(extra="forbid", strict=True),
        # WHY: Pydantic 将下划线字段视作私有属性，用 alias 保留其真实状态名并执行类型校验。
        **{("private" + name if name.startswith("_") else name):
           (hints[name], Field(default=None, alias=name)) for name in sorted(names)})


def validate_patch(operation, patch, removed):
    spec = CONTRACTS[operation]
    if set(patch) & set(removed):
        raise ValueError("action cannot set and remove the same field")
    forbidden = (set(patch) | set(removed)) - spec.outputs
    if forbidden:
        raise ValueError("unauthorized action output: " + ", ".join(sorted(forbidden)))
    schema_for(operation, True).model_validate(patch)
