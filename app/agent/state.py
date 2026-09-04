"""Agent 全局状态定义。

``ResearchAgentState`` 是贯穿整个 Agent 工作流的状态字典，
由各节点读取和更新。使用 TypedDict 保持类型安全。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from app.schemas.paper_schema import SourceDiagnostic


class ResearchAgentState(TypedDict, total=False):
    """Agent 工作流状态。

    所有字段均为可选，按需填充。节点只读写关心的字段，
    避免不必要的耦合。
    """

    # ---------- 用户输入 ----------
    user_query: str

    # ---------- 意图与槽位 ----------
    intent: str
    confidence: float
    # 意图识别的轮次角色：request / clarification_answer / working_query。
    intent_context_role: str
    topic: str
    canonical_topic: str
    keywords: List[str]
    core_keywords: List[str]
    expanded_keywords: List[str]
    # 检索批次（exact→broader→variant）来自关键词生成工具的 type 元数据
    keyword_batches: List[Dict[str, Any]]
    required_concepts: List[List[str]]
    topic_anchors: List[List[str]]
    excluded_title_terms: List[str]
    start_year: int
    end_year: int
    max_papers: int
    required_reference_count: int  # 用户要求最终综述至少使用的唯一参考文献数量
    retrieval_target: int
    generation_limit: int
    # 证据池绝对目标（详情补全阶段的候选规模上限）。
    # WHY: 预留余量此前只加在"本轮增量"上，增量轮 required_to_fetch 变小
    # 就把池目标一起缩掉（实测 60 → 15）。绝对目标跨轮持久化，增量轮取
    # max(持久化目标, 按增量算的目标)，不因增量小而丢失预留。
    evidence_pool_target: int
    # 上一轮观测到的证据成品率：evidence_availability_rate（可用/卡片）、
    # citation_realization_rate（引用/可用）、end_to_end_rate（引用/卡片）。
    # WHY: 池目标必须按端到端成品率倒推；首轮无观测时退回配置默认余量。
    evidence_yield: Dict[str, Any]
    year_range_explicit: bool
    strict_year_range: bool
    max_papers_explicit: bool
    requested_sections: List[str]
    core_deliverables: List[str]
    user_paper_profile: Dict[str, Any]
    research_request: Dict[str, Any]
    research_plan: Dict[str, Any]
    research_semantic_frame: Dict[str, Any]
    # 语义帧解析时所用的工作查询；查询变化（如澄清后追加范围确认）即失效重解析。
    semantic_frame_source_query: str
    screening_protocol: Dict[str, Any]
    search_branches: List[Dict[str, Any]]
    topic_interpretations: List[Dict[str, Any]]
    selected_scope: Dict[str, Any]
    # 本轮范围编译快照；rank 与详情复核共享同一规范化语义。
    compiled_scope: Dict[str, Any]
    scope_search_queries: List[str]
    scope_query_roles: Dict[str, str]
    search_expanded: bool
    search_failed: bool
    retrieval_requirement_met: bool
    language: str
    citation_style: str
    # 由主题语言倾向判断映射的中文分支配额；缺省时 rank 回落到全局配置。
    language_branch_zh_ratio: float
    language_branch_zh_ratio_reason: str
    # 双语覆盖最低要求在检索阶段建立，最终按实际有效引用重新验收。
    language_coverage_target: Dict[str, Any]
    language_coverage: Dict[str, Any]
    workflow: str
    planning_failed: bool
    searched_keywords: List[str]
    searched_query_windows: List[str]
    search_refinement_count: int
    search_drift_diagnostics: Dict[str, Any]
    retrieval_stop_reason: str
    focus_coverage: Dict[str, Any]
    incremental_retrieval: bool
    incremental_search_window: Dict[str, int]
    incremental_search_new_candidates: int
    incremental_new_paper_ids: List[str]
    incremental_required_new_evidence: int
    quality_recovery_attempts: int
    best_effort_generation: bool
    allow_unvalidated_taxonomy: bool
    forced_generation_issues: List[Dict[str, Any]]
    state_schema_version: str

    # ---------- 检索结果 ----------
    candidate_papers: List[Dict[str, Any]]
    ranked_papers: List[Dict[str, Any]]
    paper_details: List[Dict[str, Any]]
    source_diagnostics: List[SourceDiagnostic]  # 各数据源的检索诊断（区分空结果/失败）
    screening_report: Dict[str, Any]  # 硬过滤与 LLM 语义筛选的可解释统计
    last_search_new_results: int
    retrieval_eligible_count: int

    # ---------- PDF ----------
    pdf_paths: Dict[str, str]
    parsed_papers: Dict[str, Dict[str, Any]]

    # ---------- PaperCard ----------
    paper_cards: List[Dict[str, Any]]
    clusters: List[Dict[str, Any]]
    dynamic_taxonomy: Dict[str, Any]
    taxonomy_validation: Dict[str, Any]
    taxonomy_remediation: Dict[str, Any]
    theme_synthesis: List[Dict[str, Any]]
    search_report: Dict[str, Any]
    deliverable_readiness: List[Dict[str, Any]]
    deliverable_downgrades: List[Dict[str, Any]]
    writing_plans: List[Dict[str, Any]]
    # 写作计划阶段的路线并入事件（被并路线 / 目标路线 / 迁移论文数 / 原因）。
    # WHY: 小节名额溢出与单篇路线此前被静默丢弃，证据消失且无处可查；
    # 生命周期与 writing_plans 相同，由 _merge_and_select_themes 写入。
    route_merge_diagnostics: List[Dict[str, Any]]
    citation_allocation_plans: List[Dict[str, Any]]
    deliverable_validation: List[Dict[str, Any]]
    writer_diagnostics: List[Dict[str, Any]]
    writer_section_diagnostics: List[Dict[str, Any]]
    final_review_integrity: Dict[str, Any]
    generation_readiness: Dict[str, Any]
    quality_gate: Dict[str, Any]
    quarantined_draft: str
    generation_blocked: bool
    citation_eligible_paper_ids: List[str]
    unsupported_task_guard: Dict[str, Any]
    provisional_framework: Dict[str, Any]
    validated_routes: List[Dict[str, Any]]
    route_decisions: List[Dict[str, Any]]
    route_validation_report: Dict[str, Any]
    evidence_gap_report: Dict[str, Any]
    evidence_snapshot_version: int
    evidence_snapshot_fingerprint: str
    recovery_decision: Dict[str, Any]
    recovery_round: int
    route_recovery_attempts: Dict[str, int]
    # 每条路线的补检索进度：target / core_before / core_after / new_relevant / attempts
    route_recovery_progress: Dict[str, Dict[str, int]]
    # 恢复耗尽后仍未达目标篇数的路线；用于门禁 warning，不阻断交付
    route_evidence_deficits: List[Dict[str, Any]]
    scope_revision_count: int
    scope_revision_failed: bool
    recovery_history: List[Dict[str, Any]]
    evidence_recovery_status: str
    claim_plans: List[Dict[str, Any]]
    claim_evidence_gate: Dict[str, Any]
    claim_alignment: Dict[str, Any]
    claim_citation_consistency: Dict[str, Any]
    global_evidence_gate: Dict[str, Any]  # 综述级证据充分性门禁结果（只评估与推荐）

    # ---------- 综述 ----------
    body: str  # 纯学术正文，不包含质量门禁、证据范围或运行提示
    review: str
    related_work: str  # 相关工作章节
    related_work_data: Dict[str, Any]  # 相关工作结构化数据
    introduction: str  # 引言章节
    introduction_data: Dict[str, Any]  # 引言结构化数据
    references: List[str]
    reference_papers: List[Dict[str, Any]]
    citation_map: Dict[str, int]
    citation_registry: Dict[str, Any]
    citation_validation: Dict[str, Any]
    claim_verification: Dict[str, Any]
    claim_verification_cache: Dict[str, Dict[str, Any]]
    generation_quality: Dict[str, Any]
    evidence_quality_report: Dict[str, Any]
    unique_cited_paper_count: int
    unique_valid_cited_paper_count: int
    final_requirement_met: bool
    answer: str

    # ---------- 用户提供的本文工作信息（用于 Related Work 和 Introduction） ----------
    our_work: Dict[str, Any]  # research_problem, method_name, method_summary, innovations
    background: Dict[str, Any]  # task_definition, importance, application_scenarios
    existing_limitations: List[Dict[str, Any]]  # limitation, supporting_paper_ids
    verified_results: List[Dict[str, Any]]  # statement, source
    target_length: int  # 目标字数

    # ---------- 执行记录 ----------
    errors: List[str]
    steps: List[Dict[str, Any]]
    # @requires 契约违例记录（节点执行前缺必需输入时由装饰器写入，随结果导出供审计）
    contract_violations: List[Dict[str, Any]]
    state_invariant_check: Dict[str, Any]
    recovery_statistics: Dict[str, Any]
