"""研究对象—方法角色—终点目标语义解析测试。"""

import json

import pytest

from app.agent.example_retriever import retrieve_semantic_examples
from app.agent.research_semantic_parser import (
    _validate_llm_frame,
    derive_research_semantics,
    parse_research_semantics,
)
from app.agent.search_plan_builder import (
    build_semantic_search_branches,
    prioritized_branch_queries,
)
from app.schemas.research_plan_schema import (
    MethodRole,
    EvidenceRequirement,
    ResearchMethod,
    ResearchMode,
    ResearchSemanticFrame,
    SemanticItem,
    TerminalGoal,
)


def _domain() -> SemanticItem:
    return SemanticItem(
        id="application_domain", label="application domain",
        surface_text="应用领域", explicit=True, source="user_explicit", confidence=0.9,
    )


def _technical(role: MethodRole = MethodRole.NOT_SPECIFIED) -> ResearchMethod:
    return ResearchMethod(
        id="technical_method", label="technical method", surface_text="技术方法",
        explicit=True, source="user_explicit", confidence=0.9,
        category="technical", role=role,
    )


def _target() -> SemanticItem:
    return SemanticItem(
        id="analysis_target", label="analysis target", surface_text="分析目标",
        explicit=True, source="user_explicit", confidence=0.9,
    )


@pytest.mark.parametrize(
    ("domains", "methods", "targets", "expected_mode"),
    [
        ([], [_technical()], [], ResearchMode.TECHNOLOGY_ORIENTED),
        ([_domain()], [], [], ResearchMode.DOMAIN_ORIENTED),
        ([_domain()], [_technical()], [], ResearchMode.TECHNOLOGY_APPLIED_TO_DOMAIN),
        (
            [_domain()], [_technical()], [_target()],
            ResearchMode.TECHNOLOGY_ASSISTED_DOMAIN_ANALYSIS,
        ),
    ],
)
def test_research_mode_is_derived_from_structured_relations(
    domains, methods, targets, expected_mode,
):
    frame = derive_research_semantics(ResearchSemanticFrame(
        canonical_topic="任意主题",
        application_domains=domains,
        methods=methods,
        analysis_targets=targets,
    ))
    assert frame.research_mode == expected_mode


def test_method_role_is_attached_to_each_method_in_multistage_request():
    frame = derive_research_semantics(ResearchSemanticFrame(
        canonical_topic="跨阶段任务",
        application_domains=[_domain()],
        methods=[_technical(), _technical().model_copy(update={"id": "second_method"})],
        analysis_targets=[_target()],
        terminal_goal=TerminalGoal(type="domain_analysis", target="analysis_target"),
    ))

    assert frame.research_mode == ResearchMode.TECHNOLOGY_ASSISTED_DOMAIN_ANALYSIS
    assert {method.id for method in frame.methods} == {
        "technical_method", "second_method"
    }
    assert all(
        method.role == MethodRole.INTERMEDIATE_STEP
        for method in frame.methods
        if method.category == "technical"
    )
    assert frame.terminal_goal.type == "domain_analysis"
    assert frame.task_chain[-1] == "analysis_target"


def test_no_llm_returns_conservative_empty_semantics():
    frame = parse_research_semantics("课堂行为分析", "课堂行为分析", llm=None)

    assert frame.research_mode == ResearchMode.AMBIGUOUS
    assert frame.application_domains == []
    assert frame.methods == []
    assert "semantic_parser_unavailable" in frame.validation_warnings


def test_llm_scope_decision_is_not_overridden_by_a_local_domain_rule():
    class OverconfidentLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return """
            {
              "canonical_topic": "课堂行为分析",
              "application_domains": [],
              "research_objects": [],
              "methods": [],
              "analysis_targets": [],
              "clarification_needed": false,
              "confidence": {"overall": 0.9}
            }
            """

    frame = parse_research_semantics(
        "调研近三年课堂行为分析论文",
        "课堂行为分析",
        llm=OverconfidentLLM(),
    )

    assert frame.clarification_needed is False
    assert frame.confidence["source"] == 1.0


def test_llm_cannot_turn_broad_topic_into_explicit_method():
    class TopicAsMethodLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return json.dumps({
                "canonical_topic": "课堂行为分析",
                "application_domains": [{
                    "id": "education", "label": "education",
                    "surface_text": "课堂", "explicit": True, "confidence": 0.9,
                }],
                "research_objects": [{
                    "id": "classroom_behavior", "label": "classroom behavior",
                    "surface_text": "课堂行为分析", "explicit": True, "confidence": 0.9,
                }],
                "methods": [{
                    "id": "classroom_behavior_analysis",
                    "label": "classroom behavior analysis",
                    "surface_text": "课堂行为分析",
                    "category": "technical",
                    "explicit": True,
                    "inferred": False,
                    "confidence": 0.9,
                }],
                "research_actions": [],
                "analysis_targets": [],
                "clarification_needed": False,
            }, ensure_ascii=False)

    frame = parse_research_semantics(
        "调研近三年课堂行为分析论文，并生成研究背景和研究现状",
        "课堂行为分析",
        llm=TopicAsMethodLLM(),
    )

    assert frame.methods == []
    assert frame.clarification_needed is True
    assert "removed_ungrounded_method:classroom_behavior_analysis" in frame.validation_issues


def test_technology_only_queries_do_not_inherit_an_application_domain():
    frame = derive_research_semantics(ResearchSemanticFrame(
        canonical_topic="few-shot action recognition",
        methods=[
            _technical().model_copy(update={
                "id": "few_shot_learning", "label": "few-shot learning",
                "surface_text": "少样本",
            }),
            _technical().model_copy(update={
                "id": "action_recognition", "label": "action recognition",
                "surface_text": "动作识别",
            }),
        ],
        terminal_goal=TerminalGoal(type="method_analysis"),
    ))
    branches = build_semantic_search_branches(frame)
    queries = " ".join(prioritized_branch_queries(branches)).lower()

    assert frame.application_domains == []
    assert {branch.branch_type for branch in branches} == {"technical_method"}
    assert not any(term in queries for term in ("classroom", "student", "education"))


def test_cross_domain_request_generates_domain_technical_and_bridge_branches():
    frame = derive_research_semantics(ResearchSemanticFrame(
        canonical_topic="few-shot classroom behavior recognition",
        application_domains=[_domain()],
        research_objects=[SemanticItem(
            id="classroom_behavior", label="classroom behavior",
            surface_text="课堂行为", explicit=True, confidence=0.9,
        )],
        methods=[_technical().model_copy(update={
            "id": "few_shot_learning", "label": "few-shot learning",
            "surface_text": "少样本学习",
        })],
        terminal_goal=TerminalGoal(type="domain_task"),
    ))
    branches = build_semantic_search_branches(frame)
    branch_types = {branch.branch_type for branch in branches}

    assert branch_types == {
        "domain_foundation", "technical_method", "bridge_research"
    }
    bridge = next(branch for branch in branches if branch.branch_type == "bridge_research")
    assert "few-shot learning" in bridge.queries[0]
    assert "classroom behavior" in bridge.queries[0]


def test_downstream_request_adds_downstream_analysis_branch():
    frame = derive_research_semantics(ResearchSemanticFrame(
        canonical_topic="行为识别与互动分析",
        application_domains=[_domain()],
        research_objects=[SemanticItem(
            id="behavior", label="behavior", surface_text="行为",
            explicit=True, confidence=0.9,
        )],
        methods=[_technical()],
        analysis_targets=[_target()],
    ))
    branches = build_semantic_search_branches(frame)

    assert "downstream_analysis" in {branch.branch_type for branch in branches}


def test_explicit_classroom_pipeline_is_preserved_as_task_chain_and_focuses():
    frame = derive_research_semantics(ResearchSemanticFrame(
        canonical_topic="课堂行为分析",
        application_domains=[_domain()],
        research_objects=[SemanticItem(
            id="classroom_behavior", label="classroom behavior",
            surface_text="教师与学生行为", explicit=True, confidence=0.9,
        )],
        methods=[
            _technical().model_copy(update={
                "id": "action_recognition", "label": "action recognition",
                "surface_text": "自动识别",
            }),
            ResearchMethod(
                id="st_analysis", label="S-T analysis", surface_text="S-T分析法",
                category="analytical", explicit=True, confidence=0.9,
            ),
            ResearchMethod(
                id="lag_sequential_analysis", label="lag sequential analysis",
                surface_text="滞后分析法", category="analytical",
                explicit=True, confidence=0.9,
            ),
        ],
        analysis_targets=[SemanticItem(
            id="teaching_structure_and_interaction", label="teaching structure and interaction",
            surface_text="教学结构与师生互动解释", explicit=True, confidence=0.9,
        )],
        task_chain=[
            "teacher_student_behavior_recognition",
            "automatic_behavior_coding",
            "st_or_lag_sequential_analysis",
            "teaching_structure_and_interaction_interpretation",
        ],
        required_focuses=[
            "教师与学生行为自动识别", "自动行为编码",
            "S-T分析法或滞后序列分析法", "教学结构与师生互动解释",
        ],
    ))

    assert frame.task_chain == [
        "teacher_student_behavior_recognition",
        "automatic_behavior_coding",
        "st_or_lag_sequential_analysis",
        "teaching_structure_and_interaction_interpretation",
    ]
    assert frame.required_focuses == [
        "教师与学生行为自动识别",
        "自动行为编码",
        "S-T分析法或滞后序列分析法",
        "教学结构与师生互动解释",
    ]
    assert {method.id for method in frame.methods} >= {
        "action_recognition", "st_analysis", "lag_sequential_analysis"
    }

    branches = build_semantic_search_branches(frame)
    branch_types = {branch.branch_type for branch in branches}
    queries = " ".join(prioritized_branch_queries(branches)).lower()
    assert {"analytical_method", "pipeline_bridge"} <= branch_types
    assert "s-t analysis" in queries
    assert "lag sequential analysis" in queries


def test_or_with_etc_marks_analysis_methods_as_open_alternatives():
    frame = ResearchSemanticFrame(
        canonical_topic="任意领域行为分析",
        evidence_requirements=[EvidenceRequirement(
            requirement_id="analytical_method:alternatives",
            label="适用的分析方法",
            evidence_role="analytical_method",
            aliases=["S-T分析法", "滞后序列分析法"],
            source_ids=["st_analysis", "lag_sequential_analysis"],
            selection_mode="open_any",
            exact_method_required=False,
        )],
        task_chain=["automatic_recognition", "automatic_coding", "downstream_domain_analysis"],
        required_focuses=["自动识别", "自动编码", "适用的分析方法"],
    )
    analytical = [
        item for item in frame.evidence_requirements
        if item.evidence_role == "analytical_method"
    ]

    assert len(analytical) == 1
    assert analytical[0].selection_mode == "open_any"
    assert analytical[0].exact_method_required is False
    assert set(analytical[0].source_ids) == {"st_analysis", "lag_sequential_analysis"}
    assert frame.required_focuses[2].startswith("适用的分析方法")
    assert frame.task_chain[-1] == "downstream_domain_analysis"
    assert "教学结构与师生互动解释" not in frame.required_focuses


def test_or_in_research_objects_does_not_change_and_joined_analysis_methods():
    frame = ResearchSemanticFrame(
        canonical_topic="任意领域行为分析",
        evidence_requirements=[
            EvidenceRequirement(
                requirement_id="analytical_method:st", label="S-T分析法",
                evidence_role="analytical_method", aliases=["S-T分析法"],
                source_ids=["st_analysis"], selection_mode="all",
            ),
            EvidenceRequirement(
                requirement_id="analytical_method:lag", label="滞后序列分析法",
                evidence_role="analytical_method", aliases=["滞后序列分析法"],
                source_ids=["lag_sequential_analysis"], selection_mode="all",
            ),
        ],
        required_focuses=["自动识别", "自动编码", "S-T分析法", "滞后序列分析法"],
    )
    analytical = [
        item for item in frame.evidence_requirements
        if item.evidence_role == "analytical_method"
    ]

    assert len(analytical) == 2
    assert all(item.selection_mode == "all" for item in analytical)
    assert "S-T分析法" in frame.required_focuses
    assert "滞后序列分析法" in frame.required_focuses


def test_pipeline_derivation_is_generic_and_does_not_leak_classroom_terms():
    frame = ResearchSemanticFrame(
        canonical_topic="康复训练行为分析",
        task_chain=["action_recognition", "automatic_behavior_coding", "lag_sequential_analysis"],
        required_focuses=["康复训练动作识别", "训练行为编码", "训练质量分析"],
    )

    assert frame.task_chain[:3] == [
        "action_recognition",
        "automatic_behavior_coding",
        "lag_sequential_analysis",
    ]
    serialized = " ".join([*frame.task_chain, *frame.required_focuses]).lower()
    assert "classroom" not in serialized
    assert "教师" not in serialized
    assert "学生" not in serialized


class ImplicitFewShotLLM:
    def complete(self, prompt: str, **kwargs) -> str:
        assert "人工审核的相似回归案例" in prompt
        return json.dumps({
            "canonical_topic": "有限标注下的学生行为识别",
            "application_domains": [
                {"id": "education", "label": "education", "surface_text": "学生", "explicit": True, "confidence": 0.95}
            ],
            "research_objects": [
                {"id": "student_behavior", "label": "student behavior", "surface_text": "学生举手行为", "explicit": True, "confidence": 0.95}
            ],
            "methods": [
                {
                    "id": "few_shot_learning", "label": "few-shot learning",
                    "explicit": False, "inferred": True, "source": "llm_inference",
                    "inference_basis": "每个动作只有5段视频", "confidence": 0.63,
                    "category": "technical",
                },
                {
                    "id": "action_recognition", "label": "action recognition",
                    "surface_text": "识别学生举手行为", "explicit": True,
                    "confidence": 0.95, "category": "technical",
                },
            ],
            "research_actions": [], "analysis_targets": [],
            "terminal_goal": {"type": "domain_recognition", "target": "student_behavior"},
        }, ensure_ascii=False)


def test_inferred_method_is_preserved_but_only_creates_exploratory_constraints():
    query = "每个动作只有5段视频，想识别学生举手行为"
    frame = parse_research_semantics(query, "有限标注学生行为识别", llm=ImplicitFewShotLLM())
    inferred = next(method for method in frame.methods if method.id == "few_shot_learning")
    branches = build_semantic_search_branches(frame)
    technical = next(branch for branch in branches if branch.branch_type == "technical_method")

    assert inferred.explicit is False
    assert inferred.inferred is True
    assert inferred.inference_basis == "每个动作只有5段视频"
    assert technical.constraint_level == "exploratory"
    assert "few-shot learning" not in technical.required_concepts[0]


class TemporalShellRequirementLLM:
    """复刻 2026-08-21 少样本动作识别会话的 LLM 输出结构。

    查询动词被拆成 research_action，并派生“时间窗 + 动词壳”的 evidence
    requirement；动词用词表外的“梳理”，证明判据是实体映射而非动词枚举。
    """

    def complete(self, prompt: str, **kwargs) -> str:
        return json.dumps({
            "canonical_topic": "少样本动作识别研究综述",
            "application_domains": [],
            "research_objects": [{
                "id": "human_action", "label": "human action",
                "surface_text": "动作", "explicit": True,
            }],
            "methods": [
                {
                    "id": "few_shot_learning", "label": "few-shot learning",
                    "surface_text": "few-shot learning", "explicit": True,
                    "category": "technical",
                },
                {
                    "id": "action_recognition", "label": "action recognition",
                    "surface_text": "动作识别", "explicit": True,
                    "category": "technical",
                },
            ],
            "research_actions": [{
                "id": "literature_survey", "label": "literature survey",
                "surface_text": "调研", "explicit": True,
            }],
            "analysis_targets": [],
            "required_focuses": ["少样本学习", "动作识别", "近五年文献梳理证据"],
            "evidence_requirements": [
                {
                    "requirement_id": "method:few_shot_learning",
                    "label": "少样本学习相关证据",
                    "evidence_role": "主要研究目标",
                    "aliases": ["少样本学习", "few-shot learning"],
                    "source_ids": ["few_shot_learning"],
                },
                {
                    "requirement_id": "method:action_recognition",
                    "label": "动作识别相关证据",
                    "evidence_role": "主要研究目标",
                    "aliases": ["动作识别", "action recognition"],
                    "source_ids": ["action_recognition"],
                },
                {
                    "requirement_id": "action:literature_survey",
                    "label": "近五年文献梳理证据",
                    "evidence_role": "文献调研范围约束",
                    "aliases": ["文献梳理", "literature survey"],
                    "source_ids": ["literature_survey"],
                    "minimum_direct_sources": 60,
                },
            ],
            "terminal_goal": {
                "type": "deliverable_generation",
                "target": "research_background_and_status",
            },
        }, ensure_ascii=False)


def test_temporal_shell_requirement_dropped_by_entity_mapping_not_verb_list():
    frame = parse_research_semantics(
        "调研近五年少样本动作识别论文，并生成研究背景和研究现状，引用论文不少于60篇",
        "少样本动作识别",
        llm=TemporalShellRequirementLLM(),
    )

    # 动词壳要求（词表外的“梳理”）按实体映射判为检索范围，不进入覆盖检查
    assert [item.requirement_id for item in frame.evidence_requirements] == [
        "method:action_recognition"
    ]
    assert "近五年文献梳理证据" not in frame.required_focuses
    assert frame.required_focuses == ["少样本学习", "动作识别"]

    # few_shot_learning 方法无原文依据被 grounding 移除后，其 requirement
    # 随源实体一并清理，不再留下匹配不到论文的孤儿假门禁
    assert "removed_ungrounded_method:few_shot_learning" in frame.validation_issues
    assert "dropped_orphan_requirement:method:few_shot_learning" in frame.validation_issues


class LeakingDomainLLM:
    def complete(self, prompt: str, **kwargs) -> str:
        return json.dumps({
            "canonical_topic": "few-shot action recognition",
            "application_domains": [
                {
                    "id": "education", "label": "education", "surface_text": "课堂",
                    "explicit": True, "confidence": 0.99,
                }
            ],
            "research_objects": [],
            "methods": [
                {
                    "id": "few_shot_learning", "label": "few-shot learning",
                    "surface_text": "少样本", "explicit": True, "confidence": 0.99,
                    "category": "technical",
                },
                {
                    "id": "action_recognition", "label": "action recognition",
                    "surface_text": "动作识别", "explicit": True, "confidence": 0.99,
                    "category": "technical",
                },
            ],
            "research_actions": [], "analysis_targets": [],
            "terminal_goal": {"type": "method_analysis", "target": "action_recognition"},
        }, ensure_ascii=False)


def test_ungrounded_domain_from_llm_or_examples_is_removed():
    frame = parse_research_semantics(
        "调研少样本动作识别", "少样本动作识别", llm=LeakingDomainLLM()
    )

    assert frame.application_domains == []
    assert frame.research_mode == ResearchMode.TECHNOLOGY_ORIENTED
    assert "removed_ungrounded_application_domain:education" in frame.validation_issues


class ObjectScopeAmbiguityLLM:
    """模拟 LLM 将 scope_ambiguities/secondary_goals 元素写成字典而非字符串。"""

    def complete(self, prompt: str, **kwargs) -> str:
        return json.dumps({
            "canonical_topic": "课堂行为分析范式待澄清",
            "application_domains": [
                {"id": "education", "label": "education", "surface_text": "课堂", "explicit": True, "confidence": 0.95}
            ],
            "research_objects": [],
            "methods": [],
            "research_actions": [], "analysis_targets": [],
            "secondary_goals": [
                {"id": "goal_engagement", "description": "同时评估学生参与度、情绪等"}
            ],
            "scope_ambiguities": [
                {"id": "ambiguity_behavior_scope", "description": "研究范式未明确（参与度、情绪等）"}
            ],
            "terminal_goal": {"type": "domain_understanding"},
        }, ensure_ascii=False)


def test_dict_shaped_scope_ambiguities_and_secondary_goals_are_stringified():
    frame = parse_research_semantics(
        "课堂行为分析", "课堂行为分析", llm=ObjectScopeAmbiguityLLM()
    )

    assert frame.scope_ambiguities == ["研究范式未明确（参与度、情绪等）"]
    assert frame.secondary_goals == ["同时评估学生参与度、情绪等"]


def test_no_llm_does_not_create_explicit_methods_from_a_local_term_table():
    frame = parse_research_semantics(
        "比较元学习与度量学习在动作识别中的性能",
        "动作识别方法比较",
        llm=None,
    )

    assert frame.methods == []
    assert "semantic_parser_unavailable" in frame.validation_warnings


def test_deliverable_terms_are_isolated_from_research_focus_and_search_branches():
    class DeliverableLeakLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return json.dumps({
                "canonical_topic": "少样本动作识别研究现状与研究背景",
                "application_domains": [],
                "research_objects": [{
                    "id": "human_action", "label": "human action",
                    "surface_text": "动作", "explicit": True,
                }],
                "methods": [{
                    "id": "action_recognition", "label": "action recognition",
                    "surface_text": "动作识别", "explicit": True,
                    "category": "technical",
                }],
                "research_actions": [
                    {"id": "status_generation", "label": "research status generation",
                     "surface_text": "生成研究现状", "explicit": True},
                ],
                "analysis_targets": [],
                "terminal_goal": {"type": "deliverable_generation", "target": "research_status"},
                "task_chain": ["action_recognition", "research_status_generation"],
                "required_focuses": ["动作识别", "研究现状", "研究背景"],
                "evidence_requirements": [{
                    "requirement_id": "method:action_recognition",
                    "label": "动作识别证据", "evidence_role": "method",
                    "aliases": ["动作识别", "action recognition"],
                    "source_ids": ["action_recognition"],
                }, {
                    "requirement_id": "deliverable:status",
                    "label": "研究现状生成证据", "evidence_role": "deliverable",
                    "aliases": ["研究现状"], "source_ids": ["status_generation"],
                }],
            }, ensure_ascii=False)

    frame = parse_research_semantics(
        "调研少样本动作识别论文，并生成研究背景和研究现状",
        "少样本动作识别",
        deliverables=["background", "research_status"],
        llm=DeliverableLeakLLM(),
    )
    branches = build_semantic_search_branches(frame)
    serialized = " ".join(prioritized_branch_queries(branches) + frame.required_focuses).lower()

    assert "研究背景" not in serialized
    assert "研究现状" not in serialized
    assert frame.required_focuses == ["动作识别"]
    assert frame.methods[0].id == "action_recognition"
    assert all("deliverable" not in item.requirement_id for item in frame.evidence_requirements)


def test_reviewed_case_retrieval_returns_similar_structure_without_mutating_request():
    cases = retrieve_semantic_examples("每个类别只有几段视频，需要识别学生动作", top_k=3)

    assert cases
    assert any("implicit_few_shot" in case["error_tags"] for case in cases)
    assert all("expected_frame" in case for case in cases)


def test_temporal_qualifier_entities_are_dropped_from_llm_frame():
    raw = {
        "canonical_topic": "少样本动作识别",
        "research_objects": [
            {
                "id": "fsar", "label": "few-shot action recognition",
                "surface_text": "少样本动作识别", "explicit": True,
            },
            {
                # 模型把“近五年”提升成研究对象的实际案例（2026-08 运行日志）
                "id": "recent_literature", "label": "recent literature",
                "surface_text": "近五年文献调研", "explicit": True,
            },
        ],
        "required_focuses": ["近五年文献调研证据", "跨域泛化"],
        "evidence_requirements": [
            {
                "requirement_id": "time:recent", "label": "近五年文献调研证据",
                "evidence_role": "recency",
                "aliases": ["近五年", "recent five years"],
                "source_ids": ["recent_literature"],
            },
            {
                "requirement_id": "perception:fsar", "label": "少样本动作识别",
                "evidence_role": "perception",
                "aliases": ["few-shot action recognition", "少样本动作识别"],
                "source_ids": ["fsar"],
            },
        ],
    }

    frame = _validate_llm_frame(raw, "少样本动作识别")

    # 时间限定词不进入实体表：覆盖检查不再去论文正文里词面匹配“近五年”
    assert {item.id for item in frame.research_objects} == {"fsar"}
    assert frame.required_focuses == ["跨域泛化"]
    assert {item.requirement_id for item in frame.evidence_requirements} == {
        "perception:fsar"
    }


def test_temporal_prefixed_domain_content_survives_frame_normalization():
    raw = {
        "canonical_topic": "近五年少样本动作识别",
        "research_objects": [{
            "id": "fsar", "label": "few-shot action recognition",
            "surface_text": "近五年少样本动作识别", "explicit": True,
        }],
    }

    frame = _validate_llm_frame(raw, "少样本动作识别")

    # “近五年X”剥除时间词后仍有领域内容：保留
    assert [item.id for item in frame.research_objects] == ["fsar"]


class DeliverableActionRequirementLLM:
    """复刻 2026-08-22 会话：动作被派生成需要论文直接证据的要求。"""

    def complete(self, prompt: str, **kwargs) -> str:
        return json.dumps({
            "canonical_topic": "少样本动作识别研究综述",
            "application_domains": [],
            "research_objects": [{
                "id": "human_action", "label": "human action",
                "surface_text": "动作", "explicit": True,
            }],
            "methods": [{
                "id": "action_recognition", "label": "action recognition",
                "surface_text": "动作识别", "explicit": True, "category": "technical",
            }],
            "research_actions": [
                {"id": "literature_survey", "label": "literature survey",
                 "surface_text": "调研", "explicit": True},
                {"id": "background_generation", "label": "background generation",
                 "surface_text": "生成研究背景", "explicit": True},
                {"id": "status_generation", "label": "status generation",
                 "surface_text": "生成研究现状", "explicit": True},
            ],
            "analysis_targets": [],
            "evidence_requirements": [
                {"requirement_id": "method:action_recognition",
                 "label": "动作识别相关证据", "evidence_role": "主要研究目标",
                 "aliases": ["动作识别", "action recognition"],
                 "source_ids": ["action_recognition"]},
                {"requirement_id": "action:literature_survey",
                 "label": "少样本动作识别文献调研证据", "evidence_role": "调研范围",
                 "aliases": ["文献调研"], "source_ids": ["literature_survey"],
                 "minimum_direct_sources": 60},
                {"requirement_id": "action:background_generation",
                 "label": "研究背景生成证据", "evidence_role": "交付",
                 "aliases": ["研究背景"], "source_ids": ["background_generation"]},
                {"requirement_id": "action:status_generation",
                 "label": "研究现状生成证据", "evidence_role": "交付",
                 "aliases": ["研究现状"], "source_ids": ["status_generation"]},
            ],
            "terminal_goal": {
                "type": "deliverable_generation",
                "target": "research_background_and_status",
            },
        }, ensure_ascii=False)


def test_research_action_requirements_do_not_enter_evidence_coverage():
    """研究动作派生的要求不得作为直接证据判据。

    没有论文会声明自己“生成研究背景”，把这类要求送进覆盖检查会让命中数
    恒为 0 而必然误拦（2026-08-22 会话：少样本动作识别文献调研证据 /
    研究背景生成证据 / 研究现状生成证据）。按来源实体类型结构化排除，
    因此换任何动词或交付物名称都成立。
    """
    from app.agent.evidence_roles import evidence_coverage

    frame = parse_research_semantics(
        "调研近五年少样本动作识别论文，并生成研究背景和研究现状，引用论文不少于60篇",
        "少样本动作识别",
        llm=DeliverableActionRequirementLLM(),
    )

    requirement_ids = [item.requirement_id for item in frame.evidence_requirements]
    assert requirement_ids == ["method:action_recognition"]

    # 动作本身仍保留在语义帧中供规划与检索使用，只是不作为证据判据
    assert {item.id for item in frame.research_actions} >= {"literature_survey"}

    cards = [{
        "paper_id": "p1",
        "title": "Few-shot action recognition via metric alignment",
        "research_problem": "动作识别在少样本条件下的判别性不足",
        "method": "action recognition with metric learning",
        "year": 2024,
    }]
    coverage = evidence_coverage(frame.model_dump(mode="json"), cards)
    assert coverage["ready"] is True
    assert coverage["missing_focuses"] == []


class PerRequirementQuotaLLM:
    """复刻 2026-08-29 会话：整篇引用下限被拆成逐要求配额。

    用户只说了一次"引用论文不少于40篇"，模型却给七条要求分别写上
    40/20/20/10/10/5/10（合计 115 篇），于是 60 篇证据池必然七项全缺，
    正文生成被整体阻断。
    """

    def complete(self, prompt: str, **kwargs) -> str:
        return json.dumps({
            "canonical_topic": "课堂行为分析",
            "application_domains": [],
            "research_objects": [
                {"id": "classroom_behavior", "label": "classroom behavior",
                 "surface_text": "课堂行为", "explicit": True, "confidence": 0.95},
                {"id": "teaching_interaction", "label": "teaching interaction",
                 "surface_text": "教学互动", "explicit": True, "confidence": 0.95},
            ],
            "methods": [
                {"id": "teaching_observation", "label": "teaching observation",
                 "surface_text": "教学观察", "explicit": True,
                 "category": "methodological", "confidence": 0.9},
            ],
            "research_actions": [],
            "analysis_targets": [],
            "evidence_requirements": [
                {"requirement_id": "object:classroom_behavior",
                 "label": "课堂行为分析研究证据", "evidence_role": "perception",
                 "aliases": ["课堂行为", "classroom behavior"],
                 "source_ids": ["classroom_behavior"],
                 "minimum_direct_sources": 40},
                {"requirement_id": "object:teaching_interaction",
                 "label": "教学互动研究证据", "evidence_role": "interpretation",
                 "aliases": ["教学互动", "teaching interaction"],
                 "source_ids": ["teaching_interaction"],
                 "minimum_direct_sources": 20},
                {"requirement_id": "method:teaching_observation",
                 "label": "教学观察方法证据", "evidence_role": "analytical_method",
                 "aliases": ["教学观察", "teaching observation"],
                 "source_ids": ["teaching_observation"],
                 "minimum_direct_sources": 10},
            ],
            "terminal_goal": {"type": "literature_review", "target": "课堂行为分析研究现状"},
        }, ensure_ascii=False)


def test_per_requirement_quota_without_textual_basis_falls_back_to_one():
    """无原文依据的逐要求配额回落为 1，整篇引用下限不被重复消费。

    用户原文只有一处数量（"引用论文不少于40篇"），它约束的是整份正文的
    引用总量（由 required_reference_count 单独校验），不是每一条证据要求
    各自都要 40/20/10 篇。判据与学科无关：数字必须紧跟在该要求自身的
    概念之后、且中间没有分句标点，才算用户为该概念指定的配额。
    """
    frame = parse_research_semantics(
        "调研近三年课堂行为分析论文，并生成研究背景和研究现状，引用论文不少于40篇",
        "课堂行为分析",
        llm=PerRequirementQuotaLLM(),
    )

    minimums = {
        item.requirement_id: item.minimum_direct_sources
        for item in frame.evidence_requirements
    }
    assert minimums and set(minimums.values()) == {1}
    assert any(
        issue.startswith("clamped_ungrounded_minimum_sources:")
        for issue in frame.validation_issues
    )


def test_per_requirement_quota_kept_when_user_states_it_next_to_the_concept():
    """用户为某个概念直接指定的配额必须保留，不能被一并清零。"""
    frame = parse_research_semantics(
        "调研课堂行为分析论文，其中教学互动至少15篇，引用论文不少于40篇",
        "课堂行为分析",
        llm=PerRequirementQuotaLLM(),
    )

    minimums = {
        item.requirement_id: item.minimum_direct_sources
        for item in frame.evidence_requirements
    }
    assert minimums["object:teaching_interaction"] == 15
    # 同句里没有自己配额的要求仍回落为 1，不继承邻居的数量
    assert minimums["object:classroom_behavior"] == 1
