"""把旧意图/槽位结果编译为结构化研究计划。"""

from __future__ import annotations

import re
from typing import Any, Iterable

from app.schemas.agent_schema import IntentResult, IntentType, SlotResult
from app.schemas.research_plan_schema import (
    ClarificationState,
    DeliverableType,
    ResearchConfidence,
    ResearchConstraints,
    ResearchOperation,
    ResearchRequestPlan,
    ResearchSemanticFrame,
    ResearchScope,
    TaskNode,
    TimeConstraint,
)


_RETRIEVAL_OPERATIONS = [
    ResearchOperation.QUERY_EXPANSION,
    ResearchOperation.SEARCH,
    ResearchOperation.METADATA_VERIFICATION,
    ResearchOperation.DEDUPLICATE,
    ResearchOperation.SCREEN,
]

_WRITING_OPERATIONS = [
    ResearchOperation.EXTRACT_PAPER_CARDS,
    ResearchOperation.CLASSIFY_PAPERS,
    ResearchOperation.SYNTHESIZE,
    ResearchOperation.WRITE,
    ResearchOperation.VALIDATE_CITATIONS,
]


def build_research_request_plan(
    user_query: str,
    intent_result: IntentResult,
    slots: SlotResult,
    search_plan: dict[str, Any] | None = None,
    selected_scope: dict[str, Any] | None = None,
    topic_interpretations: list[dict[str, Any]] | None = None,
    semantic_frame: ResearchSemanticFrame | dict[str, Any] | None = None,
) -> ResearchRequestPlan:
    """兼容适配器：保留旧节点输入，额外生成统一研究计划。"""

    search_plan = search_plan or {}
    selected_scope = selected_scope or {}
    interpretations = topic_interpretations or []
    if semantic_frame and not isinstance(semantic_frame, ResearchSemanticFrame):
        semantic_frame = ResearchSemanticFrame.model_validate(semantic_frame)
    intent = str(intent_result.intent or IntentType.GENERAL_QA.value)
    topic = str(search_plan.get("topic") or slots.topic or user_query).strip()
    deliverables = _deliverables_for(intent, slots.requested_sections)
    operations = _operations_for(intent, deliverables)
    if interpretations and not selected_scope:
        operations = _unique([ResearchOperation.SCOPE_DISAMBIGUATION, *operations])

    scope = ResearchScope(
        domain=selected_scope.get("label") or selected_scope.get("scope_id"),
        included_perspectives=_strings(
            selected_scope.get("include_terms")
            or selected_scope.get("included_perspectives")
        ),
        excluded_perspectives=_strings(
            selected_scope.get("exclude_terms")
            or selected_scope.get("excluded_perspectives")
        ),
        keywords=_strings(search_plan.get("keywords")),
    )

    clarification = ClarificationState()
    if interpretations and not selected_scope:
        clarification = ClarificationState(
            needed=True,
            slot="scope",
            question="该主题存在多个研究范围，请先确认需要覆盖的方向。",
            options=[
                str(item.get("label") or item.get("scope_id") or "")
                for item in interpretations
                if item
            ],
            reason="不同范围会显著改变检索语料和分类体系。",
        )

    time_constraint = _time_constraint(user_query, slots)
    assumptions = [time_constraint.assumption] if time_constraint.assumption else []
    result = ResearchRequestPlan(
        summary=user_query.strip(),
        topic=topic,
        scope=scope,
        operations=operations,
        deliverables=deliverables,
        constraints=ResearchConstraints(
            time=time_constraint,
            minimum_references=slots.required_reference_count,
            maximum_references=slots.generation_limit,
            retrieval_target=slots.retrieval_target,
            language=slots.language,
            citation_style=slots.citation_style,
        ),
        assumptions=assumptions,
        clarification=clarification,
        confidence=ResearchConfidence(
            overall=float(intent_result.confidence),
            scope=1.0 if selected_scope else (None if not interpretations else 0.5),
            uncertain_fields=["scope"] if clarification.needed else [],
        ),
        semantic_frame=semantic_frame,
        legacy_intent=intent,
    )
    result.task_graph = compile_task_graph(result)
    return result


def compile_task_graph(plan: ResearchRequestPlan) -> list[TaskNode]:
    """把允许的操作编译成稳定、无任意代码节点的依赖链。"""

    nodes: list[TaskNode] = []
    previous: str | None = None
    for operation in plan.operations:
        node_id = operation.value
        nodes.append(
            TaskNode(
                id=node_id,
                operation=operation,
                depends_on=[previous] if previous else [],
                affected_deliverables=list(plan.deliverables),
            )
        )
        previous = node_id
    return nodes


def _operations_for(
    intent: str,
    deliverables: list[DeliverableType],
) -> list[ResearchOperation]:
    if intent == IntentType.SEARCH_PAPERS.value:
        return list(_RETRIEVAL_OPERATIONS)
    writing_outputs = {
        DeliverableType.RESEARCH_BACKGROUND,
        DeliverableType.RESEARCH_STATUS,
        DeliverableType.RELATED_WORK,
        DeliverableType.LITERATURE_REVIEW,
        DeliverableType.NARRATIVE_REVIEW,
        DeliverableType.INTRODUCTION,
    }
    if intent in {
        IntentType.GENERATE_REVIEW.value,
        IntentType.GENERATE_RELATED_WORK.value,
        IntentType.GENERATE_INTRODUCTION.value,
    } or any(item in writing_outputs for item in deliverables):
        return [*_RETRIEVAL_OPERATIONS, *_WRITING_OPERATIONS]
    if intent == IntentType.GENERATE_REFERENCES.value:
        return [
            *_RETRIEVAL_OPERATIONS,
            ResearchOperation.WRITE,
            ResearchOperation.VALIDATE_CITATIONS,
        ]
    return []


def _deliverables_for(intent: str, requested_sections: Iterable[str]) -> list[DeliverableType]:
    mapping = {
        "background": DeliverableType.RESEARCH_BACKGROUND,
        "research_background": DeliverableType.RESEARCH_BACKGROUND,
        "research_status": DeliverableType.RESEARCH_STATUS,
        "related_work": DeliverableType.RELATED_WORK,
        "review": DeliverableType.NARRATIVE_REVIEW,
        "literature_review": DeliverableType.NARRATIVE_REVIEW,
        "narrative_review": DeliverableType.NARRATIVE_REVIEW,
        "introduction": DeliverableType.INTRODUCTION,
        "references": DeliverableType.REFERENCE_LIST,
        "reference_list": DeliverableType.REFERENCE_LIST,
        "paper_table": DeliverableType.PAPER_TABLE,
    }
    values = [mapping[item] for item in requested_sections if item in mapping]
    if intent == IntentType.SEARCH_PAPERS.value:
        values.append(DeliverableType.PAPER_LIST)
    elif intent == IntentType.GENERATE_REVIEW.value and not values:
        values.append(DeliverableType.NARRATIVE_REVIEW)
    elif intent == IntentType.GENERATE_RELATED_WORK.value:
        values.append(DeliverableType.RELATED_WORK)
    elif intent == IntentType.GENERATE_INTRODUCTION.value:
        values.append(DeliverableType.INTRODUCTION)
    elif intent == IntentType.GENERATE_REFERENCES.value:
        values.append(DeliverableType.REFERENCE_LIST)
    return _unique(values)


def _time_constraint(user_query: str, slots: SlotResult) -> TimeConstraint:
    relative = re.search(r"近\s*(?:[一二两三四五六七八九十\d]+)\s*年", user_query)
    absolute = re.search(r"(?:19|20)\d{2}\s*(?:[-~—到至])\s*(?:19|20)\d{2}", user_query)
    raw = relative.group(0) if relative else (absolute.group(0) if absolute else None)
    mode = "calendar_year" if relative else ("absolute" if absolute else "unspecified")
    assumption = None
    if relative:
        assumption = (
            "论文元数据按年份过滤，因此相对年份采用含当前年在内的自然年度口径；"
            f"本次明确执行 {slots.start_year}—{slots.end_year}。"
        )
    return TimeConstraint(
        raw_expression=raw,
        mode=mode,
        start_year=slots.start_year,
        end_year=slots.end_year,
        explicit=slots.year_range_explicit,
        assumption=assumption,
    )


def _strings(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if str(item).strip()]


def _unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
