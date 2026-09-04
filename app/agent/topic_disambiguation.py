"""研究主题消歧与澄清范围解析。"""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.json_utils import parse_json_object as _parse_json_object_robust

from app.agent.intent import recognize_intent
from app.agent.slot_extractor import extract_slots
from app.core.logger import get_logger
from app.core.config import get_settings
from app.schemas.agent_schema import IntentType, TopicAmbiguityResult, TopicScope
from app.utils.date_utils import current_year as get_current_year

logger = get_logger(__name__)

_RESEARCH_INTENTS = {
    IntentType.GENERATE_REVIEW.value,
    IntentType.GENERATE_RELATED_WORK.value,
    IntentType.GENERATE_INTRODUCTION.value,
    IntentType.SEARCH_PAPERS.value,
    IntentType.FIND_TRENDS.value,
}


def analyze_topic_ambiguity(
    user_query: str,
    llm=None,
    current_year: int | None = None,
) -> dict[str, Any]:
    """构建研究请求并判断是否需要在检索前向用户澄清主题范围。"""
    intent = recognize_intent(user_query, llm=llm).intent
    year = current_year or get_current_year()
    slots = extract_slots(user_query, intent, llm=llm, current_year=year)
    research_request = {
        "task_type": intent,
        **slots.model_dump(),
        "original_query": user_query,
    }
    from app.agent.research_semantic_parser import parse_research_semantics

    semantic_frame = parse_research_semantics(
        user_query=user_query,
        topic=slots.topic or user_query,
        deliverables=slots.requested_sections,
        llm=llm,
    )
    scope_is_complete = _has_complete_explicit_scope(semantic_frame)
    if scope_is_complete and semantic_frame.clarification_needed:
        semantic_frame = semantic_frame.model_copy(update={
            "clarification_needed": False,
            "clarification_question": None,
            "validation_warnings": list(dict.fromkeys([
                *semantic_frame.validation_warnings,
                "scope_clarification_suppressed_for_complete_explicit_semantics",
            ])),
        })
    research_request["semantic_frame"] = semantic_frame.model_dump(mode="json")
    fallback = _semantic_fallback_ambiguity(semantic_frame, slots.topic or user_query)

    if intent not in _RESEARCH_INTENTS or not slots.topic or llm is None:
        return _build_analysis_response(research_request, fallback)

    # 明确给出技术方法且不存在语义歧义时，不再额外调用一次消歧模型。
    # 这既避免把清晰任务改写成宽泛问题，也缩短检索前等待时间。
    if scope_is_complete:
        return _build_analysis_response(research_request, fallback)

    try:
        from app.prompt.topic_disambiguation import TOPIC_DISAMBIGUATION_PROMPT

        constraints = {
            key: research_request.get(key)
            for key in (
                "start_year",
                "end_year",
                "required_reference_count",
                "requested_sections",
                "language",
                "citation_style",
            )
        }
        constraints["research_semantic_frame"] = semantic_frame.model_dump(mode="json")
        prompt = TOPIC_DISAMBIGUATION_PROMPT.format(
            user_query=user_query,
            topic=slots.topic,
            constraints_json=json.dumps(constraints, ensure_ascii=False),
        )
        raw = llm.complete(
            prompt,
            # 部分 reasoning 模型在 json_object 模式下只返回推理内容而正文为空；
            # 使用文本模式后由本模块做严格 JSON 解析，兼容性更好。
            response_format="text",
            temperature=0.0,
            timeout=get_settings().llm_control_plane_timeout,
            retry_empty=False,
            operation="topic_disambiguation",
        )
        data = _parse_json_object(raw)
        ambiguity = TopicAmbiguityResult(**data)
    except Exception as exc:
        logger.warning("Topic ambiguity analysis failed, using semantic fallback: %s", exc)
        ambiguity = fallback

    ambiguity = _normalize_ambiguity(ambiguity)
    # 技术、对象和终点关系已经明确时，不允许二次消歧把清晰任务改成宽泛领域追问。
    if scope_is_complete:
        ambiguity = TopicAmbiguityResult(
            ambiguous=False,
            confidence=max(ambiguity.confidence, 0.8),
            reason="研究对象、方法角色和终点目标已经明确",
            recommended_strategy="single_scope",
        )
    return _build_analysis_response(research_request, ambiguity)


def _has_complete_explicit_scope(semantic_frame) -> bool:
    """判断核心检索边界是否已由用户显式给全，不依赖领域词表。"""
    explicit_objects = [item for item in semantic_frame.research_objects if item.explicit]
    explicit_methods = [item for item in semantic_frame.methods if item.explicit]
    goal_is_defined = bool(
        str(semantic_frame.terminal_goal.type or "") not in {"", "unspecified"}
        or semantic_frame.task_chain
        or semantic_frame.analysis_targets
    )
    return bool(
        explicit_objects
        and explicit_methods
        and goal_is_defined
        and semantic_frame.research_mode.value != "ambiguous"
        and not semantic_frame.validation_issues
    )


def _semantic_fallback_ambiguity(semantic_frame, topic: str) -> TopicAmbiguityResult:
    """模型不可用时不凭固定模板编造候选研究范围。"""
    if not semantic_frame.clarification_needed:
        return TopicAmbiguityResult(
            ambiguous=False,
            confidence=0.0,
            reason="消歧模型不可用，未生成额外范围假设",
            recommended_strategy="single_scope",
        )
    return TopicAmbiguityResult(
        ambiguous=False,
        confidence=0.0,
        reason="消歧模型不可用，不能可靠构造互斥候选范围",
        recommended_strategy="single_scope",
    )


def _normalize_ambiguity(result: TopicAmbiguityResult) -> TopicAmbiguityResult:
    """拒绝结构不完整的歧义判断，并规范默认选项。"""
    unique: list[TopicScope] = []
    seen: set[str] = set()
    for scope in result.scopes:
        scope_id = re.sub(r"[^a-z0-9_]+", "_", scope.scope_id.lower()).strip("_")
        if not scope_id or scope_id in seen:
            continue
        seen.add(scope_id)
        unique.append(scope.model_copy(update={"scope_id": scope_id}))
    result.scopes = unique[:4]

    if not result.ambiguous or len(result.scopes) < 2:
        return TopicAmbiguityResult(
            ambiguous=False,
            confidence=result.confidence,
            reason=result.reason,
            recommended_strategy="single_scope",
            scopes=result.scopes,
        )

    valid_ids = {scope.scope_id for scope in result.scopes}
    if result.default_scope_id not in valid_ids:
        result.default_scope_id = result.scopes[0].scope_id
    if result.recommended_strategy not in {"ask_user", "multi_branch", "single_scope"}:
        result.recommended_strategy = "ask_user"
    if result.recommended_strategy == "ask_user" and not result.question:
        labels = "、".join(scope.label for scope in result.scopes)
        result.question = f"“该主题”存在多种研究范围。你希望选择：{labels}？"
    return result


def _build_analysis_response(
    research_request: dict[str, Any],
    ambiguity: TopicAmbiguityResult,
) -> dict[str, Any]:
    needs_clarification = bool(
        ambiguity.ambiguous
        and ambiguity.confidence >= 0.7
        and ambiguity.recommended_strategy == "ask_user"
        and len(ambiguity.scopes) >= 2
    )
    return {
        "research_request": research_request,
        "ambiguity": ambiguity.model_dump(),
        "needs_clarification": needs_clarification,
    }


def resolve_scope(clarification: dict[str, Any], answer: str) -> dict[str, Any] | None:
    """将 scope_id、序号或用户可读名称解析为一个范围。"""
    scopes = clarification.get("scopes") or []
    normalized = str(answer or "").strip().lower()
    if not normalized:
        return None

    if normalized.isdigit():
        index = int(normalized) - 1
        return scopes[index] if 0 <= index < len(scopes) else None

    for scope in scopes:
        scope_id = str(scope.get("scope_id") or "").lower()
        label = str(scope.get("label") or "").lower()
        if normalized == scope_id or normalized == label:
            return scope
    for scope in scopes:
        label = str(scope.get("label") or "").lower()
        if label and (label in normalized or normalized in label):
            return scope
    return None


def resolve_scope_conversational(
    clarification: dict[str, Any],
    answer: str,
    llm=None,
) -> dict[str, Any]:
    """理解自由文本范围回答；不明确时返回下一条单一疑问句。"""
    scopes = clarification.get("scopes") or []
    direct = resolve_scope(clarification, answer)
    if direct:
        return {
            "selected_scope": direct,
            "needs_clarification": False,
            "question": None,
        }

    if not scopes:
        return {
            "selected_scope": None,
            "needs_clarification": True,
            "question": "你希望这次研究重点关注什么对象、问题和分析视角？",
        }

    if llm is not None and str(answer or "").strip():
        try:
            from app.prompt.topic_disambiguation import SCOPE_ANSWER_RESOLUTION_PROMPT

            raw = llm.complete(
                SCOPE_ANSWER_RESOLUTION_PROMPT.format(
                    question=clarification.get("question") or "请说明希望采用的研究范围。",
                    scopes_json=json.dumps(scopes, ensure_ascii=False),
                    answer=answer,
                ),
                response_format="text",
                temperature=0.0,
                timeout=get_settings().llm_control_plane_timeout,
                retry_empty=False,
                operation="scope_answer_resolution",
            )
            data = _parse_json_object(raw)
            valid = {
                str(scope.get("scope_id")): scope
                for scope in scopes
                if scope.get("scope_id")
            }
            matched_ids = [
                str(scope_id)
                for scope_id in (data.get("matched_scope_ids") or [])
                if str(scope_id) in valid
            ]
            if matched_ids and not data.get("needs_clarification"):
                selected_scopes = [valid[scope_id] for scope_id in matched_ids]
                selected = (
                    selected_scopes[0]
                    if len(selected_scopes) == 1
                    else _combine_selected_scopes(selected_scopes)
                )
                return {
                    "selected_scope": selected,
                    "needs_clarification": False,
                    "question": None,
                }
            question = str(data.get("question") or "").strip()
            if question:
                return {
                    "selected_scope": None,
                    "needs_clarification": True,
                    "question": _single_question(question),
                }
        except Exception as exc:
            logger.warning("Free-text scope resolution failed: %s", exc)

    overlap_match = _resolve_scope_by_answer_signals(scopes, answer)
    if overlap_match:
        return {
            "selected_scope": overlap_match,
            "needs_clarification": False,
            "question": None,
        }

    return {
        "selected_scope": None,
        "needs_clarification": True,
        "question": _fallback_scope_question(scopes),
    }


def _resolve_scope_by_answer_signals(
    scopes: list[dict[str, Any]], answer: str
) -> dict[str, Any] | None:
    """模型不可用时只做候选文本重叠匹配，不创造新范围。"""
    ranked = sorted(
        ((_scope_answer_overlap(scope, answer), scope) for scope in scopes),
        key=lambda item: item[0],
        reverse=True,
    )
    if not ranked or ranked[0][0] <= 0:
        return None
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def _scope_answer_overlap(scope: dict[str, Any], answer: str) -> int:
    scope_text = " ".join([
        str(scope.get("label") or ""),
        str(scope.get("description") or ""),
        " ".join(str(term) for term in scope.get("include_terms") or []),
    ]).lower()
    answer_text = str(answer or "").lower()
    scope_tokens = set(re.findall(r"[a-z][a-z0-9-]{2,}|[\u4e00-\u9fff]{2}", scope_text))
    answer_tokens = set(re.findall(r"[a-z][a-z0-9-]{2,}|[\u4e00-\u9fff]{2}", answer_text))
    return len(scope_tokens & answer_tokens)


def _combine_selected_scopes(scopes: list[dict[str, Any]]) -> dict[str, Any]:
    """把用户明确要求的多个范围合并为可执行的交叉范围。"""
    return {
        "scope_id": "combined_" + "_".join(str(scope.get("scope_id")) for scope in scopes),
        "label": "交叉综合：" + " + ".join(str(scope.get("label") or "") for scope in scopes),
        "description": "按用户回答同时覆盖多个研究范围，并在生成时比较其分析单位和证据类型。",
        "include_terms": list(dict.fromkeys(
            term for scope in scopes for term in (scope.get("include_terms") or [])
        )),
        "exclude_terms": [],
        "seed_queries": list(dict.fromkeys(
            query for scope in scopes for query in (scope.get("seed_queries") or [])
        )),
        "branches": scopes,
    }


def _fallback_scope_question(scopes: list[dict[str, Any]]) -> str:
    labels = [str(scope.get("label") or "").strip() for scope in scopes]
    labels = [label for label in labels if label]
    if labels:
        return f"我还不能确定你的侧重点，你更偏向{'、'.join(labels)}中的哪一种，还是希望进行交叉分析？"
    return "我还不能确定你的侧重点，你希望主要研究什么对象、使用什么方法？"


def _single_question(text: str) -> str:
    """避免模型在一次澄清里连续抛出多个问题。"""
    text = re.sub(r"\s+", " ", text).strip()
    first = re.split(r"[？?]", text, maxsplit=1)[0].strip()
    return (first or "请进一步说明你的研究侧重点") + "？"


def reconcile_selected_scope_from_history(
    selected_scope: dict[str, Any] | None,
    scopes: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """在旧会话重生成时，用候选文本与原始回答做保守一致性检查。"""
    current = dict(selected_scope or {}) or None
    for item in reversed(history or []):
        if str(item.get("type") or "") != "clarification_answer":
            continue
        resolved = _resolve_scope_by_answer_signals(
            scopes or [],
            str(item.get("content") or ""),
        )
        if resolved:
            return resolved
    return current


def build_scoped_query(
    original_query: str,
    scope: dict[str, Any],
    *,
    clarification_answer: str = "",
) -> str:
    """把已确认范围显式写回原请求，供后续通用查询规划器使用。"""
    include = "、".join(scope.get("include_terms") or []) or "按该范围的标准定义检索"
    exclude = "、".join(scope.get("exclude_terms") or []) or "无额外排除项"
    seeds = "；".join(scope.get("seed_queries") or [])
    answer_clause = (
        f"\n用户澄清原文（其中明确方法、对象、先后关系和分析目标均为硬约束）：{clarification_answer.strip()}。"
        if clarification_answer.strip() else ""
    )
    suffix = answer_clause + (
        f"\n研究范围确认：{scope.get('label', '')}。"
        f"范围说明：{scope.get('description', '')}。"
        f"优先纳入：{include}。排除相邻含义：{exclude}。"
    )
    if seeds:
        suffix += f"可参考的种子检索表达：{seeds}。"
    return original_query.rstrip() + suffix


def _parse_json_object(text: str) -> dict[str, Any]:
    """健壮 JSON 解析（委托 json_utils 三段式策略）。失败时抛出 ValueError。"""
    result = _parse_json_object_robust(text)
    if not result:
        raise ValueError("主题消歧模型未返回 JSON 对象")
    return result
