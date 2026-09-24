"""搜索前的概念规划层：生成候选研究框架，指导后续定向检索。

在检索之前，LLM 根据用户意图、主题和视角生成：
1. Research Scope — 明确研究边界
2. Background Outline — 研究背景的论证结构
3. Provisional Routes — 候选研究路线（含每条路线的研究问题、核心概念、检索式）

检索完成后，这些候选路线会被证据验证和修正（KEEP/MERGE/SPLIT/DROP），
而不是让论文数据无约束地自行聚类。
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.logger import get_logger

logger = get_logger(__name__)

# 候选路线数量范围
_MIN_ROUTES = 1
_MAX_ROUTES = 5

# 每条路线的最低核心概念数
_MIN_CONCEPTS_PER_ROUTE = 3


def generate_provisional_routes(
    state: dict[str, Any],
    llm,
) -> dict[str, Any]:
    """根据用户意图和语义框架生成候选研究路线（搜索前执行）。

    返回包含 research_scope、background_outline、provisional_routes 的 dict。
    """
    topic = str(state.get("canonical_topic") or state.get("topic") or "")
    user_query = str(state.get("user_query") or "")
    semantic_frame = state.get("research_semantic_frame") or {}

    if not topic or not llm:
        return {}

    prompt = _build_provisional_route_prompt(topic, user_query, semantic_frame)

    try:
        response = llm.complete(
            prompt,
            response_format="json_object",
            temperature=0.0,
            operation="generate_provisional_routes",
        )
        from app.core.json_utils import parse_json_object

        data = parse_json_object(response if isinstance(response, str) else str(response))
        routes = data.get("provisional_routes") or []
        if not isinstance(routes, list) or len(routes) < _MIN_ROUTES:
            logger.warning(
                "Provisional routes under minimum (%d < %d), discarding",
                len(routes), _MIN_ROUTES,
            )
            return {}

        validated_routes = _validate_and_normalize_routes(routes)
        if len(validated_routes) < _MIN_ROUTES:
            return {}

        result = {
            "research_scope": data.get("research_scope") or {},
            "background_outline": data.get("background_outline") or {},
            "provisional_routes": validated_routes,
        }
        logger.info(
            "Provisional framework generated: %d routes, scope=%s",
            len(validated_routes),
            str(result["research_scope"].get("perspective") or "")[:60],
        )
        return result

    except Exception as exc:
        logger.warning("Provisional route generation failed: %s", exc)
        return {}


def route_aware_search_queries(
    provisional_routes: list[dict[str, Any]],
    global_topic: str = "",
) -> list[dict[str, Any]]:
    """从候选路线生成分路线检索式，每条路线产出中英文各 1-2 条。

    Returns:
        [{route_id, route_name, queries: [{text, language}]}, ...]
    """
    branches: list[dict[str, Any]] = []
    for route in provisional_routes:
        concepts = [
            str(item).strip()
            for item in route.get("core_concepts") or []
            if str(item).strip()
        ]
        route_queries = list(dict.fromkeys(
            str(item).strip()
            for item in route.get("search_queries") or []
            if str(item).strip()
        ))
        name = str(route.get("name") or "")
        route_id = str(route.get("route_id") or _safe_route_id(name))

        # 确保每条路线至少从 core_concepts 拼一个查询
        if not route_queries and concepts:
            route_queries = [" ".join(concepts[:3])]

        if not route_queries:
            continue

        branches.append({
            "route_id": route_id,
            "route_name": name,
            "research_question": str(route.get("research_question") or ""),
            "queries": route_queries[:4],
            "core_concepts": concepts,
        })

    return branches


def validate_routes_against_evidence(
    provisional_routes: list[dict[str, Any]],
    paper_cards: list[dict[str, Any]],
    llm=None,
    topic: str = "",
    semantic_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """用 v2 特征矩阵分别评估路线有效性与证据充分性。"""
    # Route Validator v2 owns matching and decision policy.  The implementation
    # lives in a focused module so graph orchestration and route planning remain
    # independent from feature extraction.
    from app.agent.route_validator import validate_route_evidence

    return validate_route_evidence(
        provisional_routes,
        paper_cards,
        llm=llm,
        topic=topic,
        semantic_frame=semantic_frame,
    )


def _name_cluster(
    cards: list[dict[str, Any]],
    llm,
    *,
    topic: str = "",
    parent_name: str = "",
    sibling_names: list[str] | None = None,
    reserved_names: list[str] | None = None,
) -> str:
    """用 LLM 为论文簇生成简短名称。

    给出主题、父路线、同级与全部既有路线名称作为约束上下文：缺少这些
    信息时模型会把整个研究主题搬进名称（"跨模态小样本视频动作识别"），
    擅自改写主题的任务设定（把少样本写成"零样本动作识别"），或与既有
    路线指向同一概念（"跨模态匹配"对既有"多模态与自监督"）。
    """
    titles = [str(card.get("title") or "")[:100] for card in cards[:5]]
    constraints = [
        "只用一个不超过 12 字的中文短语命名，只返回名称，不要解释。",
        "名称必须是这批论文相对其他路线的区分点（机制、模态或建模方式）。",
    ]
    if topic:
        constraints.append(
            f"研究主题是「{topic}」：名称不得包含或复述主题本身，"
            "也不得改写主题的任务设定或限定词。"
        )
    if parent_name:
        constraints.append(f"这批论文属于「{parent_name}」路线内部的一个细分方向。")
    siblings = [str(item).strip() for item in sibling_names or [] if str(item).strip()]
    if siblings:
        constraints.append(
            "已命名的同级子路线："
            + "、".join(siblings)
            + "；名称必须与它们明显不同，不得语义重叠。"
        )
    reserved = [
        str(item).strip() for item in reserved_names or [] if str(item).strip()
    ]
    if reserved:
        constraints.append(
            "正文将同时出现以下既有路线："
            + "、".join(reserved)
            + "；名称不得与它们中的任何一条指向同一概念或共享核心词"
            "（例如已有「多模态」路线时不得再起「跨模态」类名称）。"
        )
    prompt = (
        "为以下论文簇生成中文研究路线名称：\n"
        + chr(10).join(titles)
        + "\n\n要求：\n"
        + "\n".join(f"{index}. {item}" for index, item in enumerate(constraints, 1))
    )
    try:
        response = llm.complete(prompt, temperature=0.0, max_tokens=30)
        return str(response or "").strip()[:30]
    except Exception:
        return ""


def generate_global_recall_queries(
    topic: str,
    semantic_frame: dict[str, Any] | None = None,
) -> list[str]:
    """生成全局召回检索式，防止路线导向搜索产生确认偏误。

    这些查询不使用任何候选路线的概念，只用主题本身 + 语义框架的研究对象。
    """
    queries = [topic]
    frame = semantic_frame or {}
    objects = [
        str(item.get("surface_text") or item.get("label") or "")
        for item in frame.get("research_objects") or []
        if str(item.get("surface_text") or item.get("label") or "").strip()
    ]
    if objects:
        queries.append(" ".join(objects[:3]))
    # 主题 + 通用学术后缀
    queries.extend([
        f"{topic} 研究现状",
        f"{topic} 研究进展",
    ])
    return list(dict.fromkeys(q.strip() for q in queries if q.strip()))


def _build_provisional_route_prompt(
    topic: str,
    user_query: str,
    semantic_frame: dict[str, Any],
) -> str:
    """构建候选路线生成 prompt。"""
    frame_json = json.dumps(semantic_frame, ensure_ascii=False)
    return f"""你是学术综述规划器。请根据用户的研究主题和视角，在检索之前生成候选研究框架。

**用户请求**：{user_query}
**研究主题**：{topic}
**语义框架**：{frame_json}

你需要完成三件事：

1. **Research Scope（研究范围）**：
   - 明确核心研究对象、目标、可包含和应排除的内容
   - 用语义框架中的对象、方法和分析目标给出边界；将实现手段与研究对象区分。不得引用其他领域示例，也不得自行补充用户未提及的应用场景、对象或排除项。

2. **Background Outline（研究背景框架）**：
   - 3-4 个递进的段落目标，每个目标说明该段需要论证什么
   - 不写正文，只写段落目标

3. **Provisional Routes（候选研究路线）**：
   - {_MIN_ROUTES}-{_MAX_ROUTES} 条互斥的候选路线；路线数量应由当前主题和证据边界决定，不要为满足固定数量虚构路线
   - 每条路线必须有：
     * name：简洁中文名称（≤15字）
     * research_question：该路线的核心研究问题（1句话）
     * route_role：该路线在研究链中的角色（formalization / sensing / interpretation / application / synthesis）
     * core_concepts：核心概念列表（≥{_MIN_CONCEPTS_PER_ROUTE}个，用于检索）
     * semantic_anchors：3-8个中英文语义锚点或文献常用同义表达
     * method_concepts：该路线的方法机制概念，不得混入研究对象或应用场景
     * task_anchors：该路线实际解决的研究任务或输入输出关系
     * negative_anchors：容易误召回、但明确不属于该路线的概念
     * search_queries：2-3条中英文检索式
     * inclusion_criteria：明确哪些论文属于该路线
     * exclusion_criteria：明确哪些论文不属于该路线（特别是指出与相邻路线的边界）
     * boundary_note：与哪条路线容易混淆，如何区分
   - 路线之间应有清晰界限，不要重叠
   - 路线分类维度应为**研究问题+方法机制**，不是数据类型、出版类型或文件格式
   - 路线名称禁止使用：期刊论文、会议论文、实验研究、图像/视频/文本等模态名

严格返回 JSON：
{{
  "research_scope": {{
    "core_objects": ["..."],
    "goals": ["..."],
    "includes": ["..."],
    "excludes": ["..."],
    "perspective": "..."
  }},
  "background_outline": {{
    "paragraph_goals": [
      {{"id": "bg_1", "label": "...", "goal": "...", "core_question": "..."}}
    ]
  }},
  "provisional_routes": [
    {{
      "route_id": "R1",
      "name": "...",
      "research_question": "...",
      "route_role": "formalization|sensing|interpretation|application|synthesis",
      "core_concepts": ["...", "..."],
      "semantic_anchors": ["...", "..."],
      "method_concepts": ["...", "..."],
      "task_anchors": ["...", "..."],
      "negative_anchors": ["..."],
      "search_queries": ["...", "..."],
      "inclusion_criteria": ["..."],
      "exclusion_criteria": ["..."],
      "boundary_note": "...",
      "rationale": "..."
    }}
  ]
}}
"""


def _validate_and_normalize_routes(
    routes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """验证和规范化候选路线。"""
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for i, route in enumerate(routes, 1):
        name = str(route.get("name") or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        concepts = list(dict.fromkeys(
            str(c).strip()
            for c in route.get("core_concepts") or []
            if str(c).strip()
        ))

        if len(concepts) < _MIN_CONCEPTS_PER_ROUTE:
            logger.debug("Route %s has too few concepts (%d), skipping", name, len(concepts))
            continue

        normalized.append({
            "route_id": str(route.get("route_id") or f"R{i}"),
            "name": name,
            "research_question": str(route.get("research_question") or "").strip(),
            "route_role": str(route.get("route_role") or "").strip(),
            "core_concepts": concepts,
            "semantic_anchors": list(dict.fromkeys(
                str(value).strip()
                for value in route.get("semantic_anchors") or []
                if str(value).strip()
            ))[:12],
            "method_concepts": list(dict.fromkeys(
                str(value).strip()
                for value in route.get("method_concepts") or []
                if str(value).strip()
            ))[:12],
            "task_anchors": list(dict.fromkeys(
                str(value).strip()
                for value in route.get("task_anchors") or []
                if str(value).strip()
            ))[:12],
            "negative_anchors": list(dict.fromkeys(
                str(value).strip()
                for value in route.get("negative_anchors") or []
                if str(value).strip()
            ))[:12],
            "anchor_expansions": [
                dict(value) for value in route.get("anchor_expansions") or []
                if isinstance(value, dict)
            ][:24],
            "search_queries": list(dict.fromkeys(
                str(q).strip()
                for q in route.get("search_queries") or []
                if str(q).strip()
            )),
            "inclusion_criteria": [
                str(c).strip()
                for c in (route.get("inclusion_criteria") or [])
                if str(c).strip()
            ],
            "exclusion_criteria": [
                str(c).strip()
                for c in (route.get("exclusion_criteria") or [])
                if str(c).strip()
            ],
            "boundary_note": str(route.get("boundary_note") or "").strip(),
            "rationale": str(route.get("rationale") or "").strip(),
        })

    return normalized


def _safe_route_id(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9一-鿿]+", "_", str(name or "")).strip("_")
    return safe or "route"
