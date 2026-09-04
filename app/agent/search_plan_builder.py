"""依据研究语义帧构造互补检索分支。"""

from __future__ import annotations

import re

from app.schemas.research_plan_schema import (
    EvidenceRequirement,
    ResearchMode,
    ResearchSemanticFrame,
    SearchBranch,
)

# 高引用目标阈值：达到该量级时，单一宽泛查询无法为各方法学子路线提供
# 均衡的候选召回，需要为每个子方向保留独立检索分支。
_HIGH_CITATION_TARGET = 40


def build_semantic_search_branches(
    frame: ResearchSemanticFrame,
    *,
    retrieval_target: int | None = None,
) -> list[SearchBranch]:
    """生成领域、技术、桥接和下游分支，不补充用户未表达的应用领域。

    ``retrieval_target`` 达到高引用目标（≥40 篇）时，额外为每个方法学
    子方向和证据路线生成独立细分分支，确保各子路线在并集召回中都有
    专属查询位，而非被单一宽泛主题查询主导。
    """
    branches: list[SearchBranch] = []
    domain_terms = _terms(frame.application_domains)
    object_terms = _terms(frame.research_objects)
    technical_terms = _terms([
        method for method in frame.methods if method.category == "technical"
    ])
    explicit_technical_terms = _terms([
        method for method in frame.methods
        if method.category == "technical" and method.explicit
    ])
    inferred_technical = any(
        method.category == "technical" and method.inferred
        for method in frame.methods
    )
    analytical_terms = _terms([
        method for method in frame.methods if method.category != "technical"
    ])
    open_analytical_selection = any(
        requirement.evidence_role == "analytical_method"
        and requirement.selection_mode == "open_any"
        for requirement in frame.evidence_requirements
    )
    explicit_analytical_terms = _terms([
        method for method in frame.methods
        if method.category != "technical" and method.explicit
    ])
    target_terms = _terms(frame.analysis_targets)
    domain_anchor_terms = _domain_anchor_terms(frame)

    if frame.research_mode in {
        ResearchMode.DOMAIN_ORIENTED,
        ResearchMode.TECHNOLOGY_APPLIED_TO_DOMAIN,
        ResearchMode.TECHNOLOGY_ASSISTED_DOMAIN_ANALYSIS,
    }:
        queries = _unique([
            frame.canonical_topic,
            " ".join(object_terms + analytical_terms),
            " ".join(domain_terms + object_terms),
        ])
        branches.append(SearchBranch(
            branch_type="domain_foundation",
            queries=queries[:3],
            required_concepts=[domain_anchor_terms] if domain_anchor_terms else [],
            rationale="覆盖研究对象、领域定义与非技术性基础研究",
            constraint_level="soft",
        ))

    if technical_terms:
        applied_mode = frame.research_mode in {
            ResearchMode.TECHNOLOGY_APPLIED_TO_DOMAIN,
            ResearchMode.TECHNOLOGY_ASSISTED_DOMAIN_ANALYSIS,
        }
        if applied_mode and domain_anchor_terms:
            queries = _unique([
                " ".join(technical_terms + domain_anchor_terms),
                " ".join([technical_terms[0], domain_anchor_terms[0]]),
            ])
            technical_required = [group for group in (
                explicit_technical_terms,
                domain_anchor_terms,
            ) if group]
        else:
            queries = _unique([
                " ".join(technical_terms),
                *technical_terms,
            ])
            technical_required = [explicit_technical_terms] if explicit_technical_terms else []
        branches.append(SearchBranch(
            branch_type="technical_method",
            queries=queries[:3],
            required_concepts=technical_required,
            rationale=(
                "覆盖应用领域内的技术方法研究"
                if applied_mode else "覆盖用户明确指定的技术方法及其方法学研究"
            ),
            constraint_level="exploratory" if inferred_technical else "hard",
        ))

    if technical_terms and domain_terms:
        anchor = domain_anchor_terms or object_terms or domain_terms
        branches.append(SearchBranch(
            branch_type="bridge_research",
            queries=_unique([
                " ".join(technical_terms + anchor),
                " ".join([technical_terms[0], anchor[0]]),
            ]),
            required_concepts=[group for group in (
                explicit_technical_terms,
                domain_anchor_terms,
            ) if group],
            rationale="检索技术方法与应用对象同时出现的直接交叉研究",
            constraint_level="exploratory" if inferred_technical else "hard",
        ))

    if analytical_terms:
        anchor = domain_anchor_terms or object_terms or domain_terms
        analytical_queries = [
            *[" ".join([term, *anchor]) for term in analytical_terms],
            " ".join(analytical_terms + anchor),
        ]
        if open_analytical_selection and anchor:
            analytical_queries.append(" ".join([*anchor, "analysis method"]))
        branches.append(SearchBranch(
            branch_type="analytical_method",
            queries=_unique(analytical_queries)[:4],
            required_concepts=(
                [] if open_analytical_selection
                else [explicit_analytical_terms] if explicit_analytical_terms else []
            ),
            rationale=(
                "以用户列举的方法为检索种子，同时探索适用于研究对象的其他分析方法"
                if open_analytical_selection
                else "覆盖用户明确指定的领域分析方法，避免在宽泛主题检索中被技术论文淹没"
            ),
            constraint_level="exploratory" if open_analytical_selection else "soft",
        ))

    if technical_terms and analytical_terms and (domain_anchor_terms or object_terms):
        anchor = domain_anchor_terms or object_terms
        branches.append(SearchBranch(
            branch_type="pipeline_bridge",
            queries=_unique([
                " ".join(technical_terms + [term] + anchor)
                for term in analytical_terms
            ])[:3],
            required_concepts=[group for group in (
                explicit_technical_terms,
                domain_anchor_terms,
            ) if group],
            rationale="检索从自动识别或编码通向后续领域分析的端到端或衔接研究",
            constraint_level="soft",
        ))

    if frame.research_mode == ResearchMode.TECHNOLOGY_ASSISTED_DOMAIN_ANALYSIS:
        downstream_anchor = target_terms or object_terms
        queries = _unique([
            " ".join(downstream_anchor + analytical_terms),
            " ".join(technical_terms + downstream_anchor),
            " ".join(object_terms + ["sequence analysis"]),
        ])
        branches.append(SearchBranch(
            branch_type="downstream_analysis",
            queries=queries[:3],
            required_concepts=[
                _explicit_terms(frame.analysis_targets)
                or _explicit_terms(frame.research_objects)
            ] if (
                _explicit_terms(frame.analysis_targets)
                or _explicit_terms(frame.research_objects)
            ) else [],
            rationale="覆盖识别或编码结果之后的解释、评价与领域分析",
            constraint_level="exploratory" if inferred_technical else "soft",
        ))

    if retrieval_target and int(retrieval_target) >= _HIGH_CITATION_TARGET:
        # 高引用目标下按方法学子方向与证据路线细分；单方法主题的
        # 方法细分无增量，由既有 technical_method 分支直接覆盖。
        branches.extend(_method_subroute_branches(
            frame,
            technical_terms=technical_terms,
            anchor_terms=domain_anchor_terms or object_terms or domain_terms,
        ))

    if not branches:
        branches.append(SearchBranch(
            branch_type="topic_core",
            queries=[frame.canonical_topic] if frame.canonical_topic else [],
            rationale="语义信息有限时保持原主题边界",
            constraint_level="soft",
        ))
    return branches


def _method_subroute_branches(
    frame: ResearchSemanticFrame,
    *,
    technical_terms: list[str],
    anchor_terms: list[str],
) -> list[SearchBranch]:
    """高引用目标下为每个方法学子方向与证据路线保留独立检索分支。

    子方向只来自语义帧中用户表达或规范化得到的方法与证据要求，
    不预设领域专属路线（如特定领域的具体算法族）。每个子分支与
    领域锚点组合成对，保证细分路线在并集召回中拥有专属查询位。
    """
    branches: list[SearchBranch] = []
    anchor = anchor_terms[:2]

    if len(technical_terms) >= 2:
        for term in technical_terms:
            queries = _unique([
                " ".join([term, *anchor]),
                term,
            ])
            branches.append(SearchBranch(
                branch_type=f"method_subroute_{_branch_id(term)}",
                queries=queries[:3],
                required_concepts=[[term]],
                rationale=f"高引用目标下为方法学子方向「{term}」保留独立召回位",
                constraint_level="exploratory",
            ))

    grouped: dict[str, EvidenceRequirement] = {}
    for requirement in frame.evidence_requirements:
        group = str(
            requirement.route_group or requirement.evidence_role or ""
        ).strip()
        if group and group not in grouped:
            grouped[group] = requirement
    for group, requirement in grouped.items():
        route_terms = _unique([
            *[
                str(alias).strip() for alias in requirement.aliases
                if str(alias).strip()
            ],
            str(requirement.label or "").strip(),
        ])
        if not route_terms:
            continue
        queries = _unique([
            " ".join([*route_terms[:2], *anchor]),
            " ".join([route_terms[0], *anchor]) if anchor else route_terms[0],
        ])
        branches.append(SearchBranch(
            branch_type=f"requirement_route_{_branch_id(group)}",
            queries=queries[:3],
            required_concepts=[],
            rationale=f"高引用目标下为证据路线「{requirement.label or group}」保留独立召回位",
            constraint_level="exploratory",
        ))
    return branches


def _branch_id(value: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return safe or "route"


def prioritized_branch_queries(branches: list[SearchBranch], limit: int = 10) -> list[str]:
    """先取每个分支的首选查询，再补充各分支变体，避免单一分支占满检索位。"""
    values = [branch.queries[0] for branch in branches if branch.queries]
    values.extend(
        query
        for branch in branches
        for query in branch.queries[1:]
    )
    return _unique(values)[:limit]


def _terms(items) -> list[str]:
    return _unique([
        str(item.label or item.surface_text or item.id).strip()
        for item in items
        if str(item.label or item.surface_text or item.id).strip()
    ])


def _explicit_terms(items) -> list[str]:
    return _terms([item for item in items if item.explicit])


def _domain_anchor_terms(frame: ResearchSemanticFrame) -> list[str]:
    """提取应用领域中的研究对象锚点，避免用通用方法词代替领域主题。"""
    domain_surfaces = [
        str(item.surface_text or "").strip().lower()
        for item in frame.application_domains
        if item.explicit and str(item.surface_text or "").strip()
    ]
    anchored = []
    for item in frame.research_objects:
        if not item.explicit:
            continue
        label = str(item.label or "").strip()
        surface = str(item.surface_text or "").strip()
        if any(domain in surface.lower() for domain in domain_surfaces):
            anchored.extend([label, surface])
    return _unique(anchored) or _explicit_terms(frame.research_objects)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = " ".join(str(value or "").split()).strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result
