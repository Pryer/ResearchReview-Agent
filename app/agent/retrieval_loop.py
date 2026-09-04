"""ReAct 检索精化循环。

从 graph.py 编排层下沉的检索策略本体：首轮检索 → 规则排序 →
循环（覆盖度判定 → LLM 关键词精化 → 增量检索 → 再排序）→
末尾统一 LLM 重排。graph 只负责决定"何时调用本循环"，
循环的退出条件与轮次预算在这里维护。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from app.agent.execution import AgentCancelledError
from app.agent.execution import checkpoint as _checkpoint
from app.agent.nodes import rank_node, refine_search_node, search_node
from app.agent.state import ResearchAgentState
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def recover_english_screening_shortfall(
    state: ResearchAgentState,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> bool:
    """对英文硬筛低通过率执行一次有界定向补检索。

    该动作只在英文候选池足够大但通过率异常低时触发。新查询仅发送到
    英文兼容来源，结果回并到候选池后由 ``rank_node`` 重新执行统一 compiled
    scope 硬筛；因此不会用放宽规则或低相关论文填充语言配额。
    """
    report = dict(state.get("screening_report_low_pass_protection") or {})
    if not report.get("quota_transfer_blocked"):
        return False
    if state.get("english_screening_recovery_attempted"):
        return False
    state["english_screening_recovery_attempted"] = True

    from app.tools.paper_matching import compile_scope
    from app.tools.search_papers import search_papers
    from app.utils.deduplicate import deduplicate_papers

    compiled = state.get("compiled_scope") or compile_scope(
        selected_scope=state.get("selected_scope") or {},
        semantic_frame=state.get("research_semantic_frame") or {},
        screening_protocol=state.get("screening_protocol") or {},
        required_concepts=state.get("required_concepts") or [],
        topic_anchors=state.get("topic_anchors") or [],
        search_branches=state.get("search_branches") or [],
        excluded_title_terms=state.get("excluded_title_terms") or [],
        topic=str(state.get("topic") or ""),
    )
    state["compiled_scope"] = compiled

    aliases = compiled.get("aliases") or {}
    query_candidates: list[str] = []
    for role in (
        "scope_seed", "scope_include", "object", "method", "context",
        "requirement", "topic", "topic_anchor",
    ):
        query_candidates.extend(
            str(item).strip()
            for item in aliases.get(role) or []
            if str(item).strip() and not re.search(r"[\u4e00-\u9fff]", str(item))
        )
    # Prefer compact concept combinations over long protocol prose.
    query_candidates = list(dict.fromkeys(query_candidates))
    meaningful = [
        item for item in query_candidates
        if len(re.findall(r"[A-Za-z0-9]+", item)) >= 1
        and item.casefold() not in {str(q).casefold() for q in state.get("searched_keywords") or []}
    ]
    queries: list[str] = []
    for item in meaningful:
        if len(queries) >= min(2, int(get_settings().max_search_keywords or 2)):
            break
        queries.append(item)
    if len(queries) < 2:
        en_tokens = list((compiled.get("tokens") or {}).get("en") or [])
        for index in range(0, len(en_tokens), 3):
            query = " ".join(en_tokens[index:index + 3]).strip()
            if query and query.casefold() not in {str(q).casefold() for q in queries}:
                queries.append(query)
            if len(queries) >= min(2, int(get_settings().max_search_keywords or 2)):
                break

    english_sources = [
        source for source in get_settings().search_sources_list
        if source in {"arxiv", "semantic_scholar", "openalex", "crossref"}
    ]
    diagnostics: list[Any] = []
    new_papers: list[dict[str, Any]] = []
    for query in queries[: max(1, int(get_settings().max_search_keywords or 1))]:
        if should_cancel and should_cancel():
            raise AgentCancelledError("英文筛选恢复在检索前取消")
        found = search_papers(
            query=query,
            start_year=int(state.get("start_year") or 1900),
            end_year=int(state.get("end_year") or 2100),
            max_results=int(get_settings().max_results_per_keyword or 30),
            sources=english_sources,
            diagnostics=diagnostics,
            should_cancel=should_cancel,
        )
        for paper in found:
            item = paper if isinstance(paper, dict) else paper.model_dump(mode="json")
            item["_search_branches"] = list(dict.fromkeys([
                *(item.get("_search_branches") or []),
                "english_screening_recovery",
            ]))
            new_papers.append(item)

    previous = list(state.get("candidate_papers") or [])
    merged = deduplicate_papers([*previous, *new_papers])
    state["candidate_papers"] = merged
    state["searched_keywords"] = list(dict.fromkeys([
        *(state.get("searched_keywords") or []), *queries,
    ]))
    state["english_screening_recovery"] = {
        "attempted": True,
        "queries": queries,
        "sources": english_sources,
        "new_candidates": max(0, len(merged) - len(deduplicate_papers(previous))),
        "diagnostics": [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in diagnostics
        ],
        "compiled_scope_fingerprint": compiled.get("fingerprint"),
        "rescreen_required": bool(new_papers),
    }
    return bool(new_papers)


def diagnose_search_drift(
    papers: list[dict[str, Any]],
    *,
    core_keywords: list[str] | None = None,
    expanded_keywords: list[str] | None = None,
    topic_anchors: list[list[str]] | None = None,
) -> dict[str, Any]:
    """诊断扩展查询是否脱离主题锚点，并返回可行动的覆盖明细。"""
    corpus = list(papers or [])
    core = [str(item).strip().casefold() for item in (core_keywords or []) if str(item).strip()]
    expanded = [str(item).strip().casefold() for item in (expanded_keywords or []) if str(item).strip()]
    anchors = [
        [str(term).strip().casefold() for term in group if str(term).strip()]
        for group in (topic_anchors or []) if group
    ]

    def paper_text(paper: dict[str, Any]) -> str:
        return " ".join(
            str(paper.get(key) or "") for key in ("title", "abstract", "keywords")
        ).casefold()

    def hit(paper: dict[str, Any], terms: list[str]) -> bool:
        text = paper_text(paper)
        return any(term in text for term in terms)

    core_hits = sum(hit(paper, core) for paper in corpus)
    anchor_hits = sum(
        any(any(term in paper_text(paper) for term in group) for group in anchors)
        for paper in corpus
    )
    expanded_only = sum(hit(paper, expanded) and not hit(paper, core) for paper in corpus)
    total = len(corpus)
    drift_rate = expanded_only / total if total else 0.0
    anchor_coverage_rate = anchor_hits / total if total else 0.0
    drift_detected = bool(
        total
        and expanded
        and (drift_rate >= 0.5 or bool(anchors) and anchor_coverage_rate < 0.5)
    )
    reasons = []
    if drift_rate >= 0.5:
        reasons.append("扩展词独占结果占比过高")
    if anchors and anchor_coverage_rate < 0.5:
        reasons.append("主题语义锚点覆盖不足")
    return {
        "paper_count": total,
        "core_hit_count": core_hits,
        "anchor_hit_count": anchor_hits,
        "expanded_only_count": expanded_only,
        "core_coverage_rate": core_hits / total if total else 0.0,
        "anchor_coverage_rate": anchor_coverage_rate,
        "drift_rate": drift_rate,
        "drift_detected": drift_detected,
        "reasons": reasons,
        "core_keywords": list(core_keywords or []),
        "expanded_keywords": list(expanded_keywords or []),
        "topic_anchors": [list(group) for group in (topic_anchors or [])],
    }


def search_rank_with_refinement(
    state: ResearchAgentState,
    llm=None,
    should_cancel: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    total_steps: int = 14,
) -> None:
    """ReAct 式检索闭环：检索不足时把反馈交给 LLM 修正关键词。"""
    target = int(
        state.get("retrieval_target")
        or state.get("max_papers")
        or get_settings().default_max_papers
    )
    max_rounds = get_settings().search_refinement_max_rounds
    # 增量/修复场景本身是定向补召回，完整 refine 循环的后几轮边际收益
    # 很低却各花数分钟；压缩为最多一轮精化，控制修复总时长。
    if state.get("incremental_retrieval"):
        max_rounds = min(max_rounds, 1)

    search_node(state, should_cancel=should_cancel)
    _checkpoint(state, "rank_papers", 2, total_steps, should_cancel, progress_callback)
    if state.get("search_failed") and not state.get("candidate_papers"):
        return
    # 首轮不做 LLM rerank，因为可能还不足量
    rank_node(state, llm=None)
    state["search_drift_diagnostics"] = diagnose_search_drift(
        state.get("ranked_papers") or [],
        core_keywords=state.get("core_keywords") or [state.get("topic") or ""],
        expanded_keywords=state.get("expanded_keywords") or [],
        topic_anchors=state.get("topic_anchors") or state.get("required_concepts") or [],
    )

    # 英文硬筛低通过率时先做一次英文定向补召回，再进入常规 refine；新结果
    # 由 rank_node 用同一 compiled scope 反事实复筛，语言缺口不会被静默转移。
    if recover_english_screening_shortfall(state, should_cancel=should_cancel):
        _checkpoint(state, "english_screening_recovery", 3, total_steps, should_cancel, progress_callback)
        rank_node(state, llm=None)
        state["search_drift_diagnostics"] = diagnose_search_drift(
            state.get("ranked_papers") or [],
            core_keywords=state.get("core_keywords") or [state.get("topic") or ""],
            expanded_keywords=state.get("expanded_keywords") or [],
            topic_anchors=state.get("topic_anchors") or state.get("required_concepts") or [],
        )

    for _ in range(max_rounds):
        ranked_count = len(state.get("ranked_papers") or [])
        from app.agent.focus_coverage import required_focus_coverage

        semantic_frame = state.get("research_semantic_frame") or {}
        coverage = required_focus_coverage(
            semantic_frame,
            state.get("ranked_papers") or [],
        )
        state["focus_coverage"] = coverage
        eligible_count = ranked_count
        if semantic_frame.get("evidence_requirements"):
            from app.agent.evidence_roles import citation_eligible_paper_ids

            eligible_count = len(citation_eligible_paper_ids(
                semantic_frame, state.get("ranked_papers") or []
            ))
            state["retrieval_eligible_count"] = eligible_count
        required_references = int(
            state.get("required_reference_count") or target
        )
        if eligible_count >= required_references and coverage.get("ready", True):
            state["retrieval_requirement_met"] = True
            state["retrieval_stop_reason"] = "引用需求与证据覆盖均已满足"
            break  # 满足后跳出循环，在末尾统一做一次 LLM rerank
        before_keywords = list(state.get("keywords") or [])
        _checkpoint(state, "refine_search", 3, total_steps, should_cancel, progress_callback)
        refine_search_node(state, llm=llm)
        after_keywords = list(state.get("keywords") or [])
        last_refine = next(
            (
                step for step in reversed(state.get("steps") or [])
                if step.get("step_name") == "refine_search"
            ),
            {},
        )
        if last_refine.get("status") != "success" or after_keywords == before_keywords:
            break
        _checkpoint(state, "search_refined_queries", 3, total_steps, should_cancel, progress_callback)
        search_node(state, should_cancel=should_cancel)
        if not state.get("last_search_new_results"):
            break
        if state.get("search_failed") and not state.get("candidate_papers"):
            break
        rank_node(state, llm=None)
        state["search_drift_diagnostics"] = diagnose_search_drift(
            state.get("ranked_papers") or [],
            core_keywords=state.get("core_keywords") or [state.get("topic") or ""],
            expanded_keywords=state.get("expanded_keywords") or [],
            topic_anchors=state.get("topic_anchors") or state.get("required_concepts") or [],
        )

    # 只在最后做一次 LLM rerank
    if llm is not None and state.get("ranked_papers"):
        from app.tools.rank_papers import llm_rerank_papers

        from app.agent.nodes.base import _paper_identity_key

        rerank_diagnostics: dict[str, Any] = {}
        rerank_input = state["ranked_papers"]
        rerank_mode = "full"
        retained_head: list = []
        # 增量轮：旧榜单的相对顺序已由上一轮全量重排确定，全池重排
        # 纯属浪费（实测一次全池 rerank 数分钟）。只重排新增论文 +
        # 旧榜尾部边界段（可能被新论文挤出 cutoff 的候选），重排后
        # 与旧榜头部段拼接；头部段保持原顺序不变。
        new_keys = set(state.get("incremental_new_paper_keys") or [])
        if state.get("incremental_retrieval") and new_keys:
            new_papers = [
                paper for paper in state["ranked_papers"]
                if _paper_identity_key(paper) in new_keys
            ]
            old_papers = [
                paper for paper in state["ranked_papers"]
                if _paper_identity_key(paper) not in new_keys
            ]
            if new_papers and old_papers:
                boundary = max(1, min(len(old_papers), target // 2))
                retained_head = old_papers[:-boundary]
                rerank_input = old_papers[-boundary:] + new_papers
                rerank_mode = "incremental_merge"
        try:
            ranked = llm_rerank_papers(
                rerank_input,
                topic=state.get("topic", ""),
                scope=state.get("selected_scope") or {},
                llm=llm,
                top_k=target,
                research_mode=str(
                    (state.get("research_semantic_frame") or {}).get("research_mode") or ""
                ),
                screening_protocol=state.get("screening_protocol") or {},
                rerank_diagnostics=rerank_diagnostics,
                minimum_required=int(state.get("required_reference_count") or 0),
            )
        except AgentCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # rerank 只是排序增强：单次 LLM/解析异常不能炸掉整条检索链，
            # 否则也是 citation_gap_repair 增量路径的逃逸口。保留规则粗排
            # 顺序继续走，并在诊断中记录降级。
            logger.warning("llm_rerank_papers failed; keeping rule-based order: %s", exc)
            ranked = rerank_input
            rerank_diagnostics.clear()
            rerank_diagnostics.update({
                "mode": "rule_order_fallback",
                "error": str(exc),
            })
        if retained_head:
            # 头部段与重排结果集合互斥（按身份键拆分），直接拼接后截断。
            ranked = (retained_head + list(ranked))[:target]
        state["ranked_papers"] = ranked
        report = dict(state.get("screening_report") or {})
        report["llm_rerank"] = rerank_diagnostics
        report["llm_rerank_mode"] = rerank_mode
        state["screening_report"] = report
