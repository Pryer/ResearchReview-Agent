"""基础辅助函数与通用步骤记录工具。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from app.core.logger import get_logger
from app.schemas.paper_schema import SourceDiagnostic

if TYPE_CHECKING:
    from app.agent.state import ResearchAgentState

logger = get_logger(__name__)

def _paper_identity_key(paper: dict[str, Any]) -> str:
    """跨检索、详情和卡片阶段复用的稳定论文身份键。"""
    from app.utils.deduplicate import normalize_title

    return (
        str(paper.get("doi") or "").strip().lower()
        or str(paper.get("arxiv_id") or "").strip().lower()
        or normalize_title(str(paper.get("title") or ""))
        or str(paper.get("paper_id") or paper.get("id") or "")
    )


def _select_search_keywords(keywords: list[str], limit: int = 3) -> list[str]:
    """保留主题词，并优先选择不同的英文检索变体。

    只要原始关键词列表中出现过中文词，就保证最终选出的关键词里至少保留
    一个中文词，用于触发 CNKI / Crossref 等中文数据源检索。之前的实现
    只检查列表第一个词是否为中文来决定是否保留中文位，当第一个词恰好是
    英文变体（如规范化后的 canonical_topic snake_case 形式）时，中文词
    会被排到列表末尾并被 limit 截断，导致中文数据源永远不会被检索到。
    """
    unique: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        keyword = str(keyword or "").strip()
        key = keyword.lower()
        if keyword and key not in seen:
            seen.add(key)
            unique.append(keyword)
    if not unique:
        return []

    def _is_chinese(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    english = [item for item in unique[1:] if re.search(r"[A-Za-z]", item)]
    others = [item for item in unique[1:] if item not in english]
    first_is_chinese = _is_chinese(unique[0])

    if first_is_chinese and english:
        # 保留中文主题词用于 Crossref/OpenAlex 中文检索，同时优先覆盖英文标准术语。
        selected = english[: max(0, limit - 1)] + [unique[0]]
    else:
        selected = [unique[0]]
        selected.extend(english)
    selected.extend(others)
    selected = selected[:limit]

    if limit > 0 and not any(_is_chinese(item) for item in selected):
        chinese_candidate = next((item for item in unique if _is_chinese(item)), None)
        if chinese_candidate:
            selected = selected[: limit - 1] + [chinese_candidate]

    return selected


def _select_branch_diverse_keywords(
    keywords: list[str],
    search_branches: list[dict[str, Any]],
    limit: int,
) -> list[str]:
    """优先为每个语义检索分支保留一个查询，再用普通关键词补足。"""
    if limit <= 0:
        return []
    available = {
        str(item).strip().lower(): str(item).strip()
        for item in keywords
        if str(item).strip()
    }
    selected: list[str] = []
    for branch in search_branches:
        for query in branch.get("queries") or []:
            value = available.get(str(query).strip().lower())
            if value and value not in selected:
                selected.append(value)
                break
        if len(selected) >= limit:
            return selected
    remaining = [item for item in keywords if str(item).strip() not in selected]
    return [
        *selected,
        *_select_search_keywords(remaining, limit - len(selected)),
    ][:limit]


def _select_batch_first_keywords(
    keywords: list[str],
    keyword_batches: list[dict[str, Any]],
    search_branches: list[dict[str, Any]],
    limit: int,
) -> list[str]:
    """按 exact→broader→variant 批次派发检索词，批内保持分支多样。

    批次顺序来自关键词生成工具的 type 语义（完整表达先检、外扩后检），
    本函数不做任何领域词判断。不在任何批次中的词（规划拼接词、全局
    召回式、refine 追加词）排在全部批次之后作为兜底。

    例外：``targeted_recovery`` 分支的查询先预留名额。它们是为特定路线补
    篇数而生成的，若继续排在全部批次之后，批次未跑完时会被整轮挤出，该
    方向的定向补检索就不会真正发生。
    """
    if limit <= 0 or not keywords:
        return []
    pool = [str(item).strip() for item in keywords if str(item).strip()]
    pool_set = set(pool)
    recovery_queries = [
        str(query).strip()
        for branch in search_branches
        if str(branch.get("constraint_level") or "") == "targeted_recovery"
        for query in branch.get("queries") or []
        if str(query).strip() in pool_set
    ]
    reserved: list[str] = []
    if recovery_queries:
        reserve_limit = max(1, limit // 2)
        for query in dict.fromkeys(recovery_queries):
            if len(reserved) >= reserve_limit:
                break
            reserved.append(query)
    if reserved:
        reserved_set = set(reserved)
        pool = [item for item in pool if item not in reserved_set]
    remaining_limit = limit - len(reserved)
    if remaining_limit <= 0:
        return reserved[:limit]
    consumed: set[str] = set()
    groups: list[list[str]] = []
    for batch in keyword_batches:
        members = {str(word).strip() for word in batch.get("keywords") or []}
        words = [item for item in pool if item in members and item not in consumed]
        if words:
            consumed.update(words)
            groups.append(words)
    unbatched = [item for item in pool if item not in consumed]
    if unbatched:
        groups.append(unbatched)
    selected: list[str] = []
    for group in groups:
        if len(selected) >= remaining_limit:
            break
        selected.extend(_select_branch_diverse_keywords(
            group, search_branches, remaining_limit - len(selected),
        ))
    return [*reserved, *selected[:remaining_limit]]


# ============================================================
# 步骤记录
# ============================================================
def append_step(
    state: "ResearchAgentState",
    step_name: str,
    status: str,
    tool_name: str | None = None,
    input_data: dict | None = None,
    output_data: dict | None = None,
    error: str | None = None,
    duration_ms: int | None = None,
) -> "ResearchAgentState":
    """记录 Agent 执行步骤。

    在 ``state["steps"]`` 列表追加一条步骤记录，用于前端展示和调试。
    """
    step: Dict[str, Any] = {
        "step_name": step_name,
        "tool_name": tool_name,
        "input_data": input_data or {},
        "output_data": output_data,
        "status": status,
        "error": error,
        "duration_ms": duration_ms,
        "timestamp": datetime.utcnow().isoformat(),
    }
    state.setdefault("steps", []).append(step)
    logger.info(
        "AGENT_STEP_DEBUG %s",
        json.dumps(_compact_debug_value(step), ensure_ascii=False, default=str),
    )
    # P2-3 集成：统一在此记录节点级性能指标，不需要改动各节点内部逻辑。
    # 指标采集失败不应影响主流程，因此单独捕获异常。
    try:
        from app.core.metrics import get_metrics_collector

        collector = get_metrics_collector()
        collector.record_step(
            step_name=step_name, status=status, duration_ms=duration_ms
        )
        token_report = collector.get_token_report()
        state["step_metrics"] = {
            "last_step": step_name,
            "last_duration_ms": duration_ms,
            "total_tokens": token_report["total_tokens"],
            "total_llm_calls": token_report["total_calls"],
        }
    except Exception:  # noqa: BLE001 - 指标采集绝不能影响主流程
        logger.debug("metrics recording failed for step=%s", step_name, exc_info=True)
    return state


def _compact_debug_value(value: Any, depth: int = 0) -> Any:
    """仅压缩日志副本；state 中仍保留完整可审计数据。"""
    if depth >= 4:
        return "...[nested]"
    if isinstance(value, str):
        return _preview_text(value, 600)
    if isinstance(value, list):
        compact = [_compact_debug_value(item, depth + 1) for item in value[:10]]
        if len(value) > 10:
            compact.append(f"...[{len(value) - 10} more]")
        return compact
    if isinstance(value, dict):
        return {
            str(key): _compact_debug_value(item, depth + 1)
            for key, item in value.items()
        }
    return value


def _preview_text(text: str | None, limit: int = 500) -> str:
    """压缩长文本，避免调试输出过大。"""
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def _paper_debug_item(paper: dict) -> dict:
    """返回单篇论文的调试摘要。"""
    return {
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title"),
        "authors": paper.get("authors", [])[:6] if isinstance(paper.get("authors"), list) else paper.get("authors"),
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "doi": paper.get("doi"),
        "url": paper.get("url"),
        "pdf_url": paper.get("pdf_url"),
        "source": paper.get("source"),
        "rank_score": paper.get("_rank_score"),
        "relevance_score": paper.get("_relevance_score"),
        "quality_score": paper.get("_quality_score"),
        "filter_reason": paper.get("_filtered_reason"),
        "abstract_preview": _preview_text(paper.get("abstract"), limit=180),
    }


def _summarize_papers(papers: list[dict], limit: int = 10) -> list[dict]:
    """返回论文列表的调试摘要。"""
    return [_paper_debug_item(paper) for paper in (papers or [])[:limit]]


def _latest_step(state: "ResearchAgentState", step_name: str) -> dict:
    """取最近一次指定步骤记录。"""
    for step in reversed(state.get("steps") or []):
        if step.get("step_name") == step_name:
            return step
    return {}


def _needs_current_time_tool(user_query: str) -> bool:
    """判断解析年份是否需要运行时当前时间。"""
    query = str(user_query or "")
    # 用户给了闭区间年份，如 2022-2025 / 2022 到 2025，不需要当前时间。
    if re.search(r"(?:19|20)\d{2}\s*(?:[-~—到至])\s*(?:19|20)\d{2}", query):
        return False
    return True

