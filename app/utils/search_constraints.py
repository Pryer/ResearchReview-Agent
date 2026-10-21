"""纯检索任务的返回数量边界；综述引用门禁继续使用原有硬要求。"""
from __future__ import annotations

import re
from typing import Any


def exact_search_title(state: dict[str, Any]) -> str | None:
    request = state.get("research_request") or {}
    if (state.get("intent") or request.get("task_type")) != "search_papers":
        return None
    query = str(request.get("original_query") or state.get("user_query") or "")
    prefix = r"(?:标题|题名)\s*(?:严格\s*)?(?:等于|为|是)\s*"
    quoted = re.search(prefix + r'["“「](.+?)["”」]', query)
    if quoted:
        return quoted.group(1).strip()
    raw = re.search(prefix + r"(.+?)(?=\s*的(?:20\d{2}\s*年)?论文|\s*的?20\d{2}\s*年|[，。；,;\n]|$)", query)
    return raw.group(1).strip() if raw else None


def search_count_bounds(state: dict[str, Any]) -> dict[str, int | None] | None:
    request = state.get("research_request") or {}
    if (state.get("intent") or request.get("task_type")) != "search_papers":
        return None
    query = str(request.get("original_query") or state.get("user_query") or "")
    upper = re.search(r"(?:最多|至多|不超过|上限(?:为)?|at\s+most|up\s+to|no\s+more\s+than)\s*(\d+)\s*(?:篇|papers?\b)", query, re.I)
    minimum = int(state.get("required_reference_count") or request.get("required_reference_count") or 0) if (
        state.get("max_papers_explicit") or request.get("max_papers_explicit")) else 0
    if not upper:
        return {"minimum": minimum, "maximum": None}
    # WHY: “最多 N 篇”只约束返回上限，不能被内部检索目标变成至少 N 篇；
    # 若同一请求还明确写了下限，两者都保留，不能借上限抹去下限。
    lower = re.search(r"(?:至少|最少|不少于|不低于|at\s+least)\s*(\d+)\s*(?:篇|papers?\b)", query, re.I)
    return {"minimum": int(lower.group(1)) if lower else 0, "maximum": int(upper.group(1))}
