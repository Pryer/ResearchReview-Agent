"""前端查询文本处理工具。"""

from __future__ import annotations


INTENT_HINTS = (
    "综述", "文献综述", "调研", "研究现状", "survey", "review",
    "找", "搜索", "检索", "推荐", "论文", "paper",
    "对比", "比较", "总结", "参考文献", "引用", "数据集", "趋势",
)


def normalize_review_query(user_query: str) -> str:
    """将裸主题补全为当前支持的“研究现状”请求。"""
    query = user_query.strip()
    if not query:
        return query

    query_lower = query.lower()
    if any(hint.lower() in query_lower for hint in INTENT_HINTS):
        return query
    return f"帮我调研{query}相关论文，并生成研究现状"


def build_agent_request_payload(
    user_query: str,
    session_id: str | None = None,
    *,
    clarification_answer: str | None = None,
) -> dict:
    """构造异步 Agent 请求；澄清回答不得被包装成新的综述主题。"""
    is_clarification = bool(str(clarification_answer or "").strip())
    payload = {
        "user_query": (
            str(user_query).strip()
            if is_clarification
            else normalize_review_query(user_query)
        ),
        "session_id": session_id,
    }
    if is_clarification:
        payload["clarification_answer"] = str(clarification_answer).strip()
    return payload
