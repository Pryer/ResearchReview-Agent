"""前端查询处理测试。"""

from __future__ import annotations

from app.frontend.query_utils import build_agent_request_payload, normalize_review_query


def test_normalize_review_query_wraps_bare_topic():
    assert normalize_review_query("少样本动作识别") == (
        "帮我调研少样本动作识别相关论文，并生成研究现状"
    )


def test_normalize_review_query_keeps_explicit_request():
    query = "帮我生成少样本动作识别的综述"
    assert normalize_review_query(query) == query


def test_clarification_answer_is_not_rewritten_as_new_research_request():
    answer = "先自动识别和编码，再从教育学角度分析"
    payload = build_agent_request_payload(
        answer,
        "session-1",
        clarification_answer=answer,
    )
    assert payload == {
        "user_query": answer,
        "session_id": "session-1",
        "clarification_answer": answer,
    }
