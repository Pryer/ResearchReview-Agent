"""有界摘要上下文；只选择原文，不推断未读内容。"""
from __future__ import annotations


SCREENING_CONTEXT_VERSION = "abstract-head-tail.v1"


def select_abstract_context(text: str, budget: int) -> tuple[str, dict]:
    """短摘要完整保留，长摘要保留首尾并显式标记中间省略。"""
    if budget < 1:
        raise ValueError("摘要预算必须为正数")
    if len(text) <= budget:
        return text, {"strategy": "full", "truncated": False,
                      "source_chars": len(text), "ranges": [[0, len(text)]]}
    marker = "\n[摘要中间部分因上下文预算省略]\n"
    # WHY: 省略标记也计入预算；分段不能拼成貌似连续的原文证据。
    if budget <= len(marker):
        return text[:budget], {"strategy": "head", "truncated": True,
                               "source_chars": len(text), "ranges": [[0, budget]]}
    remaining = budget - len(marker)
    head = (remaining + 1) // 2
    tail = remaining // 2
    return text[:head] + marker + (text[-tail:] if tail else ""), {
        "strategy": "head_tail", "truncated": True, "source_chars": len(text),
        "ranges": [[0, head], [len(text) - tail, len(text)]],
    }
