"""确定性上下文压缩和保真校验。"""

from __future__ import annotations

import json
from typing import Any

from app.schemas.context_schema import ContextSnapshot, MainAgentContext


def _size(context: MainAgentContext) -> int:
    return len(json.dumps(context.model_dump(mode="json"), ensure_ascii=False))


def compact_main_context(context: MainAgentContext, *, max_chars: int) -> MainAgentContext:
    """按可丢弃优先级收缩上下文，永不删除目标和显式约束。"""
    compacted = context.model_copy(deep=True)
    max_chars = max(2000, int(max_chars))
    while _size(compacted) > max_chars and len(compacted.key_evidence) > 3:
        compacted.key_evidence.pop()
    while _size(compacted) > max_chars and len(compacted.decisions) > 2:
        removable = next((i for i, item in enumerate(compacted.decisions) if item.source != "user"), None)
        if removable is None:
            break
        compacted.decisions.pop(removable)
    while _size(compacted) > max_chars and len(compacted.state.completed_actions) > 3:
        compacted.state.completed_actions.pop(0)
    while _size(compacted) > max_chars and len(compacted.open_questions) > 1:
        removable = next((i for i, item in enumerate(compacted.open_questions) if not item.blocking), None)
        if removable is None:
            break
        compacted.open_questions.pop(removable)
    return compacted


def validate_context_snapshot(snapshot: ContextSnapshot, state: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    context = snapshot.context
    if snapshot.estimated_chars > snapshot.max_chars:
        errors.append("context_budget_exceeded")
    for required in _expected_explicit_constraints(state):
        if not any(
            item.get("name") == required["name"] and item.get("value") == required["value"]
            for item in context.goal.explicit_constraints
        ):
            errors.append(f"missing_explicit_constraint:{required['name']}")
    valid_papers = {str(item.get("paper_id") or "") for item in state.get("paper_cards") or []}
    for evidence in context.key_evidence:
        if any(paper_id not in valid_papers for paper_id in evidence.paper_ids):
            errors.append(f"unknown_evidence_paper:{evidence.evidence_id}")
    return errors


def _expected_explicit_constraints(state: dict[str, Any]) -> list[dict[str, Any]]:
    from app.agent.context_builder import _explicit_constraints
    return _explicit_constraints(state)
