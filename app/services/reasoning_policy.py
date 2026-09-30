"""Operation-based reasoning choices shared by completions and tool calls."""

from __future__ import annotations


def choose_reasoning(
    operation: str,
    *,
    mode: str,
    default_enabled: bool,
    default_effort: str,
    requested_enabled: bool | None = None,
    requested_effort: str | None = None,
) -> tuple[bool, str]:
    if requested_effort is not None and requested_effort not in {"low", "high", "max"}:
        raise ValueError("reasoning_effort must be low, high, or max")
    if requested_enabled is not None:
        return requested_enabled, requested_effort or default_effort
    name = str(operation or "").lower()
    if mode != "auto":
        # WHY: static 必须遵守全局开关；写作不能暗中开启思考并扩大输出预留。
        return default_enabled, requested_effort or default_effort
    if name.startswith(("generate_search_keywords", "keyword", "scope_answer")):
        return False, default_effort
    if name.startswith(("research_query_rewrite", "research_semantic", "main_agent_decision")):
        return True, "low"
    if name.startswith(("repair", "polish", "rewrite")) or "attempt_2" in name or "attempt_3" in name:
        return True, "high"
    if name.startswith(("render", "write", "backfill_citations", "synthesize")):
        return True, "low"
    return default_enabled, requested_effort or default_effort
