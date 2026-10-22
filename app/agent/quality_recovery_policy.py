"""质量恢复的约束授权；动作调度和预算仍由既有恢复器负责。"""
from __future__ import annotations

import re
from typing import Any


def forbidden_quality_actions(state: dict[str, Any], original_query: str = "") -> dict[str, str]:
    request = state.get("research_request") or {}
    query = original_query or str(state.get("quality_original_query") or request.get("original_query") or state.get("user_query") or "")
    denied: dict[str, str] = {}
    # WHY: 并列否定句共享否定词，如“不能扩大标题、年份或降低篇数”；不能
    # 只匹配每个字段前紧邻的否定词，也不能把一般的最低篇数理解为禁止用户改约。
    clauses = re.split(r"[。；;\n]", query)
    for clause in clauses:
        negative = re.search(r"不能|不得|禁止|不允许|不可|不要|不扩大|不放宽|不降低|不改变", clause)
        if not negative:
            continue
        restriction = clause[negative.start():]
        for action, pattern in {
            "accept_available": r"(?:降低|减少|放宽|改变).{0,12}(?:篇数|数量|引用要求)|(?:篇数|数量).{0,8}(?:降低|减少|放宽|改变)",
            "expand_time_range": r"年份|时间范围|时间窗",
            "broaden_scope": r"题名|标题|主题|研究范围",
        }.items():
            if re.search(pattern, restriction):
                denied[action] = "用户明确禁止放宽此项要求"
    if re.search(r"(?:标题|题名)\s*严格\s*(?:等于|为)|仅允许.{0,16}精确(?:标题|题名)", query):
        denied["broaden_scope"] = "精确题名是当前证据范围的硬约束"
    # WHY: 用户可明确改约；只撤销已记录动作对应的限制，含糊的“继续”不能解锁。
    for change in state.get("quality_constraint_changes") or []:
        denied.pop(change.get("action"), None)
    if "accept_available" in denied:
        denied["force_generate"] = "不能通过最佳努力草稿绕过禁止降低篇数的要求"
    return denied


def concrete_quality_input(issues: list[dict[str, Any]]) -> str:
    return "；".join(
        str(item.get("message") or "").strip()
        for item in issues
        if item.get("code") in {"authentication_required", "human_action_required", "missing_user_material"}
        and str(item.get("message") or "").strip()
    )


def explicit_constraint_change(answer: str, action: str) -> bool:
    targets = {
        "accept_available": r"篇|引用|数量",
        "expand_time_range": r"年|时间",
        "broaden_scope": r"题名|标题|主题|范围",
    }
    # 仅明确声明修改要求才允许旧会话中的受限动作；不能按选项回显推断授权。
    return bool(action in targets and re.search(r"修改要求|更改要求|现在允许|改为|取消.{0,12}限制", answer)
                and re.search(targets[action], answer)
                and not re.search(r"不能|不要|不允许|禁止", answer))
