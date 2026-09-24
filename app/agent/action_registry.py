"""主控动作的单一契约：工具 schema、角色与前置条件来自同一注册表。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schemas.agent_task_schema import AgentRole


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    role: AgentRole | None
    required_fields: tuple[str, ...] = ()
    output_fields: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=lambda: {
        "type": "object", "properties": {}, "additionalProperties": False,
    })


_ACTION_SPECS = (
    ActionSpec("search_and_rank", "按当前范围检索、筛选和排序论文", AgentRole.SEARCH,
               output_fields=("candidate_papers", "ranked_papers", "source_diagnostics")),
    ActionSpec("targeted_search", "针对已确认的证据缺口执行一次有界补检索", AgentRole.SEARCH,
               required_fields=("evidence_gap_report",), output_fields=("candidate_papers", "ranked_papers")),
    ActionSpec("fetch_metadata", "补全并核验已排序论文的元数据", AgentRole.SEARCH,
               required_fields=("ranked_papers",), output_fields=("paper_details", "source_diagnostics")),
    ActionSpec("extract_paper_cards", "从已核验资料提取带溯源的证据卡", AgentRole.ANALYSIS,
               required_fields=("paper_details",), output_fields=("paper_cards",)),
    ActionSpec("validate_routes", "用证据验证候选研究路线", AgentRole.ANALYSIS,
               required_fields=("paper_cards",), output_fields=("validated_routes", "route_decisions")),
    ActionSpec("cluster_papers", "没有有效候选路线时执行证据驱动聚类", AgentRole.ANALYSIS,
               required_fields=("paper_cards",), output_fields=("clusters", "dynamic_taxonomy")),
    ActionSpec("plan_claims", "为研究路线建立主张与证据授权", AgentRole.ANALYSIS,
               required_fields=("paper_cards",), output_fields=("claim_plans", "evidence_gap_report")),
    ActionSpec("generate_deliverables", "按计划和获准证据生成交付物", AgentRole.WRITING,
               required_fields=("paper_cards", "claim_plans"), output_fields=("review", "writing_plans")),
    ActionSpec(
        "rewrite_sections", "仅重写指定且已有证据授权的章节", AgentRole.WRITING,
        required_fields=("review", "writing_plans"), output_fields=("review",),
        parameters={
            "type": "object",
            "properties": {"section_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["section_ids"],
            "additionalProperties": False,
        },
    ),
    ActionSpec("validate_result", "运行确定性的主张、引用和输出验证", AgentRole.ANALYSIS,
               required_fields=("review",), output_fields=("quality_gate", "claim_verification")),
    ActionSpec(
        "request_clarification", "信息不足时向用户请求一个必要澄清", None,
        parameters={
            "type": "object",
            "properties": {"question": {"type": "string", "minLength": 1}},
            "required": ["question"], "additionalProperties": False,
        },
    ),
    ActionSpec("request_finish", "申请交付当前结果；最终门禁由代码裁决", None),
    ActionSpec(
        "report_blocked", "在无法安全继续时报告明确阻断原因", None,
        parameters={
            "type": "object",
            "properties": {"reason": {"type": "string", "minLength": 1}},
            "required": ["reason"], "additionalProperties": False,
        },
    ),
)

ACTION_REGISTRY = {item.name: item for item in _ACTION_SPECS}


def action_spec(name: str) -> ActionSpec:
    try:
        return ACTION_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown agent action: {name}") from exc


def allowed_actions(
    state: dict[str, Any], registered_handlers: set[str] | None = None
) -> list[str]:
    handlers = registered_handlers or set(ACTION_REGISTRY)
    mode = str(state.get("agent_operation_mode") or "initial")
    if mode == "verification_only":
        return [
            name for name in ("validate_result", "request_finish", "report_blocked")
            if name in ACTION_REGISTRY and (ACTION_REGISTRY[name].role is None or name in handlers)
        ]
    allowed: list[str] = []
    for spec in _ACTION_SPECS:
        if state.get("intent") == "search_papers" and spec.role is not None and spec.name != "search_and_rank":
            continue
        if spec.role is not None and spec.name not in handlers:
            continue
        if spec.name == "targeted_search":
            from app.agent.evidence_recovery import targeted_search_kind
            if not targeted_search_kind(state):
                continue
        elif any(not state.get(field) for field in spec.required_fields):
            continue
        if mode == "regeneration" and spec.name in {"search_and_rank", "fetch_metadata", "targeted_search"}:
            continue
        allowed.append(spec.name)
    # 控制动作始终可用；研究动作仍须由真实 handler 注册。
    return allowed


def tool_schemas(names: list[str]) -> list[dict[str, Any]]:
    return [{
        "type": "function",
        "function": {
            "name": ACTION_REGISTRY[name].name,
            "description": ACTION_REGISTRY[name].description,
            "parameters": ACTION_REGISTRY[name].parameters,
        },
    } for name in names if name in ACTION_REGISTRY]


def validate_arguments(name: str, arguments: dict[str, Any]) -> None:
    schema = action_spec(name).parameters
    allowed = set((schema.get("properties") or {}).keys())
    unknown = set(arguments) - allowed
    if unknown or (not schema.get("additionalProperties", True) and unknown):
        raise ValueError(f"unexpected arguments for {name}: {sorted(unknown)}")
    missing = set(schema.get("required") or []) - set(arguments)
    if missing:
        raise ValueError(f"missing arguments for {name}: {sorted(missing)}")
    if name == "rewrite_sections" and (not isinstance(arguments.get("section_ids"), list) or not arguments["section_ids"] or not all(
        isinstance(item, str) and item.strip() for item in arguments["section_ids"]
    )):
        raise ValueError("rewrite_sections.section_ids must contain strings")
    for field in ("question", "reason"):
        if field in schema.get("required", []) and (not isinstance(arguments.get(field), str) or not arguments[field].strip()):
            raise ValueError(f"{field} must be a nonempty string")
