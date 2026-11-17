"""Task-scoped, credential-free configuration; persisted usage is never reset."""
from contextlib import contextmanager
import hashlib
import json
import os
from app.core.config import get_settings, _TASK_SETTINGS

POLICY_VERSION = "draft-delivery.v1"
PREFIXES = ("llm_", "writer_", "agent_", "main_context_", "research_memory_", "retrieval_", "rerank_",
            "card_", "evidence_", "recovery_", "route_", "verify_", "claim_", "synthesis_",
            "search_", "paper_screening_", "reference_", "global_gate_", "language_branch_")
EXACT = {"llm_model", "llm_provider", "llm_backup_model", "llm_backup_provider", "review_delivery_mode", "llm_request_timeout", "llm_control_plane_timeout",
         "llm_failover_total_timeout", "default_search_sources", "min_results_per_keyword",
         "max_results_per_keyword", "enable_claim_verification", "enable_global_evidence_gate"}

def _snapshot_fields(settings):
    # WHY: 仅持久化白名单标量，避免把凭证或连接串写入任务快照。
    return {name: value for name, value in settings.model_dump().items()
            if (name in EXACT or name.startswith(PREFIXES))
            and not any(term in name for term in ("api_key", "password", "secret", "url", "dsn"))
            and isinstance(value, (bool, int, float, str))}

def delivery_mode(state):
    # WHY: 新默认值不追溯放宽旧任务；旧策略只能通过显式修订变更。
    return str(state.get("delivery_mode") or
               (state.get("effective_config") or {}).get("values", {}).get("review_delivery_mode") or "strict")

def initialize_delivery(state, query="", *, new_task=False):
    if "delivery_mode" not in state:
        value = get_settings().review_delivery_mode if new_task else "strict"
        if any(term in query for term in ("不接受草稿", "只交完整验收稿", "只要正式稿")):
            value = "strict"
        elif any(term in query for term in ("先交草稿", "允许草稿", "先给草稿")):
            value = "draft_first"
        state["delivery_mode"] = value

@contextmanager
def task_settings_scope(state):
    base = get_settings()
    if not state.get("effective_config"):
        values = _snapshot_fields(base)
        values["review_delivery_mode"] = delivery_mode(state)
        from dotenv import dotenv_values
        dotenv_keys={str(k).upper() for k in dotenv_values(".env")}
        cli_keys=set(os.environ.get("REVIEW_CLI_OVERRIDE_KEYS", "").split(","))
        sources = {key: "cli" if key.upper() in cli_keys else "environment" if key.upper() in os.environ else "dotenv" if key.upper() in dotenv_keys else "default" for key in values}
        state["effective_config"] = {"version": POLICY_VERSION, "revision": 1, "values": values,
            "sources": sources, "fingerprint": hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()}
    values = dict(state["effective_config"]["values"])
    # WHY: 旧引擎已退役；只升级执行架构，目标、交付策略、模型和累计预算保持原任务值。
    architecture = values.get('agent_architecture_version')
    if architecture not in {None, 'v1', 'v2'}:
        raise ValueError('unsupported persisted architecture version')
    if architecture != 'v2':
        values['agent_architecture_version'] = 'v2'
    values.setdefault('agent_v2_local_strategy', 'deterministic')
    values.setdefault('agent_v2_analysis_strategy', 'inherit')
    # WHY: 新角色配置不追溯改变旧任务；显式修订后才使用新的模型身份。
    values.setdefault('agent_role_model_profiles', '{}')
    # Validate task values against field bounds without letting process changes alter a task.
    settings = type(base)(**{**base.model_dump(), **values})
    if architecture != 'v2':
        # WHY: 所有历史配置通过容量/参数校验后才提交升级标记，失败时不污染原快照。
        snapshot = state['effective_config']
        snapshot.setdefault('architecture_migration', {'from': architecture or 'v1', 'to': 'v2'})
        snapshot['values'] = dict(values)
        snapshot['fingerprint'] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    token = _TASK_SETTINGS.set(settings)
    try:
        yield settings
    finally:
        _TASK_SETTINGS.reset(token)
