"""核验配置身份与当前调用回执；不保存认证信息。"""
import hashlib
import json
from contextvars import ContextVar

_RECEIPT = ContextVar("verifier_receipt", default=None)


def set_verifier_receipt(value):
    _RECEIPT.set(value)


def verifier_receipt():
    return _RECEIPT.get()


def verifier_profile(llm=None):
    from app.core.config import get_settings, get_review_threshold_policy
    from app.prompt.claim_verification import CLAIM_ENTAILMENT_PROMPT
    from app.prompt.agent_system import SYSTEM_PROMPTS
    from app.agent.tool_registry import tools_for_role
    config = get_settings()
    fields = ("provider", "base_url", "model", "backup_enabled", "backup_provider", "backup_base_url", "backup_model",
              "thinking_enabled", "reasoning_mode", "thinking_effort", "thinking_max_tokens", "max_tokens")
    values = {key: getattr(config, "llm_" + key, None) for key in fields}
    if llm is not None:
        values.update({key: getattr(llm, key, values[key]) for key in fields})
        values["implementation"] = type(llm).__module__ + "." + type(llm).__qualname__
        values["role_prompt"] = getattr(llm, "_system_prompt", "")
    # WHY: endpoint 只参与哈希，不把可能含凭证的地址写入报告。
    values.update(prompt=CLAIM_ENTAILMENT_PROMPT, system={role: SYSTEM_PROMPTS[role] for role in ("check", "analysis")},
                  thresholds=get_review_threshold_policy().snapshot(), context_limit=config.claim_verification_max_context_chars,
                  tools=tools_for_role("check"), temperature=0.0, profile_version=1)
    from app.core.role_model_profiles import profile_for
    # WHY: Check配置变化使旧报告失效；Search/Writing调参不撤销独立核验结论。
    profile = profile_for('check')
    values['check_model_profile'] = profile.model_dump(mode='json') if profile else None
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
