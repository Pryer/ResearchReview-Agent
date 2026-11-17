"""可恢复的角色模型/提示身份；地址仅参与哈希，运行报告不保存连接信息。"""
import hashlib
import json
from app.core.config import get_settings


def bind_role(llm, role):
    binder = getattr(llm, 'for_agent', None)
    return binder(role) if callable(binder) else llm


def execution_profile(llm, role):
    settings = get_settings()
    from app.core.role_model_profiles import profile_for
    from app.prompt.agent_system import SYSTEM_PROMPTS
    from app.agent.tool_registry import tools_for_role
    profile = profile_for(role)
    primary = {name: getattr(llm, name, getattr(settings, 'llm_' + name, None)) for name in (
        'provider', 'model', 'base_url', 'temperature', 'max_tokens', 'control_plane_max_tokens',
        'thinking_enabled', 'thinking_effort', 'native_tools_enabled')}
    fallback = {name: getattr(llm, 'backup_' + name, getattr(settings, 'llm_backup_' + name, None))
                for name in ('provider', 'model', 'base_url')}
    enabled = bool(getattr(llm, 'backup_enabled', settings.llm_backup_enabled))
    window = getattr(llm, 'context_window_tokens', settings.agent_context_window_tokens)
    payload = {'version': 1, 'role': role, 'primary': primary, 'fallback': fallback if enabled else None,
        'context_window_tokens': window, 'profile': profile.model_dump() if profile else None,
        'prompt': SYSTEM_PROMPTS[role], 'tools': tools_for_role(role)}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {'version': 1, 'role': role, 'provider': primary['provider'], 'model': primary['model'],
        'fallback_provider': fallback['provider'] if enabled else None,
        'fallback_model': fallback['model'] if enabled else None,
        'context_window_tokens': window, 'fingerprint': fingerprint}
