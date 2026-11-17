"""专业能力从 Action owner 派生，避免维护第二份权限列表。"""
from dataclasses import dataclass
from app.agent.action_registry import ACTION_REGISTRY
from app.schemas.agent_task_schema import AgentRole


@dataclass(frozen=True)
class AgentSpec:
    role: AgentRole
    strategy: str

    @property
    def model_profile(self):
        from app.core.role_model_profiles import profile_for
        return profile_for(self.role.value)

    @property
    def actions(self):
        return frozenset(name for name, spec in ACTION_REGISTRY.items() if spec.role == self.role)


AGENTS = {role.value: AgentSpec(role, "react" if role in {AgentRole.SEARCH, AgentRole.ANALYSIS} else "direct")
          for role in AgentRole}


def validate_registry(handlers):
    from app.agent.registry.tool_registry import validate_tool_registry
    validate_tool_registry()
    from app.agent.action_contracts import CONTRACTS
    from app.agent.policy.capability_profiles import PROFILES
    professional = {name for name, spec in ACTION_REGISTRY.items() if spec.role}
    if professional - set(handlers) or professional - set(CONTRACTS):
        raise ValueError("incomplete professional action registration")
    for profile in PROFILES.values():
        if not set(profile.actions) <= AGENTS[profile.role.value].actions:
            raise ValueError("profile exceeds agent capabilities")
