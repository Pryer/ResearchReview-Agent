"""委派授权的唯一入口，复用现有领域准入与字段契约。"""
import copy
from app.agent.action_registry import ACTION_REGISTRY, allowed_actions, validate_arguments, recovery_handoff_action
from app.agent.context_builder import _goal, _stable_hash
from app.agent.policy.capability_profiles import PROFILES
from app.schemas.delegation_schema import DelegationGrant, SCOPE_FIELDS


def goal_fingerprint(state):
    return _stable_hash(_goal(state).model_dump(mode="json"))


class DelegationAuthorizationError(ValueError):
    """授权拒绝与业务执行异常分开处理，避免自动重试错误的业务结果。"""


class AuthorizationPolicy:
    def authorize_single(self, state, *, role, operation, arguments, goal):
        from app.schemas.agent_task_schema import AgentRole
        role = AgentRole(role)
        if operation not in ACTION_REGISTRY or ACTION_REGISTRY[operation].role != role:
            raise DelegationAuthorizationError("single action role does not own action")
        validate_arguments(operation, arguments)
        return DelegationGrant(session_id=state['session_id'], role=role, profile='single_action',
            goal=goal, goal_fingerprint=goal_fingerprint(state), allowed_actions=(operation,),
            immutable_arguments=copy.deepcopy(arguments), max_steps=1,
            scope={'section_ids': copy.deepcopy(arguments.get('section_ids') or [])}
                  if operation == 'rewrite_sections' else {})

    def migrate_single(self, state, **kwargs):
        # WHY: 历史标签仅在迁移桥保留；新任务的精确恢复授权使用single_action。
        return self.authorize_single(state, **kwargs).model_copy(update={'profile': 'legacy_single_v1'})

    def issue(self, request, state):
        profile = PROFILES.get(request.profile)
        if profile is None:
            raise DelegationAuthorizationError('unknown capability profile')
        if profile.role != request.agent:
            raise DelegationAuthorizationError(
                f"Profile {profile.name} requires agent={profile.role.value}. "
                "Choose a profile and its matching role from the current catalog."
            )
        scope = copy.deepcopy(request.scope)
        # WHY: 范围只允许收窄；年份来自权威目标，不接受模型重新定义。
        if set(scope) - set(SCOPE_FIELDS):
            raise DelegationAuthorizationError("unsupported delegation scope")
        for key, values in scope.items():
            if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
                raise DelegationAuthorizationError("scope must contain nonempty identifier lists")
        if scope and profile.name not in {"section_revision", "gap_recovery", "gap_diagnosis", "evidence_build"}:
            raise DelegationAuthorizationError("profile does not accept scope")
        if "section_ids" in scope and profile.name != "section_revision":
            raise DelegationAuthorizationError("section scope belongs to writing")
        if "gap_ids" in scope and profile.name not in {"gap_recovery", "gap_diagnosis"}:
            raise DelegationAuthorizationError("gap scope belongs to recovery")
        if 'paper_ids' in scope:
            ids = scope['paper_ids']
            known = {p.get('paper_id') for p in state.get('paper_details') or []}
            if (profile.name != 'evidence_build' or set(scope) != {'paper_ids'} or len(ids) > 32
                    or len(ids) != len(set(ids)) or not set(ids) <= known):
                raise DelegationAuthorizationError('paper subset is not a valid evidence extraction scope')
            if state.get('agent_mandatory_actions'):
                raise DelegationAuthorizationError('paper subset cannot replace a mandatory recovery action')
        if profile.name == "section_revision" and not scope.get("section_ids"):
            raise DelegationAuthorizationError("section revision requires explicit scope")
        if profile.name == 'section_revision':
            known = {section.get('id') for plan in state.get('writing_plans') or []
                     for section in plan.get('sections') or []}
            requested = set(scope['section_ids'])
            permitted = set(state.get('target_section_ids') or known)
            if not requested <= known or not requested <= permitted or len(requested) > 3:
                raise DelegationAuthorizationError('section scope exceeds the current writing request')
        return DelegationGrant(session_id=state["session_id"], role=profile.role,
            profile=profile.name, goal=request.goal, scope=scope,
            goal_fingerprint=goal_fingerprint(state),
            allowed_actions=('extract_paper_cards',) if scope.get('paper_ids') else profile.actions,
            max_steps=1 if scope.get('paper_ids') else profile.max_steps)

    def assert_target(self, grant, state):
        if grant.session_id != state.get("session_id") or grant.goal_fingerprint != goal_fingerprint(state):
            raise DelegationAuthorizationError("delegation target changed")
        if grant.scope.get('paper_ids') and not set(grant.scope['paper_ids']) <= {p.get('paper_id') for p in state.get('paper_details') or []}:
            raise DelegationAuthorizationError('scoped paper is no longer in the current research input')
        if grant.scope.get('paper_ids') and state.get('agent_mandatory_actions'):
            raise DelegationAuthorizationError('paper subset cannot consume a global mandatory action')

    def available_actions(self, grant, state, handlers):
        self.assert_target(grant, state)
        names = [name for name in allowed_actions(state, set(handlers)) if name in grant.allowed_actions
                and ACTION_REGISTRY[name].role == grant.role and name in handlers]
        mandatory = state.get("agent_mandatory_actions") or []
        required = mandatory[0].get("action") if mandatory else recovery_handoff_action(state)
        return [name for name in names if name == required] if required else names

    def assert_action_allowed(self, grant, state, handlers, action, arguments):
        if action not in self.available_actions(grant, state, handlers):
            raise DelegationAuthorizationError("action not authorized")
        validate_arguments(action, arguments)
        if grant.immutable_arguments is not None and arguments != grant.immutable_arguments:
            raise DelegationAuthorizationError("single action arguments are immutable")
        if grant.profile == "section_revision":
            requested = arguments.get("section_ids") or []
            if not requested or not set(requested) <= set(grant.scope.get("section_ids") or []):
                raise DelegationAuthorizationError("section scope exceeded")
            if len(requested) > 3:
                raise DelegationAuthorizationError("section revision limit exceeded")
        if arguments.get("gap_ids") and not set(arguments["gap_ids"]) <= set(grant.scope.get("gap_ids") or []):
            raise DelegationAuthorizationError("gap scope exceeded")
