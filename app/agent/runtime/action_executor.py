"""任务级授权适配现有单一 Controller；每步保持独立事务。"""
from app.agent.context_builder import context_source_fingerprint
from app.agent.policy.authorization import AuthorizationPolicy, DelegationAuthorizationError
from app.database.delegation_repository import DelegationRepository
from app.database.runtime_repository import RuntimeConflict
from app.schemas.agent_task_schema import AgentTaskResult
from app.services.invocation_scope import invocation_scope
from app.services.durable_execution_service import active_runtime, active_task_id


class ActionExecutor:
    def __init__(self, controller, *, sidecar=None):
        self.controller = controller
        self.authorization = AuthorizationPolicy()
        self.sidecar = sidecar

    def execute(self, invocation, state, handlers, *, should_cancel=None):
        from contextlib import nullcontext
        from app.agent.execution_budget import action_time_budget
        from app.core.config import get_settings
        timing = action_time_budget(get_settings().agent_section_revision_timeout_seconds) if invocation.action == 'rewrite_sections' else nullcontext()
        # WHY: 审查和业务执行共用本次Action期限，不能审查花满预算后再重新计时。
        with timing:
            return self._execute(invocation, state, handlers, should_cancel=should_cancel)

    def _execute(self, invocation, state, handlers, *, should_cancel=None):
        runtime = active_runtime()
        if runtime is None or active_task_id():
            raise RuntimeConflict("delegation requires an independent durable action scope")
        repo = DelegationRepository(runtime)
        row, grant = repo.read(invocation.delegation_id)
        if row.status != "running" or invocation.grant_revision != grant.revision:
            raise RuntimeConflict("delegation is no longer active")
        # WHY: 重放也重新检查会话目标/授权；仅返回已提交结果，不合并旧 patch。
        self.authorization.assert_target(grant, state)
        cached = repo.committed_result(invocation)
        if cached:
            return AgentTaskResult.model_validate(cached)
        self.authorization.assert_action_allowed(grant, state, handlers, invocation.action, invocation.arguments)
        from app.core.config import get_settings
        settings = get_settings()
        if settings.agent_sidecar_enabled and invocation.action in {
            name.strip() for name in settings.agent_sidecar_review_actions.split(',')
        }:
            if self.sidecar is None:
                raise DelegationAuthorizationError('configured sidecar must be supplied to v2 executor')
            review = self.sidecar.review(operation=invocation.action, arguments=invocation.arguments)
            from uuid import uuid4
            with runtime.transaction():
                runtime._owned_row_for_write(live=True)
                runtime.event('v2-review:' + uuid4().hex, 'pre_execution_review', {
                    'invocation_id': invocation.invocation_id, 'verdict': review.verdict,
                    'reason': review.reason, 'grant_revision': grant.revision})
            if review.verdict != 'allow':
                raise DelegationAuthorizationError('pre-execution review requires Main intervention: ' + review.verdict)
            row, grant = repo.read(invocation.delegation_id)
            if row.status != 'running':
                raise RuntimeConflict('delegation revoked during review')
            self.authorization.assert_action_allowed(grant, state, handlers, invocation.action, invocation.arguments)
        if invocation.source_fingerprint != context_source_fingerprint(state):
            raise RuntimeConflict("action input changed")
        repo.prepare(invocation)
        def run_handler(working):
            from app.services.tool_execution import tool_execution_scope
            with tool_execution_scope(invocation):
                return handlers[invocation.action](working, invocation.arguments)
        with invocation_scope(invocation):
            mandatory = state.get("agent_mandatory_actions") or []
            return self.controller.execute(role=grant.role, operation=invocation.action,
                objective=grant.goal, state=state,
                handler=run_handler,
                constraints=invocation.arguments, should_cancel=should_cancel, invalidate_evidence=True,
                paper_scope=tuple(grant.scope.get('paper_ids') or ()),
                consume_mandatory_action=bool(mandatory and mandatory[0].get("action") == invocation.action))
