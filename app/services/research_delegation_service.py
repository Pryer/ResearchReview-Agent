"""任务级研究入口：复用graph组合根，恢复以数据库委派为准。"""
import copy
from app.agent.controller import AgentController
from app.agent.policy.authorization import AuthorizationPolicy
from app.agent.runtime.action_executor import ActionExecutor
from app.agent.runtime.delegation_runner import DelegationRunner
from app.agent.runtime.search_strategy import RoleReActStrategy, RolePlanAndActStrategy
from app.agent.execution_budget import AgentFinalizationRequired
from app.agent.sidecar import ActionSidecar
from app.database.delegation_repository import DelegationRepository
from app.database.runtime_repository import RuntimeConflict
from app.schemas.delegation_schema import DelegationRequest
from app.services.durable_execution_service import active_runtime, active_task_id


def run_delegation(state, request, *, llm, strategy=None, should_cancel=None, handlers=None, forced_operation=None, forced_arguments=None):
    runtime = active_runtime()
    if runtime is None or active_task_id() or state.get('active_subagent'):
        raise RuntimeConflict('task delegation requires an independent durable research scope')
    request = DelegationRequest.model_validate(request)
    from app.services.role_execution_profile import bind_role, execution_profile
    role_llm = bind_role(llm, request.agent.value)
    if strategy == 'configured':
        from app.core.config import get_settings
        settings = get_settings()
        strategy = (settings.agent_v2_analysis_strategy if request.profile == 'evidence_build'
                    and settings.agent_v2_analysis_strategy != 'inherit' else settings.agent_v2_local_strategy)
    if strategy in ('react', 'plan_and_act', 'deterministic'):
        if strategy == 'plan_and_act' and request.profile != 'evidence_build':
            raise ValueError('plan_and_act is limited to evidence_build')
        local_strategy = (RoleReActStrategy(role_llm) if strategy == 'react' else
                          RolePlanAndActStrategy(role_llm) if strategy == 'plan_and_act' else None)
    elif isinstance(strategy, str):
        raise ValueError('unknown local strategy')
    else:
        local_strategy = strategy
    # WHY: 模型请求先通过静态授权与范围验证，再允许组合根初始化运行快照。
    proposed = (AuthorizationPolicy().authorize_single(state, role=request.agent, operation=forced_operation,
        arguments=forced_arguments or {}, goal=request.goal) if forced_operation else AuthorizationPolicy().issue(request, state))
    repo = DelegationRepository(runtime)
    recorded = state.get('active_delegation_id')
    active = repo.read(recorded)[0] if recorded else repo.active()
    if active:
        _, grant = repo.read(active.delegation_id)
        if (grant.profile != request.profile or grant.role != request.agent
                or grant.scope != request.scope or grant.goal != request.goal):
            raise RuntimeConflict('resume request differs from active delegation')
        AuthorizationPolicy().assert_target(grant, state)
    metadata = execution_profile(role_llm, request.agent.value)
    control = state.setdefault('delegation_control', {}).setdefault(
        grant.delegation_id if active else proposed.delegation_id, {'rounds': 0, 'stalled': 0, 'digest': ''})
    if control.get('execution_profile', metadata) != metadata:
        raise RuntimeConflict('resume must retain the delegation model and context profile')
    control['execution_profile'] = metadata
    if handlers is None:
        from app.agent.graph import build_research_actions
        handlers, _ = build_research_actions(state, llm=llm, should_cancel=should_cancel)
    if not active:
        grant = proposed
        AuthorizationPolicy().assert_target(grant, state)
        repo.create(grant)
    # WHY: active记录是权威；在create后快照前崩溃也可从数据库找回，不另建游标副本。
    state['architecture_version'] = 2
    state['active_delegation_id'] = grant.delegation_id
    from app.services.execution_progress import record_delegation
    record_delegation(runtime, grant, execution_profile=metadata)
    runtime.snapshot(state, ledger=state.get('agent_execution_budget') or {})
    executor = ActionExecutor(AgentController(), sidecar=ActionSidecar(llm))
    runner = DelegationRunner(executor, strategy=local_strategy)
    try:
        result = runner.run(grant.delegation_id, state, handlers, should_cancel=should_cancel)
    except AgentFinalizationRequired:
        import json
        results = [json.loads(item.result_json) for item in repo.invocations(grant.delegation_id) if item.status == 'committed']
        result = runner._finish(repo, grant, results, 'needs_help', ('finalization_reserve_retained',))
    final = copy.deepcopy({key: value for key, value in state.items() if key != 'active_delegation_id'})
    from app.agent.react_context import ReActContext
    pending = (final.get('main_agent_trajectory') or {}).get('pending') or {}
    if pending.get('action') == 'delegate_agent':
        pending_request = DelegationRequest.model_validate(pending.get('decision', {}).get('arguments') or {})
        if pending_request != request:
            raise RuntimeConflict('Main observation belongs to a different delegation request')
        # WHY: 观察配对与活动标记清理必须同快照；否则汇报前崩溃会遗留不可恢复的pending。
        ReActContext(final).observe(result.model_dump(mode='json'))
    runtime.snapshot(final, ledger=state.get('agent_execution_budget') or {})
    # WHY: 清理标记须在持久快照成功后生效；汇报前崩溃仍可读取原委派汇总。
    ledger = state.get('agent_execution_budget')
    if ledger is not None:
        ledger.update(final.get('agent_execution_budget') or {})
        final['agent_execution_budget'] = ledger
    state.clear()
    state.update(final)
    return result


def run_search_delegation(state, request, *, llm, autonomous=False, should_cancel=None):
    request = DelegationRequest.model_validate(request)
    if request.profile not in {'initial_research', 'gap_recovery'}:
        raise ValueError('this entry accepts Search tasks only')
    return run_delegation(state, request, llm=llm, strategy='react' if autonomous else None,
                          should_cancel=should_cancel)
