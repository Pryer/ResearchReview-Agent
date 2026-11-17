"""任务级Main：模型选择Profile，Runtime管理多Action与确定性提交。"""
import json
import copy
from app.agent.action_registry import ACTION_REGISTRY, allowed_actions, recovery_handoff_action, tool_schemas, validate_arguments
from app.agent.context_builder import refresh_main_agent_context, _stable_hash
from app.agent.controller import AgentController
from app.agent.execution_budget import (budget_scope, check_execution, AgentBudgetExceeded,
                                       AgentExecutionCancelled, AgentExecutionStale)
from app.agent.runtime.errors import AgentDecisionError
from app.agent.policy.authorization import DelegationAuthorizationError
from app.agent.policy.capability_profiles import PROFILES
from app.agent.react_context import ReActContext, observe_terminal, execute_context_recall, ContextBudgetExceeded
from app.core.config import get_settings
from app.core.exceptions import LLMInvocationError, NativeToolsUnsupportedError
from app.core.json_utils import parse_json_object
from app.database.runtime_repository import RuntimeConflict
from app.database.delegation_repository import DelegationRepository
from app.prompt.task_main import TASK_MAIN_PROMPT
from app.schemas.agent_decision_schema import AgentDecision
from app.schemas.delegation_schema import DelegationRequest
from app.schemas.recovery_schema import RecoveryAction
from app.services.durable_execution_service import active_runtime
from app.services.research_delegation_service import run_delegation
from app.agent.execution import AgentCancelledError

CONTROLS = ('request_clarification', 'recall_context', 'search_research_memory', 'publish_draft', 'request_finish', 'report_blocked')


class TaskMainPolicy:
    def __init__(self, llm):
        from app.services.role_execution_profile import bind_role
        self.llm = bind_role(llm, 'main')
        self.tools = [{'type': 'function', 'function': {'name': 'delegate_agent',
            'description': '委派一个有界专业任务，返回完整阶段结果',
            'parameters': DelegationRequest.model_json_schema()}}] + tool_schemas(CONTROLS)
        # WHY: 模型只看到公开Profile；精确单动作授权由服务端签发，不能作为模型可选能力。
        schema = self.tools[0]['function']['parameters']
        schema['required'].remove('agent')
        parameters = schema['properties']
        parameters.pop('agent')
        parameters['profile']['enum'] = sorted(PROFILES)
        parameters['profile']['description'] = '公开能力与角色：' + '; '.join(
            f'{profile.name}={profile.role.value}' for profile in PROFILES.values())

    def decide(self, context, names, trajectory, *, profiles=None):
        # WHY: 静态目录不能把待Check锁等已禁止的Profile继续交给模型选择；执行端仍独立复核。
        tools = [copy.deepcopy(tool) for tool in self.tools if tool['function']['name'] in names]
        if profiles is not None:
            delegate = next((tool for tool in tools if tool['function']['name'] == 'delegate_agent'), None)
            if delegate is not None:
                field = delegate['function']['parameters']['properties']['profile']
                field['enum'] = list(profiles)
                field['description'] = '本轮可用能力与角色：' + '; '.join(
                    f'{name}={PROFILES[name].role.value}' for name in profiles)
        native = bool(getattr(self.llm, 'native_tools_enabled', False)) and not trajectory.data.get('native_tools_disabled')
        for _ in range(get_settings().agent_decision_repair_attempts):
            messages = trajectory.request(context, system=TASK_MAIN_PROMPT + ('' if native else '\n工具：' + json.dumps(tools, ensure_ascii=False)),
                                          tools=tools if native else [], allowed_actions=names)
            try:
                if native:
                    raw = self.llm.complete_tool_call(messages, tools=tools, operation='main_agent_decision')
                    decision = AgentDecision(action=raw['name'], arguments=raw.get('arguments') or {},
                        transport='native_tool', tool_call_id=raw.get('tool_call_id'))
                    assistant = raw.get('assistant_message')
                else:
                    raw = parse_json_object(self.llm.complete_messages(messages, response_format='json',
                        operation='main_agent_decision', retry_empty=False))
                    decision = AgentDecision.model_validate(raw)
                    assistant = None
                if decision.action == 'delegate_agent' and 'agent' not in decision.arguments:
                    name = decision.arguments.get('profile')
                    if isinstance(name, str) and name in PROFILES:
                        # WHY: 模型只选择能力，角色由服务端目录派生。历史显式角色仍独立核对。
                        decision.arguments['agent'] = PROFILES[name].role.value
                trajectory.proposed(decision, assistant_message=assistant)
                if decision.plan or decision.context_selection is not None or decision.memory_feedback:
                    raise ValueError('task-level tool metadata must use the declared protocol')
                if decision.action not in names:
                    raise ValueError('main tool is unavailable')
                if decision.action == 'delegate_agent':
                    DelegationRequest.model_validate(decision.arguments)
                else:
                    validate_arguments(decision.action, decision.arguments)
                return decision
            except NativeToolsUnsupportedError:
                native = False
                trajectory.data['native_tools_disabled'] = True
            except ContextBudgetExceeded:
                raise
            except (ValueError, TypeError, KeyError) as exc:
                trajectory.observe({'status': 'rejected', 'error': str(exc)[:300]})
        raise AgentDecisionError('task-level main decision repair exhausted')


class TaskMainLoop:
    def __init__(self, llm):
        self.llm, self.policy = llm, TaskMainPolicy(llm)

    def run(self, state, *, handlers, finish_validator, should_cancel=None):
        from app.agent.runtime.progress import progress_cache_scope
        try:
            with budget_scope(state, should_cancel), progress_cache_scope():
                status = self._run(state, handlers, finish_validator, should_cancel)
                state['result_status'] = status
                if (state.get('main_agent_trajectory') or {}).get('pending'):
                    observe_terminal(state, {'status': status})
                return status
        except (AgentBudgetExceeded, ContextBudgetExceeded, AgentCancelledError, AgentDecisionError,
                DelegationAuthorizationError, LLMInvocationError, RuntimeConflict) as exc:
            status = ('failed' if isinstance(exc, (AgentExecutionStale, RuntimeConflict)) else
                      'cancelled' if isinstance(exc, AgentCancelledError) else
                      'blocked' if isinstance(exc, (AgentBudgetExceeded, ContextBudgetExceeded)) else 'failed')
            state['result_status'] = status
            if status == 'blocked':
                state['generation_blocked'] = True
            state.setdefault('errors', []).append({'code': type(exc).__name__, 'message': str(exc)[:300]})
            observe_terminal(state, {'status': status, 'error': type(exc).__name__})
            return status

    def _run(self, state, handlers, finish_validator, should_cancel):
        if not active_runtime():
            raise RuntimeConflict('v2 Main requires durable execution')
        from app.services.legacy_checkpoint_migration import migrate_active
        migrate_active(state)
        state['architecture_version'] = 2
        repo = DelegationRepository(active_runtime())
        trajectory = ReActContext(state)
        control = state.setdefault('task_main_control', {'stalled': 0, 'previous': '', 'attempts': {}})
        limit = state['agent_execution_budget']['main_round_limit']
        while state['agent_execution_budget'].get('decision_rounds', 0) < limit:
            # WHY: 专业提交会替换state内容；恢复分支返回后也要重取运行游标，避免计数写入旧字典。
            control = state['task_main_control']
            check_execution()
            from app.agent.execution_budget import update_finalization
            from app.services.research_memory_retrieval import prepare_memory_context
            update_finalization(state)
            prepare_memory_context(state)
            if should_cancel and should_cancel():
                raise AgentExecutionCancelled('task-level main cancelled')
            recorded = state.get('active_delegation_id')
            active = repo.read(recorded)[0] if recorded else repo.active()
            if active:
                _, grant = repo.read(active.delegation_id)
                request = DelegationRequest(agent=grant.role, profile=grant.profile, goal=grant.goal, scope=grant.scope)
                result = run_delegation(state, request, llm=self.llm, handlers=handlers,
                    strategy='configured',
                    forced_operation=grant.allowed_actions[0] if grant.immutable_arguments is not None else None,
                    forced_arguments=grant.immutable_arguments, should_cancel=should_cancel)
                if trajectory.data.get('pending'):
                    trajectory.observe(result.model_dump(mode='json'))
                continue
            legacy = allowed_actions(state, set(handlers))
            mandatory = state.get('agent_mandatory_actions') or []
            required = mandatory[0] if mandatory else None
            handoff = recovery_handoff_action(state) if not required else ''
            operation = required['action'] if required else handoff
            recovery = state.get('active_quality_recovery') or {}
            previous = (state.get('pending_generation_previous') or {}).get('products') or {}
            # WHY: 任务级修订以新候选复验结算恢复账本，才能选择下一轮质量动作。
            if (not operation and state.get('pending_result_validation') and 'validate_result' in legacy
                    and recovery.get('origin') == 'main'
                    and (previous.get('check_report') or {}).get('verdict') == 'needs_revision'):
                operation = 'validate_result'
            state['agent_execution_budget']['decision_rounds'] = int(state['agent_execution_budget'].get('decision_rounds', 0)) + 1
            active_runtime().persist_budget(state['agent_execution_budget'])
            from app.agent.quality_feedback import checked_recovery, checked_section_revision
            repair = checked_recovery(state) if not operation else None
            revision = None
            if repair and repair.remaining_budget:
                if repair.action == RecoveryAction.REWRITE_SECTIONS and 'rewrite_sections' in legacy:
                    revision = checked_section_revision(state, repair)
                elif repair.action == RecoveryAction.REBUILD_CLAIMS and 'plan_claims' in legacy:
                    operation = 'plan_claims'
                    required = {'reason': repair.reason}
            if operation in handlers:
                if operation not in legacy:
                    return self._blocked(state, 'required_operation_unavailable')
                args = dict((required or {}).get('arguments') or {})
                scope = {'section_ids': args['section_ids']} if args.get('section_ids') else {}
                request = DelegationRequest(agent=ACTION_REGISTRY[operation].role, profile='single_action',
                    goal=(required or {}).get('reason') or ACTION_REGISTRY[operation].description, scope=scope)
                decision = AgentDecision(action='delegate_agent', arguments=request.model_dump(mode='json'))
                trajectory.proposed(decision)
                forced = operation
                state['allowed_agent_actions'] = ['delegate_agent']
                snapshot = refresh_main_agent_context(state)
                context = snapshot.context
            elif revision:
                forced = None
                profiles = [revision.profile]
                decision = AgentDecision(action='delegate_agent', arguments=revision.model_dump(mode='json'))
                trajectory.proposed(decision)
                state['allowed_agent_actions'] = ['delegate_agent']
                context = refresh_main_agent_context(state).context
            else:
                names = [name for name in CONTROLS if name in legacy]
                profiles = []
                for profile in PROFILES.values():
                    # WHY: 新检索任务先完成检索；仅详情动作可用时不能暴露一个无法起步的检索Profile。
                    first = {'initial_research': 'search_and_rank', 'gap_recovery': 'targeted_search'}.get(profile.name)
                    if first and first not in legacy:
                        continue
                    if set(profile.actions) & set(legacy):
                        profiles.append(profile.name)
                if profiles:
                    names.insert(0, 'delegate_agent')
                state['allowed_agent_actions'] = names
                snapshot = refresh_main_agent_context(state)
                if snapshot.validation_errors:
                    return self._blocked(state, 'main_context_invalid')
                context = snapshot.context.model_copy(deep=True)
                context.state.agent_messages.append({'sender': 'runtime', 'content': 'available_profiles: ' + ', '.join(profiles)})
                decision = self.policy.decide(context, names, trajectory, profiles=profiles)
                forced = None
            from app.services.execution_report_capture import record_main_context
            state.setdefault('main_agent_decisions', []).append(decision.model_dump(mode='json'))
            record_main_context(state, context, decision, model_input=forced is None and revision is None)
            if decision.action == 'delegate_agent':
                request = DelegationRequest.model_validate(decision.arguments)
                if not forced and request.profile not in profiles:
                    trajectory.observe({'status': 'rejected', 'error': 'profile_unavailable'})
                    control['rejections'] = control.get('rejections', 0) + 1
                    if control['rejections'] >= 3:
                        return self._blocked(state, 'task_authorization_repair_exhausted')
                    continue
                from app.agent.runtime.progress import _progress_fingerprint
                from app.agent.runtime.delegation_runner import search_progress
                digest = _stable_hash((_progress_fingerprint(state), search_progress(state)))
                task_key = _stable_hash((request.agent.value, request.profile, request.scope, digest))
                control = state['task_main_control']
                control['attempts'][task_key] = control['attempts'].get(task_key, 0) + 1
                if control['attempts'][task_key] > 3:
                    return self._blocked(state, 'repeated_delegation_without_progress')
                active_runtime().snapshot(state, ledger=state['agent_execution_budget'])
                try:
                    result = run_delegation(state, request, llm=self.llm, handlers=handlers, should_cancel=should_cancel,
                        strategy='configured',
                        forced_operation=forced, forced_arguments=args if forced else None)
                except DelegationAuthorizationError as exc:
                    trajectory.observe({'status': 'rejected', 'error': str(exc)[:300]})
                    control['rejections'] = control.get('rejections', 0) + 1
                    if control['rejections'] >= 3:
                        return self._blocked(state, 'task_authorization_repair_exhausted')
                    continue
                state['task_main_control']['rejections'] = 0
                trajectory.observe(result.model_dump(mode='json'))
            elif decision.action == 'request_finish':
                if finish_validator(state):
                    trajectory.observe({'status': state.get('result_status') or 'completed'})
                    return state.get('result_status') or 'completed'
                trajectory.observe({'status': 'rejected', 'error': 'finish_rejected_by_quality_gates'})
            elif decision.action == 'publish_draft':
                try:
                    AgentController().publish_draft(state)
                except ValueError as exc:
                    from app.agent.execution_budget import is_control_exception
                    if is_control_exception(exc):
                        raise
                    # WHY: 来源/版本预检拒绝是可观察的业务结果，交Main重新规划；不绕过发布门禁。
                    trajectory.observe({'status': 'rejected', 'error': str(exc)[:300]})
            elif decision.action == 'recall_context':
                execute_context_recall(trajectory, decision.arguments)
            elif decision.action == 'search_research_memory':
                from app.services.research_memory_retrieval import execute_memory_search
                execute_memory_search(state, trajectory, decision.arguments)
            elif decision.action == 'request_clarification':
                # WHY: 服务持久化按kind/needed识别主控澄清，字段不全会丢失可恢复暂停状态。
                state['clarification'] = {'needed': True, 'kind': 'main_agent', 'question': decision.arguments['question']}
                trajectory.observe({'status': 'waiting_user'})
                return 'waiting_user'
            else:
                trajectory.observe({'status': 'blocked', 'reason': decision.arguments['reason']})
                return self._blocked(state, 'main_agent_blocked', message=decision.arguments['reason'])
            from app.agent.runtime.progress import _progress_fingerprint
            from app.agent.runtime.delegation_runner import search_progress
            digest = _stable_hash((_progress_fingerprint(state), search_progress(state)))
            control = state['task_main_control']
            control['stalled'] = control['stalled'] + 1 if digest == control['previous'] else 0
            control['previous'] = digest
            if control['stalled'] >= get_settings().agent_main_no_progress_limit:
                return self._blocked(state, 'no_task_progress')
            active_runtime().snapshot(state, ledger=state['agent_execution_budget'])
        return self._blocked(state, 'main_round_budget_exhausted')

    @staticmethod
    def _blocked(state, reason, *, message=None):
        state['result_status'] = 'blocked'
        state['generation_blocked'] = True
        state.setdefault('errors', []).append({'code': reason, 'message': message if message is not None else reason})
        return 'blocked'
