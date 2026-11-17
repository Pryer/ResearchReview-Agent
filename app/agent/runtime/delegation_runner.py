"""四角色共享的有界任务运行器；每一步复用独立授权与提交。"""
import copy
import json
from app.agent.execution_budget import check_execution, AgentExecutionCancelled
from app.agent.context_builder import context_source_fingerprint, _stable_hash
from app.agent.policy.acceptance import accept_delegation
from app.agent.runtime.strategies import deterministic_decision, PlanAndActStrategy
from app.agent.policy.authorization import AuthorizationPolicy
from app.database.delegation_repository import DelegationRepository
from app.schemas.delegation_schema import ActionInvocation, DelegationResult
from app.services.durable_execution_service import active_runtime


def search_progress(state):
    ledger = state.get('search_query_ledger') or {}
    explored = sorted(key for key, entry in ledger.items()
                      if isinstance(entry, dict) and entry.get('status') in {'success', 'empty'})
    keys = ('paper_id', 'doi', 'title', 'authors', 'year', 'abstract')
    return _stable_hash({'artifacts': {field: sorted(
        (_stable_hash({key: paper.get(key) for key in keys}) for paper in state.get(field) or []))
        for field in ('candidate_papers', 'paper_details')}, 'explored': explored})


class DelegationRunner:
    def __init__(self, executor, *, strategy=None):
        self.executor = executor
        self.strategy = strategy
        self.authorization = AuthorizationPolicy()

    def run(self, delegation_id, state, handlers, *, should_cancel=None):
        repo = DelegationRepository(active_runtime())
        _, grant = repo.read(delegation_id)
        self.authorization.assert_target(grant, state)
        completed = repo.completed_result(delegation_id)
        if completed:
            return DelegationResult.model_validate(completed)
        control = state.setdefault('delegation_control', {}).setdefault(delegation_id,
                    {'rounds': 0, 'stalled': 0, 'digest': ''})
        mode = ('direct' if grant.scope.get('paper_ids') or grant.profile not in {'initial_research', 'gap_recovery', 'evidence_build'} else
                'plan_and_act' if isinstance(self.strategy, PlanAndActStrategy) else 'react' if self.strategy else 'deterministic')
        if control.setdefault('strategy_mode', mode) != mode:
            raise ValueError('resume must retain the delegation strategy')
        while True:
            check_execution()
            if should_cancel and should_cancel():
                raise AgentExecutionCancelled('delegation cancelled')
            row, grant = repo.read(delegation_id)
            self.authorization.assert_target(grant, state)
            records = repo.invocations(delegation_id)
            results = [json.loads(item.result_json) for item in records if item.status == 'committed']
            marker = control.get('plan_invocation')
            if marker and any(item.invocation_id == marker and item.status == 'committed' for item in records):
                control['plan'] = (control.get('plan') or [])[1:]
                control.pop('plan_invocation', None)
                active_runtime().snapshot(state, ledger=state.get('agent_execution_budget') or {})
            acceptance = accept_delegation(grant, state, results)
            if row.observation_cursor < row.next_step:
                repo.consume(delegation_id, row.next_step)
            if acceptance.satisfied:
                return self._finish(repo, grant, results, 'satisfied', ())
            if row.next_step >= grant.max_steps or control['rounds'] >= control.get('round_limit', grant.max_steps * 2):
                return self._finish(repo, grant, results, 'needs_help', acceptance.missing)
            pending = next((item for item in records if item.step_index == row.next_step), None)
            available = self.authorization.available_actions(grant, state, handlers)
            if not available:
                return self._finish(repo, grant, results, 'needs_help', acceptance.missing)
            if pending is not None:
                invocation = ActionInvocation.model_validate_json(pending.invocation_json)
            else:
                control['rounds'] += 1
                # WHY: 决策轮与调用身份先保存；已持久Invocation恢复时不再请求模型决定。
                active_runtime().snapshot(state, ledger=state.get('agent_execution_budget') or {})
                direct = bool(grant.scope.get('paper_ids')) or grant.profile not in {'initial_research', 'gap_recovery', 'evidence_build'}
                if direct:
                    action, arguments = deterministic_decision(grant, state, available, results)
                elif isinstance(self.strategy, PlanAndActStrategy):
                    plan = control.get('plan') or []
                    if not plan:
                        if control.get('replans', 0) >= 2:
                            return self._finish(repo, grant, results, 'needs_help', ('local_plan_exhausted',))
                        plan = self.strategy.propose_plan(grant, self._planning_context(grant, state, results), tuple(available))
                        control['plan'] = copy.deepcopy(plan)
                        control['replans'] = control.get('replans', 0) + 1
                        active_runtime().snapshot(state, ledger=state.get('agent_execution_budget') or {})
                    step = plan[0]
                    if step['action'] not in available:
                        control['plan'] = []
                        active_runtime().snapshot(state, ledger=state.get('agent_execution_budget') or {})
                        continue
                    action, arguments = step['action'], copy.deepcopy(step['arguments'])
                elif self.strategy:
                    action, arguments = self.strategy(grant, self._planning_context(grant, state, results),
                                                      tuple(available))
                else:
                    action, arguments = deterministic_decision(grant, state, available, results)
                if action == 'report_to_main':
                    return self._finish(repo, grant, results, 'needs_help', acceptance.missing)
                from app.agent.quality_feedback import begin_main_recovery
                begin_main_recovery(state, action, arguments)
                self.authorization.assert_action_allowed(grant, state, handlers, action, arguments)
                invocation = ActionInvocation(delegation_id=delegation_id, step_index=row.next_step,
                    action=action, arguments=arguments, expected_version=repo.version(),
                    source_fingerprint=context_source_fingerprint(state), grant_revision=grant.revision)
                repo.prepare(invocation)
            if isinstance(self.strategy, PlanAndActStrategy):
                control['plan_invocation'] = invocation.invocation_id
                active_runtime().snapshot(state, ledger=state.get('agent_execution_budget') or {})
            self.executor.execute(invocation, state, handlers, should_cancel=should_cancel)
            # 只计确定的业务产物/探索账本，不把Invocation和报告文本当作进展。
            if grant.role.value == 'search':
                digest = search_progress(state)
            else:
                from app.agent.runtime.progress import _progress_fingerprint
                digest = _progress_fingerprint(state)
            control = state['delegation_control'][delegation_id]
            if control.get('plan') and control['plan'][0]['action'] == invocation.action:
                control['plan'] = control['plan'][1:]
                control.pop('plan_invocation', None)
            control['stalled'] = control['stalled'] + 1 if control['digest'] == digest else 0
            control['digest'] = digest
            active_runtime().snapshot(state, ledger=state.get('agent_execution_budget') or {})
            if control['stalled'] >= 3:
                records = repo.invocations(delegation_id)
                return self._finish(repo, grant,
                    [json.loads(item.result_json) for item in records if item.status == 'committed'],
                    'needs_help', ('no_business_progress',))

    @staticmethod
    def _planning_context(grant, state, results):
        # WHY: 规划模型读取有限摘要，不能获得全部研究状态或多个Action输入的并集。
        return {'goal': grant.goal, 'scope': copy.deepcopy(grant.scope),
                'state_version': DelegationRepository(active_runtime()).version(),
                'source_fingerprint': context_source_fingerprint(state),
                'topic': str(state.get('topic') or ''),
                'constraints': {k: state.get(k) for k in ('start_year', 'end_year', 'required_reference_count')},
                'candidates': [{k: p.get(k) for k in ('paper_id', 'title', 'year')}
                               for p in (state.get('ranked_papers') or [])[:12]],
                'details_count': len(state.get('paper_details') or []),
                'cards': [{'paper_id': c.get('paper_id'), 'quality_status': c.get('quality_status'),
                           'supported_fields': sorted((c.get('field_claims') or {}).keys())}
                          for c in (state.get('paper_cards') or [])[:12]],
                'routes': [{'route_id': r.get('route_id'), 'route_name': r.get('route_name')}
                           for r in (state.get('validated_routes') or [])[:8]],
                'observations': [{k: r.get(k) for k in ('operation', 'status', 'outcome', 'findings')}
                                 for r in results[-4:]]}

    @staticmethod
    def _finish(repo, grant, results, outcome, missing):
        result = DelegationResult(delegation_id=grant.delegation_id, status='completed', outcome=outcome,
            action_result_refs=tuple(r['task_id'] for r in results),
            findings=tuple(f for r in results for f in r.get('findings') or []), unresolved=tuple(missing))
        repo.finish(result)
        return result
