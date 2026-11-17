"""共享执行器上的确定性局部策略与持久计划策略。"""
from app.agent.policy.direct_arguments import direct_arguments
from app.agent.action_registry import validate_arguments


def deterministic_decision(grant, state, available, results):
    if grant.scope.get('paper_ids'):
        return ('extract_paper_cards' if 'extract_paper_cards' in available else 'report_to_main'), {}
    if grant.profile not in {'initial_research', 'gap_recovery', 'evidence_build'}:
        action, arguments = direct_arguments(grant)
        return (action, arguments) if action in available else ('report_to_main', {})
    done = {r['operation'] for r in results}
    if grant.profile == 'evidence_build':
        if not state.get('paper_cards') and 'extract_paper_cards' in available:
            return 'extract_paper_cards', {}
        if not (state.get('validated_routes') or state.get('dynamic_taxonomy')):
            for action in ('validate_routes', 'cluster_papers'):
                if action in available and action not in done:
                    return action, {}
        if 'plan_claims' in available and 'plan_claims' not in done:
            return 'plan_claims', {}
        return 'report_to_main', {}
    search = 'targeted_search' if grant.profile == 'gap_recovery' else 'search_and_rank'
    action = 'fetch_metadata' if search in done else search
    # WHY: Grant内某个动作可用不代表固定流程的下一步可用；交回缺口，不提出越权动作。
    if action not in available:
        return 'report_to_main', {}
    arguments = ({'gap_ids': list(grant.scope['gap_ids'])}
                 if action == 'targeted_search' and grant.scope.get('gap_ids') else {})
    return action, arguments


class PlanAndActStrategy:
    """规划回调只提出操作序列；Runner保存计划，每一步仍动态准入。"""
    def __init__(self, planner):
        self.planner = planner

    def propose_plan(self, grant, context, available):
        plan = self.planner(grant, context, available)
        if not isinstance(plan, list) or not 1 <= len(plan) <= grant.max_steps:
            raise ValueError('local plan must be bounded')
        for step in plan:
            if (not isinstance(step, dict) or set(step) != {'action', 'arguments'}
                    or step['action'] not in grant.allowed_actions or not isinstance(step['arguments'], dict)):
                raise ValueError('invalid local plan step')
            validate_arguments(step['action'], step['arguments'])
        return plan
