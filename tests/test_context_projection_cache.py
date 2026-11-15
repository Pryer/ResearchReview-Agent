import copy

from app.agent.execution_budget import budget_scope
from app.agent.context_builder import _prewriting_projection, build_main_agent_context


def test_prewriting_cache_reuses_and_invalidates_all_policy_dependencies(monkeypatch):
    import app.agent.deliverable_router as router
    original = router.check_generation_readiness
    calls = []
    def counted(state):
        calls.append(True)
        return original(state)
    monkeypatch.setattr(router, 'check_generation_readiness', counted)
    current = {'user_query': 'topic', 'core_deliverables': ['research_background'],
               'required_reference_count': 5, 'max_papers_explicit': True}
    with budget_scope(current):
        first = _prewriting_projection(current)
        first.authorized_reference_count = 999
        assert _prewriting_projection(current).authorized_reference_count != 999
        assert len(calls) == 1
        changes = {
            'required_reference_count': 10,
            'paper_details': [{'paper_id': 'p', '_screening_decision': 'rule_screened_reserve'}],
            'citation_allocation_plans': [{'sections': [{'paper_ids': ['p']}]}],
            'core_deliverables': ['research_status'],
            'research_semantic_frame': {'required_focuses': ['test']},
            'quality_original_query': '禁止扩大年份',
            'quality_constraint_changes': [{'action': 'expand_time_range'}],
        }
        for key, value in changes.items():
            current[key] = value
            _prewriting_projection(current)
        assert len(calls) == 1 + len(changes)
    assert 'prewriting_projection' not in current


def test_recent_results_only_deduplicated_when_feedback_remains_visible():
    import json
    from app.agent.react_context import ReActContext
    from app.schemas.agent_decision_schema import AgentDecision
    result = {'operation': 'search_and_rank', 'status': 'completed', 'outcome': 'succeeded',
              'findings': ['found papers'], 'evidence_refs': ['state://paper/p']}
    current = {'user_query': 'topic', 'topic': 'topic', 'steps': [], 'errors': [],
               'agent_task_results': [copy.deepcopy(result)]}
    assert build_main_agent_context(current).state.recent_trajectory
    trajectory = ReActContext(current)
    trajectory.proposed(AgentDecision(action=result['operation'], reason='test'))
    trajectory.observe(result, archive=False)
    context = build_main_agent_context(current)
    messages = trajectory.request(context, system='fixed', tools=[], allowed_actions=[])
    assert not json.loads(messages[-1]['content'])['state']['recent_trajectory']
    assert context.state.recent_trajectory
    trajectory.data['memory'] = []
    messages = trajectory.request(context, system='fixed', tools=[], allowed_actions=[])
    assert json.loads(messages[-1]['content'])['state']['recent_trajectory']
