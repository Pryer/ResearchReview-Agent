import copy
import json
from unittest.mock import patch

from app.agent.context_compaction import compact_main_context
from app.agent.context_builder import build_main_agent_context, _prewriting_projection
from app.agent.execution_budget import budget_scope
from app.agent.react_context import status_projection
from app.schemas.context_schema import MainAgentContext, KeyEvidence, OpenQuestion


def test_communication_is_trimmed_before_evidence_and_source_is_unchanged():
    context = MainAgentContext()
    context.state.agent_messages = [{'sender': 'search', 'kind': 'progress', 'content': 'x' * 1200} for _ in range(8)]
    context.key_evidence = [KeyEvidence(evidence_id=str(i), finding='e' * 200) for i in range(10)]
    before = context.model_copy(deep=True)
    result = compact_main_context(context, max_chars=9000)
    assert len(json.dumps(result.model_dump(mode='json'), ensure_ascii=False)) <= 9000
    assert len(result.key_evidence) == 10
    assert len(result.state.agent_messages) < 8
    assert context == before


def test_latest_unresolved_help_and_hard_constraints_survive_compaction():
    context = MainAgentContext()
    context.goal.explicit_constraints = [{'name': 'required_reference_count', 'value': 40}]
    context.open_questions = [OpenQuestion(question_id='blocking', category='access', question='access needed', blocking=True)]
    for role in ('search', 'analysis', 'writing', 'check'):
        context.state.agent_messages.extend([
            {'sender': role, 'kind': 'progress', 'content': 'old' * 400},
            {'sender': role, 'kind': 'needs_help', 'content': role * 200},
        ])
    result = compact_main_context(context, max_chars=2000)
    assert len(result.state.agent_messages) == 4
    assert all(m['kind'] == 'needs_help' for m in result.state.agent_messages)
    assert result.goal == context.goal and result.open_questions == context.open_questions


def test_readiness_cache_retains_alternating_views_and_is_bounded(monkeypatch):
    import app.agent.deliverable_router as router
    original = router.check_generation_readiness
    calls = []
    def counted(state):
        calls.append(True)
        return original(state)
    monkeypatch.setattr(router, 'check_generation_readiness', counted)
    current = {'core_deliverables': ['research_background']}
    with budget_scope(current) as active:
        for _ in range(3):
            _prewriting_projection({**current, 'required_reference_count': 1})
            _prewriting_projection({**current, 'required_reference_count': 2})
        assert len(calls) == 2
        for i in range(20):
            _prewriting_projection({**current, 'required_reference_count': i})
        assert len(active['prewriting_projection']) == 8


def test_model_projection_omits_aliases_without_changing_snapshot_schema():
    context = MainAgentContext()
    context.state.remaining_action_budget = 3
    context.state.budget_remaining = {'recovery': 3}
    raw = context.model_dump(mode='json')
    before = copy.deepcopy(raw)
    projected = status_projection(raw, [])
    assert 'status' not in projected['state']
    assert 'remaining_action_budget' not in projected['state']
    assert projected['state']['research_status'] == raw['state']['research_status']
    assert raw == before
    assert MainAgentContext.model_validate(raw) == context


def test_subagent_builds_one_context_and_reads_fresh_authoritative_controls():
    from app.agent.delegation_context import subagent_context
    current = {'topic': 'topic', 'user_query': 'topic', 'research_plan': {'task_graph': []},
               'subagent_sessions': {'search': {'messages': []}},
               'quality_gate': {'passed': False}, 'result_status': 'blocked',
               'agent_execution_budget': {'token_limit': 100, 'llm_tokens': 20}}
    active = {'id': 'd', 'operation': 'search_and_rank', 'objective': 'search', 'arguments': {},
              'evidence_refs': [], 'completed_operations': [], 'rounds': 0}
    with patch('app.agent.delegation_context.build_main_agent_context', wraps=build_main_agent_context) as build:
        result = subagent_context(current, 'search', active, ['search_and_rank'])
    assert build.call_count == 1
    assert result.state.gate_status['quality'] is False
    assert result.state.budget_remaining['tokens'] == 80
    assert result.goal.request == 'topic'


def test_history_experiment_is_offline_and_reports_tradeoff():
    from scripts.benchmark_status_history import experiment
    from scripts.benchmark_review_quality import offline
    with offline():
        data = experiment(3)
    append = data['results']['append']
    transient = data['results']['transient_experiment']
    assert transient['sum_upper_bound'] < append['sum_upper_bound']
    assert append['full_prior_request_prefix_matches'] == 2
    assert transient['full_prior_request_prefix_matches'] == 0
    assert transient['captured_contexts'] == 3
    assert data['provider_requests'] == 0 and data['quality_effect'] is None
