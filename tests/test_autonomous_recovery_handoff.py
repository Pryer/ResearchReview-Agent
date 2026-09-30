"""恢复耗尽不空转；新证据必须重新授权后才进入写作，终态关闭恢复记录。"""
import json

import pytest

from app.agent import graph, recovery_loop
from app.agent.action_registry import allowed_actions, recovery_handoff_action
from app.agent.context_builder import build_main_agent_context
from app.agent.evidence_recovery import targeted_search_kind
from app.agent.main_loop import _progress_fingerprint
from app.agent.nodes.verification import claim_plan_node, claim_evidence_gate_node
from app.agent.deliverable_router import check_generation_readiness
from app.core.config import get_settings


def test_pending_hybrid_candidates_do_not_offer_empty_fetch():
    from app.agent.nodes.retrieval import fetch_detail_node

    state = _state()
    state.update(retrieval_profile={"mode": "hybrid"},
                 ranked_papers=[{"paper_id": "pending", "_screening_decision": "pending_semantic_check"}],
                 active_quality_recovery={"action": "TARGETED_SEARCH"})
    assert "fetch_metadata" not in allowed_actions(state)
    assert recovery_handoff_action(state) == ""
    fetch_detail_node(state, llm=None)
    assert state["paper_details"] == []
    assert state["steps"][-1]["output_data"]["reason"] == "pending_semantic_admission"


def test_admitted_recovery_candidates_force_evidence_handoff_before_more_search():
    from app.services.retrieval_ranking_service import _document, admission_scope_fingerprint

    state = _state()
    state.update(retrieval_profile={"mode": "hybrid"},
                 active_quality_recovery={"action": "TARGETED_SEARCH"})
    paper = {"paper_id": "fresh", "title": "A verified study", "year": 2025,
             "abstract": "Research methods and evidence", "_screening_decision": "include",
             "_topic_relation": "direct", "_eligible_deliverables": ["research_background"],
             "_semantic_conditions_confirmed": True, "_semantic_contract_complete": True,
             "_screening_confidence": 0.95, "_pending_semantic_check": False}
    paper["_screened_content_fingerprint"] = _document(paper).content_fingerprint
    paper["_screened_scope_fingerprint"] = admission_scope_fingerprint(state)
    state["ranked_papers"] = [paper]
    assert recovery_handoff_action(state) == "fetch_metadata"
    state["paper_details"] = [paper]
    assert recovery_handoff_action(state) == "extract_paper_cards"
    state["paper_cards"] = [{**_card(0), "paper_id": "fresh", "title": paper["title"]}]
    assert recovery_handoff_action(state) == "validate_routes"
    state["validated_routes"] = [{"route_id": "r", "paper_ids": ["fresh"]}]
    assert recovery_handoff_action(state) == "plan_claims"
    state["claim_plans"] = [{"claim_id": "c"}]
    assert recovery_handoff_action(state) == ""


def _state():
    return {
        'user_query': '近三年课堂行为研究，不少于40篇', 'topic': '课堂行为分析',
        'intent': 'generate_review', 'required_reference_count': 40,
        'max_papers_explicit': True, 'year_range_explicit': True,
        'start_year': 2024, 'end_year': 2026,
        'core_deliverables': ['research_background'],
        'generation_blocked': True, 'review': '正文生成已阻止',
        'paper_cards': [_card(0)],
        'evidence_gap_report': {'needs_recovery': True},
        'quality_recovery_decision': {'action': 'TARGETED_SEARCH'},
        'steps': [], 'errors': [],
    }


def _card(i):
    return {
        'paper_id': f'p{i}', 'title': f'测试研究{i}', 'year': 2024,
        'quality_status': 'valid', 'evidence_source': 'abstract',
        'relation_type': 'direct', 'eligible_deliverables': ['research_background'],
        'field_claims': {'research_problem': [{
            'claim': f'研究{i}分析课堂行为与学生参与度的关系',
            'explicitly_reported': True, 'evidence_id': f'p{i}:e001',
        }]},
    }


@pytest.mark.parametrize('issue_code, expected_searches', [
    ('required_focus_evidence_not_met', 1),
    ('minimum_references_not_met', 0),
])
def test_autonomous_quality_recovery_executes_focus_search_before_main_loop(
    monkeypatch, issue_code, expected_searches,
):
    state = _state()
    state.update(
        agent_orchestration_mode='autonomous',
        paper_details=[{'paper_id': 'p0'}],
        active_quality_recovery={
            'action': 'TARGETED_SEARCH',
            'issues': [{'code': issue_code, 'details': {'missing_focuses': ['重点甲']}}],
        },
    )
    calls = []

    def search(current, **kwargs):
        calls.append('focus_search')
        assert current['agent_operation_mode'] == 'incremental'
        assert kwargs['objective'].startswith('针对缺失的用户研究重点')

    def main_loop(current, **kwargs):
        calls.append('main_loop')
        return {'research_state': current}

    monkeypatch.setattr(graph, '_get_llm', lambda: object())
    monkeypatch.setattr(graph, '_run_search_subagent', search)
    monkeypatch.setattr(graph, '_run_autonomous_pipeline', main_loop)

    graph.continue_research_agent(state)

    assert calls == ['focus_search'] * expected_searches + ['main_loop']


@pytest.mark.parametrize('boundary', ['rounds', 'marginal_gain', 'actions', 'permission'])
def test_exhausted_route_search_is_removed_and_direct_execution_makes_no_model_call(monkeypatch, boundary):
    state = _state()
    if boundary == 'rounds':
        state['recovery_round'] = get_settings().evidence_recovery_max_rounds
    elif boundary == 'marginal_gain':
        state['evidence_recovery_status'] = 'EXHAUSTED'
    elif boundary == 'actions':
        state['recovery_action_count'] = get_settings().recovery_total_action_budget
    else:
        state['allow_evidence_expansion'] = False
    assert 'targeted_search' not in allowed_actions(state)
    assert 'plan_claims' in allowed_actions(state)
    context = build_main_agent_context(state)
    assert 'TARGETED_SEARCH' not in context.state.next_actions
    monkeypatch.setattr(recovery_loop, '_get_llm', lambda: pytest.fail('exhausted recovery requested LLM'))
    recovery_loop.run_route_evidence_recovery(state)
    assert state['recovery_decision']['action'] == 'DEGRADE'
    assert state['required_reference_count'] == 40


def test_route_exhaustion_does_not_disable_unattempted_citation_repair():
    state = _state()
    state.update(generation_blocked=False, review='已生成正文 [p1]', unique_cited_paper_count=23,
                 candidate_papers=[{'paper_id': 'p1'}], evidence_recovery_status='EXHAUSTED')
    assert targeted_search_kind(state) == 'citation'
    assert 'targeted_search' in allowed_actions(state)
    state['citation_gap_repair_attempted'] = True
    assert 'targeted_search' not in allowed_actions(state)


def test_blocked_draft_can_repair_real_citation_gap_without_route_gap():
    state = _state()
    state.update(
        generation_blocked=True, review='已有正文 [p1]',
        unique_cited_paper_count=23,
        candidate_papers=[{'paper_id': 'p1'}],
        evidence_gap_report={'needs_recovery': False, 'status': 'NOT_REQUIRED'},
    )
    assert targeted_search_kind(state) == 'citation'
    state['citation_gap_repair_attempted'] = True
    assert targeted_search_kind(state) == ''


def test_blocking_notice_without_body_citations_cannot_start_citation_search():
    state = _state()
    state.update(
        generation_blocked=True, review='当前研究未满足交付要求',
        unique_cited_paper_count=23, candidate_papers=[{'paper_id': 'p1'}],
        evidence_gap_report={'needs_recovery': False, 'status': 'NOT_REQUIRED'},
    )
    assert targeted_search_kind(state) == ''


class Decisions:
    native_tools_enabled = True

    def __init__(self):
        self.calls = 0

    def complete_tool_call(self, messages, **kwargs):
        self.calls += 1
        context = json.loads(messages[-1]['content'])
        if self.calls == 1:
            assert 'targeted_search' in context['state']['allowed_actions']
            return {'name': 'targeted_search', 'arguments': {}}
        assert self.calls == 2
        assert 'targeted_search' not in context['state']['allowed_actions']
        return {'name': 'report_blocked', 'arguments': {'reason': '仍有交付要求未通过验证'}}


@pytest.mark.parametrize('usable', [39, 40])
def test_recovery_commits_fresh_claims_and_checks_writing_readiness(monkeypatch, usable):
    state = _state()
    llm = Decisions()
    monkeypatch.setattr(graph, '_get_llm', lambda: llm)
    calls = []

    def recover(current, **kwargs):
        calls.append('recover')
        # 复现卡片很多、可用证据有限；不能将总卡片数量当成最终引用能力。
        current['paper_cards'] = [_card(i) for i in range(133)]
        for card in current['paper_cards'][usable:]:
            card['quality_status'] = 'invalid'
        current['validated_routes'] = [{'route_id': 'r1', 'name': '行为分析',
                                       'core_paper_ids': [f'p{i}' for i in range(usable)]}]
        current['evidence_recovery_status'] = 'EXHAUSTED'

    def claims(current, **kwargs):
        calls.append('claims')
        claim_plan_node(current, llm=None)

    def gate(current):
        calls.append('claim_gate')
        claim_evidence_gate_node(current, llm=None)

    def write(current, **kwargs):
        calls.append('readiness')
        ready = check_generation_readiness(current)
        assert ready.ready == (usable >= 40)
        assert current['reference_coverage_stats']['claim_authorized'] == usable
        current['generation_readiness'] = ready.model_dump(mode='json')
        current['generation_blocked'] = not ready.ready
        current['quality_gate'] = {'passed': False, 'draft_released': False,
                                   'blocking_issues': ready.blocking_issues}

    monkeypatch.setattr(graph, '_run_route_evidence_recovery', recover)
    monkeypatch.setattr(graph, 'claim_plan_node', claims)
    monkeypatch.setattr(graph, '_run_claim_evidence_gate', gate)
    monkeypatch.setattr(graph, '_generate_deliverables_or_block', write)
    monkeypatch.setattr(graph, '_verify_generated_draft', lambda current: None)
    monkeypatch.setattr(graph, 'final_answer_node', lambda current: None)
    output = graph._run_autonomous_pipeline(state)
    assert calls == ['recover', 'claims', 'claim_gate', 'readiness']
    assert llm.calls == 2
    assert output['status'] == 'blocked'
    assert state['reference_coverage_stats']['claim_authorized'] == usable
    assert state['reference_coverage_stats']['final_valid'] == 0
    assert state['required_reference_count'] == 40
    assert (state['start_year'], state['end_year']) == (2024, 2026)
    assert not state['quality_gate']['draft_released']


def test_route_validation_counts_as_progress_but_diagnostic_history_does_not():
    state = _state()
    original = _progress_fingerprint(state)
    state['recovery_history'] = [{'reason': '重复诊断'}]
    assert _progress_fingerprint(state) == original
    state['validated_routes'] = [{'route_id': 'new', 'core_paper_ids': ['p0']}]
    assert _progress_fingerprint(state) != original


@pytest.mark.parametrize('terminal', ['blocked', 'failed', 'cancelled'])
def test_terminal_recovery_does_not_remain_started(monkeypatch, terminal):
    from app.agent.main_loop import MainAgentLoop
    state = _state()
    state['active_quality_recovery'] = {'action': 'TARGETED_SEARCH'}
    state['quality_recovery_history'] = [{'action': 'TARGETED_SEARCH', 'outcome': 'started',
                                         'progress_before': {'valid_reference_shortfall': 40}}]
    monkeypatch.setattr(graph, '_get_llm', lambda: object())
    monkeypatch.setattr(MainAgentLoop, 'run', lambda *args, **kwargs: terminal)
    graph._run_autonomous_pipeline(state)
    assert 'active_quality_recovery' not in state
    assert state['quality_recovery_history'][-1]['outcome'] != 'started'
    assert 'progress_after' in state['quality_recovery_history'][-1]


def test_blocked_recovery_with_pending_mandatory_action_is_closed(monkeypatch):
    from app.agent.main_loop import MainAgentLoop
    state = _state()
    state['active_quality_recovery'] = {'action': 'TARGETED_SEARCH'}
    state['quality_recovery_history'] = [{'action': 'TARGETED_SEARCH', 'outcome': 'started',
                                         'progress_before': {'valid_reference_shortfall': 40}}]
    state['agent_mandatory_actions'] = [{'action': 'targeted_search'}]
    monkeypatch.setattr(graph, '_get_llm', lambda: object())
    monkeypatch.setattr(MainAgentLoop, 'run', lambda *args, **kwargs: 'blocked')
    graph._run_autonomous_pipeline(state)
    assert 'active_quality_recovery' not in state
    assert state['quality_recovery_history'][-1]['outcome'] == 'failed'
