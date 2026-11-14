"""轨迹中的目标丢失与重复发布回归，外部模型不参与。"""
import copy
from types import SimpleNamespace
import pytest

from tests.test_draft_delivery_policy import draft_state


def test_published_candidate_is_not_offered_again():
    from app.agent.delivery import publish_draft
    from app.agent.action_registry import allowed_actions
    state = draft_state()
    assert 'publish_draft' in allowed_actions(state)
    publish_draft(state)
    assert 'publish_draft' not in allowed_actions(state)
    state['references'].append('新增经核验书目信息')
    assert 'publish_draft' in allowed_actions(state)


def test_recovery_targets_survive_summary_gate_and_feedback_truncation():
    from app.agent.generation_recovery import decide_generation_recovery
    state = {'review': '待修复正文', 'quality_gate': {'passed': False, 'blocking_issues': [
        {'code': 'claim_evidence_quality_not_met', 'message': '证据不足'}]},
        'claim_verification': {'claims': [
            {'claim_id': f'c{i}', 'section_id': 'background_body',
             'support_status': 'unsupported', 'sentence': f'未获支持论述{i}',
             'evidence_ids': ['e1']} for i in range(8)]}}
    decision = decide_generation_recovery(state, max_actions=8)
    assert set(decision.target_claim_ids) == {f'c{i}' for i in range(8)}
    assert decision.target_section_ids == ['background_body']
    assert all(i.details.get('sentence') for i in decision.issues if i.affected_claim_ids)


def test_section_type_maps_to_plan_but_unknown_target_is_rejected():
    from app.agent.generation_recovery import resolve_rewrite_sections
    state = {'writing_plans': [{'deliverable_type': 'research_background',
             'sections': [{'id': 'background_body'}]}]}
    assert resolve_rewrite_sections(state, ['research_background']) == ['background_body']
    with pytest.raises(ValueError, match='rewrite target'):
        resolve_rewrite_sections(state, ['unknown'])
    with pytest.raises(ValueError, match='rewrite target'):
        resolve_rewrite_sections(state, [])
    state['writing_plans'].append(copy.deepcopy(state['writing_plans'][0]))
    with pytest.raises(ValueError, match='ambiguous'):
        resolve_rewrite_sections(state, ['research_background'])


def test_report_refresh_alone_does_not_create_publication():
    from app.agent.delivery import publish_draft
    from app.agent.checking import check_input_fingerprint
    state = draft_state()
    state['check_report'] = {'input_fingerprint': check_input_fingerprint(state),
                             'verdict': 'needs_revision', 'issues': []}
    first = publish_draft(state)
    state['check_report']['report_id'] = 'different-report-id'
    state['check_report']['issues'] = []
    assert publish_draft(state)['version'] == first['version']


def test_unchanged_quality_rewrite_does_not_reset_attempt_after_rewording():
    from app.agent.generation_recovery import repair_basis_fingerprint, rewrite_attempt_exhausted
    state = {'review': '旧措辞', 'claim_verification': {'unsupported': 1},
             'quality_gate': {'passed': False, 'blocking_issues': [{'code': 'claim_evidence_quality_not_met'}]}}
    state['quality_recovery_history'] = [{'action': 'REWRITE_SECTIONS', 'outcome': 'no_progress',
        'repair_basis_fingerprint': repair_basis_fingerprint(state)}]
    restored = copy.deepcopy(state)
    restored['review'] = '换了措辞仍无支持'
    assert rewrite_attempt_exhausted(restored)
    restored['paper_cards'] = [{'paper_id': 'new-evidence'}]
    assert not rewrite_attempt_exhausted(restored)


def test_inferred_requirement_cannot_be_promoted_by_explicit_topic_entity():
    from app.agent.research_semantic_parser import _annotate_requirement_provenance
    from app.schemas.research_plan_schema import EvidenceRequirement, ResearchSemanticFrame, SemanticItem
    frame = ResearchSemanticFrame(canonical_topic='RAG', research_objects=[
        SemanticItem(id='rag', label='RAG', explicit=True, source='user_explicit')])
    requirement = EvidenceRequirement(requirement_id='framework', label='原始框架',
        evidence_role='framework', source_ids=['rag'], source='llm_inference', inferred=True)
    result = _annotate_requirement_provenance([requirement], frame)[0]
    assert result.source == 'llm_inference' and result.inferred and not result.explicit


def test_delegation_summary_does_not_double_count_quality_failure(monkeypatch):
    import app.services.research_memory_retrieval as memory
    events = []
    monkeypatch.setattr(memory, 'memory_service', lambda s: SimpleNamespace(
        memory=SimpleNamespace(append_event=lambda **kw: events.append(kw))))
    trajectory = SimpleNamespace(data={'scope_id': 'tool'})
    state = {'session_id': 'test'}
    feedback = {'action': 'validate_result', 'status': 'completed', 'outcome': 'degraded', 'sequence': 1}
    memory.record_failure(state, trajectory, feedback)
    memory.record_failure(state, trajectory, {**feedback, 'delegation_status': 'completed'})
    assert len(events) == 1
    assert events[0]['payload']['event_category'] == 'quality_rejected'


def test_stale_report_is_not_used_to_authorize_claim_targets():
    from app.agent.generation_recovery import diagnose_generation_issues
    state = {'review': '新稿', 'check_report': {'input_fingerprint': 'old'},
        'quality_gate': {'passed': False, 'blocking_issues': [{'code': 'claim_evidence_quality_not_met'}]},
        'claim_verification': {'claims': [{'claim_id': 'c1', 'support_status': 'unsupported'}]}}
    assert not any(i.affected_claim_ids for i in diagnose_generation_issues(state))
