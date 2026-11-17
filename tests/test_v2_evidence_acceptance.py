"""阶段验收遵守既有逐字摘要弱授权，不能升级成实验或强结论。"""
import copy
import hashlib
import json

import pytest

from app.agent.claim_plan import claim_has_authorized_support
from app.agent.policy.acceptance import accept_delegation
from app.agent.policy.authorization import AuthorizationPolicy
from app.agent.react_context import ReActContext, working_memory
from app.core.config import get_settings
from app.schemas.agent_decision_schema import AgentDecision
from app.schemas.delegation_schema import DelegationRequest


def source():
    text = 'The paper studies retrieval augmented generation.'
    card = {'paper_id': 'arxiv:one', 'quality_status': 'partial', 'evidence_source': 'abstract',
        'evidence_state': {'access_level': 'abstract'}, 'field_claims': {}, 'evidence_spans': [{
            'evidence_id': 'arxiv:one:e001', 'source_type': 'abstract', 'source_status': 'source_verified',
            'text': text, 'char_start': 0, 'char_end': len(text),
            'source_text_hash': hashlib.sha256(text.encode()).hexdigest()}]}
    claim = {'claim_text': text, 'claim_type': 'background_fact', 'evidence_ids': ['arxiv:one:e001'],
             'support_level': 'single', 'evidence_access_limit': 'abstract_only', 'allowed_verbs': ['提及', '涉及']}
    return card, claim


def test_stage_accepts_existing_literal_abstract_coverage_contract():
    card, claim = source()
    current = {'session_id': 'test', 'topic': 'RAG', 'paper_cards': [card],
               'claim_plans': [{'claims': [claim]}], 'claim_evidence_gate': {'passed': True}}
    grant = AuthorizationPolicy().issue(DelegationRequest(agent='analysis', profile='evidence_build', goal='evidence'), current)
    before = copy.deepcopy(current)
    assert claim_has_authorized_support(claim, {'arxiv:one': card})
    assert accept_delegation(grant, current, [{'operation': 'plan_claims', 'status': 'completed'}]).satisfied
    assert current == before


@pytest.mark.parametrize('change', [
    {'claim_type': 'quantitative_result'}, {'claim_type': 'finding'}, {'support_level': 'moderate'},
    {'evidence_access_limit': 'full_text'}, {'allowed_verbs': ['证明']}, {'evidence_ids': ['arxiv:one:e999']},
    {'claim_text': 'The paper does not study retrieval augmented generation.'}, {'evidence_ids': []},
])
def test_abstract_fallback_does_not_grant_stronger_or_unbound_claim(change):
    card, claim = source()
    claim.update(change)
    assert not claim_has_authorized_support(claim, {'arxiv:one': card})


@pytest.mark.parametrize('change', [{'source_type': 'metadata'}, {'source_status': 'source_unverified'},
                                    {'source_text_hash': 'changed'}, {'char_end': 999}])
def test_abstract_fallback_still_requires_independent_readable_source(change):
    card, claim = source()
    card['evidence_spans'][0].update(change)
    assert not claim_has_authorized_support(claim, {'arxiv:one': card})


@pytest.mark.parametrize('native', [False, True])
def test_needs_help_keeps_bounded_reasons_in_model_observation_and_memory(native):
    current = {}
    context = ReActContext(current)
    decision = AgentDecision(action='delegate_agent', arguments={},
                             transport='native_tool' if native else 'json', tool_call_id='call' if native else None)
    context.proposed(decision)
    context.observe({'status': 'completed', 'outcome': 'needs_help', 'findings': ['阶段未满足'],
        'unresolved': ['claim_evidence_authorization_incomplete', *(['x' * 300] * 7)]}, archive=False)
    visible = json.loads(context.data['messages'][-1]['content'])
    if not native:
        visible = visible['observation']
    assert visible['unresolved'][0] == 'claim_evidence_authorization_incomplete'
    assert len(visible['unresolved']) == 5 and all(len(x) <= 240 for x in visible['unresolved'])
    retained, _ = working_memory(context.data['memory'], get_settings())
    assert retained[-1]['unresolved'] == visible['unresolved']
