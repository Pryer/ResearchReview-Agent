"""真实验收中暴露的准入错位；不扩大任何角色权限。"""
import pytest
from app.agent.runtime.strategies import deterministic_decision
from app.agent.policy.authorization import AuthorizationPolicy, goal_fingerprint
from app.agent.action_registry import ACTION_REGISTRY, recovery_handoff_action
from app.schemas.delegation_schema import DelegationGrant
from tests.test_draft_delivery_policy import draft_state


@pytest.mark.parametrize('profile,available,results',[
    ('gap_recovery', ['fetch_metadata'], []),
    ('initial_research', ['search_and_rank'], [{'operation':'search_and_rank'}]),
    ('independent_check', ['fetch_metadata'], []),
])
def test_deterministic_next_step_never_proposes_unavailable_action(profile,available,results):
    grant=DelegationGrant(session_id='s',role='check' if profile=='independent_check' else 'search',
        profile=profile,goal='goal',goal_fingerprint='goal',allowed_actions=tuple(available))
    assert deterministic_decision(grant,{},available,results)==('report_to_main',{})


def test_main_publication_handoff_does_not_starve_active_check_grant():
    state=draft_state();state['pending_result_validation']=True
    assert recovery_handoff_action(state)=='publish_draft'
    state['active_delegation_id']='checking'
    grant=DelegationGrant(delegation_id='checking',session_id=state['session_id'],role='check',
        profile='independent_check',goal='check',goal_fingerprint=goal_fingerprint(state),
        allowed_actions=('validate_result',))
    assert recovery_handoff_action(state)==''
    assert AuthorizationPolicy().available_actions(grant,state,set(ACTION_REGISTRY))==['validate_result']
