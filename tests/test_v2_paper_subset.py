"""显式论文子集的输入/工具授权与局部合并，不覆盖其他论文。"""
import copy
import pytest

from app.agent.execution_budget import budget_scope
from app.agent.policy.authorization import AuthorizationPolicy
from app.database.runtime_repository import RuntimeConflict
from app.schemas.delegation_schema import DelegationRequest
from app.services.durable_execution_service import durable_scope
from app.services.research_delegation_service import run_delegation
from app.services.research_artifact_service import ResearchArtifactService
from app.services.tool_execution import execute_tool
from tests.test_v2_delegation_runtime import db, state
from tests.test_tool_execution import events


def current_state():
    current = state()
    current.update(paper_details=[{'paper_id': 'p1', 'title': 'one'}, {'paper_id': 'p2', 'title': 'two'}],
        paper_cards=[{'paper_id': 'p1', 'title': 'old one'}, {'paper_id': 'p2', 'title': 'kept two'}],
        extraction_cache={'one': {'paper_id': 'p1'}, 'two': {'paper_id': 'p2'}},
        pending_card_rebuild_ids=['p1', 'p2'], required_reference_count=3)
    return current


def request(ids=('p1',)):
    return DelegationRequest(agent='analysis', profile='evidence_build', goal='rebuild selected evidence', scope={'paper_ids': list(ids)})


def test_subset_projects_and_merges_only_selected_papers(db):
    current = current_state()
    before = copy.deepcopy(current['paper_cards'][1])
    def extract(w, args):
        assert [p['paper_id'] for p in w['paper_details']] == ['p1']
        assert set(w['extraction_cache']) == {'one'}
        assert w['required_reference_count'] == 3
        w.update(paper_cards=[{'paper_id': 'p1', 'title': 'new one'}],
                 extraction_cache={'new': {'paper_id': 'p1'}}, pending_card_rebuild_ids=[])
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        result = run_delegation(current, request(), llm=object(), handlers={'extract_paper_cards': extract}, strategy='react')
        assert result.outcome == 'needs_help'
        assert 'global_claim_planning_required_after_scoped_extraction' in result.unresolved
        assert current['paper_cards'] == [{'paper_id': 'p1', 'title': 'new one'}, before]
        assert current['extraction_cache'] == {'new': {'paper_id': 'p1'}, 'two': {'paper_id': 'p2'}}
        assert current['pending_card_rebuild_ids'] == ['p2']
        assert current['required_reference_count'] == 3
        lineage = events(db, 'action_artifacts_committed')[0]
        assert [i['paper_id'] for i in lineage['outputs']['paper_cards']['items']] == ['p1']
        packet = events(db, 'action_resources_bound')[0]
        assert set(packet['paper_resources']['papers']) == {'p1'}


@pytest.mark.parametrize('mutation', ['card', 'cache', 'artifact', 'tool'])
def test_subset_rejects_outside_paper_writes_and_tools(db, mutation):
    current = current_state()
    before = copy.deepcopy(current['paper_cards'])
    def extract(w, args):
        if mutation == 'card': w['paper_cards'] = [{'paper_id': 'p2', 'title': 'unauthorized'}]
        elif mutation == 'cache': w['extraction_cache']['new'] = {'paper_id': 'p2'}
        elif mutation == 'artifact':
            ResearchArtifactService(db).persist_payload(session_id='v2', artifact_type='paper_card', payload={'value': {'paper_id': 'p2'}})
        else:
            execute_tool('vector_read', lambda: pytest.fail('outside vector request'), arguments={'paper_id': 'p2'})
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        with pytest.raises(RuntimeConflict):
            run_delegation(current, request(), llm=object(), handlers={'extract_paper_cards': extract})
        assert current['paper_cards'] == before


@pytest.mark.parametrize('ids', [[], ['unknown'], ['p1', 'p1']])
def test_invalid_subset_is_rejected_before_execution(ids):
    with pytest.raises(ValueError):
        AuthorizationPolicy().issue(request(ids), current_state())


def test_subset_cannot_consume_mandatory_rebuild_queue():
    current = current_state()
    current['agent_mandatory_actions'] = [{'action': 'extract_paper_cards', 'arguments': {}}]
    with pytest.raises(ValueError, match='mandatory'):
        AuthorizationPolicy().issue(request(), current)
