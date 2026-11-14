"""Behavioral coverage for draft delivery, immutable publication and shared budgets."""
import copy
import json
from types import SimpleNamespace as NS
import pytest

from tests.test_context_writer_repair import state as writer_state, plan
from tests.test_durable_runtime import db
from app.agent.delivery import draft_preflight, publish_draft, delivery_output, published_delivery, supplement_draft_plans
from app.agent.execution_budget import (budget_scope, budgeted_create, AgentBudgetExceeded,
    AgentFinalizationRequired, AgentExecutionCancelled, research_stage, update_finalization)
from app.agent.runtime_config import initialize_delivery
from app.core.config import get_settings


def draft_state():
    s=writer_state()
    s.update(delivery_mode="draft_first", intent="generate_review", topic="RAG", required_reference_count=5,
        writing_version=1, writing_plans=[plan().model_dump(mode="json")],
        review="## 研究背景\n\n"+"相关研究通过外部知识检索为开放域问答提供可更新的文本材料，并将检索结果用于生成过程[p1]。"*4)
    card=s['paper_cards'][0]
    card.update(relation_type='direct',eligible_deliverables=['research_background'],year=2021,
        authors=['Author'],source='arxiv',url='https://example.org/paper',venue='arXiv',
        arxiv_id='2005.11401',publication_status='preprint')
    s['paper_details']=[{k:card[k] for k in ('paper_id','title','year','authors','source','url','venue','arxiv_id','publication_status')}]
    from app.agent.nodes.verification import citation_check_node
    citation_check_node(s)
    assert s['citation_validation']['valid']
    return s


def test_shortfall_draft_publishes_without_changing_final_goal():
    s=draft_state(); assert draft_preflight(s)['ready']
    record=publish_draft(s)
    out=delivery_output(s,{'status':'blocked'})
    assert record['grade']=='draft' and out['status']=='partial'
    assert out['delivery_status']=='draft' and not out['requirements_satisfied']
    assert s['required_reference_count']==5 and '[1]' in out['body']
    assert '最终目标5篇' in out['body'] and out['references']


@pytest.mark.parametrize('terminal',['blocked','failed','cancelled'])
def test_published_version_survives_failed_later_candidate(terminal):
    s=draft_state(); record=publish_draft(s)
    s.update(review='坏的新稿',writing_version=2,result_status=terminal)
    out=delivery_output(s,{'status':terminal})
    assert out['body']==record['body'] and out['published_draft_version']==1
    assert out['execution_status']=={'blocked':'stopped','failed':'failed','cancelled':'cancelled'}[terminal]


def test_unpublished_candidate_stays_private_after_hard_stop():
    s=draft_state();s['result_status']='blocked'
    from app.agent.graph import _build_output
    out=_build_output(s)
    assert not out['body'] and out['delivery_status']=='none'


def test_strict_policy_does_not_publish_shortfall():
    s=draft_state();s['delivery_mode']='strict'
    with pytest.raises(ValueError,match='not authorized'):publish_draft(s)
    assert not s.get('published_versions')


@pytest.mark.parametrize('mutation',['source','citation','placeholder','stale_report','contradiction'])
def test_bad_source_or_known_error_blocks_draft(mutation):
    s=draft_state()
    if mutation=='source':s['paper_cards'][0]['evidence_spans'][0]['source_text_hash']='wrong'
    elif mutation=='citation':s['citation_validation']={'valid':False}
    elif mutation=='placeholder':s['review']='## 正文生成已阻止\n'+'材料不足'*30
    elif mutation=='stale_report':s['check_report']={'input_fingerprint':'old','issues':[]}
    else:
        from app.agent.checking import check_input_fingerprint
        s['check_report']={'input_fingerprint':check_input_fingerprint(s),'issues':[{'blocks_draft':True}]}
    assert not draft_preflight(s)['ready']
    with pytest.raises(ValueError):publish_draft(s)


def test_incomplete_semantic_check_can_publish_as_draft_only():
    s=draft_state()
    from app.agent.checking import build_check_report,check_input_fingerprint
    s['quality_gate']={'passed':False,'blocking_issues':[{'code':'claim_verification_incomplete','message':'服务超时'}]}
    build_check_report(s,check_input_fingerprint(s),failure='timeout')
    assert not any(i['blocks_draft'] for i in s['check_report']['issues'])
    out=publish_draft(s)
    assert out['grade']=='draft' and out['verification_status']=='partial'


def test_source_bound_plan_does_not_masquerade_as_verified():
    s=draft_state();s['claim_plans']=[]
    supplement_draft_plans(s)
    claim=s['claim_plans'][0]['claims'][0]
    assert claim['authorization_kind']=='draft_only' and claim['verification_status']=='unverified'
    from app.services.writer_support import authorized_writer_claims
    assert authorized_writer_claims(s,'p1')[0]['authorization_kind']=='draft_only'


def test_publication_is_idempotent_and_scoped_to_research():
    s=draft_state();a=publish_draft(s);b=publish_draft(s)
    assert a==b and len(s['published_versions'])==1
    s['user_query']='another research request'
    assert published_delivery(s) is None


def test_cancel_cannot_publish_new_version():
    s=draft_state()
    with pytest.raises(AgentExecutionCancelled):
        with budget_scope(s,lambda:True):publish_draft(s)
    assert not s.get('published_versions')


def test_effective_settings_are_restored_without_resetting_usage(monkeypatch):
    s=draft_state()
    monkeypatch.setattr(get_settings(),'agent_main_max_rounds',30)
    with budget_scope(s):
        assert get_settings().agent_main_max_rounds==30
        s['agent_execution_budget']['llm_requests']=12
    monkeypatch.setattr(get_settings(),'agent_main_max_rounds',80)
    with budget_scope(s):
        assert get_settings().agent_main_max_rounds==30
        assert s['agent_execution_budget']['llm_requests']==12
    assert not any('api_key' in k or k.endswith('_url') for k in s['effective_config']['values'])


def test_old_task_defaults_strict_and_explicit_request_wins():
    old={'agent_execution_budget':{'llm_requests':2}}
    initialize_delivery(old,new_task=False);assert old['delivery_mode']=='strict'
    new={};initialize_delivery(new,'只交完整验收稿，不接受草稿',new_task=True)
    assert new['delivery_mode']=='strict'


def test_soft_reserve_returns_budget_to_main_and_writer(monkeypatch):
    s=draft_state()
    monkeypatch.setattr(get_settings(),'agent_model_request_limit',10)
    monkeypatch.setattr(get_settings(),'agent_finalization_reserve_requests',3)
    calls=[]
    client=NS(chat=NS(completions=NS(create=lambda **kw:(calls.append(kw) or NS(usage=NS(total_tokens=10,prompt_tokens=5,completion_tokens=5))))))
    with budget_scope(s):
        s['agent_execution_budget']['llm_requests']=6
        with research_stage(s,'search_and_rank'):
            with pytest.raises(AgentFinalizationRequired):budgeted_create(client,messages=[],max_tokens=10)
        assert s['finalization_phase']=='finalizing' and not calls
        with research_stage(s,'generate_deliverables'):budgeted_create(client,messages=[],max_tokens=10)
        assert len(calls)==1 and s['agent_execution_budget']['llm_requests']==7


def test_hard_request_limit_never_gives_final_model_call(monkeypatch):
    s=draft_state();monkeypatch.setattr(get_settings(),'agent_model_request_limit',2)
    client=NS(chat=NS(completions=NS(create=lambda **kw:pytest.fail('must not call provider'))))
    with budget_scope(s):
        s['agent_execution_budget']['llm_requests']=2
        with research_stage(s,'generate_deliverables'):
            with pytest.raises(AgentBudgetExceeded):budgeted_create(client,messages=[])


def test_runtime_idle_revision_preserves_usage_and_uses_cas(db):
    from app.database.runtime_repository import ResearchRuntimeRepository,RuntimeConflict
    s=draft_state()
    with budget_scope(s):pass
    ledger=s['agent_execution_budget'];ledger['llm_requests']=9
    repo=ResearchRuntimeRepository(db,'s');repo.acquire(s,ledger);repo.snapshot(s,ledger=ledger);repo.release(ledger)
    revision=s['effective_config']['revision']
    result=repo.revise_configuration(expected_revision=revision,values={'agent_model_request_limit':2048})
    assert result['revision']==revision+1
    restored=repo.restore()
    assert restored['agent_execution_budget']['llm_requests']==9
    assert restored['agent_execution_budget']['model_request_limit']==2048
    with pytest.raises(RuntimeConflict):repo.revise_configuration(expected_revision=revision,values={'agent_model_request_limit':3000})
    with pytest.raises(ValueError):repo.revise_configuration(expected_revision=revision+1,values={'agent_model_request_limit':2})


def test_runtime_rejects_revision_while_owned(db):
    from app.database.runtime_repository import ResearchRuntimeRepository,RuntimeConflict
    s=draft_state()
    with budget_scope(s):pass
    repo=ResearchRuntimeRepository(db,'s');repo.acquire(s,s['agent_execution_budget']);repo.snapshot(s,ledger=s['agent_execution_budget'])
    try:
        with pytest.raises(RuntimeConflict):repo.revise_configuration(expected_revision=1,values={'agent_main_max_rounds':100})
    finally:repo.release(s['agent_execution_budget'])


def test_published_versions_round_trip_as_service_artifacts(db):
    from app.services.research_artifact_service import ResearchArtifactService
    s=draft_state();publish_draft(s)
    service=ResearchArtifactService(db)
    stored=service.externalize_state('s',s);restored=service.hydrate_state('s',stored)
    assert published_delivery(restored)['body']==published_delivery(s)['body']
    with pytest.raises(PermissionError):service.hydrate_state('s',stored,role='search')


def test_actual_retrieval_tokens_reduce_status_balance():
    from app.agent.context_builder import build_main_agent_context
    s=draft_state();s['agent_execution_budget']={'token_limit':1000,'llm_tokens':100,'retrieval_model_tokens':200,'tokens_reserved':50}
    assert build_main_agent_context(s).state.budget_remaining['tokens']==650


def test_committed_subagent_result_needs_no_second_model_report(monkeypatch):
    from tests.test_subagent_react import ScriptedLLM,state
    from app.agent.subagents.loop import SubagentLoop
    from app.agent.controller import AgentController
    from app.schemas.agent_decision_schema import AgentDecision
    s=state();s['delivery_mode']='strict'
    monkeypatch.setattr(get_settings(),'agent_deterministic_subagent_execute',False)
    llm=ScriptedLLM({'search_agent_decision':['search_and_rank']})
    calls=[]
    def search(working,args):
        calls.append('search');working['candidate_papers']=[{'paper_id':'p1','title':'source'}]
    result=SubagentLoop(llm,AgentController()).run(s,decision=AgentDecision(action='search_and_rank'),
        handlers={'search_and_rank':search})
    assert calls==['search'] and len(llm.calls)==1
    assert result.operation=='search_and_rank' and result.changed_state_fields


def test_task_scope_propagates_policy_to_nested_graph_state():
    outer={'delivery_mode':'draft_first'};inner={}
    with budget_scope(outer):
        with budget_scope(inner):
            assert inner['delivery_mode']=='draft_first'
            assert inner['effective_config']==outer['effective_config']
            assert inner['agent_execution_budget'] is outer['agent_execution_budget']


def test_download_marks_exact_unverified_claims():
    s=draft_state();s['claim_verification']={'claims':[{'sentence':'某个实验比较仍缺直接证据。','support_status':'unsupported'}]}
    record=publish_draft(s)
    assert '待完善内容' in record['body'] and '某个实验比较仍缺直接证据' in record['body']


def test_first_usable_draft_is_published_before_more_research():
    from app.agent.main_loop import MainAgentLoop
    from tests.test_subagent_react import ScriptedLLM
    s=draft_state();s['pending_result_validation']=True
    llm=ScriptedLLM({'main_agent_decision':['request_finish']})
    status=MainAgentLoop(llm).run(s,handlers={'validate_result':lambda *a:pytest.fail('should first publish')},
        finish_validator=lambda current:bool(published_delivery(current)))
    assert status=='completed' and s['main_agent_decisions'][0]['action']=='publish_draft'
    assert published_delivery(s)['grade']=='draft' and len(llm.calls)==1


def test_exact_main_arguments_are_dispatched_without_model_replanning():
    from app.agent.subagents.loop import SubagentLoop
    from app.agent.controller import AgentController
    from app.schemas.agent_decision_schema import AgentDecision
    from tests.test_subagent_react import ScriptedLLM,state
    s=state();s.update(delivery_mode='draft_first',paper_cards=[{'paper_id':'p1'}],
        evidence_gap_report={'needs_recovery':True},recovery_decision={'action':'TARGETED_SEARCH','kind':'quality','queries':['original']})
    llm=ScriptedLLM({})
    # Ordinary Search suffices to prove exact dispatch; no source/model access occurs in the handler.
    result=SubagentLoop(llm,AgentController()).run(s,decision=AgentDecision(action='search_and_rank'),
        handlers={'search_and_rank':lambda working,args:working.update(candidate_papers=[{'paper_id':'p2'}])})
    assert result.changed_state_fields and not llm.calls


def test_live_delivery_poll_does_not_hydrate_or_hash_pdf(db):
    from app.services.research_artifact_service import ResearchArtifactService
    from app.database.runtime_repository import ResearchRuntimeRepository
    s=draft_state();publish_draft(s)
    with budget_scope(s):pass
    repo=ResearchRuntimeRepository(db,'s');repo.acquire(s,s['agent_execution_budget']);repo.snapshot(s,ledger=s['agent_execution_budget']);repo.release(s['agent_execution_budget'])
    service=ResearchArtifactService(db)
    original=service.hydrate_state
    def minimal(session_id,state,**kw):
        assert set(state.get('artifact_manifest',{}))=={'published_versions'}
        assert 'controlled_pdf_files' not in state
        return original(session_id,state,**kw)
    service.hydrate_state=minimal
    assert service.read_published_delivery('s')['body']==published_delivery(s)['body']


def test_later_contradiction_withdraws_only_the_same_published_version():
    from app.agent.delivery import review_published_delivery
    from app.agent.checking import check_input_fingerprint
    s=draft_state();record=publish_draft(s)
    s['check_report']={'report_id':'r','input_fingerprint':check_input_fingerprint(s),
        'issues':[{'blocks_draft':True,'code':'claim_contradicted_by_evidence'}]}
    review_published_delivery(s)
    assert published_delivery(s) is None
    assert s['published_versions'][0]['body']==record['body']


def test_bad_new_candidate_check_keeps_earlier_valid_publication():
    from app.agent.delivery import review_published_delivery
    from app.agent.checking import check_input_fingerprint
    s=draft_state();record=publish_draft(s)
    s.update(review='different bad candidate',writing_version=2)
    s['check_report']={'report_id':'r','input_fingerprint':check_input_fingerprint(s),
        'issues':[{'blocks_draft':True,'code':'claim_contradicted_by_evidence'}]}
    review_published_delivery(s)
    assert published_delivery(s)['body']==record['body']


@pytest.mark.parametrize('year',[None,2019,2024])
def test_draft_does_not_relax_explicit_years(year):
    s=draft_state();s.update(start_year=2020,end_year=2023,year_range_explicit=True)
    s['paper_details'][0]['year']=year
    assert not draft_preflight(s)['ready']


def test_published_body_uses_its_own_citation_mapping():
    s=draft_state();record=publish_draft(s)
    s['citation_map']={'new_bad_paper':1}
    out=delivery_output(s,{'status':'failed','citation_map':s['citation_map']})
    assert out['citation_map']==record['citation_map']
    assert out['status']=='partial' and out['execution_status']=='failed'


def test_main_can_finish_with_retained_good_version_after_new_candidate_fails(monkeypatch):
    from app.agent import graph
    from tests.test_subagent_react import ScriptedLLM
    s=draft_state();record=publish_draft(s)
    s.update(review='bad new candidate',pending_result_validation=True,writing_version=2)
    monkeypatch.setattr(graph,'_get_llm',lambda:ScriptedLLM({'main_agent_decision':['request_finish']}))
    out=graph._run_autonomous_pipeline(s)
    assert out['status']=='partial' and out['body']==record['body']
    assert out['execution_status']=='completed'


def test_preflight_model_service_uses_task_snapshot_not_new_process_defaults(monkeypatch):
    from app.services.llm_service import LLMService
    s={'delivery_mode':'strict'}
    monkeypatch.setattr(get_settings(),'llm_request_timeout',31)
    with budget_scope(s):pass
    monkeypatch.setattr(get_settings(),'llm_request_timeout',91)
    original=LLMService();original._client=object()
    with budget_scope(s):
        bound=original.with_task_settings()
        assert bound.request_timeout==31 and original.request_timeout==91
        assert bound._client is original._client


def test_soft_stop_commits_only_completed_authorized_stage_results(db,monkeypatch):
    from app.agent.controller import AgentController
    from app.services.durable_execution_service import durable_scope
    from app.schemas.agent_task_schema import AgentRole
    s=draft_state();monkeypatch.setattr(get_settings(),'agent_model_request_limit',10)
    monkeypatch.setattr(get_settings(),'agent_finalization_reserve_requests',3)
    client=NS(chat=NS(completions=NS(create=lambda **kw:pytest.fail('reserve must prevent this call'))))
    def handler(working):
        working['candidate_papers']=[{'paper_id':'new','title':'completed discovery'}]
        budgeted_create(client,messages=[],max_tokens=10)
    with budget_scope(s):
        s['agent_execution_budget']['llm_requests']=6
        with durable_scope(db,'s',s,s['agent_execution_budget']) as runtime:
            result=AgentController().execute(role=AgentRole.SEARCH,operation='search_and_rank',objective='search',state=s,handler=handler)
            assert result.outcome.value=='degraded' and any(i.code=='budget_low' for i in result.issues)
            assert s['candidate_papers'][0]['paper_id']=='new'
            assert runtime.restore()['candidate_papers']==s['candidate_papers']
            assert s['agent_execution_budget']['llm_requests']==6


def test_retrieval_subquota_preserves_total_budget_for_writer(monkeypatch):
    from app.agent.execution_budget import record_retrieval_request
    s=draft_state();monkeypatch.setattr(get_settings(),'retrieval_model_request_limit',1)
    monkeypatch.setattr(get_settings(),'agent_model_request_limit',10)
    calls=[]
    client=NS(chat=NS(completions=NS(create=lambda **kw:(calls.append(kw) or NS(usage=NS(total_tokens=2,prompt_tokens=1,completion_tokens=1))))))
    with budget_scope(s):
        record_retrieval_request('embedding')
        with pytest.raises(AgentFinalizationRequired):record_retrieval_request('rerank')
        with research_stage(s,'generate_deliverables'):budgeted_create(client,messages=[],max_tokens=10)
        assert len(calls)==1 and s['agent_execution_budget']['retrieval_model_requests']==1
