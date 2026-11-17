"""真实故障的行为回归：Main保留决策权，原文/授权/复验协议不可绕过。"""
import copy

import pytest

from app.agent.action_registry import allowed_actions
from app.agent.controller import AgentController
from tests.reference_v1.main_loop import MainAgentLoop, _progress_fingerprint
from app.agent.quality_feedback import diagnose_quality_recovery, quality_feedback, complete_main_recovery
from app.agent.delegation_recovery import search_once, settle_after_analysis
from app.schemas.agent_task_schema import AgentRole, AgentTaskOutcome
from tests.test_subagent_react import ScriptedLLM, report, state, store
from tests.agent_decision_fakes import checked_result


def card(i):
    return {'paper_id':f'p{i}','title':f'RAG evidence {i}','year':2021,'quality_status':'partial',
            'relation_type':'direct','eligible_deliverables':['research_background'],
            'evidence_source':'abstract','evidence_spans':[{'evidence_id':f'p{i}:e1','text':'原文支持研究问题'}]}


def prepared():
    current=state()
    current.update(intent='generate_review',topic='RAG open domain question answering',
                   start_year=2020,end_year=2023,year_range_explicit=True,required_reference_count=3,
                   core_deliverables=['research_background'],allow_evidence_expansion=True,
                   ranked_papers=[paper(i) for i in range(3)],
                   paper_details=[paper(i) for i in range(3)],
                   paper_cards=[card(i) for i in range(3)],claim_plans=[{'claims':[{'claim_id':'allowed'}]}])
    return current


def paper(i):
    return {k:card(i)[k] for k in ('paper_id','title','year')}


def failed_verification(current):
    current.update(
                   quality_gate={'passed':False,'phase':'post_generation','draft_released':False,
                       'blocking_issues':[{'code':'claim_evidence_quality_not_met','message':'c_bad缺直接证据'}]},
                   claim_verification={'valid':False,'supported':2,'unsupported':1,'unverified':0,'support_rate':2/3,
                       'claims':[{'claim_id':'c_bad','sentence':'Retrieval reduces unsupported answers in open domain QA',
                                  'support_status':'unsupported','verification_status':'verified',
                                  'evidence_ids':['p0:e1'],'issues':['缺少具体对照结果']} ]},
                   unique_valid_cited_paper_count=3)


@pytest.mark.parametrize('native',[True,False])
def test_main_selects_search_then_analysis_authorization_then_write_and_verify(monkeypatch,native,store):
    import app.agent.nodes as nodes
    monkeypatch.setattr(nodes,'diagnose_evidence_gaps_node',lambda s,**kw:s.update(evidence_gap_report={'needs_recovery':False,'gaps':[]}))
    monkeypatch.setattr(nodes,'global_evidence_gate_node',lambda s,**kw:s.update(global_evidence_gate={'passed':True}))
    current=prepared();seen=[];writes=[];checks=[]
    def write(s,args):
        if writes:
            assert len(s['paper_cards'])==4 and not s.get('pending_evidence_recovery')
            assert s['claim_plans'][0]['claims'][0]['evidence_id']=='p3:e1'
        writes.append(len(s['paper_cards']));seen.append('write')
        s.update(review=('初稿' if len(writes)==1 else '新旧证据支持的改写稿')*100,
                 pending_result_validation=True)
        s.pop('quality_gate',None)
    def verify(s,args):
        seen.append('verify');checks.append(s['review'])
        if len(checks)==1:failed_verification(s)
        else:
            from app.agent.graph import _draft_fingerprint
            s.update(quality_gate={'passed':True,'phase':'post_generation','draft_released':True},
                     unique_valid_cited_paper_count=3,unique_cited_paper_count=3,
                     claim_verification={'valid':True,'supported':3,'unsupported':0,'unverified':0,'support_rate':1.0})
        checked_result(s)
    def diagnose(s,args):seen.append('diagnose');diagnose_quality_recovery(s)
    def search(s,args):
        seen.append('search')
        def raw(value,**kwargs):
            assert (value['start_year'],value['end_year'],value['required_reference_count'])==(2020,2023,3)
            value['candidate_papers']=[paper(i) for i in range(4)]
            value['ranked_papers']=copy.deepcopy(value['candidate_papers'])
            value['last_search_new_results']=1
        search_once(s,search=raw,rank=lambda x:None)
    def fetch(s,args):seen.append('fetch');s['paper_details'].append(paper(3))
    def extract(s,args):seen.append('extract');s['paper_cards'].append(card(3))
    def claims(s,args):
        seen.append('claims')
        s['claim_plans']=[{'claims':[{'claim_id':'new','evidence_id':'p3:e1'}]}]
        settle_after_analysis(s)
    llm=ScriptedLLM({
        'main_agent_decision':['generate_deliverables','validate_result','diagnose_evidence_gaps','targeted_search',
                               'generate_deliverables','validate_result','request_finish'],
        'writing_agent_decision':['generate_deliverables',report(),'generate_deliverables',report()],
        'check_agent_decision':['validate_result',report('needs_help','c_bad缺证据',['diagnose_evidence_gaps']), 'validate_result',report()],
        'analysis_agent_decision':['diagnose_evidence_gaps',report(), 'extract_paper_cards',report(), 'plan_claims',report()],
        'search_agent_decision':['targeted_search',report(),'fetch_metadata',report()]},native=native)
    status=MainAgentLoop(llm).run(current,handlers={'generate_deliverables':write,'validate_result':verify,
        'diagnose_evidence_gaps':diagnose,'targeted_search':search,'fetch_metadata':fetch,
        'extract_paper_cards':extract,'plan_claims':claims},
        finish_validator=lambda s:not s.get('pending_result_validation') and (s.get('quality_gate') or {}).get('passed'))
    assert status=='completed', {'seen':seen,'errors':current.get('errors'),'rejections':current.get('main_agent_rejections')}
    assert seen==['write','verify','diagnose','search','fetch','extract','claims','write','verify']
    assert writes==[3,4]
    assert current['recovery_action_count']==1
    assert current['agent_execution_budget']['recovery_used']==1
    assert current['quality_recovery_history'][-1]['outcome']=='improved'
    assert not current.get('active_quality_recovery')
    assert all('c_bad' in str(messages) for op,messages in llm.calls
               if op=='main_agent_decision' and '缺证据' in str(messages))


def test_pending_candidate_cannot_be_displaced_by_diagnosis_or_replanning():
    current=prepared();current.update(review='待验证',pending_result_validation=True)
    names=allowed_actions(current)
    assert names==['validate_result','report_blocked']
    from app.agent.context_builder import build_main_agent_context
    assert build_main_agent_context(current).state.next_actions==['validate_result']
    with pytest.raises(ValueError,match='awaiting independent validation'):
        AgentController().execute(role=AgentRole.ANALYSIS,operation='plan_claims',objective='不能绕过',state=current,handler=lambda s:None)


def test_short_invalid_candidate_is_reported_blocked_with_structured_feedback():
    current=prepared()
    def write(s):
        s.update(review='以原文为准，不作外推。',pending_result_validation=True,
                 deliverable_validation=[{'valid':False,'errors':['正文不足且没有引用']}])
    result=AgentController().execute(role=AgentRole.WRITING,operation='generate_deliverables',objective='写正文',state=current,handler=write)
    assert result.outcome==AgentTaskOutcome.BLOCKED
    assert any('unusable' in x for x in result.findings)
    assert not any('available' in x for x in result.findings)
    feedback=result.proposed_decisions[0]['feedback']
    assert not feedback['candidate']['usable'] and feedback['candidate']['pending_validation']
    assert feedback['suggested_actions']==['validate_result']


def test_feedback_distinguishes_previous_verification_from_current_candidate():
    current=prepared();current['review']='初稿';failed_verification(current)
    first=quality_feedback(current)
    assert first['weak_claims'][0]['claim_id']=='c_bad'
    current.update(review='新候选',pending_result_validation=True)
    second=quality_feedback(current)
    assert second['verification_state']=='previous_candidate'
    assert second['suggested_actions']==['validate_result']
    assert _progress_fingerprint(current)!=_progress_fingerprint({**current,'pending_result_validation':False})


@pytest.mark.parametrize('stop',['permission','budget'])
def test_quality_search_respects_existing_permission_and_recovery_budget(stop):
    from app.agent.evidence_recovery import targeted_search_kind
    current=prepared();current['review']='已有候选';failed_verification(current)
    if stop=='permission':current['allow_evidence_expansion']=False
    else:current['recovery_action_count']=999
    diagnose_quality_recovery(current)
    assert targeted_search_kind(current)==''


def test_empty_quality_retrieval_settles_as_exhausted_without_resetting_history(monkeypatch):
    import app.agent.nodes as nodes
    monkeypatch.setattr(nodes,'diagnose_evidence_gaps_node',lambda s,**kw:s.update(evidence_gap_report={'gaps':[]}))
    monkeypatch.setattr(nodes,'global_evidence_gate_node',lambda s,**kw:None)
    current=prepared();current.update(recovery_round=1,recovery_action_count=2,
        pending_evidence_recovery={'kind':'quality','decision':{'queries':['补证']},'before_card_ids':['p0','p1','p2']})
    settle_after_analysis(current)
    assert current['evidence_recovery_status']=='EXHAUSTED'
    assert 'pending_evidence_recovery' not in current
    assert current['recovery_action_count']==2


@pytest.mark.parametrize('change',['review','claim_plans'])
def test_stale_quality_queries_cannot_be_reused_for_a_new_input(change):
    from app.agent.evidence_recovery import targeted_search_kind
    current=prepared();current['review']='待恢复正文';failed_verification(current)
    diagnose_quality_recovery(current)
    assert targeted_search_kind(current)=='quality'
    current[change]='新正文' if change=='review' else [{'claims':[{'claim_id':'new'}]}]
    assert targeted_search_kind(current)==''


@pytest.mark.parametrize('minimum',[3,5,10,40])
def test_default_ranked_reserve_keeps_surplus_before_first_evidence_batch(minimum):
    from app.agent.slot_extractor import _derive_count_targets
    from app.agent.nodes.retrieval import absolute_evidence_pool_target
    ranked,_=_derive_count_targets(minimum,True)
    first_batch=absolute_evidence_pool_target(minimum,{},0.5)
    assert ranked>=2*minimum
    assert minimum<first_batch<=ranked


@pytest.mark.parametrize('native',[True,False])
def test_post_commit_report_failure_returns_main_and_does_not_repeat_work(native):
    current=prepared();seen=[]
    def write(s,args):seen.append('write');s.update(review='研究正文'*100,pending_result_validation=True)
    def verify(s,args):seen.append('verify');s.update(quality_gate={'passed':True});checked_result(s)
    llm=ScriptedLLM({'main_agent_decision':['generate_deliverables','validate_result','request_finish'],
        'writing_agent_decision':['generate_deliverables','generate_deliverables','generate_deliverables'],
        'check_agent_decision':['validate_result',report()]},native=native)
    status=MainAgentLoop(llm).run(current,handlers={'generate_deliverables':write,'validate_result':verify},
        finish_validator=lambda s:not s.get('pending_result_validation') and (s.get('quality_gate') or {}).get('passed'))
    assert status=='completed' and seen==['write','verify']
    assert len([x for x in llm.calls if x[0]=='writing_agent_decision'])==3
    assert current['agent_execution_budget']['actions_committed']==2
    assert any('汇报协议' in str(r.get('message')) for r in current['main_agent_rejections'])
    retained=current['subagent_sessions']['writing']['last_delegation']['last_result']
    assert retained['proposed_decisions'][0]['feedback']['candidate']['pending_validation']


def test_passed_gate_suggests_finish_instead_of_old_recovery_and_pending_still_requires_validation():
    current=prepared();current.update(review='当前已通过的正文',quality_gate={'passed':True},
        claim_verification={'valid':True,'unsupported':0},
        quality_recovery_decision={'action':'REWRITE_SECTIONS'},recovery_decision={'queries':['旧补检索建议']})
    assert quality_feedback(current)['suggested_actions']==['request_finish']
    current['pending_result_validation']=True
    assert quality_feedback(current)['suggested_actions']==['validate_result']
