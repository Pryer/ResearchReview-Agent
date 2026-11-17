"""Check只读、版本绑定、统一反馈与Main恢复的行为证明。"""
import copy
import pytest

from app.agent import graph
from app.agent.checking import build_check_report, check_input_fingerprint, check_is_current
from app.agent.controller import AgentController
from tests.reference_v1.main_loop import MainAgentLoop
from app.agent.subagents.base import AgentBoundaryViolation
from app.schemas.agent_task_schema import AgentRole, AgentTaskResult
from app.schemas.check_schema import SuggestedAction
from tests.test_subagent_react import ScriptedLLM, report, state, store
from tests.test_main_quality_loop import prepared


def checked(s, passed=True, unverified=0):
    s['claim_verification'] = {'valid': passed, 'supported': 2 if passed else 1,
        'unsupported': 0 if passed else 1, 'factual_claims': 2,
        'unverified': unverified, 'verification_coverage': 1.0 if not unverified else .5,
        'claims': [] if passed else [{'claim_id': 'c2', 'sentence': '夸大的主张',
            'support_status': 'unsupported', 'evidence_ids': ['p1:e1'], 'issues': ['超出原文范围']}]}
    s['quality_gate'] = {'passed': passed, 'draft_released': passed, 'phase': 'post_generation',
        'blocking_issues': [] if passed else [{'code': 'claim_evidence_quality_not_met', 'message': '主张不受支持'}]}
    build_check_report(s, check_input_fingerprint(s))


def check_task(s, handler):
    return AgentController().execute(role=AgentRole.CHECK, operation='validate_result',
        objective='独立检查', state=s, handler=handler)


def test_invalid_repair_target_returns_rejection_without_executing_writer(store):
    s = prepared()
    s.update(review='待修复正文', writing_plans=[{'deliverable_type': 'research_background',
        'sections': [{'id': 'background_body'}]}])
    checked(s, passed=False)
    llm = ScriptedLLM({'main_agent_decision': [
        ('rewrite_sections', {'section_ids': ['unknown_section']}),
        ('report_blocked', {'reason': '当前目标无法定位'})]}, True)
    status = MainAgentLoop(llm).run(s, handlers={'rewrite_sections': lambda *a: pytest.fail('invalid target must not execute')},
        finish_validator=lambda w: False)
    assert status == 'blocked'
    assert any(r['reason'] == 'invalid_rewrite_target' for r in s['main_agent_rejections'])
    assert not s.get('active_quality_recovery')


@pytest.mark.parametrize('field,value', [
    ('review', '改写正文'), ('paper_cards', []), ('claim_plans', []),
    ('required_reference_count', 1), ('writing_version', 10), ('citation_source_text', '伪造引用来源'),
])
def test_check_cannot_modify_candidate_evidence_or_constraints(field, value):
    s=prepared();s.update(review='候选正文', pending_result_validation=True)
    before=copy.deepcopy(s)
    with pytest.raises(AgentBoundaryViolation):
        check_task(s, lambda w: w.update({field:value}))
    for k in ('review', 'paper_cards', 'claim_plans', 'required_reference_count'):
        assert s[k]==before[k]


@pytest.mark.parametrize('field,value', [
    ('review','新版正文'), ('paper_cards',[]), ('claim_plans',[]),
    ('start_year',2021), ('required_reference_count',4), ('target_length',900),
    ('language','en'), ('citation_source_text','另一引用来源'),
])
def test_report_cannot_authorize_changed_inputs(field,value):
    s=prepared();s['review']='原候选';checked(s)
    assert check_is_current(s)
    s[field]=value
    assert not check_is_current(s)


def test_report_binding_ignores_dispatch_and_history_changes():
    s=prepared();s['review']='正文';checked(s)
    s.update(state_revision=99, pending_result_validation=False, errors=['调度记录'])
    assert check_is_current(s)


def test_check_completion_and_content_failure_are_separate():
    s=prepared();s.update(review='有问题的正文',pending_result_validation=True)
    result=check_task(s,lambda w:checked(w,False))
    assert result.status.value=='completed' and result.outcome.value=='degraded'
    assert not s['pending_result_validation']
    assert s['check_report']['execution_status']=='completed'
    assert s['check_report']['verdict']=='needs_revision'
    issue=next(i for i in result.issues if i.claim_id=='c2')
    assert issue.evidence_ids==['p1:e1'] and issue.reason=='超出原文范围'
    assert result.check_report_ref.endswith(s['check_report']['report_id'])


def test_incomplete_check_never_becomes_passed():
    s=prepared();s.update(review='正文',pending_result_validation=True)
    result=check_task(s,lambda w:checked(w,True,unverified=1))
    assert result.outcome.value=='blocked'
    assert s['check_report']['verdict']=='unable_to_verify'
    assert not s['quality_gate']['passed'] and not s['quality_gate']['draft_released']


@pytest.mark.parametrize('native',[True,False])
def test_check_failure_returns_to_main_for_local_rewrite_and_recheck(native,store):
    s=prepared();s.update(review='初稿',pending_result_validation=True,
        writing_plans=[{'deliverable_type':'research_background','sections':[{'id':'s2'}]}])
    seen=[]
    def validate(w,args):
        seen.append('check');checked(w,passed=w['review']=='局部修复后的完整正文')
    def write(w,args):
        assert args['section_ids']==['s2']
        assert w['check_report']['issues']
        seen.append('write');w.update(review='局部修复后的完整正文',pending_result_validation=True)
    llm=ScriptedLLM({
        'main_agent_decision':['validate_result',('rewrite_sections',{'section_ids':['s2']}),'validate_result','request_finish'],
        'check_agent_decision':['validate_result',report('needs_help','c2超出范围',['rewrite_sections']), 'validate_result',report()],
        'writing_agent_decision':[('rewrite_sections',{'section_ids':['s2']}),report()]},native)
    status=MainAgentLoop(llm).run(s,handlers={'validate_result':validate,'rewrite_sections':write},
        finish_validator=lambda w:check_is_current(w) and w['check_report']['verdict']=='passed' and not w['pending_result_validation'])
    assert status=='completed' and seen==['check','write','check']
    assert s['agent_execution_budget']['actions_committed']==3
    assert s['check_report']['verdict']=='passed'
    assert s['subagent_sessions']['check']['last_delegation']['last_result']['issues']==[]


def test_failed_tool_returns_structured_reason_for_main_decision():
    s=state();s['ranked_papers']=[{'paper_id':'p1','title':'候选'}]
    llm=ScriptedLLM({'main_agent_decision':['fetch_metadata',('report_blocked',{'reason':'来源暂时不可用'})],
        'search_agent_decision':['fetch_metadata']})
    calls=[]
    def fail(w,args):calls.append(1);raise RuntimeError('temporary provider unavailable')
    assert MainAgentLoop(llm).run(s,handlers={'fetch_metadata':fail},finish_validator=lambda w:False)=='blocked'
    last=s['subagent_sessions']['search']['last_delegation']['last_result']
    assert last['status']=='failed' and last['error']=='temporary provider unavailable'
    assert last['issues'][0]['code']=='execution_failed' and calls==[1]
    assert len([op for op,_ in llm.calls if op=='main_agent_decision'])==2


def test_search_success_is_not_failed_by_previous_draft_gate():
    s=state();s.update(generation_blocked=True,quality_gate={'passed':False})
    result=AgentController().execute(role=AgentRole.SEARCH,operation='search_and_rank',objective='补充论文',state=s,
        handler=lambda w:w.update(candidate_papers=[{'paper_id':'p1'}],ranked_papers=[{'paper_id':'p1'}]))
    assert result.outcome.value=='succeeded'


def test_check_report_roundtrips_as_immutable_artifact(store):
    s=prepared();s['review']='正文';checked(s)
    stored=store.externalize_state('s',s)
    assert 'check_report' not in stored and stored['artifact_manifest']['check_report']['shape']=='json_object'
    restored=store.hydrate_state('s',stored)
    assert restored['check_report']==s['check_report'] and check_is_current(restored)
    ref=stored['artifact_manifest']['check_report']['items'][0]['ref']
    assert store.resolve('s',ref,role='check')['artifact_type']=='check_report'
    with pytest.raises(PermissionError):store.resolve('s',ref,role='search')


def test_old_task_results_remain_readable_and_suggestions_are_registered():
    old={'task_id':'old','role':'analysis','operation':'validate_result','status':'completed','input_state_fingerprint':'old'}
    assert AgentTaskResult.model_validate(old).role.value=='analysis'
    assert AgentTaskResult.model_validate(old).issues==[]
    with pytest.raises(ValueError):SuggestedAction(action='随便搜一下')


def test_production_check_does_not_publish_and_finish_requires_same_version(monkeypatch):
    s=prepared();s.update(review='候选正文',pending_result_validation=True)
    calls=[]
    class LLM:
        def for_agent(self,role):calls.append(role);return self
    monkeypatch.setattr(graph,'_get_llm',lambda:LLM())
    def verify(w,**kw):
        assert kw['read_only'] is True
        calls.append('verify');checked(w)
    monkeypatch.setattr(graph,'_verify_generated_draft',verify)
    def final(w,**kw):
        assert kw['prevalidated'] is True
        calls.append('publish');w['body']=w['review'];w['answer']=w['review']
    monkeypatch.setattr(graph,'final_answer_node',final)
    def run(loop,w,*,handlers,finish_validator,**kw):
        result=loop.controller.execute(role=AgentRole.CHECK,operation='validate_result',objective='核验',state=w,
            handler=lambda work:handlers['validate_result'](work,{}))
        assert result.status.value=='completed' and 'publish' not in calls
        assert finish_validator(w)
        w['review']='更换正文'
        assert not finish_validator(w)
        return 'blocked'
    monkeypatch.setattr(MainAgentLoop,'run',run)
    output=graph._run_autonomous_pipeline(s)
    assert calls==['check','verify','publish'] and output['body']==''


def test_failed_check_does_not_reuse_prior_passed_report(monkeypatch):
    s=prepared();s.update(review='正文',pending_result_validation=True);checked(s)
    monkeypatch.setattr(graph,'_get_llm',lambda:object())
    def fail(w,**kw):raise RuntimeError('verifier unavailable')
    monkeypatch.setattr(graph,'_verify_generated_draft',fail)
    def run(loop,w,*,handlers,finish_validator,**kw):
        loop.controller.execute(role=AgentRole.CHECK,operation='validate_result',objective='重验',state=w,
            handler=lambda work:handlers['validate_result'](work,{}))
        assert w['check_report']['execution_status']=='failed'
        assert not finish_validator(w)
        return 'blocked'
    monkeypatch.setattr(MainAgentLoop,'run',run)
    assert graph._run_autonomous_pipeline(s)['body']==''


def test_readonly_semantic_node_does_not_delete_or_rewrite(monkeypatch):
    from app.agent.nodes.verification import verify_claims_node
    s=prepared();s.update(review='该方法在所有任务均有效[p0]。',writing_plans=[{}])
    original=s['review']
    monkeypatch.setattr('app.tools.verify_claims.verify_review_claims',lambda *a,**k:{
        'valid':False,'factual_claims':1,'unsupported':1,'supported':0,'unverified':0,'support_rate':0,
        'claims':[{'claim_id':'c1','sentence':original,'factual':True,'support_status':'unsupported'}]})
    monkeypatch.setattr('app.agent.nodes.verification._remove_unsupported_claims',lambda *a:pytest.fail('Check cannot delete'))
    monkeypatch.setattr('app.agent.nodes.verification._rewrite_and_weaken_unsupported_claims',lambda *a,**k:pytest.fail('Check cannot rewrite'))
    verify_claims_node(s,llm=object(),read_only=True)
    assert s['review']==original and s['claim_verification']['unsupported']==1
    assert s['claim_repairs']['strategy']=='main_delegation_required'


@pytest.mark.parametrize('changed_evidence',[True,False])
def test_controller_rolls_back_only_same_dependency_verified_candidate(changed_evidence):
    from app.agent.graph import _snapshot_generation_products
    s=prepared();s['review']='完整旧稿';checked(s)
    previous={'products':_snapshot_generation_products(s),'operation':'rewrite_sections'}
    s.update(review='退化残稿',pending_result_validation=True,pending_generation_previous=previous)
    if changed_evidence:s['paper_cards'][0]['new_evidence']='新增证据'
    def worse(w):
        checked(w,False)
    check_task(s,worse)
    assert s['review']==('退化残稿' if changed_evidence else '完整旧稿')
    assert bool(s.get('recovery_candidate_rejections')) is not changed_evidence
    assert check_is_current(s)


def test_blocking_claim_issue_cannot_coexist_with_passed_verdict():
    s=prepared();s['review']='有未支持主张的正文';checked(s,False)
    s['quality_gate']={'passed':True,'draft_released':True}
    build_check_report(s,check_input_fingerprint(s))
    assert s['check_report']['verdict']=='needs_revision'
    assert not s['quality_gate']['passed'] and not s['quality_gate']['draft_released']


def test_unfinished_legacy_analysis_validation_is_redelegated_to_check(store):
    from tests.reference_v1.subagent_loop import SubagentLoop
    from app.schemas.agent_decision_schema import AgentDecision
    from app.agent.context_builder import context_source_fingerprint
    s=prepared();s.update(review='历史候选',pending_result_validation=True,active_subagent='analysis')
    s['subagent_sessions']={'analysis':{'active':{'operation':'validate_result','completed_operations':[],
        'objective':'恢复历史核验','consume_mandatory_action':False}}}
    llm=ScriptedLLM({'check_agent_decision':['validate_result',report()]})
    result=SubagentLoop(llm,AgentController()).run(s,handlers={'validate_result':lambda w,a:checked(w)})
    assert result.role==AgentRole.CHECK and s['check_report']['verdict']=='passed'
    assert s['subagent_sessions']['analysis']['last_delegation']['operation']=='validate_result'
    assert not s.get('active_subagent')


def test_rejected_report_reference_survives_rollback_and_restart(store):
    from app.agent.graph import _snapshot_generation_products
    s=prepared();s['review']='完整旧稿';checked(s)
    previous={'products':_snapshot_generation_products(s),'operation':'rewrite_sections'}
    s.update(review='退化残稿',pending_result_validation=True,pending_generation_previous=previous)
    result=check_task(s,lambda w:checked(w,False))
    assert s['review']=='完整旧稿'
    restored=store.hydrate_state('s',store.externalize_state('s',s))
    record=store.resolve('s',result.check_report_ref,state=restored,allowed_types={'check_report'},role='analysis')
    assert record['payload']['verdict']=='needs_revision'
    assert record['payload']['report_id']!=restored['check_report']['report_id']


@pytest.mark.parametrize('after_commit',[True,False])
def test_model_protocol_failure_returns_structured_result_without_replaying(after_commit):
    from tests.reference_v1.subagent_loop import SubagentLoop
    from app.schemas.agent_decision_schema import AgentDecision
    from app.core.exceptions import LLMInvocationError
    s=prepared();s['review']='待核验';calls=[]
    items=(['validate_result'] if after_commit else [])+[LLMInvocationError('provider must return exactly one tool call')]
    llm=ScriptedLLM({'check_agent_decision':items})
    def verify(w,args):calls.append(1);checked(w)
    result=SubagentLoop(llm,AgentController()).run(s,decision=AgentDecision(action='validate_result'),handlers={'validate_result':verify})
    assert result.delegation_status=='needs_help' and len(calls)==int(after_commit)
    if after_commit:
        assert result.outcome.value=='succeeded' and check_is_current(s)
    else:
        assert result.issues[0].code=='subagent_decision_failed'


def test_model_provider_balance_failure_still_stops_execution():
    from tests.reference_v1.subagent_loop import SubagentLoop
    from app.schemas.agent_decision_schema import AgentDecision
    from app.core.exceptions import LLMProviderUnavailableError
    s=prepared();s['review']='待核验'
    llm=ScriptedLLM({'check_agent_decision':[LLMProviderUnavailableError('unfunded')]})
    with pytest.raises(LLMProviderUnavailableError):
        SubagentLoop(llm,AgentController()).run(s,decision=AgentDecision(action='validate_result'),handlers={'validate_result':lambda w,a:checked(w)})


def test_report_suggestions_are_preserved_in_unified_result():
    from tests.reference_v1.subagent_loop import SubagentLoop
    from app.schemas.agent_decision_schema import AgentDecision
    s=state()
    llm=ScriptedLLM({'search_agent_decision':['search_and_rank',report(suggestions=['fetch_metadata'])]})
    result=SubagentLoop(llm,AgentController()).run(s,decision=AgentDecision(action='search_and_rank'),
        handlers={'search_and_rank':lambda w,a:w.update(ranked_papers=[{'paper_id':'p1'}])})
    assert result.suggested_actions[0].action=='fetch_metadata'



def test_main_instruction_and_selected_context_reach_actual_execution_model():
    from app.services.llm_service import _RoleBoundLLM
    from app.agent.task_context import current_task_brief
    s=prepared();s['subagent_sessions']={'writing':{'active':{'operation':'generate_deliverables',
        'context_packet':{'open_questions':[{'question':'保留第一段，只修复第二段'}]}}}}
    calls=[]
    class Service:
        def complete_messages(self,messages,**kwargs):calls.append(messages);return '修改后的候选正文'
    writing=_RoleBoundLLM(Service(),'writing')
    check=_RoleBoundLLM(Service(),'check')
    def handler(w):
        w['review']=writing.complete('工具原来的写作提示',operation='write_deliverable')
        check.complete('独立角色不应收到Writing私有任务说明')
    AgentController().execute(role=AgentRole.WRITING,operation='generate_deliverables',objective='按Check反馈只修复c2，保留其他段落',
        state=s,handler=handler)
    assert '按Check反馈只修复c2' in calls[0][1]['content']
    assert '保留第一段，只修复第二段' in calls[0][1]['content']
    assert '按Check反馈只修复c2' not in str(calls[1])
    assert current_task_brief('writing') is None
    assert 'research_execution_brief' not in str(s)


def test_execution_brief_resets_on_failed_tool():
    from app.agent.task_context import current_task_brief
    s=prepared()
    def fail(w):
        assert current_task_brief('analysis')['objective']=='定向分析'
        raise RuntimeError('failed')
    with pytest.raises(RuntimeError):
        AgentController().execute(role=AgentRole.ANALYSIS,operation='extract_paper_cards',objective='定向分析',state=s,handler=fail)
    assert current_task_brief('analysis') is None


@pytest.mark.parametrize('native',[True,False])
def test_subagent_reason_cannot_replace_main_execution_objective(native):
    from tests.reference_v1.subagent_loop import SubagentLoop
    from app.schemas.agent_decision_schema import AgentDecision
    from app.agent.task_context import current_task_brief
    s=prepared();seen=[]
    class LLM(ScriptedLLM):
        def complete_tool_call(self,*a,**kw):
            result=super().complete_tool_call(*a,**kw);result['reason']='子Agent的短工具理由';return result
        def complete_messages(self,messages,**kwargs):
            import json
            result=json.loads(super().complete_messages(messages,**kwargs));result['reason']='子Agent的短工具理由';return json.dumps(result)
    llm=LLM({'writing_agent_decision':['generate_deliverables',report()]},native=native)
    def write(w,args):seen.append(current_task_brief('writing')['objective']);w['review']='候选正文'
    SubagentLoop(llm,AgentController()).run(s,decision=AgentDecision(action='generate_deliverables',reason='Main明确要求：只修复c2并保留已支持内容'),handlers={'generate_deliverables':write})
    assert seen==['Main明确要求：只修复c2并保留已支持内容']



def test_main_terminal_result_does_not_start_outer_recovery_or_quality_clarification(monkeypatch,store):
    from app.services.research_conversation_service import ResearchConversationService, _has_hard_execution_stop
    s=prepared();s.update(check_protocol_version=1,result_status='blocked',generation_blocked=True,
        quality_gate={'passed':False,'blocking_issues':[{'code':'claim_evidence_quality_not_met','message':'证据不足'}]},
        errors=[{'code':'main_agent_blocked','message':'Main已完成有界恢复并停止'}])
    result=graph._build_output(s)
    assert _has_hard_execution_stop(result)
    service=ResearchConversationService(store.repo.db,llm=None)
    monkeypatch.setattr(service,'_auto_recover_actionable_result',lambda **kw:pytest.fail('Main停止后不能另起恢复'))
    monkeypatch.setattr(service,'_quality_clarification',lambda *a,**kw:pytest.fail('不能把Main终止包装为质量澄清'))
    saved=service._persist_or_pause_result('s',s['user_query'],s,result)
    assert saved['status']=='blocked'
    assert service.repo.get('s')['clarification'] is None


def test_legacy_recovery_remains_available_without_new_main_protocol():
    from app.services.research_conversation_service import _has_hard_execution_stop
    assert not _has_hard_execution_stop({'status':'blocked','research_state':{'result_status':'blocked'},'errors':[]})
