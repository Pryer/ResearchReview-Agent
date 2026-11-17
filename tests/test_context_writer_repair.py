"""冻结故障的行为回归：保留真实候选、授权投影与跨委派隔离。"""
import copy
import hashlib
import json
import pytest

from app.schemas.deliverable_schema import WritingPlan
from app.services.writer_support import authorized_writer_claims, read_candidate, record_candidate, substantive
from app.agent.delegation_memory import prepare_work_context, complete_work_context, begin_delegation


def state():
    text = 'We propose retrieval augmented generation for open domain question answering.'
    span = {'evidence_id': 'p1:e001', 'text': text, 'source_type': 'abstract',
            'source_url': 'https://example.org/paper', 'char_start': 0, 'char_end': len(text),
            'source_text_hash': hashlib.sha256(text.encode()).hexdigest()}
    return {'user_query': '写RAG研究背景约600字', 'session_id': 's', 'target_length': 600,
        'canonical_topic': 'RAG', 'core_deliverables': ['research_background'],
        'paper_cards': [{'paper_id': 'p1', 'title': 'RAG', 'quality_status': 'partial',
            'evidence_source': 'abstract', 'evidence_spans': [span], 'field_claims': {}}],
        'claim_plans': [{'route_id': 'reference_coverage', 'claims': [{'claim_id': 'c1',
            'claim_text': text, 'evidence_ids': ['p1:e001'], 'support_level': 'single',
            'evidence_access_limit': 'abstract_only'}]}], 'subagent_sessions': {'writing': {}}}


def plan():
    return WritingPlan(deliverable_type='research_background', purpose='说明研究背景', organizing_strategy='evidence_driven',
        sections=[{'id': 'background_body', 'title': '研究背景', 'purpose': '背景',
                   'supporting_paper_ids': ['p1'], 'supporting_claim_ids': []}],
        citation_policy={'minimum_unique_references': 1})


def test_coverage_authorization_survives_absence_of_field_claims():
    from app.tools.write_deliverable import _writer_inputs
    cards, _ = _writer_inputs(plan(), state())
    assert len(cards[0]['claims']) == 1
    assert cards[0]['claims'][0]['evidence_access_limit'] == 'abstract_only'
    assert cards[0]['claims'][0]['source_records'][0]['char_start'] == 0


@pytest.mark.parametrize('mutation', ['bad_hash', 'missing_source', 'unsupported'])
def test_coverage_cannot_restore_unverified_evidence(mutation):
    s = state()
    if mutation == 'bad_hash': s['paper_cards'][0]['evidence_spans'][0]['source_text_hash'] = 'wrong'
    elif mutation == 'missing_source': s['paper_cards'][0]['evidence_spans'] = []
    else: s['claim_plans'][0]['claims'][0]['support_level'] = 'unsupported'
    assert authorized_writer_claims(s, 'p1') == []


def test_raw_candidates_remain_when_every_local_validation_fails(monkeypatch):
    from app.deliverables.renderers import get_renderer
    import app.tools.validate_deliverable as v
    monkeypatch.setattr(v, 'validate_deliverable', lambda *a: {'valid': False, 'errors': ['需修复一个结构问题']})
    class LLM:
        def complete(self, *a, **kw): return '## 研究背景\n\n' + '检索增强生成通过外部知识为开放域问答提供可更新的信息来源[p1]。' * 8
    s=state(); r=get_renderer('research_background')
    text=r.render(plan(), s, [{'paper_id': 'p1', 'claims': []}], [], llm=LLM())
    assert len(text) > 100 and '[p1]' in text
    assert len(s['writer_candidates']) >= 6
    assert read_candidate(s, s['selected_candidate_id'])['validation']['valid'] is False
    assert any(c['stage'].endswith(':received') and c['raw_text'] for c in s['writer_candidates'])


def test_unmapped_numeric_citations_are_preserved_and_reported():
    from app.services.writer_support import normalize_writer_citations
    text,issues=normalize_writer_citations('真实候选内容[1]。', [{'paper_id': 'p1'}])
    assert '[1]' in text and issues[0]['code']=='citation_protocol_invalid'


def test_executor_report_suffix_cannot_become_false_numeric_citations():
    from app.services.writer_support import normalize_writer_citations,strip_writer_report
    body='## 研究背景\n\n'+'检索增强生成通过检索外部文档为问答提供可追溯的知识材料[p1]。'*6
    raw=body+'\n\n**生成诊断**\n\n- 字数约600字，落在target_char_range [480,720]内。\n\n**结果状态：partial**\n\n- 建议动作：由Main决定是否补检索。'
    text,issues=normalize_writer_citations(raw,[{'paper_id':'p1'}])
    assert text==body and not issues
    assert strip_writer_report(raw)==body
    assert '[480,720]' in raw


def test_report_cleaning_keeps_unmapped_real_citations_and_academic_sections():
    from app.services.writer_support import normalize_writer_citations,strip_writer_report
    body='## 研究背景\n\n'+'实验结果需要结合原文给定的设置和比较条件进行讨论[1]。'*6
    text,issues=normalize_writer_citations(body,[{'paper_id':'p1'}])
    assert '[1]' in text and issues
    with_section=body+'\n\n**生成诊断**\n\n目标target_char_range [480,720]。\n\n## 研究现状\n\n后续的学术章节应当保留。'
    assert strip_writer_report(with_section)==with_section


def test_substantive_rejects_observed_64_character_stub():
    assert not substantive('## 一、研究背景\n\n各文献的具体问题设定与方法细节以其原文报告为准，本节不作外推。\n\n现有证据的适用范围以各文献原文报告为准。')


def test_explicit_600_char_target_overrides_default():
    from app.agent.writing_plan import build_writing_plan
    from app.agent.slot_extractor import extract_slots
    slots=extract_slots('写RAG研究背景约600字，至少引用3篇', 'generate_review')
    assert slots.target_length==600
    p=build_writing_plan('research_background',state())
    assert p.target_char_range==(480,720)
    assert sum(x.target_word_count for x in p.sections)==600


def test_total_length_is_allocated_across_deliverables():
    from app.agent.writing_plan import build_writing_plan
    s=state(); s['target_length']=1000; s['core_deliverables']=['research_background','research_status']
    plans=[build_writing_plan(k,s) for k in s['core_deliverables']]
    assert sum(round(sum(p.target_char_range)/2) for p in plans) in range(999,1002)


def test_cross_operation_summary_tracks_dependencies_and_is_task_scoped():
    s=state();complete_work_context(s,'writing',{'task_id':'old','operation':'generate_deliverables','outcome':'blocked','issues':[{'code':'missing'}]})
    s['review']='新的稿件'
    c=prepare_work_context(s,'writing')
    assert c['prior_result']['operation']=='generate_deliverables'
    assert 'review' in c['input_delta']['new_fields']
    s.pop('paper_cards')
    assert 'paper_cards' in prepare_work_context(s,'writing')['input_delta']['removed_fields']
    s['user_query']='全新研究'
    assert prepare_work_context(s,'writing')['prior_result']=={}


def test_new_delegation_does_not_replay_prior_messages():
    s=state(); s['subagent_sessions']['writing']['main_agent_trajectory']={'version':1,'messages':[{'role':'assistant','content':'old'}], 'memory':[], 'goals':[], 'archive_refs':[], 'epoch':0}
    begin_delegation(s,'writing')
    assert s['subagent_sessions']['writing']['main_agent_trajectory']['messages']==[]
    assert s['subagent_sessions']['writing']['delegation_history'][0]['messages'][0]['content']=='old'


def test_check_summary_does_not_accept_writer_assurances():
    s=state();s['subagent_sessions']['check']={};s['check_report']={'report_id':'x','verdict':'passed'}
    c=prepare_work_context(s,'check')
    assert c['previous_check']['current'] is False
    assert '自述' in c['independence']


def test_blocked_writing_has_structured_issues():
    from app.agent.subagents.writing_agent import WritingAgent
    from app.schemas.agent_task_schema import AgentTask, AgentRole, AgentTaskStatus
    s=state();s['review']='';s['writer_attempt_issues']=[{'code':'citation_plan_input_mismatch','reason':'缺少授权输入'}]
    task=AgentTask(role=AgentRole.WRITING,operation='generate_deliverables',objective='生成',source_state_version='1',source_state_fingerprint='f')
    result=WritingAgent()._result(task,s,AgentTaskStatus.COMPLETED,[])
    assert result.outcome.value=='blocked'
    assert any(i.code=='citation_plan_input_mismatch' for i in result.issues)


def test_candidate_permission_and_state_roundtrip():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.database.models import Base
    from app.services.research_artifact_service import ResearchArtifactService
    from app.services.research_execution_service import _STORE
    engine=create_engine('sqlite:///:memory:');Base.metadata.create_all(engine)
    with Session(engine) as db:
        store=ResearchArtifactService(db);token=_STORE.set(store)
        try:
            s=state();cid=record_candidate(s,plan(),'initial','原始正文','规范正文',{'valid':False,'errors':['失败']})
            saved=store.externalize_state('s',s);restored=store.hydrate_state('s',saved)
            assert read_candidate(restored,cid)['raw_text']=='原始正文'
            with pytest.raises(PermissionError):read_candidate(restored,cid,role='search')
        finally:_STORE.reset(token)


def test_writer_control_stop_is_never_fallback(monkeypatch):
    from app.deliverables.renderers import get_renderer
    from app.agent.execution_budget import AgentBudgetExceeded
    class LLM:
        def complete(self,*a,**kw): raise AgentBudgetExceeded('stop')
    with pytest.raises(AgentBudgetExceeded):get_renderer('research_background').render(plan(),state(),[],[],llm=LLM())


def test_old_task_candidate_is_not_readable_after_new_research():
    s=state();cid=record_candidate(s,plan(),'initial','原稿','规范稿',{'valid':False})
    s['user_query']='全新的研究主题'
    with pytest.raises(PermissionError,match='another research task'):read_candidate(s,cid,role='writing')


def test_candidate_raw_is_saved_before_validation_exception(monkeypatch):
    from app.deliverables.renderers import get_renderer
    import app.tools.validate_deliverable as v
    monkeypatch.setattr(v,'validate_deliverable',lambda *a: (_ for _ in ()).throw(ValueError('validator failed')))
    class LLM:
        def complete(self,*a,**kw):return '原始模型输出需要在验证报错前保存'
    s=state()
    with pytest.raises(ValueError):get_renderer('research_background').render(plan(),s,[],[],llm=LLM())
    assert s['writer_candidates'][0]['raw_text']=='原始模型输出需要在验证报错前保存'


def test_failed_writer_keeps_audit_without_committing_partial_research_state():
    from app.agent.controller import AgentController
    from app.schemas.agent_task_schema import AgentRole
    s=state();s['writing_plans']=[plan().model_dump(mode='json')]
    def fail(working):
        record_candidate(working,plan(),'initial','审计稿','审计稿',{'valid':False})
        working['review']='错误执行写入的未完成正文'
        raise ValueError('validation failed')
    with pytest.raises(ValueError):AgentController().execute(role=AgentRole.WRITING,operation='generate_deliverables',
        objective='失败审计测试',state=s,handler=fail)
    assert s.get('review') is None
    assert read_candidate(s,s['writer_candidates'][0]['candidate_id'])['raw_text']=='审计稿'


def test_controller_injects_constraints_even_when_model_omits_them():
    from app.agent.controller import AgentController
    from app.schemas.agent_task_schema import AgentRole
    from app.agent.task_context import current_task_brief
    s=state();s.update(start_year=2020,end_year=2023,required_reference_count=3)
    received=[]
    AgentController().execute(role=AgentRole.WRITING,operation='generate_deliverables',objective='写作',
        state=s,handler=lambda w:received.append(current_task_brief('writing')))
    assert received[0]['constraints']['target_length']==600
    assert received[0]['constraints']['minimum_references']==3


def test_backfill_cannot_call_model_for_paper_without_current_authorization():
    from app.agent.nodes.synthesis import _backfill_global_citation_union
    s=state();s['required_reference_count']=2
    c=copy.deepcopy(s['paper_cards'][0]);c['paper_id']='p2';s['paper_cards'].append(c)
    class Forbidden:
        def complete(self,*a,**kw):pytest.fail('must not supplement an unauthorized paper')
    original='## 研究背景\n\n已有研究讨论了开放域问答[p1]。'
    outputs,_=_backfill_global_citation_union(s,Forbidden(),[original],[{'valid':False,'errors':[]}],
        [plan().model_dump(mode='json')],[{'sections':[{'paper_ids':['p1','p2']}]}])
    assert outputs==[original]
    assert s['writer_attempt_issues'][0]['code']=='citation_plan_input_mismatch'


def test_rejected_backfill_retains_original_response_and_reason():
    from app.agent.nodes.synthesis import _backfill_global_citation_union
    s=state();s.pop('claim_plans');s['required_reference_count']=2
    c=copy.deepcopy(s['paper_cards'][0]);c['paper_id']='p2';s['paper_cards'].append(c)
    class Invalid:
        def complete(self,*a,**kw):return '未补齐引用的原始返回[p1]。'
    original='## 研究背景\n\n已有研究讨论了开放域问答[p1]。'
    outputs,_=_backfill_global_citation_union(s,Invalid(),[original],[{'valid':False,'errors':[]}],
        [plan().model_dump(mode='json')],[{'sections':[{'paper_ids':['p1','p2']}]}])
    assert outputs==[original]
    assert any(c.get('raw_text')=='未补齐引用的原始返回[p1]。' for c in s['writer_candidates'])
    assert any('citation_coverage_not_monotonic' in c['validation'].get('errors',[]) for c in s['writer_candidates'])


def test_historical_completed_operation_does_not_mark_new_delegation_completed():
    from tests.reference_v1.delegation_context import subagent_context
    s=state();s['subagent_sessions']['analysis']={'messages':[]}
    s['agent_task_results']=[{'task_id':'old','role':'analysis','operation':'extract_paper_cards','status':'completed','outcome':'succeeded','findings':['旧卡片完成']}]
    active={'id':'new','operation':'extract_paper_cards','objective':'补检索后重新抽取','arguments':{},
            'evidence_refs':[],'completed_operations':[],'rounds':0,'work_context':{'prior_result':s['agent_task_results'][0]}}
    c=subagent_context(s,'analysis',active,['extract_paper_cards','report_to_main'])
    assert c.state.completed_actions==[] and c.state.last_task_result=={}
    assert c.state.delegation['execution_status']=='not_started'
    assert c.state.delegation['work_context']['prior_result']['status']=='completed'


def test_early_help_report_cannot_claim_an_execution_observation():
    from tests.reference_v1.subagent_loop import SubagentLoop
    from app.agent.controller import AgentController
    from app.schemas.agent_decision_schema import AgentDecision
    from tests.test_subagent_react import ScriptedLLM, state as agent_state, report
    s=agent_state()
    llm=ScriptedLLM({'search_agent_decision':[report(status='needs_help',message='历史检索已经完成')]})
    result=SubagentLoop(llm,AgentController()).run(s,decision=AgentDecision(action='search_and_rank'),
        handlers={'search_and_rank':lambda w:pytest.fail('preflight report did not execute')})
    assert any(i.code=='delegated_action_not_executed' for i in result.issues)
    summary=s['subagent_sessions']['search']['work_summary']['last_result']
    assert summary['task_executed'] is False
    assert summary['findings']==['本次没有已提交工具观察']


def test_pre_writer_status_uses_actual_authorization_and_marks_uncomputed_counts():
    from app.agent.context_builder import build_main_agent_context
    s=state();before=copy.deepcopy(s)
    c=build_main_agent_context(s)
    assert c.state.artifact_summary['claim_authorized_paper_count']==1
    assert c.state.artifact_summary['writer_authorized_paper_count']=='not_projected'
    assert c.state.artifact_summary['valid_citations']=='not_checked'
    assert s==before


def test_missing_authorization_input_is_unknown_not_zero():
    from app.agent.context_builder import build_main_agent_context
    s=state();s.pop('claim_plans')
    assert build_main_agent_context(s).state.artifact_summary['claim_authorized_paper_count']=='not_planned'


def test_frozen_real_three_paper_authorization_is_ready_before_writer_projection():
    from pathlib import Path
    from app.agent.context_builder import build_main_agent_context
    s=json.loads((Path(__file__).parent/'fixtures/writer_t06_20261110.json').read_text(encoding='utf8'))
    s.update(core_deliverables=['research_background'],required_reference_count=3,max_papers_explicit=True,target_length=600)
    c=build_main_agent_context(s)
    assert c.state.artifact_summary['claim_authorized_paper_count']==3
    assert c.state.artifact_summary['prewriting_ready'] is True
    assert c.state.artifact_summary['writer_authorized_paper_count']=='not_projected'


def test_known_unambiguous_citation_prefix_and_metadata_alias_expand():
    from app.services.writer_support import normalize_writer_citations
    pid='s2:659bf9ce7175e1ec266ff54359e2bd76e0b7ff31'
    cards=[{'paper_id':pid,'doi':'10.48550/arXiv.2005.11401'}]
    text,issues=normalize_writer_citations('事实[659bf9ce]与方法[s2:659bf9ce…]及原文[arxiv:2005.11401]',cards)
    assert text.count('['+pid+']')==3 and not issues


def test_ambiguous_prefix_is_never_assigned_to_one_paper():
    from app.services.writer_support import normalize_writer_citations
    cards=[{'paper_id':'s2:12345678'+'a'*32},{'paper_id':'s2:12345678'+'b'*32}]
    text,_=normalize_writer_citations('事实[12345678]',cards)
    assert text=='事实[12345678]'


def test_numbered_alias_requires_explicit_prompt_mapping():
    from app.services.writer_support import normalize_writer_citations
    cards=[{'paper_id':'p1'}]
    assert normalize_writer_citations('事实[1]',cards)[1]
    assert normalize_writer_citations('事实[1]',cards,trusted_numbers={'1':'p1'})==('事实[p1]',[])


def test_trusted_short_labels_never_override_canonical_paper_ids():
    from app.services.writer_support import normalize_writer_citations
    cards=[{'paper_id':'p2'},{'paper_id':'p1'}]
    text,issues=normalize_writer_citations('事实[p1]与另一来源[1]',cards,trusted_numbers={'1':'p2','2':'p1'})
    assert text=='事实[p1]与另一来源[p2]' and not issues


def test_background_outline_cannot_use_fields_outside_current_claim_authorization():
    from app.agent.writing_plan import build_writing_plan
    s=state();quote='UNAUTHORIZED_RESULT_WITH_100_PERCENT_ACCURACY'
    span={'evidence_id':'p1:e002','text':quote,'source_type':'abstract','char_start':0,'char_end':len(quote)}
    s['paper_cards'][0]['evidence_spans'].append(span)
    s['paper_cards'][0]['field_claims']={'results':[{'evidence_id':'p1:e002','claim':quote,'source_text':quote,'explicitly_reported':True}]}
    class Outline:
        def complete(self,prompt,**kw):
            assert quote not in prompt
            assert s['claim_plans'][0]['claims'][0]['claim_text'] in prompt
            return json.dumps({'paragraph_goals':[{'writing_goal':str(i)} for i in range(3)]})
    build_writing_plan('research_background',s,llm=Outline())


def test_section_writer_translates_authorized_sources_when_fallback_has_no_facts(monkeypatch):
    from app.tools.write_deliverable import _writer_inputs
    from app.deliverables.renderers import get_renderer
    import app.deliverables.renderers.base_renderer as r
    s=state();p=plan();cards,_=_writer_inputs(p,s)
    draft=get_renderer('research_background').render_fallback(p,s,cards)
    calls=[]
    class Writer:
        def complete(self,prompt,**kw):
            calls.append(prompt)
            assert s['claim_plans'][0]['claims'][0]['claim_text'] in prompt
            return '## 研究背景\n\n检索增强生成在开放域问答中结合检索与生成过程，为答案组织提供外部上下文信息[p1]。'
    monkeypatch.setattr(r,'_validate_rewritten_section',lambda *a,**kw:[])
    text=r._write_sections_in_chinese(draft,p,s,cards,Writer())
    assert calls and '[p1]' in text
    assert not s['writer_section_diagnostics'][0]['sections'][0]['status']=='already_valid'
