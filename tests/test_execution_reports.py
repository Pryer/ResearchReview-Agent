import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine,select
from sqlalchemy.orm import Session,sessionmaker

from app.database.models import Base,ResearchCheckpointModel
from app.core.config import get_settings
from app.services.execution_report_capture import Capture,capture_scope,record_main_context,with_execution_report,preview
from app.services.execution_report_service import build_report,finalize_capture,freeze_source,get_report,list_reports,report_file,report_id


@pytest.fixture
def reports(monkeypatch,tmp_path):
    monkeypatch.setattr(get_settings(),'execution_report_enabled',True)
    monkeypatch.setattr(get_settings(),'execution_report_dir',str(tmp_path/'reports'))
    from sqlalchemy.pool import StaticPool
    engine=create_engine('sqlite://',connect_args={'check_same_thread':False},poolclass=StaticPool);Base.metadata.create_all(engine)
    with Session(engine) as db:yield db,tmp_path/'reports'
    engine.dispose()


def add_round(cap,model=True):
    from app.agent.context_builder import build_main_agent_context
    from app.schemas.agent_decision_schema import AgentDecision
    state={'session_id':cap.session_id,'user_query':cap.request,'topic':'RAG','main_agent_decisions':[],
        'agent_execution_budget':{'model_request_limit':1024,'llm_requests':1}}
    decision=AgentDecision(action='request_finish',arguments={},reason='结束已有草稿的优化')
    with capture_scope(cap):record_main_context(state,build_main_agent_context(state),decision,model_input=model)
    state['main_agent_decisions'].append(decision.model_dump(mode='json'));cap.latest_state=state
    return state


@pytest.mark.parametrize('status',['completed','partial','blocked','failed','cancelled','needs_clarification'])
def test_every_terminal_status_has_readable_report_without_model_calls(reports,status):
    db,directory=reports;cap=Capture('s','RAG研究请求');add_round(cap,model=False)
    result={'status':status,'body':'已有正文','references':[]}
    info=finalize_capture(db,cap,result)
    assert info['state']=='ready' and info['status']==status
    html=report_file(db,'s',info['report_id'],'report.html').read_text(encoding='utf-8')
    assert '按时间顺着读' in html and 'Main 看到了什么' in html
    data=json.loads(gzip.decompress((directory/info['report_id']/'report.json.gz').read_bytes()))
    assert data['summary']['main_rounds']==1 and data['summary']['policy_rounds']==1
    assert data['rounds'][0]['context'] and not data['rounds'][0]['model_input']
    assert result['body']=='已有正文'


def test_failure_and_export_failure_never_hide_original_exception(reports,monkeypatch):
    db,directory=reports
    @with_execution_report
    def fail():raise RuntimeError('private-provider sk-do-not-publish')
    with pytest.raises(RuntimeError,match='private-provider'):fail()
    files=list(directory.glob('*/report.html'));assert len(files)==1
    assert 'private-provider' not in files[0].read_text(encoding='utf-8')
    monkeypatch.setattr('app.services.execution_report_service.write_report',lambda *a:(_ for _ in ()).throw(OSError('private path')))
    with pytest.raises(RuntimeError,match='private-provider'):fail()


def test_export_failure_is_visible_and_does_not_change_body(reports,monkeypatch):
    db,_=reports;result={'status':'partial','body':'正文保留'}
    monkeypatch.setattr('app.services.execution_report_service.write_report',lambda *a:(_ for _ in ()).throw(OSError('private')))
    info=finalize_capture(db,Capture('s','q'),result)
    assert info['state']=='failed' and info['error_type']=='OSError'
    assert result=={'status':'partial','body':'正文保留'} and list_reports(db,'s')[0]==info


def test_session_isolation_filename_allowlist_and_idempotency(reports):
    db,directory=reports;cap=Capture('own','q');info=finalize_capture(db,cap,{'status':'blocked'})
    first=(directory/info['report_id']/'report.html').read_bytes()
    assert finalize_capture(db,cap,{'status':'completed'})==info
    assert (directory/info['report_id']/'report.html').read_bytes()==first
    assert get_report(db,'other',info['report_id']) is None
    assert report_file(db,'own',info['report_id'],'../report.html') is None
    assert report_file(db,'own','../../evil','report.html') is None
    assert report_file(db,'own',info['report_id'],'unregistered.txt') is None


def test_diagnostics_are_service_only(reports):
    db,_=reports;info=finalize_capture(db,Capture('s','q'),{'status':'blocked'})
    from app.services.research_artifact_service import ResearchArtifactService
    with pytest.raises(PermissionError):ResearchArtifactService(db).resolve('s','artifact://report-'+info['report_id'],role='main')


def test_long_preview_redacts_credentials_before_serializing():
    data=preview({'api_key':'top-secret','value':'x'*10000,'nested':{'reasoning_content':'hidden-thought'}},40)
    text=json.dumps(data)
    assert 'top-secret' not in text and 'hidden-thought' not in text and data['preview_only']


def test_script_breakout_is_escaped_in_html(reports):
    db,_=reports;cap=Capture('s','</script><script>ATTACK</script>')
    info=finalize_capture(db,cap,{'status':'blocked'})
    html=report_file(db,'s',info['report_id'],'report.html').read_text(encoding='utf-8')
    assert '</script><script>ATTACK' not in html and '\\u003c/script' in html


def test_standalone_entry_exports_and_nested_entry_does_not_duplicate(reports):
    _,directory=reports
    @with_execution_report
    def inner():return {'status':'blocked','body':'无正文'}
    @with_execution_report
    def outer():return inner()
    result=outer()
    assert result['execution_report']['state']=='ready'
    assert len(list(directory.glob('*/report.html')))==1


@pytest.mark.parametrize('raises',[False,True])
def test_conversation_completion_hook_exports_real_checkpoint(reports,raises):
    db,directory=reports
    from app.services.research_execution_service import with_artifact_store
    from app.schemas.agent_schema import AgentRequest
    class Service:
        def __init__(self):
            self.db=db;self.repo=SimpleNamespace(get=lambda *a:None);self.llm=object();self.should_cancel=None
        @with_artifact_store
        def handle(self,request):
            if raises:raise RuntimeError('private error')
            return {'status':'partial','body':'原正文','research_state':{'session_id':request.session_id,'user_query':request.user_query}}
    service=Service();request=AgentRequest(session_id='session',user_query='RAG')
    if raises:
        with pytest.raises(RuntimeError):service.handle(request)
    else:
        result=service.handle(request)
        assert result['body']=='原正文' and result['execution_report']['state']=='ready'
    assert list_reports(db,'session')[0]['state']=='ready'
    assert len(list(directory.glob('*/report.html')))==1


def test_two_runs_same_session_have_separate_files_and_decisions(reports):
    db,_=reports;caps=[Capture('same','first'),Capture('same','second')]
    for cap in caps:add_round(cap);finalize_capture(db,cap,{'status':'partial'})
    infos=list_reports(db,'same');assert len(infos)==2 and len({x['report_id'] for x in infos})==2
    for cap in caps:
        data=json.loads(gzip.decompress(report_file(db,'same',report_id(cap.report_id),'report.json.gz').read_bytes()))
        assert len(data['rounds'])==1 and data['request']==cap.request


def test_report_api_requires_key_and_rejects_other_session(reports,monkeypatch):
    db,_=reports;info=finalize_capture(db,Capture('owned','q'),{'status':'partial'})
    from app.main import app
    from app.database.db import get_db
    from fastapi.testclient import TestClient
    monkeypatch.setattr(get_settings(),'app_api_key','key')
    app.dependency_overrides[get_db]=lambda:db
    try:
        client=TestClient(app);url=info['html_url']
        assert client.get(url).status_code in {401,403}
        response=client.get(url,headers={'X-API-Key':'key'})
        assert response.status_code==200 and 'text/html' in response.headers['content-type']
        assert "connect-src 'none'" in response.headers['content-security-policy']
        assert client.get(url.replace('/owned/','/other/'),headers={'X-API-Key':'key'}).status_code==404
    finally:app.dependency_overrides.pop(get_db,None)


@pytest.mark.parametrize('cancel',[False,True])
def test_background_job_auto_exports_after_actual_terminal_status(reports,monkeypatch,cancel):
    _,directory=reports
    from app.services.research_job_service import ResearchJobService
    from app.services.research_conversation_service import ResearchConversationService
    from app.database.repositories import ResearchJobRepository
    engine=create_engine('sqlite:///:memory:');Base.metadata.create_all(engine);maker=sessionmaker(bind=engine)
    with maker() as db:
        repo=ResearchJobRepository(db);repo.create('job','s',{'user_query':'RAG'})
        if cancel:repo.update('job',status='cancel_requested')
        db.commit()
    monkeypatch.setattr('app.services.research_job_service.SessionLocal',maker)
    monkeypatch.setattr(ResearchConversationService,'handle',lambda *a:{'status':'partial','body':'保留正文'})
    ResearchJobService._run_job('job')
    with maker() as db:
        job=ResearchJobService(db).get('job')
        assert job['status']==('cancelled' if cancel else 'partial')
        assert job['execution_report']['state']=='ready' and job['execution_report']['status']==job['status']
    engine.dispose()


def test_real_main_subagent_loop_captures_context_and_role_continuity(reports):
    db,_=reports
    from tests.test_subagent_react import ScriptedLLM,state,search,fetch
    from app.agent.main_loop import MainAgentLoop
    from app.agent.execution_budget import budget_scope
    from app.services.durable_execution_service import durable_scope
    from app.services.research_artifact_service import ResearchArtifactService
    from app.services.research_execution_service import _STORE
    s=state();s['delivery_mode']='draft_first';cap=Capture('s',s['user_query'])
    llm=ScriptedLLM({'main_agent_decision':['search_and_rank','fetch_metadata','request_finish']})
    token=_STORE.set(ResearchArtifactService(db))
    try:
        with capture_scope(cap),budget_scope(s):
            with durable_scope(db,'s',s,s['agent_execution_budget']) as runtime:
                cap.start_version=runtime.version
                status=MainAgentLoop(llm).run(s,handlers={'search_and_rank':search,'fetch_metadata':fetch},
                    finish_validator=lambda current:bool(current.get('paper_details')))
                runtime.snapshot(s,ledger=s['agent_execution_budget']);cap.end_version=runtime.version
                cap.source=freeze_source(db,cap)
        info=finalize_capture(db,cap,{'status':status})
    finally:_STORE.reset(token)
    assert info['state']=='ready'
    d=json.loads(gzip.decompress(report_file(db,'s',info['report_id'],'report.json.gz').read_bytes()))
    assert len(d['rounds'])==3 and len(d['calls'])==2
    assert all(set(r['context'])=={'goal','state','key_evidence','decisions','open_questions'} for r in d['rounds'])
    assert d['calls'][1]['work_context']['prior_result']['task_id']==d['calls'][0]['delegation_id']
    assert d['calls'][0]['returned_to_main']['status']=='completed'
    assert all(op=='main_agent_decision' for op,_ in llm.calls)


def test_policy_handoff_has_its_own_context_without_extra_llm_call(reports):
    db,_=reports
    from tests.test_draft_delivery_policy import draft_state
    from tests.test_subagent_react import ScriptedLLM
    from app.agent.main_loop import MainAgentLoop
    from app.agent.delivery import published_delivery
    s=draft_state();s['pending_result_validation']=True;cap=Capture('s','已有材料先交草稿')
    llm=ScriptedLLM({'main_agent_decision':['request_finish']})
    with capture_scope(cap):
        assert MainAgentLoop(llm).run(s,handlers={},finish_validator=lambda current:bool(published_delivery(current)))=='completed'
    assert len(cap.rounds)==2 and len(llm.calls)==1
    assert cap.rounds[0]['context'] and not cap.rounds[0]['model_input']
    assert cap.rounds[1]['model_input']


def test_freeze_source_excludes_prior_decisions_and_later_execution(reports):
    db,_=reports
    from app.schemas.agent_decision_schema import AgentDecision
    a=AgentDecision(action='request_finish',reason='first').model_dump(mode='json')
    b=AgentDecision(action='request_finish',reason='second').model_dump(mode='json')
    for version,decisions in [(1,[a]),(2,[a,b])]:
        db.add(ResearchCheckpointModel(session_id='s',version=version,state_json=json.dumps({'main_agent_decisions':decisions})))
    db.commit();cap=Capture('s','second');cap.start_version=1;cap.end_version=2
    source=freeze_source(db,cap)
    db.add(ResearchCheckpointModel(session_id='s',version=3,state_json=json.dumps({'main_agent_decisions':[]})));db.commit()
    assert [x['reason'] for x in source['final']['main_agent_decisions']]==['second']
    assert max(x['version'] for x in source['checkpoints'])==2


def test_terminal_job_exposes_retryable_link_before_export_finishes(reports):
    db,_=reports
    from app.database.repositories import ResearchJobRepository
    from app.services.research_job_service import ResearchJobService
    repo=ResearchJobRepository(db);repo.create('pending-job','s',{'user_query':'q'})
    repo.update('pending-job',status='partial',result={'status':'partial','body':'原文'});db.commit()
    pending=ResearchJobService(db).get('pending-job')
    assert pending['execution_report']['state']=='pending'
    cap=Capture('s','q','pending-job');info=finalize_capture(db,cap,{'status':'partial'})
    ready=ResearchJobService(db).get('pending-job')
    assert ready['execution_report']['state']=='ready'
    assert ready['execution_report']['download_url']==pending['execution_report']['download_url']
    assert ready['result']['body']=='原文'


def test_live_pydantic_inputs_and_nonfinite_values_remain_json_readable():
    from pydantic import BaseModel
    from app.services.execution_report_collector import safe
    class Input(BaseModel):
        title:str='原始输入'
        api_key:str='never-export'
    value=preview({2023:Input(),'unknown':float('nan')})
    assert value['2023']=={'title':'原始输入'} and value['unknown'] is None
    json.dumps(safe(value),allow_nan=False)


def test_report_observer_error_is_diagnostic_only(reports):
    db,_=reports;cap=Capture('s','q')
    class InvalidContext:
        def model_dump(self,**kw):raise ValueError('private context detail')
    from app.schemas.agent_decision_schema import AgentDecision
    with capture_scope(cap):record_main_context({},InvalidContext(),AgentDecision(action='request_finish'),model_input=True)
    result={'status':'partial','body':'正文不受影响'}
    info=finalize_capture(db,cap,result)
    assert info['state']=='failed' and info['error_type']=='ValueError'
    assert result['body']=='正文不受影响' and result['status']=='partial'


def test_e2e_export_is_directly_readable_without_api(reports,tmp_path):
    db,_=reports
    from app.services.execution_report_service import copy_report_files
    info=finalize_capture(db,Capture('s','q'),{'status':'partial'})
    exported=copy_report_files(db,'s',tmp_path/'run')
    assert exported['state']=='ready'
    assert (tmp_path/'run/execution_report.html').read_bytes()==report_file(db,'s',info['report_id'],'report.html').read_bytes()
    assert (tmp_path/'run/execution_report.zip').is_file()


def test_queue_submission_failure_also_emits_report(reports,monkeypatch):
    db,_=reports
    from app.services.research_job_service import ResearchJobService
    from app.schemas.agent_schema import AgentRequest
    def fail(*args,**kwargs):raise RuntimeError('queue closed')
    monkeypatch.setattr(ResearchJobService,'_enqueue',fail)
    with pytest.raises(RuntimeError,match='queue closed'):
        ResearchJobService(db).submit(AgentRequest(session_id='queue-session',user_query='RAG'))
    infos=list_reports(db,'queue-session')
    assert len(infos)==1 and infos[0]['state']=='ready' and infos[0]['status']=='failed'


@pytest.mark.parametrize('absolute',[False,True])
def test_report_output_does_not_drift_with_working_directory(reports,tmp_path,monkeypatch,absolute):
    _,_=reports
    from app.services import execution_report_service as service
    project=tmp_path/'project';project.mkdir()
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir()
    monkeypatch.setattr(service,'PROJECT_ROOT',project)
    configured=tmp_path/'custom' if absolute else Path('data/execution_reports')
    monkeypatch.setattr(get_settings(),'execution_report_dir',str(configured))
    monkeypatch.chdir(elsewhere)
    cap=Capture('s','q');info=finalize_capture(None,cap,{'status':'blocked'})
    assert info['state']=='ready'
    expected=(configured if absolute else project/configured)/info['report_id']/'report.html'
    assert Path(info['html_path'])==expected.resolve() and expected.is_file()
    assert not (elsewhere/'data/execution_reports').exists()
