"""任务结束后导出独立运行报告；失败不覆盖研究结果或取消异常。"""
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo
import copy
import gzip
import hashlib
import json
import logging
import re
import time
import zipfile

from sqlalchemy import select
from app.core.config import get_settings
from app.database.models import ResearchArtifactModel, ResearchCheckpointModel, ResearchTaskModel, ResearchRuntimeModel
from app.services.execution_report_collector import collect_report, safe
from app.services.execution_report_capture import preview

logger=logging.getLogger(__name__)
PROJECT_ROOT=Path(__file__).resolve().parents[2]


def report_directory():
    # WHY: 相对报告路径属于项目配置，不能随启动终端的工作目录漂移。
    configured=Path(get_settings().execution_report_dir).expanduser()
    return (configured if configured.is_absolute() else PROJECT_ROOT/configured).resolve()



def report_id(value):
    value=str(value)
    return value if re.fullmatch(r'[a-f0-9]{32}',value) else hashlib.sha256(value.encode()).hexdigest()[:32]


def freeze_source(db,cap):
    """只读已知版本范围；不把同会话后续执行混入本次报告。"""
    if db is None or cap.start_version is None:return None
    sid=cap.session_id;end=cap.end_version if cap.end_version is not None else cap.start_version
    baseline_row=db.scalar(select(ResearchCheckpointModel).where(
        ResearchCheckpointModel.session_id==sid,ResearchCheckpointModel.version<=cap.start_version
    ).order_by(ResearchCheckpointModel.version.desc()).limit(1))
    baseline=json.loads(baseline_row.state_json) if baseline_row else {}
    rows=list(db.scalars(select(ResearchCheckpointModel).where(
        ResearchCheckpointModel.session_id==sid,ResearchCheckpointModel.version>cap.start_version,
        ResearchCheckpointModel.version<=end).order_by(ResearchCheckpointModel.version)))
    cps=[{'version':x.version,'state':json.loads(x.state_json)} for x in rows]
    if baseline_row:cps.insert(0,{'version':baseline_row.version,'state':copy.deepcopy(baseline)})
    final=copy.deepcopy(cps[-1]['state']) if cps else {}
    # WHY: 查询只跟随本次快照的不可变引用；禁止扫描其它会话或后续运行。
    refs=set(re.findall(r'artifact://(art-[a-f0-9]+)',json.dumps(cps,ensure_ascii=False)))
    artifacts={}
    while refs:
        batch=list(refs)[:400];refs.difference_update(batch)
        fetched=list(db.scalars(select(ResearchArtifactModel).where(
            ResearchArtifactModel.session_id==sid,ResearchArtifactModel.artifact_id.in_(batch))))
        for row in fetched:
            artifacts[row.artifact_id]={'artifact_type':row.artifact_type,'payload':json.loads(row.payload_json)}
            refs.update(x for x in re.findall(r'artifact://(art-[a-f0-9]+)',row.payload_json) if x not in artifacts)
    tasks=[{'task_id':x.task_id,'expected_version':x.expected_version,'operation':x.operation,
        'status':x.status,'result':json.loads(x.result_json or '{}')} for x in db.scalars(select(ResearchTaskModel).where(
            ResearchTaskModel.session_id==sid,ResearchTaskModel.expected_version>=cap.start_version,
            ResearchTaskModel.expected_version<end).order_by(ResearchTaskModel.expected_version))]
    runtime=db.scalar(select(ResearchRuntimeModel).where(ResearchRuntimeModel.session_id==sid,ResearchRuntimeModel.owner==getattr(cap,'owner',None)))
    if runtime is not None:final['agent_execution_budget']=json.loads(runtime.budget_json)
    old={x.get('decision_id') for x in baseline.get('main_agent_decisions',[])}
    for state in [final,*[x['state'] for x in cps]]:
        state['main_agent_decisions']=[x for x in state.get('main_agent_decisions',[]) if x.get('decision_id') not in old]
    return {'checkpoints':cps,'final':final,'baseline':baseline,'tasks':tasks,'artifacts':artifacts,'events':[]}


def _descriptor(db,sid,rid):
    if db is None:return None
    row=db.get(ResearchArtifactModel,'report-'+rid)
    if row is None or row.session_id!=sid or row.artifact_type!='execution_report':return None
    return json.loads(row.payload_json)


def report_urls(sid,rid):
    base=f'/api/reviews/sessions/{quote(sid,safe="")}/reports/{rid}/files/'
    return {'html_url':base+'report.html','download_url':base+'bundle.zip'}


def get_report(db,sid,rid):
    if not re.fullmatch(r'[a-f0-9]{32}',rid):return None
    return _descriptor(db,sid,rid)


def list_reports(db,sid):
    rows=db.scalars(select(ResearchArtifactModel).where(ResearchArtifactModel.session_id==sid,
        ResearchArtifactModel.artifact_type=='execution_report').order_by(ResearchArtifactModel.created_at.desc()))
    return [json.loads(x.payload_json) for x in rows]


def report_file(db,sid,rid,name):
    info=get_report(db,sid,rid)
    if not info or name not in info.get('files',[]):return None
    base=(report_directory()/rid).resolve()
    path=(base/name).resolve()
    if path.parent!=base or not path.is_file():return None
    return path


def copy_report_files(db,sid,directory):
    """E2E保留可直接打开的副本；不要求本地同时运行API。"""
    import shutil
    infos=list_reports(db,sid)
    if not infos:return None
    info=infos[0]
    if info.get('state')!='ready':return {'state':info['state'],'error_type':info.get('error_type')}
    output=Path(directory);output.mkdir(parents=True,exist_ok=True)
    for source_name,name in [('report.html','execution_report.html'),('bundle.zip','execution_report.zip')]:
        source=report_file(db,sid,info['report_id'],source_name)
        if source is None:return {'state':'failed','error_type':'ReportFilesMissing'}
        shutil.copyfile(source,output/name)
    return {'state':'ready','html_file':'execution_report.html','bundle_file':'execution_report.zip','report_id':info['report_id']}


def _save_descriptor(db,cap,info):
    if db is None:return
    from app.services.research_artifact_service import ResearchArtifactService
    ResearchArtifactService(db).repo.save(artifact_id='report-'+info['report_id'],session_id=cap.session_id,
        artifact_type='execution_report',version=1,fingerprint=hashlib.sha256(json.dumps(info,sort_keys=True).encode()).hexdigest(),
        payload=info,provenance={'kind':'execution_diagnostic','start_version':cap.start_version,'end_version':cap.end_version})
    db.commit()


def build_report(cap,result=None,status=None):
    capture=cap.data();source=copy.deepcopy(cap.source) if cap.source else {
        'checkpoints':[],'final':copy.deepcopy(cap.latest_state),'baseline':{},'tasks':[],'artifacts':{},'events':[]}
    final=source['final']
    if not final.get('main_agent_decisions'):final['main_agent_decisions']=capture['decisions']
    # 来源不完整时保留已观察状态，不能让一个没有进入Main的任务借用旧正文。
    ledger=final.get('agent_execution_budget') or {};old=source['baseline'].get('agent_execution_budget') or {}
    cumulative=int(ledger.get('llm_requests') or 0)+int(ledger.get('retrieval_model_requests') or 0)
    start=int(old.get('llm_requests') or 0)+int(old.get('retrieval_model_requests') or 0)
    terminal=status or ('cancelled' if cap.error_type and 'Cancel' in cap.error_type else 'failed' if cap.error_type else (result or {}).get('status') or 'completed')
    meta={'report_id':report_id(cap.report_id),'session_id':cap.session_id,'status':terminal,
        'seconds':round(time.monotonic()-cap.started,3),'created_at':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds'),
        'request':cap.request,'error_type':cap.error_type,'start_version':cap.start_version,'end_version':cap.end_version,
        'references':len((result or {}).get('references') or final.get('references') or []),
        'publication_version':final.get('published_draft_version') or 0,'cumulative_requests':cumulative,
        'request_limit':ledger.get('model_request_limit',get_settings().agent_model_request_limit),
        'source_fingerprint':hashlib.sha256(json.dumps(source['checkpoints'],sort_keys=True,default=str).encode()).hexdigest()}
    source['meta']=meta
    source['run']={'task':{'request':{'user_query':cap.request}},'latency_seconds':meta['seconds'],
        'metrics':{'tokens':{'by_operation':{'main_agent_decision':{'calls':sum(x['model_input'] for x in capture['rounds']) or sum(bool(x.get('usage')) for x in final.get('main_agent_decisions',[]))}}}},
        'delivery_observed':{'delivery_metrics':final.get('delivery_metrics') or {}},
        'budget_ledger':{'llm_requests':max(0,cumulative-start),'retrieval_model_requests':0}}
    report=collect_report(source,capture)
    report['clarification']=(result or {}).get('clarification')
    task_versions={x['task_id']:x['expected_version'] for x in source['tasks']}
    for call in report['calls']:
        if call.get('task_id') in task_versions:call['commit_checkpoint']=task_versions[call['task_id']]+1
        call['input_state']={k:preview(v) for k,v in call.get('input_state',{}).items()}
        for row in call.get('same_role_input_changes',[]):
            row['before']=preview(row['before'],1000);row['after']=preview(row['after'],1000)
        call['steps']=[{k:preview(v) for k,v in step.items()} for step in call.get('steps',[])]
        if not call.get('returned_to_main'):
            call['returned_to_main']={'status':'failed' if cap.error_type else 'not_completed','findings':['本次委派尚未返回完整结果'],'issues':[]}
    return safe(report)


def write_report(report,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    data=json.dumps(report,ensure_ascii=False,separators=(',',':'))
    template=(Path(__file__).parent/'templates/execution_report.html').read_text(encoding='utf-8')
    embedded=data.replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    files={'report.html':template.replace('REPORT_DATA_PLACEHOLDER',embedded).encode('utf-8'),
        'report.json.gz':gzip.compress(data.encode('utf-8'),compresslevel=5,mtime=0)}
    for kind,key,prefix in [('rounds','round','main'),('calls','call','subagent')]:
        for item in report[kind]:files[f'{prefix}-{item[key]:02d}.json']=json.dumps(item,ensure_ascii=False,indent=2).encode('utf-8')
    # WHY: HTML最后原子发布；下载方不会读到写了一半的页面。
    for name,content in files.items():
        if name=='report.html':continue
        (directory/name).write_bytes(content)
    temp=directory/'report.html.tmp';temp.write_bytes(files['report.html']);temp.replace(directory/'report.html')
    with zipfile.ZipFile(directory/'bundle.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as bundle:
        for name,content in files.items():bundle.writestr(name,content)
    return [*files,'bundle.zip']


def finalize_capture(db,cap,result=None,status=None):
    rid=report_id(cap.report_id)
    try:
        existing=_descriptor(db,cap.session_id,rid)
        if existing:return existing
        if getattr(cap,'capture_error_type',None):
            info={'report_id':rid,'state':'failed','error_type':cap.capture_error_type,'files':[]}
            _save_descriptor(db,cap,info);return info
        report=build_report(cap,result,status)
        files=write_report(report,report_directory()/rid)
        info={'report_id':rid,'state':'ready','status':report['status'],'created_at':report['meta']['created_at'],
            **report_urls(cap.session_id,rid),'files':files,
            'main_rounds':report['summary']['main_rounds'],'delegations':report['summary']['delegations']}
        if db is None:info['html_path']=str((report_directory()/rid/'report.html').resolve())
        _save_descriptor(db,cap,info);return info
    except Exception as exc:
        # 报告失败必须可见，但不得替换真实研究结果、掩盖取消或额外调用模型。
        logger.warning('Execution report failed (%s)',type(exc).__name__)
        info={'report_id':rid,'state':'failed','error_type':type(exc).__name__,'files':[]}
        if db is not None:
            db.rollback()
            try:_save_descriptor(db,cap,info)
            except Exception:db.rollback()
        return info
