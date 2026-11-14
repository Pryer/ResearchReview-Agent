"""执行局部的可观察上下文；不进入模型状态，不增加模型调用。"""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
import copy
import hashlib
import json
import time
from uuid import uuid4

_CAPTURE=ContextVar('execution_report_capture',default=None)


def preview(value,limit=3000):
    from app.services.execution_report_collector import safe
    value=safe(value)
    text=json.dumps(value,ensure_ascii=False,default=str)
    if len(text)<=limit:return copy.deepcopy(value)
    return {'preview_only':True,'json_characters':len(text),
        'sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),'preview':text[:limit],
        'full_value':'大字段原值由源检查点和持久化资料引用定位'}


def state_stats(state):
    b=state.get('agent_execution_budget') or {};check=state.get('check_report') or {}
    return {'候选论文':len(state.get('candidate_papers') or []),'排序保留':len(state.get('ranked_papers') or []),
        '详情论文':len(state.get('paper_details') or []),'PaperCard':len(state.get('paper_cards') or []),
        '候选稿字符':len(state.get('review') or ''),'发布版本':state.get('published_draft_version') or 0,
        '书目数量':len(state.get('references') or []),'Check问题':len(check.get('issues') or []),
        'Check结论':check.get('verdict') or '未开始','LLM请求':b.get('llm_requests',0),
        '检索模型请求':b.get('retrieval_model_requests',0),'Main轮次':b.get('decision_rounds',0)}


class Capture:
    def __init__(self,session_id,request,report_id=None):
        self.session_id=session_id;self.request=request;self.report_id=report_id or uuid4().hex
        self.started=time.monotonic();self.rounds=[];self.calls=[];self.latest_state={}
        self.source=None;self.error_type=None;self.start_version=None;self.end_version=None

    def data(self):
        if self.rounds:self.rounds[-1]['after']=state_stats(self.latest_state)
        return {'rounds':self.rounds,'calls':self.calls,'decisions':[x['decision'] for x in self.rounds]}


@contextmanager
def capture_scope(capture):
    token=_CAPTURE.set(capture)
    try:yield capture
    finally:_CAPTURE.reset(token)


def _observer(fn):
    @wraps(fn)
    def wrapped(*args,**kwargs):
        try:return fn(*args,**kwargs)
        except Exception as exc:
            cap=_CAPTURE.get()
            if cap is not None:cap.capture_error_type=type(exc).__name__
            import logging
            logging.getLogger(__name__).warning('Report observation failed (%s)',type(exc).__name__)
    return wrapped


@_observer
def record_main_context(state,context,decision,*,model_input):
    cap=_CAPTURE.get()
    if cap is None:return
    if cap.rounds:cap.rounds[-1]['after']=state_stats(state)
    cap.latest_state=state
    cap.rounds.append({'decision':decision.model_dump(mode='json'),'context':context.model_dump(mode='json'),
        'model_input':model_input,'origin':'模型决策' if model_input else '策略交接（无新模型请求）',
        'before':state_stats(state)})


@_observer
def record_delegation(state,role,active):
    cap=_CAPTURE.get()
    if cap is None or any(x['delegation_id']==active['id'] for x in cap.calls):return
    from app.agent.action_contracts import CONTRACTS
    from app.services.durable_execution_service import active_runtime
    from app.services.execution_report_collector import changes
    runtime=active_runtime();fields=CONTRACTS[active['operation']].inputs
    inputs={k:preview(state[k]) for k in sorted(fields) if k in state}
    prior=next((x for x in reversed(cap.calls) if x['role']==role),None)
    cap.latest_state=state
    cap.calls.append({'call':len(cap.calls)+1,'main_round':len(cap.rounds),'role':role,'operation':active['operation'],
        'task_id':None,'delegation_id':active['id'],'objective':active['objective'],'arguments':copy.deepcopy(active['arguments']),
        'context_packet':copy.deepcopy(active.get('context_packet')),'work_context':copy.deepcopy(active.get('work_context')),
        'contract_fields':sorted(fields),'input_state':inputs,'input_artifacts':copy.deepcopy({k:v for k,v in (state.get('artifact_manifest') or {}).items() if k in fields}),
        'controlled_file_references':copy.deepcopy(state.get('controlled_pdf_files')),
        'before':state_stats(state),'after':{},'before_checkpoint':runtime.version if runtime else None,
        'commit_checkpoint':None,'summary_checkpoint':None,'previous_same_role_call':prior['call'] if prior else None,
        'same_role_input_changes':changes(prior['input_state'] if prior else {},inputs),
        'result':{},'returned_to_main':None,'work_summary_after':None,'messages':[],'control_trajectory':None,
        'steps':[],'step_offset':len(state.get('steps') or []),'context_kind':'角色契约输入和工作记忆；大字段为带哈希的预览'})


@_observer
def record_delegation_result(state,role,active,result):
    cap=_CAPTURE.get()
    if cap is None:return
    call=next((x for x in cap.calls if x['delegation_id']==active['id']),None)
    if call is None:return
    from app.services.durable_execution_service import active_runtime
    runtime=active_runtime();session=state['subagent_sessions'][role]
    call.update(after=state_stats(state),result=copy.deepcopy(active.get('last_result') or {}),
        returned_to_main=result.model_dump(mode='json'),work_summary_after=copy.deepcopy(session.get('work_summary')),
        task_id=(active.get('last_result') or {}).get('task_id'),
        commit_checkpoint=runtime.version if runtime else None,summary_checkpoint=runtime.version if runtime else None,
        messages=copy.deepcopy([x for x in session.get('messages',[]) if x.get('delegation_id')==active['id']]),
        steps=copy.deepcopy((state.get('steps') or [])[call.pop('step_offset',0):]))
    cap.latest_state=state


def with_execution_report(fn):
    """直接Python图入口也生成报告；会话入口已有作用域时不重复生成。"""
    @wraps(fn)
    def wrapped(*args,**kwargs):
        from app.core.config import get_settings
        if _CAPTURE.get() is not None or not get_settings().execution_report_enabled:return fn(*args,**kwargs)
        state=kwargs.get('initial_state') or kwargs.get('state') or (args[0] if args and isinstance(args[0],dict) else {})
        query=kwargs.get('user_query') or (args[0] if args and isinstance(args[0],str) else state.get('user_query',''))
        cap=Capture(str(state.get('session_id') or 'standalone'),str(query));result=None
        try:
            with capture_scope(cap):
                result=fn(*args,**kwargs)
                cap.latest_state=result.get('research_state') or cap.latest_state
                return result
        except BaseException as exc:
            cap.error_type=type(exc).__name__;raise
        finally:
            from app.services.execution_report_service import finalize_capture
            descriptor=finalize_capture(None,cap,result)
            if isinstance(result,dict):result['execution_report']=descriptor
    return wrapped
