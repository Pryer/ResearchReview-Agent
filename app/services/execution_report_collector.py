"""从单会话冻结检查点和真实归档构建报告，不执行研究动作。"""
from collections import Counter
import copy
import hashlib
import json
from app.agent.action_contracts import CONTRACTS

def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def safe(value):
    """报告采用JSON值域，去除凭证和内部思考协议，不改变研究状态。"""
    from datetime import date,datetime
    from pathlib import Path
    import math
    blocked={'reasoning_content','reasoning','chain_of_thought','thinking','api_key','authorization','password','secret','access_token','refresh_token','client_secret','proxy_authorization'}
    if hasattr(value,'model_dump'):return safe(value.model_dump(mode='json'))
    if isinstance(value,dict):
        return {safe(str(k)):safe(v) for k,v in value.items() if str(k).lower() not in blocked and not str(k).lower().endswith('_api_key')}
    if isinstance(value,(list,tuple,set)):return [safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,(date,datetime)):return value.isoformat()
    from app.services.report_redaction import redact_text
    if isinstance(value,Path):return redact_text(str(value))
    if isinstance(value,str):return redact_text(value)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return {'unserialized_type':type(value).__name__}


def changes(old, new, path=''):
    if dumps(old)==dumps(new):return []
    if isinstance(old,dict) and isinstance(new,dict):
        rows=[]
        for key in sorted(set(old)|set(new)):
            child=(path+'.'+key).lstrip('.')
            if key not in old:rows.append({'path':child,'before':None,'after':new[key],'kind':'added'})
            elif key not in new:rows.append({'path':child,'before':old[key],'after':None,'kind':'removed'})
            else:rows+=changes(old[key],new[key],child)
        return rows
    return [{'path':path,'before':old,'after':new,'kind':'changed'}]


def collect_report(source, capture=None):
    capture=capture or {}
    artifacts=source['artifacts'];checkpoints=source['checkpoints'];final=source['final']
    tasks=source['tasks'];events=source.get('events',[])
    delegation_results={e['payload']['delegation_id']:e['payload'] for e in events if e['event_type']=='delegation_completed'}
    if not checkpoints:checkpoints=[{'version':0,'state':final}]
    by_version={p['version']:p for p in checkpoints}

    def payload(ref):return artifacts[ref.removeprefix('artifact://')]['payload']
    def field(state,key):
        m=(state.get('artifact_manifest') or {}).get(key)
        if not m:return copy.deepcopy(state.get(key))
        values=[payload(item['ref'])['value'] for item in m['items']]
        if m['shape'].startswith('json'):return json.loads(''.join(values))
        if m['shape']=='text':return ''.join(values)
        if m['shape']=='documents':
            docs={}
            for entry,value in zip(m['items'],values):docs[entry['paper_id']]=docs.get(entry['paper_id'],'')+value
            return {key:json.loads(value) for key,value in docs.items()}
        return values

    def stats(state):
        cards=field(state,'paper_cards') or []
        plans=field(state,'claim_plans') or []
        budget=state.get('agent_execution_budget') or {}
        check=field(state,'check_report') or {}
        return {'候选论文':len(field(state,'candidate_papers') or []),'排序保留':len(field(state,'ranked_papers') or []),
            '详情论文':len(field(state,'paper_details') or []),'PaperCard':len(cards),
            '原文证据':sum(len(x.get('evidence_spans') or []) for x in cards),
            '计划主张':sum(len(x.get('claims') or []) for x in plans),'候选稿字符':len(field(state,'review') or ''),
            '发布版本':state.get('published_draft_version') or 0,'书目数量':len(state.get('references') or []),
            'Check问题':len(check.get('issues') or []),'Check结论':check.get('verdict') or '未开始',
            'LLM请求':budget.get('llm_requests',0),'检索模型请求':budget.get('retrieval_model_requests',0),
            'Main轮次':budget.get('decision_rounds',0)}

    for cp in checkpoints:
        cp['trajectory']=field(cp['state'],'main_agent_trajectory') or {}
        cp['sessions']=field(cp['state'],'subagent_sessions') or {}
    scope=next((cp['trajectory']['scope_id'] for cp in checkpoints if cp['trajectory']), '')
    streams=[]
    for cp in checkpoints:
        streams.append((cp['trajectory'],{'checkpoint':cp['version']}))
    epoch_archives=[];observed_results={};observed_delegations={}
    for aid,a in artifacts.items():
        p=a['payload']
        if a['artifact_type']=='context_archive_index' and p.get('artifact_type')=='tool_observation':
            observed=json.loads(''.join(payload(ref)['value'] for ref in p['fragments']))
            if observed.get('task_id'):observed_results[observed['task_id']]=observed
            if observed.get('delegation_id'):observed_delegations[observed['delegation_id']]=observed
        if a['artifact_type']=='context_archive_index' and p.get('artifact_type')=='context_epoch':
            value=json.loads(''.join(payload(ref)['value'] for ref in p['fragments']))
            epoch_archives.append({'ref':'artifact://'+aid,'scope_id':p.get('scope_id'),
                'epoch':value.get('epoch'),'messages':len(value.get('messages') or []),
                'observations':value.get('observation_count'),'fragments':p['fragments']})
            if p.get('scope_id')==scope:streams.append((value,{'archive_ref':'artifact://'+aid}))
    matches={}
    for stream,stream_source in streams:
        msgs=stream.get('messages') or []
        for idx,msg in enumerate(msgs):
            calls=msg.get('tool_calls') or []
            if not calls:continue
            context=None
            for before in reversed(msgs[:idx]):
                try:value=json.loads(before.get('content') or '{}')
                except (ValueError,TypeError):continue
                if isinstance(value,dict) and {'goal','state','key_evidence','decisions','open_questions'}<=set(value):
                    context=value;break
            if context is not None:
                for call in calls:
                    matches.setdefault(call['id'],{'context':context,'messages':msgs[:idx],
                        'epoch':stream.get('epoch'),'source':stream_source})

    decisions=final.get('main_agent_decisions') or capture.get('decisions') or []
    rounds=[];last_context={};last_input_round=None
    for number,decision in enumerate(decisions,1):
        cp=next((p for p in checkpoints if len(p['state'].get('main_agent_decisions') or [])>=number), checkpoints[-1])
        pre=by_version.get(cp['version']-1,cp)
        post_candidates=[p for p in checkpoints if len(p['state'].get('main_agent_decisions') or [])==number]
        post=post_candidates[-1] if post_candidates else cp
        match=matches.get(decision.get('tool_call_id'))
        captured=next((x for x in capture.get('rounds',[]) if x['decision']['decision_id']==decision.get('decision_id')),None)
        if not match and captured:
            match={'context':captured['context'],'messages':[], 'epoch':None, 'source':{'captured_at_decision':True}}
        item={'round':number,'action':decision['action'],'decision':decision,
            'origin':captured['origin'] if captured else '模型决策' if decision.get('usage') else '策略交接（无新模型请求）',
            'before_checkpoint':pre['version'],'first_checkpoint':cp['version'],'after_checkpoint':post['version'],
            'before':captured['before'] if captured else stats(pre['state']),
            'after':captured.get('after',stats(post['state'])) if captured else stats(post['state']),
            'context':match['context'] if match else None,
            'model_input':captured['model_input'] if captured else bool(decision.get('usage')),
            'messages_without_system':match['messages'] if match else [],
            'context_source':match['source'] if match else {'policy_only':True},
            'epoch':match['epoch'] if match else cp['trajectory'].get('epoch'),
            'previous_input_round':last_input_round,'context_changes':changes(last_context,match['context']) if match else []}
        if match:last_context=match['context'];last_input_round=number
        rounds.append(item)

    calls=[];prior={}
    for index,task in enumerate([] if capture.get('calls') else tasks,1):
        before=by_version.get(task['expected_version'])
        if before is None:continue
        after=by_version.get(task['expected_version']+1,before)
        result=task['result']
        v2 = next((e['payload'] for e in events if e['event_type'] == 'task_started'
                   and e['payload'].get('task_id') == task['task_id'] and e['payload'].get('delegation_id')), None)
        if v2 and not result:
            result={'role':v2['role'],'operation':task['operation'],'status':task['status'],'findings':[]}
        if not result or not result.get('role'):continue
        role=result['role'];session=before['sessions'].get(role,{})
        active=session.get('active') or {}
        if not active and v2:
            active={'id':v2['delegation_id'],'objective':'任务级专业委派','arguments':{}}
        if not active or task['operation'] not in CONTRACTS:continue
        # Result submission is followed by a role-summary checkpoint. Capture it
        # separately so a summary is never presented as the original input.
        completed=None
        for cp in checkpoints:
            s=cp['sessions'].get(role,{})
            if cp['version']>before['version'] and (s.get('work_summary',{}).get('last_result') or {}).get('task_id')==active['id']:
                completed=cp;break
        inputs={k:field(before['state'],k) for k in sorted(CONTRACTS[task['operation']].inputs)
                if k in before['state'] or k in before['state'].get('artifact_manifest',{})}
        previous=prior.get(role)
        after_role=(completed or after)['sessions'].get(role,{})
        role_messages=[m for m in after_role.get('messages',[]) if m.get('delegation_id')==active['id']]
        pre_steps=before['state'].get('steps') or [];after_steps=after['state'].get('steps') or []
        item={'call':index,'main_round':len(before['state'].get('main_agent_decisions') or []),
            'role':role,'operation':task['operation'],'task_id':task['task_id'],'delegation_id':active['id'],
            'before_checkpoint':before['version'],'commit_checkpoint':after['version'],
            'summary_checkpoint':completed['version'] if completed else None,
            'objective':active['objective'],'arguments':active['arguments'],
            'context_packet':active.get('context_packet'),'work_context':active.get('work_context'),
            'controlled_file_references':before['state'].get('controlled_pdf_files') if 'pdf_paths' in CONTRACTS[task['operation']].inputs else None,
            'contract_fields':sorted(CONTRACTS[task['operation']].inputs),'input_state':inputs,
            'before':stats(before['state']),'after':stats(after['state']),
            'result':result,'returned_to_main':(observed_delegations.get(active['id']) or delegation_results.get(active['id'])) if v2 else observed_results.get(active['id']),
            'work_summary_after':after_role.get('work_summary'),
            'previous_same_role_call':previous['call'] if previous else None,
            'same_role_input_changes':changes(previous['input_state'],inputs) if previous else changes({},inputs),
            'messages':role_messages,'control_trajectory':after_role.get('main_agent_trajectory'),
            'steps':after_steps[len(pre_steps):] if after_steps[:len(pre_steps)]==pre_steps else [],
            'context_kind':'确定性委派的契约输入和角色记忆；业务LLM完整Prompt未持久化'}
        prior[role]=item;calls.append(item)

    epochs=[];last=None
    for cp in checkpoints:
        epoch=cp['trajectory'].get('epoch')
        if epoch is not None and epoch!=last:
            epochs.append({'checkpoint':cp['version'],'epoch':epoch,'main_round':len(cp['state'].get('main_agent_decisions') or []),
                'messages':len(cp['trajectory'].get('messages') or []),'archive_refs':cp['trajectory'].get('archive_refs') or []})
            last=epoch
    if capture.get('calls'):calls=capture['calls']
    from app.services.execution_progress import tool_timeline
    tools=tool_timeline(events)
    completions=delegation_results
    delegations=[{**e['payload'],'result':completions.get(e['payload']['delegation_id']),
        'actions':[c['call'] for c in calls if c['delegation_id']==e['payload']['delegation_id']]}
        for e in events if e['event_type']=='delegation_execution']
    run=source['run'];meta=source['meta']
    report=safe({'run_id':meta['report_id'],'session_id':meta['session_id'], 'status':meta['status'], 'meta':meta,'request':run['task']['request']['user_query'],
        'summary':{'main_rounds':len(rounds),'main_model_calls':run['metrics']['tokens']['by_operation']['main_agent_decision']['calls'],
            'policy_rounds':sum(not x['model_input'] for x in rounds),'delegations':len(delegations) if delegations else len(calls),
            'action_attempts':len(calls),'tool_calls':len(tools),
            'roles':dict(Counter(x['role'] for x in calls)),'checkpoints':len(checkpoints),'epoch_archives':len(epoch_archives),
            'native_contexts_recovered':sum(x['context'] is not None for x in rounds),'source_checkpoint_sha256':meta.get('source_fingerprint',''), 'references':meta.get('references',0), 'publication_version':meta.get('publication_version',0), 'cumulative_requests':meta.get('cumulative_requests',0), 'request_limit':meta.get('request_limit',0),
            'seconds':run['latency_seconds'],'first_draft_seconds':run['delivery_observed']['delivery_metrics'].get('first_draft_seconds'),
            'total_requests':run['budget_ledger']['llm_requests']+run['budget_ledger']['retrieval_model_requests']},
        'limitations':['模型决策、代码策略交接和专业工具执行分别标记。', '大文本以带长度和哈希的预览保存，原值可定位源检查点及资料引用。', '未持久化的HTTP完整Prompt不事后伪造；不包含内部思考协议。'],
        'rounds':rounds,'calls':calls,'epochs':epochs,'epoch_archives':epoch_archives,
        'delegation_runs':delegations,'tool_calls':tools,
        'events':[{'id':x['event_id'],'type':x['event_type'],'payload':x['payload']} for x in events]})
    return report
