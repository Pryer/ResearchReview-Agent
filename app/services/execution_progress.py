"""按实际execution owner读取进度，避免同会话后续任务串入旧Job/报告。"""
import json
from sqlalchemy import select
from app.database.models import ResearchEventModel as Event, ResearchJobModel as Job, ResearchTaskModel as Task
from app.database.runtime_repository import RuntimeConflict


def bind_job(runtime, job_id):
    with runtime.transaction():
        runtime._owned_row_for_write(live=True)
        job = runtime.db.get(Job, job_id)
        if not job or job.session_id != runtime.session_id or job.status != 'running':
            raise RuntimeConflict('job is not the active research execution')
        runtime.event('job-exec:' + job_id + ':' + runtime.owner, 'job_execution_bound',
                      {'job_id': job_id, 'owner': runtime.owner,
                       'architecture_version': 2})


def record_delegation(runtime, grant, *, execution_profile=None):
    key = 'delegation-exec:' + grant.delegation_id + ':' + runtime.owner
    with runtime.transaction():
        runtime._owned_row_for_write(live=True)
        exists = runtime.db.scalar(select(Event.event_id).where(Event.session_id == runtime.session_id, Event.event_key == key))
        if exists is None:
            runtime.event(key, 'delegation_execution', {'owner': runtime.owner, 'delegation_id': grant.delegation_id,
                'role': grant.role.value, 'profile': grant.profile, 'max_steps': grant.max_steps,
                'execution_profile': execution_profile})


def events_for_owner(db, session_id, owner):
    if not owner:
        return []
    start = db.scalar(select(Event.event_id).where(Event.session_id == session_id, Event.event_key == 'acquire:' + owner))
    if start is None:
        return []
    end = db.scalar(select(Event.event_id).where(Event.session_id == session_id,
        Event.event_type == 'execution_started', Event.event_id > start).order_by(Event.event_id).limit(1))
    query = select(Event).where(Event.session_id == session_id, Event.event_id > start)
    if end is not None:
        query = query.where(Event.event_id < end)
    allowed = {'task_started', 'tool_started', 'tool_finished', 'tool_interrupted', 'delegation_execution',
        'delegation_completed', 'action_resources_bound', 'action_resources_extended', 'action_artifacts_committed', 'artifact_staged', 'legacy_checkpoint_migrated'}
    rows = []
    for event in db.scalars(query.where(Event.event_type.in_(allowed)).order_by(Event.event_id)):
        payload = json.loads(event.payload_json)
        # WHY: 产品进度和报告只展示绑定计数，不外传受控本地文件路径或资料正文。
        if event.event_type == 'action_resources_bound':
            payload = {k: payload[k] for k in ('task_id', 'invocation_id', 'delegation_id')}
        elif event.event_type == 'action_resources_extended':
            payload = {'task_id': payload['task_id'], 'paper_count': len(payload.get('papers') or {}),
                       'file_count': len(payload.get('files') or {})}
        rows.append({'event_id': event.event_id, 'event_type': event.event_type, 'payload': payload})
    return rows


def tool_timeline(events):
    finished = {e['payload']['call_id']: e['payload'] for e in events if e['event_type'] in {'tool_finished', 'tool_interrupted'}}
    return [{**e['payload'], **finished.get(e['payload']['call_id'], {'status': 'pending'})}
            for e in events if e['event_type'] == 'tool_started']


def job_progress(db, job):
    binding = db.scalar(select(Event).where(Event.session_id == job['session_id'],
        Event.event_type == 'job_execution_bound',
        Event.event_key.startswith('job-exec:' + job['job_id'] + ':', autoescape=True))
        .order_by(Event.event_id.desc()).limit(1))
    if binding is None:
        return None
    execution = json.loads(binding.payload_json)
    events = events_for_owner(db, job['session_id'], execution['owner'])
    tasks = [e['payload'] for e in events if e['event_type'] == 'task_started' and e['payload'].get('delegation_id')]
    delegations = [e['payload'] for e in events if e['event_type'] == 'delegation_execution']
    # WHY: 主控可能先澄清或在首次委派前停止，架构归属不能以“至少执行过一个子动作”推断。
    if not tasks and not delegations and execution.get('architecture_version') != 2:
        return None
    states = {row.task_id: row.status for row in db.scalars(select(Task).where(
        Task.session_id == job['session_id'], Task.task_id.in_([t['task_id'] for t in tasks])))}
    completed = sum(states.get(t['task_id']) == 'completed' for t in tasks)
    # WHY: 终态Job不再执行动作；崩溃留下的running行只用于恢复，不能让界面仍显示正在写作。
    active = next((t for t in reversed(tasks) if states.get(t['task_id']) == 'running'), {}) if job['status'] in {
        'running', 'cancel_requested'} else {}
    tools = tool_timeline(events)
    return {'architecture_version': 2, 'job_status': job['status'],
        'delegations': len(delegations), 'action_attempts': len(tasks),
        'completed_actions': completed, 'current_role': active.get('role'), 'current_action': active.get('operation'),
        'current_profile': next((d['profile'] for d in delegations if d['delegation_id'] == active.get('delegation_id')), None),
        'tool_calls': len(tools), 'completed_tools': sum(t['status'] == 'completed' for t in tools),
        'failed_tools': sum(t['status'] == 'failed' for t in tools),
        'uncertain_tools': sum(t['status'] == 'uncertain' for t in tools),
        'pending_tools': sum(t['status'] == 'pending' for t in tools)}
