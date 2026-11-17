"""动作资源清单：输入冻结、候选暂存与已提交派生产物分别授权。"""
import hashlib
import json

from sqlalchemy import select
from app.agent.action_contracts import CONTRACTS
from app.database.models import ResearchEventModel as Event
from app.database.runtime_repository import RuntimeConflict
from app.schemas.artifact_schema import ARTIFACT_TYPES, STATE_ARTIFACT_FIELDS


def _event(runtime, key):
    row = runtime.db.scalar(select(Event).where(Event.session_id == runtime.session_id, Event.event_key == key))
    return json.loads(row.payload_json) if row else None


def artifact_types(action, role, *, output=False):
    fields = CONTRACTS[action].outputs if output else CONTRACTS[action].inputs
    types = {kind for field, kind in STATE_ARTIFACT_FIELDS.items() if field in fields}
    if 'parsed_papers' in fields:
        types.add('document_fragment')
    return {kind for kind in types if role in ARTIFACT_TYPES[kind].roles}


def register_inputs(runtime, task, invocation, state):
    from app.agent.delegation_memory import task_identity
    from app.services.research_artifact_service import ResearchArtifactService
    role = task.role.value
    if task.paper_scope:
        from app.agent.policy.paper_scope import project
        state = project(state, task.paper_scope)
    work = task_identity(state)
    readable = artifact_types(task.operation, role)
    selected = {field: state[field] for field in CONTRACTS[task.operation].inputs if field in state
                and ('document_fragment' if field == 'parsed_papers' else STATE_ARTIFACT_FIELDS.get(field)) in readable}
    if 'writer_candidates' in selected:
        # WHY: 历史审计可留在父状态，但不会因字段同名进入本研究的Artifact读取范围。
        selected['writer_candidates'] = [record for record in selected['writer_candidates']
                                         if record.get('task_identity') == work]
    store = ResearchArtifactService(runtime.db)
    # WHY: 从权威输入重新生成清单，不把旧manifest或模型提交的任意ref变成读取权限。
    frozen = store.externalize_state(runtime.session_id, selected)
    entries = [dict(item) for manifest in frozen.get('artifact_manifest', {}).values() for item in manifest['items']]
    if 'writer_candidate' in readable:
        for record in state.get('writer_candidates') or []:
            if record.get('task_identity') != work:
                continue
            for ref in record.get('artifact_refs') or []:
                value = store.resolve(runtime.session_id, ref, role=role, allowed_types={'writer_candidate'})
                lineage = value['provenance'].get('v2_lineage') or {}
                if lineage and lineage.get('work_identity') != work:
                    raise RuntimeConflict('candidate belongs to another research work')
                entries.append({'ref': ref, 'type': 'writer_candidate', 'version': value['version']})
    packet = {'task_id': task.task_id, 'invocation_id': invocation.invocation_id,
              'delegation_id': invocation.delegation_id, 'expected_version': invocation.expected_version,
              'source_fingerprint': invocation.source_fingerprint, 'work_identity': work,
              'items': entries, 'paper_scope': task.paper_scope}
    from app.services.paper_resources import bind
    packet['paper_resources'] = bind(state, task.operation)
    runtime.event('resources:' + task.task_id, 'action_resources_bound', packet)


def record_commit(runtime, invocation, task, stored):
    allowed = CONTRACTS[task.operation].outputs
    outputs = {field: manifest for field, manifest in stored.get('artifact_manifest', {}).items() if field in allowed}
    if task.paper_scope:
        outputs = {field: {**manifest, 'items': [entry for entry in manifest['items'] if entry.get('paper_id') in task.paper_scope]}
                   for field, manifest in outputs.items() if field == 'paper_cards'}
    runtime.event('lineage:' + invocation.invocation_id, 'action_artifacts_committed', {
        'delegation_id': invocation.delegation_id, 'invocation_id': invocation.invocation_id,
        'task_id': task.task_id, 'business_version': invocation.expected_version + 1,
        'input_scope': 'resources:' + task.task_id, 'outputs': outputs})


def _scope(runtime, context, tool):
    # WHY: 回执确认和真正资料访问分属短事务；必须在实际读写锁内再查Grant，关闭撤销窗口。
    grant, _ = runtime._authorize_tool_context({**context.__dict__, 'tool': tool})
    packet = _event(runtime, 'resources:' + context.task_id)
    if not packet or packet['invocation_id'] != context.invocation_id:
        raise RuntimeConflict('action resource scope is missing')
    return grant, packet


def _read_entry(runtime, context, packet, ref):
    entry = next((item for item in packet['items'] if item['ref'] == ref), None)
    if entry:
        return entry
    key = hashlib.sha256((context.task_id + ':' + ref).encode()).hexdigest()
    produced = _event(runtime, 'artifact-produced:' + key)
    # WHY: 同Attempt可读自己暂存的候选；失败Attempt产物不能自动扩展下一次读取范围。
    if produced and produced['task_id'] == context.task_id:
        return produced
    raise RuntimeConflict('artifact is outside the current action resource scope')


def persist(runtime, context, kwargs):
    from app.services.research_artifact_service import ResearchArtifactService
    from app.services.tool_execution import artifact_storage_scope, current_tool_call_id
    with runtime.transaction():
        grant, packet = _scope(runtime, context, 'artifact_write')
        kind = kwargs['artifact_type']
        if kwargs['session_id'] != runtime.session_id or kind not in artifact_types(context.action, grant.role.value, output=True):
            raise RuntimeConflict('artifact write exceeds action outputs')
        if packet.get('paper_scope') and (not isinstance(kwargs['payload'].get('value'), dict)
                or kwargs['payload']['value'].get('paper_id') not in packet['paper_scope']):
            raise RuntimeConflict('artifact exceeds the paper subset')
        provenance = dict(kwargs.get('provenance') or {})
        parents = provenance.get('source_refs') or []
        if not isinstance(parents, list) or any(not isinstance(ref, str) for ref in parents):
            raise RuntimeConflict('artifact parent references are invalid')
        for ref in parents:
            _read_entry(runtime, context, packet, ref)
        provenance['v2_lineage'] = {'delegation_id': context.delegation_id, 'invocation_id': context.invocation_id,
            'task_id': context.task_id, 'work_identity': packet['work_identity'],
            'input_scope': 'resources:' + context.task_id}
        with artifact_storage_scope():
            ref = ResearchArtifactService(runtime.db).persist_payload(**{**kwargs, 'provenance': provenance})
        key = hashlib.sha256((context.task_id + ':' + ref).encode()).hexdigest()
        if _event(runtime, 'artifact-produced:' + key) is None:
            runtime.event('artifact-produced:' + key, 'artifact_staged', {'ref': ref, 'type': kind,
                'version': kwargs.get('version', 1), 'task_id': context.task_id,
                'delegation_id': context.delegation_id, 'invocation_id': context.invocation_id,
                'tool_call_id': current_tool_call_id()})
        return ref


def resolve(runtime, context, kwargs):
    from app.services.research_artifact_service import ResearchArtifactService
    from app.services.tool_execution import artifact_storage_scope
    with runtime.transaction():
        grant, packet = _scope(runtime, context, 'artifact_read')
        if kwargs['session_id'] != runtime.session_id:
            raise RuntimeConflict('artifact belongs to another session')
        entry = _read_entry(runtime, context, packet, kwargs['artifact_ref'])
        allowed = artifact_types(context.action, grant.role.value)
        # 本Attempt产物仍须满足输出契约；读取不会将其升级为当前证据。
        allowed |= artifact_types(context.action, grant.role.value, output=True)
        if entry['type'] not in allowed or (kwargs.get('allowed_types') is not None and entry['type'] not in kwargs['allowed_types']):
            raise RuntimeConflict('artifact type is outside the action contract')
        if kwargs.get('expected_version') not in (None, entry['version']):
            raise RuntimeConflict('artifact version differs from the frozen scope')
        with artifact_storage_scope():
            return ResearchArtifactService(runtime.db).resolve(runtime.session_id, kwargs['artifact_ref'],
                allowed_types={entry['type']}, role=grant.role.value, expected_version=entry['version'])
