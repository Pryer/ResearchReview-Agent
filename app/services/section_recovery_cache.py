"""章节计算缓存独立持久化，不更新研究正文、Check报告或业务版本。"""
import hashlib
import json
from sqlalchemy import select
from app.database.models import ResearchEventModel
from app.database.runtime_repository import RuntimeConflict
from app.services.tool_execution import execute_tool, current_tool_context, artifact_storage_scope


def descriptor(plan, section, state, llm, prompt):
    from app.tools.write_deliverable import _section_input_fingerprint
    from app.services.role_execution_profile import execution_profile
    from app.agent.policy.authorization import goal_fingerprint
    payload = {'version': 1, 'inputs': _section_input_fingerprint(plan, section, state),
        'goal': goal_fingerprint(state), 'plan': plan.model_dump(mode='json'), 'base_review': state.get('review') or '',
        'targets': state.get('target_section_ids') or [],
        'writer': execution_profile(llm, 'writing')['fingerprint'], 'prompt': prompt}
    return {'deliverable': plan.deliverable_type.value, 'section_id': section.id,
        'fingerprint': hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}


def cached_section(descriptor, *, text=None):
    context = current_tool_context()
    if context is None:
        return None
    from app.services.durable_execution_service import active_runtime
    name = 'section_cache_read' if text is None else 'section_cache_write'
    return execute_tool(name, lambda: active_runtime().section_cache(context, descriptor, text),
        arguments={'descriptor': descriptor, 'text': text}, boundary='section_recovery_cache')


def enabled(section_id, state):
    context = current_tool_context()
    return context is not None and (context.action == 'generate_deliverables' or
        section_id in (state.get('target_section_ids') or []))


def access(runtime, context, descriptor, text):
    from app.services.action_resources import _event
    from app.services.research_artifact_service import ResearchArtifactService
    tool = 'section_cache_read' if text is None else 'section_cache_write'
    with runtime.transaction():
        grant, invocation = runtime._authorize_tool_context({**context.__dict__, 'tool': tool})
        packet = _event(runtime, 'resources:' + context.task_id)
        if not packet or grant.role.value != 'writing' or context.action not in {'generate_deliverables', 'rewrite_sections'}:
            raise RuntimeConflict('section cache requires a scoped writing task')
        if (set(descriptor) != {'deliverable', 'section_id', 'fingerprint'} or
                not all(isinstance(value, str) and value for value in descriptor.values()) or
                len(descriptor['fingerprint']) != 64):
            raise RuntimeConflict('invalid section cache descriptor')
        if context.action == 'rewrite_sections' and descriptor['section_id'] not in invocation.arguments['section_ids']:
            raise RuntimeConflict('section cache exceeds the granted revision scope')
        key = 'section-cache:' + hashlib.sha256(json.dumps([runtime.session_id, packet['work_identity'],
            descriptor], sort_keys=True).encode()).hexdigest()
        previous = runtime.db.scalar(select(ResearchEventModel).where(
            ResearchEventModel.session_id == runtime.session_id, ResearchEventModel.event_key == key))
        store = ResearchArtifactService(runtime.db)
        with artifact_storage_scope():
            if previous is not None:
                ref = json.loads(previous.payload_json)['ref']
                try:
                    payload = store.resolve(runtime.session_id, ref, role='writing', allowed_types={'writer_candidate'})
                except (ValueError, KeyError, LookupError, PermissionError) as exc:
                    raise RuntimeConflict('section cache artifact is invalid') from exc
                if payload['payload'].get('descriptor') != descriptor:
                    raise RuntimeConflict('section cache payload does not match its key')
                return payload['payload']['text']
            if text is None:
                return None
            if not isinstance(text, str) or not text.strip():
                raise RuntimeConflict('empty section cannot enter recovery cache')
            ref = store.persist_payload(session_id=runtime.session_id, artifact_type='writer_candidate',
                identity=key, payload={'descriptor': descriptor, 'text': text, 'calculation_only': True},
                provenance={'task_id': context.task_id, 'invocation_id': context.invocation_id,
                    'delegation_id': context.delegation_id, 'work_identity': packet['work_identity']})
            runtime.event(key, 'section_calculation_cached', {'ref': ref, **descriptor,
                'task_id': context.task_id, 'invocation_id': context.invocation_id})
            return text
