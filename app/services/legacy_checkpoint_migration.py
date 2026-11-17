"""将已知v1活动委派迁入精确single授权；提交证明必须来自数据库。"""
import copy
import json
from dataclasses import dataclass

from app.agent.action_registry import ACTION_REGISTRY, validate_arguments
from app.agent.context_builder import context_source_fingerprint
from app.agent.policy.authorization import AuthorizationPolicy, goal_fingerprint
from app.database.models import ResearchTaskModel, ResearchDelegationModel, ResearchInvocationModel, ResearchWorkVersionModel
from app.database.runtime_repository import RuntimeConflict, dumps
from app.schemas.agent_task_schema import AgentTaskResult
from app.schemas.delegation_schema import ActionInvocation, DelegationGrant


@dataclass(frozen=True)
class MigrationMetadata:
    grant: DelegationGrant
    legacy_id: str
    legacy_rounds: int
    completed_result: dict | None
    imported_invocation: ActionInvocation | None
    business_version: int
    expected_business_version: int


def apply_metadata(runtime, migration):
    """仅由snapshot的CAS事务调用，不在迁移器内开启嵌套提交。"""
    from app.database.delegation_repository import DelegationRepository
    if DelegationRepository(runtime).active() is not None:
        raise RuntimeConflict('legacy checkpoint conflicts with an active v2 delegation')
    grant = migration.grant
    # WHY: 迁移准备与提交之间可能有其他会话写入；不能用Session缓存覆盖新版本。
    row = runtime.db.get(ResearchWorkVersionModel, runtime.session_id, populate_existing=True)
    if (row.version if row else 0) != migration.expected_business_version:
        raise RuntimeConflict('business version changed while preparing legacy migration')
    if row is None:
        runtime.db.add(ResearchWorkVersionModel(session_id=runtime.session_id, version=migration.business_version))
    else:
        row.version = migration.business_version
    committed = migration.imported_invocation is not None
    runtime.db.add(ResearchDelegationModel(delegation_id=grant.delegation_id, session_id=runtime.session_id,
        grant_json=grant.model_dump_json(), status='running', next_step=int(committed), observation_cursor=int(committed)))
    if committed:
        invocation = migration.imported_invocation
        runtime.db.add(ResearchInvocationModel(invocation_id=invocation.invocation_id,
            delegation_id=grant.delegation_id, step_index=0, invocation_json=invocation.model_dump_json(),
            task_id=migration.completed_result['task_id'], status='committed', result_json=dumps(migration.completed_result)))
    runtime.event('legacy-migration:' + grant.delegation_id, 'legacy_checkpoint_migrated', {
        'legacy_id': migration.legacy_id, 'delegation_id': grant.delegation_id, 'role': grant.role.value,
        'operation': grant.allowed_actions[0], 'legacy_rounds': migration.legacy_rounds,
        'imported_committed_task': migration.completed_result['task_id'] if committed else None})


def migrate_active(state):
    from app.agent.execution_budget import check_execution
    from app.core.config import get_settings
    from app.services.durable_execution_service import active_runtime, active_task_id
    from app.database.delegation_repository import DelegationRepository
    from app.agent.react_context import ReActContext
    role = state.get('active_subagent')
    if not role:
        return False
    if not isinstance(role, str) or role not in {'search', 'analysis', 'writing', 'check'}:
        raise RuntimeConflict('unknown historical subagent role')
    check_execution()
    runtime = active_runtime()
    if runtime is None or active_task_id() or state.get('active_delegation_id') or DelegationRepository(runtime).active():
        raise RuntimeConflict('legacy migration requires an unambiguous durable execution')
    old = ((state.get('subagent_sessions') or {}).get(role) or {}).get('active') or {}
    operation = old.get('operation')
    spec = ACTION_REGISTRY.get(operation)
    historical_check = role == 'analysis' and operation == 'validate_result'
    if (not isinstance(old.get('id'), str) or not old['id'] or len(old['id']) > 128 or spec is None or spec.role is None
            or (spec.role.value != role and not historical_check)):
        raise RuntimeConflict('unknown historical delegation role or operation')
    if old.get('goal_fingerprint') != goal_fingerprint(state):
        raise RuntimeConflict('historical delegation target no longer matches current constraints')
    rounds = old.get('rounds')
    completed = old.get('completed_operations')
    arguments = old.get('arguments')
    if (type(rounds) is not int or rounds < 0 or not isinstance(completed, list)
            or not all(isinstance(item, str) for item in completed) or set(completed) - {operation}
            or not isinstance(arguments, dict) or not isinstance(old.get('source_fingerprint'), str) or not old['source_fingerprint']):
        raise RuntimeConflict('unknown historical delegation state format')
    validate_arguments(operation, arguments)
    if operation == 'rewrite_sections' and len(arguments['section_ids']) > 3:
        raise RuntimeConflict('historical section scope exceeds the current action limit')
    last = old.get('last_result') or {}
    proof = None
    if operation in completed and not historical_check:
        row = runtime.db.get(ResearchTaskModel, last.get('task_id')) if last.get('task_id') else None
        if row is None or row.session_id != runtime.session_id or row.operation != operation or row.status != 'completed':
            raise RuntimeConflict('historical completion has no durable task proof')
        try:
            proof = json.loads(row.result_json)
            parsed = AgentTaskResult.model_validate(proof)
        except ValueError as exc:
            raise RuntimeConflict('historical committed result format is unsupported') from exc
        if (parsed.role != spec.role or proof != last or parsed.operation != operation or parsed.status.value != 'completed'
                or parsed.input_state_fingerprint != old['source_fingerprint']
                or parsed.output_state_fingerprint != context_source_fingerprint(state)):
            raise RuntimeConflict('historical committed result does not match the checkpoint')
        mandatory = state.get('agent_mandatory_actions') or []
        if (old.get('consume_mandatory_action') and mandatory and mandatory[0].get('action') == operation
                and (mandatory[0].get('arguments') or {}) == arguments):
            raise RuntimeConflict('historical completed action conflicts with the pending mandatory queue')
    elif operation not in completed and old.get('source_fingerprint') != context_source_fingerprint(state):
        raise RuntimeConflict('historical uncommitted action inputs changed')
    candidate = copy.deepcopy(state)
    if historical_check:
        if not str(candidate.get('review') or '').strip():
            raise RuntimeConflict('historical verification has no candidate body to recheck')
        candidate['pending_result_validation'] = True
        candidate.pop('check_report', None)
        candidate.pop('autonomous_verified_fingerprint', None)
    grant = AuthorizationPolicy().migrate_single(candidate, role=spec.role, operation=operation,
        arguments=arguments, goal=str(old.get('objective') or spec.description))
    version = DelegationRepository(runtime).version()
    expected_business_version = version
    if proof or historical_check:
        version = max(1, version + int(historical_check))
    imported = ActionInvocation(delegation_id=grant.delegation_id, step_index=0, action=operation,
        arguments=copy.deepcopy(arguments), expected_version=max(0, version - 1),
        source_fingerprint=old['source_fingerprint']) if proof else None
    migration = MigrationMetadata(grant, old['id'], rounds, proof, imported, version, expected_business_version)
    session = candidate['subagent_sessions'][role]
    # WHY: 原轨迹与参数保留用于审计；仅撤掉活动指针，不让旧循环解释v2检查点。
    archived = copy.deepcopy(old)
    if historical_check:
        archived['check_report_before_migration'] = copy.deepcopy(state.get('check_report'))
    session['last_delegation'] = archived
    trajectory = ReActContext(candidate, namespace=role)
    trajectory.observe(proof or {'status': 'interrupted', 'findings': ['原委派已转入受控single恢复']}, archive=False)
    session.pop('active', None)
    candidate.pop('active_subagent', None)
    main = ReActContext(candidate)
    pending = main.data.get('pending') or {}
    if pending and pending['action'] != operation:
        raise RuntimeConflict('historical Main tool call does not match the delegated operation')
    if proof:
        main.observe(proof, archive=False)
    candidate['architecture_version'] = 2
    candidate['active_delegation_id'] = grant.delegation_id
    candidate.setdefault('delegation_control', {})[grant.delegation_id] = {
        'rounds': rounds, 'round_limit': get_settings().agent_subagent_max_rounds, 'stalled': 0, 'digest': ''}
    candidate['legacy_checkpoint_migration'] = {'version': 1, 'legacy_id': old['id'], 'delegation_id': grant.delegation_id}
    runtime.snapshot(candidate, ledger=state.get('agent_execution_budget') or {}, legacy_migration=migration)
    ledger = state.get('agent_execution_budget')
    if ledger is not None:
        candidate['agent_execution_budget'] = ledger
    state.clear()
    state.update(candidate)
    return True
