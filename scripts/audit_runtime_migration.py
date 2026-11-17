"""只读统计历史运行态与迁移结构，不输出会话身份、目标、论文或正文。"""
import json
import time
from collections import Counter

from sqlalchemy import inspect, select

from app.database.models import ResearchRuntimeModel, ResearchDelegationModel, ResearchTaskModel


def audit(engine):
    from app.agent.action_registry import ACTION_REGISTRY
    tables = set(inspect(engine).get_table_names())
    from app.database.architecture_migrations import TABLES
    v2_schema_present = {table.name for table in TABLES} <= tables
    summary = Counter()
    if ResearchRuntimeModel.__tablename__ not in tables:
        return {'runtime_rows': 0, 'schema_present': False, 'v2_schema_present': v2_schema_present,
                'classification': {}, 'read_only': True}
    with engine.connect() as conn:
        rows = conn.execute(select(ResearchRuntimeModel.__table__)).mappings()
        for row in rows:
            summary['runtime_rows'] += 1
            if row['owner'] and float(row['lease_until'] or 0) > time.time():
                summary['active_leases'] += 1
            try:
                state = json.loads(row['state_json'] or '{}')
                if not isinstance(state, dict):
                    raise ValueError('state is not an object')
            except (ValueError, TypeError):
                summary['malformed_snapshot'] += 1
                continue
            if state.get('architecture_version') == 2:
                summary['v2_snapshot'] += 1
            else:
                summary['historical_or_unstarted_snapshot'] += 1
            role = state.get('active_subagent')
            if not role:
                continue
            sessions = state.get('subagent_sessions')
            session = sessions.get(role) if isinstance(sessions, dict) and isinstance(role, str) else None
            old = session.get('active') if isinstance(session, dict) else None
            operation = old.get('operation') if isinstance(old, dict) else None
            spec = ACTION_REGISTRY.get(operation) if isinstance(operation, str) else None
            if (spec and spec.role and role in {spec.role.value, 'analysis' if old['operation'] == 'validate_result' else spec.role.value}
                    and isinstance(old.get('id'), str) and isinstance(old.get('arguments'), dict)
                    and type(old.get('rounds')) is int and isinstance(old.get('completed_operations'), list)):
                summary['historical_active_known_shape'] += 1
            else:
                summary['historical_active_needs_review'] += 1
        if ResearchDelegationModel.__tablename__ in tables:
            summary['active_v2_delegations'] = len(conn.execute(select(ResearchDelegationModel.delegation_id).where(
                ResearchDelegationModel.status == 'running')).all())
        if ResearchTaskModel.__tablename__ in tables:
            summary['running_action_attempts'] = len(conn.execute(select(ResearchTaskModel.task_id).where(
                ResearchTaskModel.status == 'running')).all())
    return {'runtime_rows': summary.pop('runtime_rows', 0), 'schema_present': True, 'v2_schema_present': v2_schema_present,
            'classification': dict(summary), 'read_only': True,
            'limitations': ['known_shape仅表示结构可识别；恢复时仍需目标/源指纹、提交证明和CAS校验。',
                            '在途任务应使用当前v2部署恢复；二进制回退须停写并恢复升级前备份。']}


if __name__ == '__main__':
    from app.database.db import engine
    if engine.dialect.name != 'sqlite':
        raise SystemExit('Local inventory supports SQLite only')
    print(json.dumps(audit(engine), ensure_ascii=False, indent=2))
