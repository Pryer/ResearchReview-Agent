"""Agent v2加法迁移：版本登记、结构校验；回退使用停写后的数据库备份。"""
import hashlib
import json
from sqlalchemy import Column, Integer, MetaData, String, Table, inspect, select
from app.database.models import ResearchDelegationModel, ResearchInvocationModel, ResearchWorkVersionModel

metadata = MetaData()
versions = Table('research_schema_versions', metadata,
    Column('component', String(64), primary_key=True), Column('version', Integer, nullable=False),
    Column('checksum', String(64), nullable=False))
TABLES = (ResearchWorkVersionModel.__table__, ResearchDelegationModel.__table__, ResearchInvocationModel.__table__)
VERSION = 2


def migrate_architecture(engine):
    signature = {table.name: [(column.name, str(column.type), column.nullable) for column in table.columns]
                 for table in TABLES}
    checksum = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    with engine.begin() as conn:
        if conn.dialect.name == 'sqlite':
            # WHY: 两个应用进程首次启动必须串行检查/建表/登记；同时让DDL可回滚。
            conn.exec_driver_sql('BEGIN IMMEDIATE')
        versions.create(conn, checkfirst=True)
        current = conn.execute(select(versions).where(versions.c.component == 'agent')).mappings().first()
        if current and (current['version'] != VERSION or current['checksum'] != checksum):
            raise ValueError('unsupported agent schema version; explicit migration required')
        for table in TABLES:
            table.create(conn, checkfirst=True)
            actual = {column['name']: column for column in inspect(conn).get_columns(table.name)}
            if set(actual) != {column.name for column in table.columns}:
                raise ValueError('agent schema mismatch: ' + table.name)
            for column in table.columns:
                found = actual[column.name]
                if (found['type']._type_affinity is not column.type._type_affinity
                        or found['nullable'] != column.nullable
                        or getattr(found['type'], 'length', None) != getattr(column.type, 'length', None)):
                    raise ValueError('agent column type mismatch: ' + table.name + '.' + column.name)
            primary = set(inspect(conn).get_pk_constraint(table.name)['constrained_columns'])
            if primary != {column.name for column in table.primary_key.columns}:
                raise ValueError('agent primary key mismatch: ' + table.name)
        constraints = inspect(conn).get_unique_constraints('research_action_invocations')
        if not any(set(c['column_names']) == {'delegation_id', 'step_index'} for c in constraints):
            raise ValueError('invocation step uniqueness is required')
        if current is None:
            conn.execute(versions.insert().values(component='agent', version=VERSION, checksum=checksum))
    return {'component': 'agent', 'version': VERSION, 'checksum': checksum}
