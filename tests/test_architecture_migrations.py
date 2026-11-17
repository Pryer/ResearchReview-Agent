"""数据库副本升级与备份回退；不对用户数据库执行降级或删除。"""
import sqlite3
import pytest
from sqlalchemy import create_engine, inspect, select, update
from app.database.models import ResearchRuntimeModel
from app.database.architecture_migrations import migrate_architecture, versions


def test_additive_upgrade_is_repeatable_preserves_old_data_and_backup(tmp_path):
    path, backup = tmp_path / 'legacy.db', tmp_path / 'before-upgrade.db'
    engine = create_engine('sqlite:///' + str(path))
    ResearchRuntimeModel.__table__.create(engine)
    with engine.begin() as conn:
        conn.execute(ResearchRuntimeModel.__table__.insert().values(session_id='legacy', version=7,
            budget_json='{"llm_tokens":123}', state_json='{"topic":"preserved"}'))
    # 停写副本使用sqlite backup保留一致快照，回退只发生在此测试的临时文件。
    with sqlite3.connect(path) as source, sqlite3.connect(backup) as target:
        source.backup(target)
    first = migrate_architecture(engine)
    assert migrate_architecture(engine) == first
    with engine.connect() as conn:
        row = conn.execute(select(ResearchRuntimeModel.__table__)).mappings().one()
        assert row['version'] == 7 and row['budget_json'] == '{"llm_tokens":123}'
        assert conn.execute(select(versions)).mappings().one()['version'] == 2
    assert {'research_delegations', 'research_action_invocations'} <= set(inspect(engine).get_table_names())
    engine.dispose()
    with sqlite3.connect(backup) as source, sqlite3.connect(path) as target:
        source.backup(target)
    engine = create_engine('sqlite:///' + str(path))
    assert 'research_delegations' not in inspect(engine).get_table_names()
    with engine.connect() as conn:
        assert conn.execute(select(ResearchRuntimeModel.__table__.c.version)).scalar_one() == 7
    engine.dispose()


def test_unknown_schema_version_rejected_without_touching_work(tmp_path):
    engine = create_engine('sqlite:///' + str(tmp_path / 'unknown.db'))
    migrate_architecture(engine)
    with engine.begin() as conn:
        conn.execute(update(versions).values(version=99))
    with pytest.raises(ValueError, match='unsupported agent schema'):
        migrate_architecture(engine)
    with engine.connect() as conn:
        assert conn.execute(select(versions.c.version)).scalar_one() == 99
    engine.dispose()


def test_malformed_existing_schema_is_not_silently_accepted(tmp_path):
    engine = create_engine('sqlite:///' + str(tmp_path / 'broken.db'))
    with engine.begin() as conn:
        conn.exec_driver_sql('CREATE TABLE research_delegations (delegation_id TEXT PRIMARY KEY)')
    with pytest.raises(ValueError, match='schema mismatch'):
        migrate_architecture(engine)
    assert set(inspect(engine).get_table_names()) == {'research_delegations'}
    engine.dispose()


def test_two_startups_install_one_consistent_version(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    path = tmp_path / 'startup.db'
    barrier = Barrier(2)
    def migrate():
        engine = create_engine('sqlite:///' + str(path))
        try:
            barrier.wait(timeout=5)
            return migrate_architecture(engine)
        finally:
            engine.dispose()
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(migrate) for _ in range(2)]
        assert tasks[0].result(timeout=10) == tasks[1].result(timeout=10)
    engine = create_engine('sqlite:///' + str(path))
    with engine.connect() as conn:
        assert len(conn.execute(select(versions)).all()) == 1
    engine.dispose()
