"""实际SQLite备份/升级/副本恢复，不覆盖活跃任务或丢弃费用。"""
import json
import sqlite3
import time

import pytest
from sqlalchemy import create_engine

from app.database.models import ResearchRuntimeModel, ResearchTaskModel, ResearchEventModel
from scripts import validate_v2_deployment as script


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(script, 'ROOT', tmp_path)
    path = tmp_path / 'business.db'
    engine = create_engine('sqlite:///' + str(path))
    for table in (ResearchRuntimeModel.__table__, ResearchTaskModel.__table__, ResearchEventModel.__table__):
        table.create(engine)
    with engine.begin() as db:
        db.execute(ResearchRuntimeModel.__table__.insert().values(session_id='historical', version=7,
            budget_json='{"llm_tokens":1234,"tokens_reserved":21}', state_json='{"topic":"private"}'))
        db.execute(ResearchTaskModel.__table__.insert().values(task_id='old-task', session_id='historical',
            idempotency_key='once', operation='fetch_metadata', expected_version=7, status='running'))
    engine.dispose()
    return path, tmp_path / 'data'


def test_rehearsal_only_restores_copy_without_changing_business(legacy):
    path, output = legacy
    before = script.capture(path)
    result = script.validate(path, output / 'rehearsal')
    assert result['rollback_copy_restored'] and result['status'] == 'passed'
    assert script.capture(path) == before
    assert script.capture(output / 'rehearsal/before-upgrade.db') == before
    assert script.capture(output / 'rehearsal/rollback-rehearsal.db') == before


def test_upgrade_and_interrupted_reconciliation_keep_spend_and_proof(legacy):
    path, output = legacy
    baseline = script.capture(path)
    result = script.validate(path, output / 'apply', apply=True)
    assert result['original_data_preserved'] and result['interrupted_tasks_reconciled'] == 1
    assert script.capture(output / 'apply/before-upgrade.db') == baseline
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT version,budget_json,state_json FROM research_runtime').fetchone() == (
            7, '{"llm_tokens":1234,"tokens_reserved":21}', '{"topic":"private"}')
        assert db.execute('SELECT status FROM research_tasks').fetchone()[0] == 'interrupted'
        assert db.execute('SELECT event_type FROM research_events').fetchone()[0] == 'startup_task_interrupted'
        assert db.execute("SELECT count(*) FROM sqlite_master WHERE name='research_delegations'").fetchone()[0] == 1
    again = script.validate(path, output / 'again', apply=True)
    assert again['interrupted_tasks_reconciled'] == 0


def test_live_lease_prevents_backup_migration_and_reconciliation(legacy):
    path, output = legacy
    with sqlite3.connect(path) as db:
        db.execute('UPDATE research_runtime SET owner=?,lease_until=?', ('live', time.time() + 300))
    before = script.capture(path)
    with pytest.raises(RuntimeError, match='active_execution'):
        script.validate(path, output / 'refused', apply=True)
    assert script.capture(path) == before
    assert not (output / 'refused/before-upgrade.db').exists()


def test_initializer_failure_keeps_backup_and_does_not_leak_error(legacy):
    path, output = legacy
    before = script.capture(path)
    def fail():
        raise ValueError('private credential must not appear')
    with pytest.raises(ValueError):
        script.validate(path, output / 'failed', apply=True, initializer=fail)
    assert script.capture(path) == before
    assert script.capture(output / 'failed/before-upgrade.db') == before
    report = json.loads((output / 'failed/deployment.json').read_text(encoding='utf-8'))
    assert report['status'] == 'failed' and report['stage'] == 'actual_upgrade'
    assert 'credential' not in json.dumps(report)


def test_output_reuse_is_refused(legacy):
    path, output = legacy
    script.validate(path, output / 'once')
    with pytest.raises(ValueError, match='output_must_be_new'):
        script.validate(path, output / 'once', apply=True)
