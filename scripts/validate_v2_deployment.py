"""备份、只读清点与v2加法升级；仅在数据库副本演练覆盖恢复。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def capture(path, baseline=None):
    result = {}
    with sqlite3.connect(path) as db:
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for table in (baseline or dict.fromkeys(tables)):
            columns = baseline[table]['columns'] if baseline else [r[1] for r in db.execute('PRAGMA table_info(' + quoted(table) + ')')]
            keys = sorted((r for r in db.execute('PRAGMA table_info(' + quoted(table) + ')') if r[5]), key=lambda r: r[5])
            order = ','.join(quoted(r[1]) for r in keys) or 'rowid'
            digest, count = hashlib.sha256(), 0
            for row in db.execute('SELECT ' + ','.join(map(quoted, columns)) + ' FROM ' + quoted(table) + ' ORDER BY ' + order):
                digest.update(json.dumps(row, ensure_ascii=True, default=str).encode())
                count += 1
            result[table] = {'columns': columns, 'rows': count, 'sha256': digest.hexdigest()}
    return result


def assert_quiescent(db):
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'research_runtime' in tables:
        active = db.execute("SELECT count(*) FROM research_runtime WHERE owner IS NOT NULL AND lease_until > (julianday('now')-2440587.5)*86400.0").fetchone()[0]
        if active:
            raise RuntimeError('active_execution_refuses_upgrade')
    if 'research_jobs' in tables and db.execute("SELECT count(*) FROM research_jobs WHERE status IN ('queued','running','cancel_requested')").fetchone()[0]:
        raise RuntimeError('active_job_refuses_upgrade')


def backup(source, target):
    with sqlite3.connect(source) as origin, sqlite3.connect(target) as destination:
        origin.backup(destination)


def upgrade(path):
    from sqlalchemy import create_engine
    from app.database.architecture_migrations import migrate_architecture
    from app.database.models import Base
    engine = create_engine('sqlite:///' + path.as_posix())
    try:
        migrate_architecture(engine)
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()


def reconcile_interrupted(path):
    with sqlite3.connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        assert_quiescent(db)
        pending = db.execute("SELECT task_id,session_id FROM research_tasks WHERE status='running'").fetchall()
        for task_id, session_id in pending:
            # WHY: 只收敛无活跃执行的动作状态；费用预留/未知回执仍由下一次acquire结算，不能清零。
            db.execute("UPDATE research_tasks SET status='interrupted' WHERE task_id=? AND status='running'", (task_id,))
            db.execute('INSERT INTO research_events(session_id,event_key,event_type,payload_json) VALUES (?,?,?,?)',
                (session_id, 'deployment-interrupt:' + task_id, 'startup_task_interrupted', json.dumps({'task_id': task_id, 'reason': 'quiescent_deployment'})))
        return len(pending)


def validate(path, output, *, apply=False, initializer=None):
    path, output = Path(path).resolve(), Path(output).resolve()
    if not path.is_file() or not path.is_relative_to(ROOT):
        raise ValueError('business_database_must_exist_inside_workspace')
    if not output.is_relative_to(ROOT / 'data') or output.exists():
        raise ValueError('output_must_be_new_under_data')
    output.mkdir(parents=True)
    report = {'started_at': datetime.now(ZoneInfo('Asia/Singapore')).isoformat(), 'mode': 'apply' if apply else 'rehearsal',
              'python_environment': Path(sys.executable).parent.name, 'status': 'running', 'stage': 'quiescence'}
    try:
        with sqlite3.connect(path) as db:
            assert_quiescent(db)
        baseline = capture(path)
        saved = output / 'before-upgrade.db'
        backup(path, saved)
        if capture(saved) != baseline:
            raise RuntimeError('concurrent_writes_during_backup')
        report.update(backup_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(), original_table_counts={k: v['rows'] for k,v in baseline.items()})
        report['stage'] = 'rollback_rehearsal'
        copy = output / 'rollback-rehearsal.db'
        backup(saved, copy)
        upgrade(copy)
        with sqlite3.connect(copy) as db:
            db.execute("INSERT INTO research_runtime(session_id,version,lease_until,cancelled,budget_json,state_json) VALUES (?,0,0,0,?,?)",
                       ('synthetic-rehearsal-' + uuid4().hex, '{}', '{"architecture_version":2}'))
        backup(saved, copy)
        report['rollback_copy_restored'] = capture(copy) == baseline
        if not report['rollback_copy_restored']:
            raise RuntimeError('rollback_copy_mismatch')
        report['stage'] = 'actual_upgrade' if apply else 'complete'
        if apply:
            with sqlite3.connect(path) as db:
                assert_quiescent(db)
            (initializer or (lambda: upgrade(path)))()
            report['original_data_preserved'] = capture(path, baseline) == baseline
            if not report['original_data_preserved']:
                raise RuntimeError('existing_data_changed_during_upgrade')
            report['interrupted_tasks_reconciled'] = reconcile_interrupted(path)
        report.update(status='passed', stage='complete')
    except Exception as exc:
        known = {'active_execution_refuses_upgrade', 'active_job_refuses_upgrade', 'concurrent_writes_during_backup',
                 'rollback_copy_mismatch', 'existing_data_changed_during_upgrade'}
        report.update(status='failed', error_type=type(exc).__name__,
                      reason=str(exc) if str(exc) in known else 'see_private_diagnostics')
        raise
    finally:
        report['finished_at'] = datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
        (output / 'deployment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    from app.database.db import engine, init_db
    if engine.dialect.name != 'sqlite':
        parser.error('only the current SQLite business database is supported')
    try:
        result = validate(Path(engine.url.database), args.output_dir, apply=args.apply, initializer=init_db)
        print(json.dumps({k: result[k] for k in ('status','mode','rollback_copy_restored','original_data_preserved','interrupted_tasks_reconciled') if k in result}))
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
