"""租约领取等待写锁时，截止时间必须按真正执行SQL时的时钟生成。"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from sqlalchemy import create_engine, event, func, update
from sqlalchemy.orm import Session
from app.database.models import Base, ResearchRuntimeModel
from app.database.runtime_repository import ResearchRuntimeRepository


@pytest.mark.parametrize('initially_expired', [False, True], ids=['expires_while_waiting', 'already_expired'])
def test_acquire_waiting_for_sqlite_writer_uses_database_execution_time(tmp_path, monkeypatch, initially_expired):
    logical_time = [time.time()]
    entered = threading.Event()
    engine = create_engine('sqlite:///' + str(tmp_path / 'clock.db'), connect_args={'check_same_thread': False, 'timeout': 5})
    def connect(connection, record):
        connection.create_function('race_clock', 0, lambda: logical_time[0])
    event.listen(engine, 'connect', connect)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(ResearchRuntimeRepository, '_start_heartbeat', lambda self: None)
    monkeypatch.setattr('app.database.runtime_repository.time.time', lambda: logical_time[0])
    monkeypatch.setattr('app.database.runtime_repository.lease_clock', lambda db: func.race_clock())
    with Session(engine) as db:
        original = ResearchRuntimeRepository(db, 'clock', ttl=10)
        original.acquire({}, {})
        if initially_expired:
            db.execute(update(ResearchRuntimeModel).values(lease_until=0))
            db.commit()
    def before_sql(conn, cursor, sql, params, context, many):
        if threading.current_thread().name.startswith('claim') and sql.startswith('UPDATE research_runtime SET owner='):
            entered.set()
    def claim():
        with Session(engine) as db:
            runtime = ResearchRuntimeRepository(db, 'clock', ttl=10)
            runtime.acquire({}, {})
            row = runtime.check()
            assert row.lease_until == logical_time[0] + 10
            runtime.release({})
    event.listen(engine, 'before_cursor_execute', before_sql)
    try:
        with Session(engine) as writer, ThreadPoolExecutor(max_workers=1, thread_name_prefix='claim') as pool:
            writer.execute(update(ResearchRuntimeModel).values(version=ResearchRuntimeModel.version))
            task = pool.submit(claim)
            try:
                assert entered.wait(5)
                logical_time[0] += 20
            finally:
                writer.commit()
            task.result(timeout=10)
    finally:
        event.remove(engine, 'before_cursor_execute', before_sql)
        engine.dispose()
