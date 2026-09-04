"""数据库连接与会话管理。

提供 SQLAlchemy engine、sessionmaker 以及 FastAPI 依赖 ``get_db``。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

settings = get_settings()

# SQLite 需要 check_same_thread=False 以支持多线程（如 Streamlit）
_is_sqlite = settings.database_url.startswith("sqlite")
connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=settings.app_debug,
    pool_pre_ping=True,
)

if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_write_friendly_pragma(dbapi_connection, connection_record):
        """降低多线程写锁冲突的影响。

        后台任务工作线程逐节点 commit 进度，与 API/前端进程并发写同一库；
        pysqlite 默认 delete journal + 5s busy timeout，瞬时锁竞争即可让
        长任务整体失败。WAL 允许读写并行，busy_timeout 给写入方 30s 等待。
        """
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """建表（如果不存在）。"""
    from app.database.models import Base

    Base.metadata.create_all(bind=engine)
    _migrate_sqlite_evidence_card_columns()
    logger.info("Database initialized: %s", settings.database_url)


def _migrate_sqlite_evidence_card_columns() -> None:
    """为既有 SQLite 库补充 Evidence Card 列。

    新部署由 ``create_all`` 直接建列；该函数只处理已经存在的旧版 SQLite 表。
    """
    if not settings.database_url.startswith("sqlite"):
        return
    inspector = inspect(engine)
    if "paper_cards" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("paper_cards")}
    additions = {
        "evidence_spans": "TEXT DEFAULT ''",
        "field_evidence": "TEXT DEFAULT ''",
        "evidence_state": "TEXT DEFAULT ''",
        "field_claims": "TEXT DEFAULT ''",
        "unsupported_fields": "TEXT DEFAULT ''",
        "quality_status": "VARCHAR(32) DEFAULT 'partial'",
        "quality_issues": "TEXT DEFAULT ''",
        "relation_type": "VARCHAR(64)",
        "authors": "TEXT DEFAULT ''",
        "doi": "VARCHAR(255)",
        "url": "TEXT",
        "publication_type": "VARCHAR(64) DEFAULT 'unknown'",
        "peer_review_status": "VARCHAR(64) DEFAULT 'unknown'",
        "evidence_level": "VARCHAR(64) DEFAULT 'unknown'",
        "study_design": "TEXT DEFAULT ''",
        "sample_size": "TEXT",
        "data_modalities": "TEXT DEFAULT ''",
        "behavior_categories": "TEXT DEFAULT ''",
    }
    with engine.begin() as connection:
        for column, definition in additions.items():
            if column not in existing:
                connection.exec_driver_sql(
                    f"ALTER TABLE paper_cards ADD COLUMN {column} {definition}"
                )
                logger.info("Migrated paper_cards column: %s", column)

    review_columns = {column["name"] for column in inspector.get_columns("reviews")}
    if "claim_verification_json" not in review_columns:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "ALTER TABLE reviews ADD COLUMN claim_verification_json TEXT"
            )
        logger.info("Migrated reviews column: claim_verification_json")


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖注入使用的数据库会话生成器。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """上下文管理器形式的数据库会话，供非 FastAPI 场景（如 Streamlit、CLI）使用。"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
