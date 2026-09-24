"""SQLAlchemy 数据库模型。

定义论文、论文卡片、综述的持久化结构。
使用 SQLite 作为 MVP 数据库，可平滑迁移到 PostgreSQL。
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from sqlalchemy import DateTime, Integer, String, Text, Float, Boolean, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类。"""


class Paper(Base):
    """论文元数据表。"""

    __tablename__ = "papers"

    paper_id: Mapped[str] = mapped_column(String(255), primary_key=True, index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    authors: Mapped[str] = mapped_column(Text, default="")  # JSON 序列化
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(String(512), nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    pdf_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    citation_count_by_source: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON 序列化 dict[str, int]
    source: Mapped[str] = mapped_column(String(64), default="unknown")
    is_open_access: Mapped[int] = mapped_column(Integer, default=0)
    keywords: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON 序列化
    full_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class PaperCardModel(Base):
    """论文卡片表。"""

    __tablename__ = "paper_cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    authors: Mapped[str] = mapped_column(Text, default="")
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(String(512), nullable=True)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_type: Mapped[str] = mapped_column(String(64), default="unknown")
    peer_review_status: Mapped[str] = mapped_column(String(64), default="unknown")
    evidence_level: Mapped[str] = mapped_column(String(64), default="unknown")
    research_problem: Mapped[str] = mapped_column(Text, default="")
    study_design: Mapped[str] = mapped_column(Text, default="")
    sample_size: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_modalities: Mapped[str] = mapped_column(Text, default="")
    behavior_categories: Mapped[str] = mapped_column(Text, default="")
    method: Mapped[str] = mapped_column(Text, default="")
    dataset: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[str] = mapped_column(Text, default="")  # JSON 序列化
    results: Mapped[str | None] = mapped_column(Text, nullable=True)
    contributions: Mapped[str] = mapped_column(Text, default="")  # JSON 序列化
    limitations: Mapped[str] = mapped_column(Text, default="")  # JSON 序列化
    relevance_reason: Mapped[str] = mapped_column(Text, default="")
    evidence_source: Mapped[str] = mapped_column(String(32), default="metadata")
    evidence_spans: Mapped[str] = mapped_column(Text, default="")  # JSON 序列化
    field_evidence: Mapped[str] = mapped_column(Text, default="")  # JSON 序列化
    evidence_state: Mapped[str] = mapped_column(Text, default="")  # JSON 序列化
    field_claims: Mapped[str] = mapped_column(Text, default="")  # JSON 序列化
    unsupported_fields: Mapped[str] = mapped_column(Text, default="")  # JSON 序列化
    quality_status: Mapped[str] = mapped_column(String(32), default="partial")
    quality_issues: Mapped[str] = mapped_column(Text, default="")  # JSON 序列化
    relation_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReviewModel(Base):
    """文献综述表。"""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic: Mapped[str] = mapped_column(Text, default="")
    review_text: Mapped[str] = mapped_column(Text, default="")
    sections_json: Mapped[str] = mapped_column(Text, default="")  # JSON 序列化
    references_json: Mapped[str] = mapped_column(Text, default="")  # JSON 序列化
    paper_ids_json: Mapped[str] = mapped_column(Text, default="")  # JSON 序列化
    citation_validation_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    claim_verification_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_style: Mapped[str] = mapped_column(String(32), default="gbt7714")
    language: Mapped[str] = mapped_column(String(16), default="zh")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ResearchSessionModel(Base):
    """可恢复的多轮研究会话。"""

    __tablename__ = "research_sessions"

    # 长度与 AgentRequest.session_id 的 max_length=128 对齐，避免客户端
    # 传入合法长 ID 时落库失败（内部生成的 uuid hex 为 32 字符）。
    session_id: Mapped[str] = mapped_column(String(128), primary_key=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="created", index=True)
    original_query: Mapped[str] = mapped_column(Text, default="")
    state_json: Mapped[str] = mapped_column(Text, default="{}")
    clarification_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class ResearchArtifactModel(Base):
    """会话范围内可寻址的研究资料和上下文快照。"""

    __tablename__ = "research_artifacts"

    artifact_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    artifact_type: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    provenance_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ResearchJobModel(Base):
    """后台研究任务及其可取消执行状态。"""

    __tablename__ = "research_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    # 与 research_sessions.session_id 同步放宽到 128
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    operation: Mapped[str] = mapped_column(String(32), default="run")
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    request_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_step: Mapped[str | None] = mapped_column(String(128), nullable=True)
    progress_current: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=14)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class ResearchRuntimeModel(Base):
    """执行租约、当前检查点和消耗；与公开会话快照分离。"""
    __tablename__ = "research_runtime"
    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, default=0)
    owner: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_until: Mapped[float] = mapped_column(Float, default=0)
    cancelled: Mapped[bool] = mapped_column(Boolean, default=False)
    budget_json: Mapped[str] = mapped_column(Text, default="{}")
    state_json: Mapped[str] = mapped_column(Text, default="{}")


class ResearchTaskModel(Base):
    __tablename__ = "research_tasks"
    __table_args__ = (UniqueConstraint("session_id", "idempotency_key"),)
    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(64))
    operation: Mapped[str] = mapped_column(String(64))
    expected_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="running")
    result_json: Mapped[str] = mapped_column(Text, default="{}")


class ResearchAttemptModel(Base):
    __tablename__ = "research_attempts"
    attempt_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running")
    reserved_tokens: Mapped[int] = mapped_column(Integer)
    usage_json: Mapped[str] = mapped_column(Text, default="{}")


class ResearchCheckpointModel(Base):
    __tablename__ = "research_checkpoints"
    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    state_json: Mapped[str] = mapped_column(Text)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class ResearchEventModel(Base):
    __tablename__ = "research_events"
    __table_args__ = (UniqueConstraint("session_id", "event_key"),)
    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    event_key: Mapped[str] = mapped_column(String(128))
    event_type: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str] = mapped_column(Text)


class ResearchMemoryModel(Base):
    __tablename__ = "research_memory_cursors"
    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    cursor: Mapped[int] = mapped_column(Integer, default=0)
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    summary_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
