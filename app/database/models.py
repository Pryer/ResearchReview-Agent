"""SQLAlchemy 数据库模型。

定义论文、论文卡片、综述的持久化结构。
使用 SQLite 作为 MVP 数据库，可平滑迁移到 PostgreSQL。
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from sqlalchemy import DateTime, Integer, String, Text
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
