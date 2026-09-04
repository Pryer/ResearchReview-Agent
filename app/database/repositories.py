"""数据仓储层。

封装对 Paper、PaperCard、Review 表的 CRUD 操作，
向上层服务提供清晰的接口，隔离 SQL 细节。
"""

from __future__ import annotations

import json
from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.logger import get_logger
from app.database.models import (
    Paper,
    PaperCardModel,
    ResearchJobModel,
    ResearchSessionModel,
    ReviewModel,
)
from app.schemas.paper_schema import PaperCard, PaperMetadata
from app.schemas.review_schema import LiteratureReview

logger = get_logger(__name__)


# ============================================================
# Paper 仓储
# ============================================================
class PaperRepository:
    """论文元数据仓储。"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def save(self, paper: PaperMetadata) -> Paper:
        """保存或更新论文元数据。"""
        existing = self.db.get(Paper, paper.paper_id)
        authors_json = json.dumps(paper.authors, ensure_ascii=False)
        keywords_json = (
            json.dumps(paper.keywords, ensure_ascii=False) if paper.keywords else None
        )
        citation_by_source_json = (
            json.dumps(paper.citation_count_by_source, ensure_ascii=False)
            if paper.citation_count_by_source
            else None
        )

        if existing:
            if paper.title:
                existing.title = paper.title
            if paper.authors:
                existing.authors = authors_json
            if paper.year is not None:
                existing.year = paper.year
            if paper.venue:
                existing.venue = paper.venue
            if paper.abstract:
                existing.abstract = paper.abstract
            existing.doi = paper.doi or existing.doi
            existing.arxiv_id = paper.arxiv_id or existing.arxiv_id
            existing.url = paper.url or existing.url
            existing.pdf_url = paper.pdf_url or existing.pdf_url
            if paper.citation_count is not None:
                existing.citation_count = max(existing.citation_count or 0, paper.citation_count)
            # citation_count_by_source: 合并策略，保留已有来源且取 max
            if citation_by_source_json:
                existing_by_source = {}
                if existing.citation_count_by_source:
                    try:
                        existing_by_source = json.loads(existing.citation_count_by_source)
                    except (json.JSONDecodeError, TypeError):
                        pass
                new_by_source = paper.citation_count_by_source or {}
                merged = {**existing_by_source}
                for source, count in new_by_source.items():
                    merged[source] = max(merged.get(source, 0), count)
                existing.citation_count_by_source = json.dumps(merged, ensure_ascii=False)
            existing.is_open_access = int(bool(existing.is_open_access) or paper.is_open_access)
            if keywords_json:
                existing.keywords = keywords_json
            self.db.flush()
            logger.debug("Updated paper: %s", paper.paper_id)
            return existing

        db_paper = Paper(
            paper_id=paper.paper_id,
            title=paper.title,
            authors=authors_json,
            year=paper.year,
            venue=paper.venue,
            abstract=paper.abstract,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            url=paper.url,
            pdf_url=paper.pdf_url,
            citation_count=paper.citation_count,
            citation_count_by_source=citation_by_source_json,
            source=paper.source,
            is_open_access=int(paper.is_open_access),
            keywords=keywords_json,
        )
        self.db.add(db_paper)
        self.db.flush()
        logger.debug("Saved paper: %s", paper.paper_id)
        return db_paper

    def get_by_id(self, paper_id: str) -> Optional[PaperMetadata]:
        """根据 ID 查询论文。"""
        row = self.db.get(Paper, paper_id)
        if not row:
            return None
        return self._to_schema(row)

    def find_by_title(self, title: str) -> Optional[PaperMetadata]:
        """根据标题精确查询。"""
        stmt = select(Paper).where(Paper.title == title)
        row = self.db.execute(stmt).scalar_one_or_none()
        return self._to_schema(row) if row else None

    def list(self, limit: int = 50) -> List[PaperMetadata]:
        """列出最新论文。"""
        stmt = select(Paper).order_by(Paper.created_at.desc()).limit(limit)
        rows = self.db.execute(stmt).scalars().all()
        return [self._to_schema(r) for r in rows]

    @staticmethod
    def _to_schema(row: Paper) -> PaperMetadata:
        """将 ORM 对象转为 Pydantic Schema。"""
        try:
            authors = json.loads(row.authors) if row.authors else []
        except (json.JSONDecodeError, TypeError):
            authors = []
        try:
            keywords = json.loads(row.keywords) if row.keywords else None
        except (json.JSONDecodeError, TypeError):
            keywords = None
        try:
            citation_by_source = (
                json.loads(row.citation_count_by_source)
                if row.citation_count_by_source
                else None
            )
        except (json.JSONDecodeError, TypeError):
            citation_by_source = None
        return PaperMetadata(
            paper_id=row.paper_id,
            title=row.title,
            authors=authors,
            year=row.year,
            venue=row.venue,
            abstract=row.abstract,
            doi=row.doi,
            arxiv_id=row.arxiv_id,
            url=row.url,
            pdf_url=row.pdf_url,
            citation_count=row.citation_count,
            citation_count_by_source=citation_by_source,
            source=row.source,
            is_open_access=bool(row.is_open_access),
            keywords=keywords,
        )


# ============================================================
# PaperCard 仓储
# ============================================================
class PaperCardRepository:
    """论文卡片仓储。"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def save(self, card: PaperCard) -> PaperCardModel:
        """保存或更新论文卡片。"""
        existing = (
            self.db.query(PaperCardModel)
            .filter(PaperCardModel.paper_id == card.paper_id)
            .first()
        )
        data = dict(
            title=card.title,
            authors=json.dumps(card.authors, ensure_ascii=False),
            year=card.year,
            venue=card.venue,
            doi=card.doi,
            url=card.url,
            publication_type=card.publication_type,
            peer_review_status=card.peer_review_status,
            evidence_level=card.evidence_level,
            research_problem=card.research_problem,
            study_design=card.study_design,
            sample_size=card.sample_size,
            data_modalities=json.dumps(card.data_modalities, ensure_ascii=False),
            behavior_categories=json.dumps(card.behavior_categories, ensure_ascii=False),
            method=card.method,
            dataset=card.dataset,
            metrics=json.dumps(card.metrics, ensure_ascii=False),
            results=card.results,
            contributions=json.dumps(card.contributions, ensure_ascii=False),
            limitations=json.dumps(card.limitations, ensure_ascii=False),
            relevance_reason=card.relevance_reason,
            evidence_source=card.evidence_source,
            evidence_spans=json.dumps(
                [span.model_dump() for span in card.evidence_spans],
                ensure_ascii=False,
            ),
            field_evidence=json.dumps(card.field_evidence, ensure_ascii=False),
            evidence_state=json.dumps(card.evidence_state.model_dump(mode="json"), ensure_ascii=False),
            field_claims=json.dumps(
                {
                    field: [claim.model_dump(mode="json") for claim in claims]
                    for field, claims in card.field_claims.items()
                },
                ensure_ascii=False,
            ),
            unsupported_fields=json.dumps(card.unsupported_fields, ensure_ascii=False),
            quality_status=card.quality_status,
            quality_issues=json.dumps(card.quality_issues, ensure_ascii=False),
            relation_type=card.relation_type,
        )
        if existing:
            for k, v in data.items():
                setattr(existing, k, v)
            self.db.flush()
            return existing

        db_card = PaperCardModel(paper_id=card.paper_id, **data)
        self.db.add(db_card)
        self.db.flush()
        return db_card

    def get_by_paper_id(self, paper_id: str) -> Optional[PaperCard]:
        """根据 paper_id 查询卡片。"""
        row = (
            self.db.query(PaperCardModel)
            .filter(PaperCardModel.paper_id == paper_id)
            .first()
        )
        return self._to_schema(row) if row else None

    def list_all(self, limit: int = 100) -> List[PaperCard]:
        """列出所有卡片。"""
        rows = (
            self.db.query(PaperCardModel)
            .order_by(PaperCardModel.created_at.desc())
            .limit(limit)
            .all()
        )
        return [self._to_schema(r) for r in rows]

    @staticmethod
    def _to_schema(row: PaperCardModel) -> PaperCard:
        """ORM → Schema。"""
        def _load_json(text: str, default):
            try:
                return json.loads(text) if text else default
            except (json.JSONDecodeError, TypeError):
                return default

        return PaperCard(
            paper_id=row.paper_id,
            title=row.title,
            authors=_load_json(getattr(row, "authors", ""), []),
            year=row.year,
            venue=row.venue,
            doi=getattr(row, "doi", None),
            url=getattr(row, "url", None),
            publication_type=getattr(row, "publication_type", "unknown"),
            peer_review_status=getattr(row, "peer_review_status", "unknown"),
            evidence_level=getattr(row, "evidence_level", "unknown"),
            research_problem=row.research_problem,
            study_design=getattr(row, "study_design", ""),
            sample_size=getattr(row, "sample_size", None),
            data_modalities=_load_json(getattr(row, "data_modalities", ""), []),
            behavior_categories=_load_json(getattr(row, "behavior_categories", ""), []),
            method=row.method,
            dataset=row.dataset,
            metrics=_load_json(row.metrics, []),
            results=row.results,
            contributions=_load_json(row.contributions, []),
            limitations=_load_json(row.limitations, []),
            relevance_reason=row.relevance_reason,
            evidence_source=row.evidence_source,
            evidence_spans=_load_json(getattr(row, "evidence_spans", ""), []),
            field_evidence=_load_json(getattr(row, "field_evidence", ""), {}),
            evidence_state=_load_json(getattr(row, "evidence_state", ""), {}),
            field_claims=_load_json(getattr(row, "field_claims", ""), {}),
            unsupported_fields=_load_json(getattr(row, "unsupported_fields", ""), []),
            quality_status=getattr(row, "quality_status", "partial"),
            quality_issues=_load_json(getattr(row, "quality_issues", ""), []),
            relation_type=getattr(row, "relation_type", None),
        )


# ============================================================
# Review 仓储
# ============================================================
class ReviewRepository:
    """文献综述仓储。"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def save(self, review: LiteratureReview, paper_ids: List[str]) -> ReviewModel:
        """保存综述。"""
        row = ReviewModel(
            topic=review.topic,
            review_text=review.full_text,
            sections_json=json.dumps(
                [s.dict() if hasattr(s, "dict") else s for s in review.sections],
                ensure_ascii=False,
            ),
            references_json=json.dumps(review.references, ensure_ascii=False),
            paper_ids_json=json.dumps(paper_ids, ensure_ascii=False),
            citation_validation_json=(
                json.dumps(review.citation_validation, ensure_ascii=False)
                if review.citation_validation
                else None
            ),
            claim_verification_json=(
                json.dumps(review.claim_verification, ensure_ascii=False)
                if review.claim_verification
                else None
            ),
            language=review.language,
            citation_style=review.citation_style,
        )
        self.db.add(row)
        self.db.flush()
        logger.info("Saved review: id=%s topic=%s", row.id, row.topic)
        return row

    def get_by_id(self, review_id: int) -> Optional[ReviewModel]:
        """根据 ID 查询综述。"""
        return self.db.get(ReviewModel, review_id)

    def list(self, limit: int = 20) -> List[ReviewModel]:
        """列出最新综述。"""
        stmt = select(ReviewModel).order_by(ReviewModel.created_at.desc()).limit(limit)
        return list(self.db.execute(stmt).scalars().all())


# ============================================================
# Research Session 仓储
# ============================================================
class ResearchSessionRepository:
    """多轮研究会话状态仓储。"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def save(
        self,
        session_id: str,
        status: str,
        original_query: str,
        state: dict,
        clarification: dict | None = None,
    ) -> ResearchSessionModel:
        row = self.db.get(ResearchSessionModel, session_id)
        values = {
            "status": status,
            "original_query": original_query,
            "state_json": json.dumps(state or {}, ensure_ascii=False),
            "clarification_json": (
                json.dumps(clarification, ensure_ascii=False)
                if clarification is not None
                else None
            ),
        }
        if row:
            for key, value in values.items():
                setattr(row, key, value)
        else:
            row = ResearchSessionModel(session_id=session_id, **values)
            self.db.add(row)
        self.db.flush()
        return row

    def get(self, session_id: str) -> dict | None:
        row = self.db.get(ResearchSessionModel, session_id)
        if not row:
            return None

        def load(value: str | None, default):
            try:
                return json.loads(value) if value else default
            except (json.JSONDecodeError, TypeError):
                return default

        return {
            "session_id": row.session_id,
            "status": row.status,
            "original_query": row.original_query,
            "state": load(row.state_json, {}),
            "clarification": load(row.clarification_json, None),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }


# ============================================================
# Research Job 仓储
# ============================================================
class ResearchJobRepository:
    """后台研究任务仓储。"""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        job_id: str,
        session_id: str,
        request: dict,
        operation: str = "run",
    ) -> ResearchJobModel:
        row = ResearchJobModel(
            job_id=job_id,
            session_id=session_id,
            operation=operation,
            status="queued",
            request_json=json.dumps(request or {}, ensure_ascii=False),
        )
        self.db.add(row)
        self.db.flush()
        return row

    def get_row(self, job_id: str) -> ResearchJobModel | None:
        return self.db.get(ResearchJobModel, job_id)

    def get(self, job_id: str) -> dict | None:
        row = self.get_row(job_id)
        if not row:
            return None

        def load(value: str | None, default):
            try:
                return json.loads(value) if value else default
            except (json.JSONDecodeError, TypeError):
                return default

        return {
            "job_id": row.job_id,
            "session_id": row.session_id,
            "operation": row.operation,
            "status": row.status,
            "request": load(row.request_json, {}),
            "result": load(row.result_json, None),
            "current_step": row.current_step,
            "progress_current": row.progress_current,
            "progress_total": row.progress_total,
            "error": row.error,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def update(self, job_id: str, **values) -> ResearchJobModel | None:
        row = self.get_row(job_id)
        if not row:
            return None
        if "result" in values:
            result = values.pop("result")
            values["result_json"] = (
                json.dumps(result, ensure_ascii=False) if result is not None else None
            )
        for key, value in values.items():
            if hasattr(row, key):
                setattr(row, key, value)
        self.db.flush()
        return row

    def update_status_if_in(
        self,
        job_id: str,
        from_statuses: set[str],
        **values,
    ) -> int:
        """条件状态迁移（CAS）：仅当当前 status ∈ from_statuses 时更新。

        返回受影响行数；为 0 表示竞争失败（cancel 与 worker 终态写入同时
        发生时，输掉的一方必须放弃写入），避免 last-writer-wins 把任务
        永久卡在 cancel_requested 等中间态。updated_at 由列级 onupdate
        自动刷新。
        """
        if not from_statuses:
            return 0
        if "result" in values:
            result = values.pop("result")
            values["result_json"] = (
                json.dumps(result, ensure_ascii=False) if result is not None else None
            )
        stmt = (
            update(ResearchJobModel)
            .where(
                ResearchJobModel.job_id == job_id,
                ResearchJobModel.status.in_(from_statuses),
            )
            .values(**{k: v for k, v in values.items() if hasattr(ResearchJobModel, k)})
            .execution_options(synchronize_session=False)
        )
        affected = self.db.execute(stmt).rowcount or 0
        self.db.flush()
        return int(affected)

    def list_by_statuses(self, statuses: set[str]) -> list[dict]:
        """列出待恢复任务，供应用重启时进行确定性状态修复。"""
        if not statuses:
            return []
        rows = self.db.execute(
            select(ResearchJobModel).where(ResearchJobModel.status.in_(statuses))
        ).scalars().all()
        return [self.get(row.job_id) for row in rows]

    def find_active_for_session(self, session_id: str) -> dict | None:
        """返回同一会话尚未结束的任务，防止并发改写会话状态。"""
        row = self.db.execute(
            select(ResearchJobModel)
            .where(
                ResearchJobModel.session_id == session_id,
                ResearchJobModel.status.in_({"queued", "running", "cancel_requested"}),
            )
            .order_by(ResearchJobModel.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        return self.get(row.job_id) if row else None
