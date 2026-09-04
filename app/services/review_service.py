"""综述服务。

封装文献综述生成的业务流程。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.logger import get_logger
from app.database.repositories import PaperCardRepository, PaperRepository, ReviewRepository
from app.schemas.paper_schema import PaperCard
from app.schemas.review_schema import LiteratureReview, ReviewRequest, ReviewSection
from app.services.llm_service import LLMService
from app.services.paper_service import PaperService
from app.tools.cluster_papers import cluster_papers
from app.tools.extract_paper_card import extract_paper_card
from app.tools.generate_citation import generate_and_validate_citations
from app.tools.rank_papers import deduplicate_and_rank
from app.tools.search_papers import search_papers
from app.tools.verify_claims import verify_review_claims

logger = get_logger(__name__)


def _write_narrative_review(
    topic: str,
    card_dicts: list[dict[str, Any]],
    cluster_result: dict[str, Any],
    llm,
    search_report: dict[str, Any],
    language: str = "zh",
) -> str:
    """兼容旧ReviewService，但统一使用四交付物的新写作链路。"""
    from app.agent.deliverable_router import check_deliverable_readiness
    from app.agent.writing_plan import build_writing_plan
    from app.schemas.deliverable_schema import CoreDeliverableType
    from app.tools.synthesize_themes import synthesize_themes
    from app.tools.validate_deliverable import validate_deliverable
    from app.tools.verify_claims import build_evidence_quality_report
    from app.tools.write_deliverable import write_deliverable

    state = {
        "topic": topic,
        "paper_cards": card_dicts,
        "clusters": cluster_result.get("clusters") or [],
        "dynamic_taxonomy": cluster_result.get("dynamic_taxonomy") or {},
        "taxonomy_validation": cluster_result.get("taxonomy_validation") or {},
        "search_report": search_report,
        "language": language,
        "evidence_quality_report": build_evidence_quality_report(card_dicts),
    }
    state["theme_synthesis"] = synthesize_themes(
        card_dicts, state["dynamic_taxonomy"]
    )
    readiness = check_deliverable_readiness(
        CoreDeliverableType.NARRATIVE_REVIEW, state, phase="post_evidence"
    )
    if not readiness.ready:
        reasons = "；".join(readiness.insufficient_evidence or readiness.missing_inputs)
        return (
            "## 叙述性综述初稿暂未生成\n\n"
            f"当前证据尚不满足完整叙述性综述初稿的条件：{reasons}。"
            "建议改为研究现状，或补充更多已获得摘要/全文的高相关论文。"
        )
    plan = build_writing_plan(CoreDeliverableType.NARRATIVE_REVIEW, state)
    text = write_deliverable(plan, state, llm=llm)
    validation = validate_deliverable(text, plan, state)
    if not validation.get("valid"):
        return (
            "> **交付物结构质量检查未通过：** "
            + "；".join(validation.get("errors") or [])
            + "\n\n"
            + text
        )
    return text


class ReviewService:
    """文献综述生成业务流程。"""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.paper_repo = PaperRepository(db)
        self.card_repo = PaperCardRepository(db)
        self.review_repo = ReviewRepository(db)
        self.llm = LLMService()
        self.paper_service = PaperService(db)

    def generate_review(self, request: ReviewRequest) -> LiteratureReview:
        """根据主题生成完整综述。

        完整流程：检索 → 排序 → 详情 → PDF → PaperCard → 聚类 → 综述 → 引用校验。
        """
        if request.review_type == "systematic":
            raise ValueError("当前流程不具备系统综述所需的双人筛选与偏倚评价，不能标记为 systematic")

        # 1-2. 检索 + 排序
        from app.schemas.paper_schema import PaperSearchRequest
        search_req = PaperSearchRequest(
            query=request.topic,
            start_year=request.start_year,
            end_year=request.end_year,
            max_results=request.max_papers * 2,
        )
        papers = self.paper_service.search_and_rank(search_req)

        # 3. 详情
        papers = self.paper_service.fetch_details(papers)

        # 4-5. PDF + PaperCard
        cards = self.paper_service.build_paper_cards(papers, request.topic)

        # 6. 聚类
        card_dicts = [c.model_dump(mode="json") for c in cards]
        cluster_result = cluster_papers(
            card_dicts, llm=self.llm, topic=request.topic
        )

        # 7. 综述
        review_text = _write_narrative_review(
            topic=request.topic,
            card_dicts=card_dicts,
            cluster_result=cluster_result,
            llm=self.llm,
            search_report={
                "start_year": request.start_year,
                "end_year": request.end_year,
                "writing_pool_count": len(card_dicts),
                "not_performed": ["双人独立筛选", "PRISMA流程", "偏倚风险评价"],
            },
            language=request.language,
        )

        # 8. 引用
        citation_result = generate_and_validate_citations(
            review_text=review_text,
            paper_cards=papers,
            citation_style=request.citation_style,
            llm=self.llm,
        )
        claim_verification = verify_review_claims(review_text, card_dicts, llm=self.llm)

        # 组装
        review = LiteratureReview(
            topic=request.topic,
            language=request.language,
            citation_style=request.citation_style,
            sections=[ReviewSection(title="全文", content=review_text)],
            references=citation_result["references"],
            paper_cards=card_dicts,
            citation_validation=citation_result["validation"],
            claim_verification=claim_verification,
        )

        # 持久化
        paper_ids = [c.paper_id for c in cards]
        self.review_repo.save(review, paper_ids)
        self.db.commit()

        return review

    def generate_review_from_paper_ids(self, paper_ids: List[str]) -> Optional[LiteratureReview]:
        """基于指定论文 ID 列表生成综述。"""
        papers = []
        for pid in paper_ids:
            p = self.paper_repo.get_by_id(pid)
            if p:
                papers.append(p.model_dump())

        if not papers:
            return None

        card_dicts = []
        for p in papers:
            card = self.paper_service.build_paper_card(p["paper_id"])
            if card:
                card_dicts.append(card.model_dump(mode="json"))

        cluster_result = cluster_papers(
            card_dicts, llm=self.llm, topic="Selected Papers"
        )
        review_text = _write_narrative_review(
            topic="Selected Papers",
            card_dicts=card_dicts,
            cluster_result=cluster_result,
            llm=self.llm,
            search_report={
                "writing_pool_count": len(card_dicts),
                "not_performed": ["数据库系统检索", "双人独立筛选", "PRISMA流程", "偏倚风险评价"],
            },
        )

        citation_result = generate_and_validate_citations(
            review_text, papers, llm=self.llm,
        )
        claim_verification = verify_review_claims(review_text, card_dicts, llm=self.llm)

        review = LiteratureReview(
            topic="Selected Papers",
            sections=[ReviewSection(title="全文", content=review_text)],
            references=citation_result["references"],
            paper_cards=card_dicts,
            citation_validation=citation_result["validation"],
            claim_verification=claim_verification,
        )
        self.review_repo.save(review, paper_ids)
        self.db.commit()
        return review

    def compare_papers(self, paper_ids: List[str], language: str = "zh") -> str:
        """对比多篇论文。"""
        cards = []
        for pid in paper_ids:
            card = self.card_repo.get_by_paper_id(pid)
            if card:
                cards.append(card.model_dump())

        if not cards:
            return "未找到指定论文的卡片。"

        parts = ["# 论文对比\n"]
        for card in cards:
            parts.append(f"\n## {card.title}")
            parts.append(f"\n- 方法：{card.method}")
            parts.append(f"- 数据集：{card.dataset}")
            parts.append(f"- 贡献：{', '.join(card.contributions)}")

        return "\n".join(parts)

    def get_by_id(self, review_id: int):
        """根据 ID 获取综述。"""
        row = self.review_repo.get_by_id(review_id)
        return self._review_row_to_dict(row) if row else None

    def list_reviews(self, limit: int = 20):
        """列出历史综述。"""
        return [self._review_row_to_dict(row) for row in self.review_repo.list(limit=limit)]

    def generate_summary_table(self, cards: List[dict]) -> List[dict]:
        """生成论文对比表。"""
        return [
            {
                "paper_id": c.get("paper_id", ""),
                "title": c.get("title", ""),
                "year": c.get("year"),
                "method": c.get("method", "")[:100],
                "dataset": c.get("dataset"),
                "evidence_source": c.get("evidence_source"),
            }
            for c in cards
        ]

    @staticmethod
    def _review_row_to_dict(row) -> dict:
        """将 ReviewModel 转为 API 可序列化字典。"""
        def _load_json(text: str | None, default):
            if not text:
                return default
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return default

        return {
            "id": row.id,
            "topic": row.topic,
            "full_text": row.review_text,
            "sections": _load_json(row.sections_json, []),
            "references": _load_json(row.references_json, []),
            "paper_ids": _load_json(row.paper_ids_json, []),
            "citation_validation": _load_json(row.citation_validation_json, None),
            "claim_verification": _load_json(
                getattr(row, "claim_verification_json", None),
                None,
            ),
            "citation_style": row.citation_style,
            "language": row.language,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
