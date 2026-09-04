"""论文服务。

封装论文检索、排序、详情补全的业务流程。
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from app.core.logger import get_logger
from app.database.repositories import PaperCardRepository, PaperRepository
from app.schemas.paper_schema import PaperCard, PaperMetadata, PaperSearchRequest
from app.tools.download_pdf import batch_download_pdfs
from app.tools.extract_paper_card import extract_paper_card
from app.tools.fetch_metadata import fetch_batch_details
from app.tools.parse_pdf import batch_parse_pdfs
from app.tools.rank_papers import deduplicate_and_rank
from app.tools.search_papers import search_papers
from app.utils.date_utils import default_year_range, current_year

logger = get_logger(__name__)


class PaperService:
    """论文相关业务流程。"""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.paper_repo = PaperRepository(db)
        self.card_repo = PaperCardRepository(db)

    def search(self, request: PaperSearchRequest) -> List[PaperMetadata]:
        """检索论文，并尽力保存到本地论文库。"""
        default_start, default_end = default_year_range(current_year())
        papers = search_papers(
            query=request.query,
            start_year=request.start_year or default_start,
            end_year=request.end_year or default_end,
            max_results=request.max_results,
            sources=request.sources,
        )
        self._save_search_results(papers)
        return papers

    def search_and_rank(self, request: PaperSearchRequest) -> List[dict]:
        """检索并排序。"""
        papers = self.search(request)
        paper_dicts = [p.model_dump() for p in papers]
        ranked = deduplicate_and_rank(paper_dicts, request.query, request.max_results)
        return ranked

    def get_by_id(self, paper_id: str) -> Optional[PaperMetadata]:
        """获取论文详情。"""
        return self.paper_repo.get_by_id(paper_id)

    def list_papers(self, limit: int = 50) -> List[PaperMetadata]:
        """列出本地论文。"""
        return self.paper_repo.list(limit=limit)

    def fetch_details(self, papers: List[dict]) -> List[dict]:
        """补全元数据。"""
        return fetch_batch_details(papers)

    def download_and_parse(self, papers: List[dict]) -> dict:
        """下载并解析开放 PDF。

        Returns:
            {"pdf_paths": {...}, "parsed": {...}}
        """
        pdf_paths = batch_download_pdfs(papers)
        parsed = batch_parse_pdfs(pdf_paths)
        return {"pdf_paths": pdf_paths, "parsed": parsed}

    def build_paper_cards(self, papers: List[dict], topic: str = "") -> List[PaperCard]:
        """构建 PaperCard 并持久化。"""
        from app.services.llm_service import LLMService

        download_result = self.download_and_parse(papers)
        parsed = download_result["parsed"]

        llm = LLMService()
        cards = []
        for paper in papers:
            paper_id = paper.get("paper_id", "")
            card = extract_paper_card(paper, parsed.get(paper_id), llm, topic)
            self.card_repo.save(card)
            cards.append(card)

        self.db.commit()
        return cards

    def build_paper_card(self, paper_id: str) -> Optional[PaperCard]:
        """为单篇论文构建卡片。"""
        paper = self.paper_repo.get_by_id(paper_id)
        if not paper:
            return None
        return extract_paper_card(paper.model_dump(), None, None)

    def _save_search_results(self, papers: List[PaperMetadata]) -> None:
        """保存检索结果；失败时不影响主检索流程。"""
        if not papers:
            return
        try:
            for paper in papers:
                self.paper_repo.save(paper)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.warning("Failed to persist search results: %s", e)
