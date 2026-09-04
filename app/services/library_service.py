"""本地论文库服务。

管理本地论文导入、搜索、索引重建。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from app.core.logger import get_logger
from app.core.config import get_settings
from app.database.repositories import PaperCardRepository, PaperRepository
from app.schemas.paper_schema import PaperCard, PaperMetadata

logger = get_logger(__name__)


class LibraryService:
    """本地论文库管理。"""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.paper_repo = PaperRepository(db)
        self.card_repo = PaperCardRepository(db)

    def add_paper(self, paper: PaperMetadata) -> None:
        """保存论文元数据。"""
        self.paper_repo.save(paper)
        self.db.commit()

    def add_paper_card(self, card: PaperCard) -> None:
        """保存 PaperCard。"""
        self.card_repo.save(card)
        self.db.commit()

    def search_local_library(self, query: str, top_k: int = 5) -> List[PaperCard]:
        """检索本地论文库（关键词匹配）。

        MVP 阶段做简单的标题/摘要包含匹配，后续升级为向量检索。
        """
        import re
        query_lower = query.lower()
        all_cards = self.card_repo.list_all(limit=200)

        scored: list[tuple[float, PaperCard]] = []
        for card in all_cards:
            score = 0.0
            if query_lower in (card.title or "").lower():
                score += 3.0
            if query_lower in (card.research_problem or "").lower():
                score += 2.0
            if query_lower in (card.method or "").lower():
                score += 1.5
            # 关键词匹配
            for word in re.split(r"\s+", query_lower):
                if len(word) > 2:
                    if word in (card.title or "").lower():
                        score += 0.5
            if score > 0:
                scored.append((score, card))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [card for _, card in scored[:top_k]]

    def get_paper_by_id(self, paper_id: str) -> Optional[PaperMetadata]:
        """根据 ID 查询论文。"""
        return self.paper_repo.get_by_id(paper_id)

    def list_papers(self, limit: int = 50) -> List[PaperCard]:
        """列出本地论文卡片。"""
        return self.card_repo.list_all(limit=limit)

    def import_pdf(self, file_path: str) -> Optional[dict]:
        """导入本地 PDF。

        解析 PDF 生成 PaperCard 并入库。
        """
        from app.tools.parse_pdf import parse_pdf
        from app.tools.extract_paper_card import extract_paper_card

        import_root = Path(get_settings().library_import_dir).resolve()
        candidate = Path(file_path)
        path = (import_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            path.relative_to(import_root)
        except ValueError:
            logger.warning("Rejected PDF path outside import directory: %s", path)
            return None

        if not path.is_file() or path.suffix.lower() != ".pdf":
            return None
        if path.stat().st_size > get_settings().pdf_download_max_mb * 1024 * 1024:
            logger.warning("Rejected oversized imported PDF: %s", path.name)
            return None

        # 后缀名不足以证明文件类型；在交给 PDF 解析器之前先检查文件签名。
        try:
            with path.open("rb") as stream:
                if stream.read(5) != b"%PDF-":
                    return None
        except OSError:
            return None

        try:
            parsed = parse_pdf(str(path))
            paper_dict = {
                "paper_id": path.stem,
                "title": path.stem,
                "source": "local_import",
            }
            card = extract_paper_card(paper_dict, parsed, None)
            self.card_repo.save(card)
            self.db.commit()
            return card.model_dump()
        except Exception as e:
            logger.error("Import PDF failed: %s", e)
            return None

    def rebuild_index(self) -> int:
        """重建向量索引。

        MVP 阶段仅返回卡片数量，后续实现 FAISS/Chroma 索引。
        """
        cards = self.card_repo.list_all(limit=10000)
        logger.info("Rebuild index: %d cards", len(cards))
        return len(cards)
