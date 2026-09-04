"""引用服务。

提供参考文献生成、引用校验、引用格式转换。
"""

from __future__ import annotations

from typing import Dict, List

from app.core.logger import get_logger
from app.tools.generate_citation import (
    generate_and_validate_citations,
    generate_reference,
    generate_references,
    validate_citations,
)

logger = get_logger(__name__)


class CitationService:
    """引用生成与校验。"""

    def __init__(self, llm=None) -> None:
        from app.services.llm_service import LLMService
        self.llm = llm or LLMService()

    def generate_references(self, papers: List[dict], style: str) -> List[str]:
        """生成参考文献。"""
        return generate_references(papers, style)

    def validate_review_citations(
        self,
        review_text: str,
        references: List[str],
        paper_cards: List[dict] | None = None,
    ) -> Dict:
        """校验引用。"""
        return validate_citations(review_text, references, paper_cards)

    def convert_citation_style(self, references: List[str], target_style: str) -> List[str]:
        """拒绝对缺少元数据的纯文本参考文献伪装执行格式转换。"""
        raise NotImplementedError(
            "引用格式转换需要结构化论文元数据；请调用 generate_references(papers, style)"
        )

    def generate_and_validate(
        self,
        review_text: str,
        paper_cards: List[dict],
        style: str = "gbt7714",
    ) -> Dict:
        """一站式生成 + 校验。"""
        return generate_and_validate_citations(
            review_text, paper_cards, style, self.llm,
        )
