"""综述相关 Schema。

定义文献综述请求、综述输出、综述段落等数据结构。
"""

from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    """结构化综述生成请求。"""

    topic: str = Field(..., description="研究主题", min_length=1)
    start_year: Optional[int] = Field(default=None, description="起始年份")
    end_year: Optional[int] = Field(default=None, description="结束年份")
    max_papers: int = Field(default=30, ge=5, le=50, description="最多引用论文数")
    language: Literal["zh", "en"] = Field(default="zh", description="综述语言：zh / en")
    citation_style: Literal["gbt7714", "apa", "ieee", "bibtex"] = Field(
        default="gbt7714", description="引用格式：gbt7714 / apa / ieee / bibtex"
    )
    review_type: Literal["survey", "systematic"] = Field(default="survey", description="综述类型：survey / systematic")


class ReviewSection(BaseModel):
    """综述段落。"""

    title: str = Field(default="", description="段落标题")
    content: str = Field(default="", description="段落正文")
    citations: List[str] = Field(
        default_factory=list, description="本段引用的 paper_id 列表"
    )


class LiteratureReview(BaseModel):
    """文献综述完整输出。"""

    topic: str = Field(default="", description="综述主题")
    language: Literal["zh", "en"] = Field(default="zh", description="综述语言")
    citation_style: Literal["gbt7714", "apa", "ieee", "bibtex"] = Field(default="gbt7714", description="引用格式")
    sections: List[ReviewSection] = Field(
        default_factory=list, description="综述各段落"
    )
    references: List[str] = Field(
        default_factory=list, description="参考文献列表"
    )
    paper_cards: List[dict] = Field(
        default_factory=list, description="引用的 PaperCard 列表"
    )
    citation_validation: Optional[dict] = Field(
        default=None, description="引用校验结果"
    )
    claim_verification: Optional[dict] = Field(
        default=None, description="句子级主张—证据验证结果"
    )

    @property
    def full_text(self) -> str:
        """拼接完整综述文本。"""
        parts = [f"# {self.topic}\n"]
        for section in self.sections:
            parts.append(f"\n## {section.title}\n\n{section.content}")
        if self.references:
            parts.append("\n\n## 参考文献\n")
            for i, ref in enumerate(self.references, 1):
                parts.append(f"[{i}] {ref}")
        return "\n".join(parts)


class ReviewCompareRequest(BaseModel):
    """论文对比请求。"""

    paper_ids: List[str] = Field(..., min_length=2, description="待对比的论文 ID")
    language: str = Field(default="zh", description="输出语言")
