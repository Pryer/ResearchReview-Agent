"""基于多轮研究上下文生成的论文筛选协议。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


CriterionSource = Literal["user_explicit", "confirmed_scope", "inferred"]


class ScreeningCriterion(BaseModel):
    """一组同义概念；组内 OR，不同硬条件之间 AND。

    ``terms_zh`` / ``terms_en`` 为双语术语，分别用于中文/英文分支的匹配。
    向后兼容：任一为空时回退到 ``terms``。
    """

    criterion_id: str
    label: str = ""
    terms: list[str] = Field(default_factory=list)
    terms_zh: list[str] = Field(default_factory=list)
    terms_en: list[str] = Field(default_factory=list)
    source: CriterionSource = "inferred"
    applies_to_each_paper: bool = False
    rationale: str = ""

    def effective_terms(self, language: str | None = None) -> list[str]:
        """返回指定语言的有效 term 列表。

        Args:
            language: ``"zh"`` 取中文术语，``"en"`` 取英文术语，
                      ``None`` 取 ``terms``（向后兼容）。

        Returns:
            去重后的 term 列表，优先取对应语言字段，为空时回退到 ``terms``。
        """
        if language == "zh":
            return self.terms_zh or self.terms
        if language == "en":
            return self.terms_en or self.terms
        return self.terms

    @field_validator("terms")
    @classmethod
    def normalize_terms(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for value in values:
            term = str(value or "").strip()
            key = term.lower()
            if term and key not in seen:
                seen.add(key)
                normalized.append(term)
        return normalized[:12]

    @field_validator("terms_zh", "terms_en")
    @classmethod
    def normalize_language_terms(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for value in values:
            term = str(value or "").strip()
            key = term.lower()
            if term and key not in seen:
                seen.add(key)
                normalized.append(term)
        return normalized[:16]


class ScreeningRoute(BaseModel):
    """证据池中的一条研究路线，路线之间是并集而非逐篇 AND。"""

    route_id: str
    label: str
    terms: list[str] = Field(default_factory=list)
    weight: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""

    @field_validator("terms")
    @classmethod
    def normalize_terms(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for value in values:
            term = str(value or "").strip()
            key = term.lower()
            if term and key not in seen:
                seen.add(key)
                normalized.append(term)
        return normalized[:16]


class ScreeningProtocol(BaseModel):
    """筛选阶段使用的稳定、可解释上下文协议。"""

    version: str = "1.0"
    corpus_goal: str = ""
    hard_include_criteria: list[ScreeningCriterion] = Field(default_factory=list)
    soft_include_criteria: list[ScreeningCriterion] = Field(default_factory=list)
    hard_exclude_title_terms: list[str] = Field(default_factory=list)
    routes: list[ScreeningRoute] = Field(default_factory=list)
    generated_by: Literal["llm", "deterministic_fallback", "minimal_fallback", "enhanced_fallback"] = "deterministic_fallback"
    notes: list[str] = Field(default_factory=list)

    @field_validator("hard_exclude_title_terms")
    @classmethod
    def normalize_exclusions(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for value in values:
            term = str(value or "").strip()
            key = term.lower()
            if term and key not in seen:
                seen.add(key)
                normalized.append(term)
        return normalized[:16]
