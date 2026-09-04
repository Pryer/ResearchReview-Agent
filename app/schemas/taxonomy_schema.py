"""动态综述分类体系的数据结构。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaxonomyStatus(str, Enum):
    """分类体系验证结果的状态。

    - VALID: 结构完整，无需修改。
    - VALID_WITH_WARNING: 结构完整，但存在可观察的集中或碎片，有警告。
    - REFINEMENT_REQUIRED: 主导主题占比过高且内部可细分，需进行二级分类。
    - INVALID: 存在覆盖率不足、重复归属等结构性错误，必须修复才能写作。
    """

    VALID = "valid"
    VALID_WITH_WARNING = "valid_with_warning"
    REFINEMENT_REQUIRED = "refinement_required"
    INVALID = "invalid"


class ResearchTheme(BaseModel):
    """从当前研究主题和论文证据中归纳出的一个综述主题。"""

    theme_id: str
    name: str
    description: str = ""
    level: int = 1
    parent_theme_id: str | None = None
    child_theme_ids: list[str] = Field(default_factory=list)
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    representative_papers: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class PaperThemeAssignment(BaseModel):
    """论文的主主题唯一，次主题仅用于跨主题比较。"""

    paper_id: str
    primary_theme_id: str
    secondary_theme_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    evidence_fields: list[str] = Field(default_factory=list)


class DynamicTaxonomy(BaseModel):
    topic: str
    scope: dict[str, Any] = Field(default_factory=dict)
    organizing_principle: str = "research_problem_or_route"
    rationale: str = ""
    themes: list[ResearchTheme] = Field(default_factory=list)
    assignments: list[PaperThemeAssignment] = Field(default_factory=list)
    version: int = 1
    source: str = "llm"


class TaxonomyValidationResult(BaseModel):
    valid: bool = False
    requires_revision: bool = False

    # 细粒度状态（比 valid/requires_revision 更丰富）
    status: TaxonomyStatus = TaxonomyStatus.INVALID
    # 主导主题过于集中时置为 True；不等同于 invalid，而是触发二级细分
    concentration_requires_split: bool = False
    # 需要细分的主导主题 ID
    dominant_theme_id: str | None = None

    paper_count: int = 0
    assigned_count: int = 0
    theme_count: int = 0
    paper_coverage: float = 0.0
    primary_assignment_ratio: float = 0.0
    largest_theme_ratio: float = 0.0
    unassigned_ratio: float = 0.0
    undersized_theme_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

