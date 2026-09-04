"""论文相关 Schema。

定义论文检索请求、论文元数据、等数据结构。
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PaperSearchRequest(BaseModel):
    """论文检索请求。"""

    query: str = Field(..., description="检索关键词", min_length=1)
    start_year: Optional[int] = Field(default=None, description="起始年份")
    end_year: Optional[int] = Field(default=None, description="结束年份")
    max_results: int = Field(default=20, ge=1, le=100, description="最大返回数量")
    sources: List[str] = Field(
        default_factory=lambda: ["arxiv", "semantic_scholar", "openalex"],
        description="数据源列表",
    )


class PublicationStatus(str, Enum):
    """文献出版生命周期状态；无法确认时必须保留 unknown。"""

    PUBLISHED = "published"
    ONLINE_FIRST = "online_first"
    ACCEPTED = "accepted"
    PREPRINT = "preprint"
    WITHDRAWN = "withdrawn"
    RETRACTED = "retracted"
    UNKNOWN = "unknown"


class PaperCrawlMetadata(BaseModel):
    """统一爬取字段。

    这是前端、导出和综述生成可依赖的最小论文元数据集合。
    """

    title: str = Field(default="", description="论文标题")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    year: Optional[int] = Field(default=None, description="发表年份")
    abstract: str = Field(default="", description="论文摘要")
    venue: Optional[str] = Field(default=None, description="发表期刊/会议")
    doi: Optional[str] = Field(default=None, description="DOI")
    url: Optional[str] = Field(default=None, description="论文 URL")
    pdf_url: Optional[str] = Field(default=None, description="PDF URL")
    keywords: Optional[List[str]] = Field(default=None, description="关键词")
    publication_status: PublicationStatus = Field(
        default=PublicationStatus.UNKNOWN,
        description="出版状态；未知时不得推断为正式发表",
    )


class PaperMetadata(BaseModel):
    """论文元数据。

    统一的论文描述结构，遮蔽不同数据源的差异。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "paper_id": "arxiv:2401.00001",
                "title": "A Survey on Vision Transformers",
                "authors": ["Alice", "Bob"],
                "year": 2024,
                "venue": "CVPR",
                "abstract": "This paper surveys...",
                "doi": "10.1000/xyz123",
                "arxiv_id": "2401.00001",
                "url": "https://arxiv.org/abs/2401.00001",
                "pdf_url": "https://arxiv.org/pdf/2401.00001",
                "citation_count": 100,
                "source": "arxiv",
                "is_open_access": True,
            }
        }
    )

    paper_id: str = Field(..., description="论文唯一标识")
    title: str = Field(default="", description="论文标题")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    year: Optional[int] = Field(default=None, description="发表年份")
    venue: Optional[str] = Field(default=None, description="发表期刊/会议")
    abstract: Optional[str] = Field(default=None, description="摘要")
    doi: Optional[str] = Field(default=None, description="DOI")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID")
    url: Optional[str] = Field(default=None, description="论文主页 URL")
    pdf_url: Optional[str] = Field(default=None, description="PDF URL")
    citation_count: Optional[int] = Field(default=None, description="引用量（各数据源的 max 聚合值，仅用于展示和弱排序）")
    citation_count_by_source: Optional[dict[str, int]] = Field(
        default=None,
        description="按数据源分别记录的引用量，如 {'semantic_scholar': 100, 'openalex': 95}",
    )
    source: str = Field(default="unknown", description="数据来源")
    is_open_access: bool = Field(default=False, description="是否开放获取")
    keywords: Optional[List[str]] = Field(default=None, description="关键词")
    publication_status: PublicationStatus = Field(
        default=PublicationStatus.UNKNOWN,
        description="出版状态；未知时不得推断为正式发表",
    )

    def to_crawl_metadata(self) -> PaperCrawlMetadata:
        """返回统一爬取字段视图。"""
        return PaperCrawlMetadata(
            title=self.title or "",
            authors=[a for a in self.authors if a],
            year=self.year,
            abstract=self.abstract or "",
            venue=self.venue,
            doi=self.doi,
            url=self.url,
            pdf_url=self.pdf_url,
            keywords=self.keywords,
            publication_status=self.publication_status,
        )

class EvidenceSpan(BaseModel):
    """可追溯到论文原文或元数据的最小证据单元。"""

    evidence_id: str = Field(..., description="论文内唯一证据标识")
    text: str = Field(default="", description="证据原文片段")
    section: Optional[str] = Field(default=None, description="所在章节")
    page: Optional[int] = Field(default=None, description="PDF 页码（从 1 开始）")
    source_type: Literal["metadata", "title", "abstract", "full_text", "table"] = Field(
        default="metadata",
        description="证据来源层级",
    )
    char_start: Optional[int] = Field(default=None, description="在对应来源文本中的起始字符位置")
    char_end: Optional[int] = Field(default=None, description="在对应来源文本中的结束字符位置")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="证据抽取置信度")
    provider: Optional[str] = Field(default=None, description="证据提供方，如 crossref/openalex/cnki")
    source_url: Optional[str] = Field(default=None, description="证据原始来源链接")


class AccessLevel(str, Enum):
    METADATA_ONLY = "metadata_only"
    TITLE_AND_KEYWORDS = "title_and_keywords"
    ABSTRACT = "abstract"
    PARTIAL_FULL_TEXT = "partial_full_text"
    FULL_TEXT = "full_text"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    METADATA_VERIFIED = "metadata_verified"
    SOURCE_VERIFIED = "source_verified"
    CONTENT_VERIFIED = "content_verified"


class PaperEvidenceState(BaseModel):
    """论文可访问内容与允许执行的抽取操作；不等同于文献学术等级。"""

    access_level: AccessLevel = AccessLevel.METADATA_ONLY
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    available_sections: List[str] = Field(default_factory=list)
    missing_sections: List[str] = Field(default_factory=list)
    can_extract_method: bool = False
    can_extract_results: bool = False
    can_extract_limitations: bool = False
    can_compare_metrics: bool = False


class EvidenceClaim(BaseModel):
    """直接绑定原文的字段声明，而非脱离证据的自由摘要。"""

    claim: str
    source_text: str
    source_section: str
    evidence_id: Optional[str] = None
    evidence_level: AccessLevel
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    explicitly_reported: bool = False


class PaperCard(BaseModel):
    """论文卡片。

    从论文全文或摘要中抽取的结构化知识卡片，
    是综述生成的核心输入。

    合规约束：
    - 必填字段缺失时不要编造，保持为空字符串。
    - ``evidence_source`` 必须为 "abstract"、"full_text" 或 "metadata"。
    """

    paper_id: str = Field(..., description="论文唯一标识")
    title: str = Field(default="", description="论文标题")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    year: Optional[int] = Field(default=None, description="发表年份")
    venue: Optional[str] = Field(default=None, description="发表期刊/会议")
    doi: Optional[str] = Field(default=None, description="DOI")
    url: Optional[str] = Field(default=None, description="稳定论文链接")
    source: str = Field(default="unknown", description="检索来源标识")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID")
    publication_status: PublicationStatus = Field(
        default=PublicationStatus.UNKNOWN,
        description="出版状态；未确认时保持 unknown，不得由 LLM 升级",
    )
    publication_type: str = Field(
        default="unknown",
        description="journal_article / conference_paper / conference_short_paper / preprint / systematic_review / meta_analysis / unknown",
    )
    peer_review_status: str = Field(
        default="unknown",
        description="peer_reviewed / likely_peer_reviewed / not_peer_reviewed / unknown",
    )
    evidence_level: str = Field(default="unknown", description="用于综述分层的证据等级")
    research_problem: str = Field(default="", description="研究问题")
    study_design: str = Field(default="", description="研究设计或证据类型")
    sample_size: Optional[str] = Field(default=None, description="原文明确报告的样本或数据规模")
    data_modalities: List[str] = Field(default_factory=list, description="视频、语音、文本、姿态等数据模态")
    behavior_categories: List[str] = Field(default_factory=list, description="原文明示的行为或状态类别")
    method: str = Field(default="", description="核心方法")
    dataset: Optional[str] = Field(default=None, description="使用的数据集")
    metrics: List[str] = Field(default_factory=list, description="评价指标")
    results: Optional[str] = Field(default=None, description="主要实验结果")
    contributions: List[str] = Field(default_factory=list, description="主要贡献")
    limitations: List[str] = Field(default_factory=list, description="局限性")
    relevance_reason: str = Field(default="", description="与主题的相关性说明")
    evidence_source: str = Field(
        default="metadata",
        description="证据来源：abstract / full_text / metadata",
    )
    evidence_spans: List[EvidenceSpan] = Field(
        default_factory=list,
        description="可回溯到摘要、全文或元数据的证据片段",
    )
    field_evidence: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="PaperCard 字段到 evidence_id 的映射",
    )
    evidence_state: PaperEvidenceState = Field(default_factory=PaperEvidenceState)
    field_claims: Dict[str, List[EvidenceClaim]] = Field(
        default_factory=dict,
        description="字段到声明级原文证据的映射",
    )
    unsupported_fields: List[str] = Field(
        default_factory=list,
        description="因当前访问级别或原文未报告而禁止补全的字段",
    )
    relation_type: Optional[str] = Field(
        default=None,
        description="语义筛选确定的主题关系：direct / near / indirect / unrelated",
    )
    anchor_low_confidence: bool = Field(
        default=False,
        description="主题锚点仅由宽松匹配或语言护栏放行，相关性未经词法硬确认，可能偏题",
    )
    eligible_deliverables: List[str] = Field(
        default_factory=list,
        description="语义筛选允许该论文直接支撑的正式交付物类型",
    )
    quality_status: Literal["valid", "partial", "invalid"] = Field(
        default="partial",
        description="结构化字段质量门禁结果",
    )
    quality_issues: List[str] = Field(
        default_factory=list,
        description="质量门禁发现的问题；字段被清空时保留原因",
    )


class PaperDetail(PaperMetadata):
    """论文详情（包含解析后的文本）。"""

    full_text: Optional[str] = Field(default=None, description="全文文本")
    sections: Optional[dict] = Field(
        default=None, description="分段文本 (abstract / method / experiment 等)"
    )
    parsed_references: List[str] = Field(
        default_factory=list, description="解析出的参考文献列表"
    )


class SourceDiagnostic(BaseModel):
    """单个数据源的检索诊断。

    ``status`` 是旧版兼容摘要；``outcome`` 和 ``failure_category`` 提供可操作的
    细分结果，避免把正常空结果、查询不适配和客户端故障混为一谈。
    """

    source: str = Field(..., description="数据源标识，如 cnki / arxiv / semantic_scholar")
    status: Literal["success", "empty", "failed", "skipped", "human_action_required"] = Field(
        default="success",
        description="兼容字段：success=有结果; empty=正常空结果; failed=异常; skipped=能力不兼容而跳过; human_action_required=需要人工处理",
    )
    outcome: Literal[
        "success_with_results", "success_empty", "query_not_adapted", "rate_limited",
        "timeout", "authentication_failed", "api_failed", "human_action_required", "skipped",
    ] = Field(default="success_empty", description="结构化来源结果")
    failure_category: Optional[str] = Field(
        default=None,
        description="失败分类；通常与 outcome 对应，保留更细的客户端分类",
    )
    returned_count: int = Field(default=0, description="本数据源返回的论文数")
    error_code: Optional[str] = Field(default=None, description="错误码(如 TIMEOUT / CAPTCHA_REQUIRED / LOGIN_REQUIRED)")
    message: Optional[str] = Field(default=None, description="人类可读的错误/说明信息")

    @model_validator(mode="after")
    def _infer_outcome_for_legacy_status(self) -> "SourceDiagnostic":
        """让只提供旧 status 的调用方得到一致的细分结果。"""
        if self.outcome != "success_empty":
            return self
        if self.status == "success" and self.returned_count > 0:
            self.outcome = "success_with_results"
        elif self.status == "skipped":
            self.outcome = "query_not_adapted" if self.error_code == "INCOMPATIBLE_QUERY_LANGUAGE" else "skipped"
        elif self.status == "human_action_required":
            self.outcome = "human_action_required"
        elif self.status == "failed":
            code = str(self.error_code or "").upper()
            if code in {"TIMEOUT", "TIME_OUT"}:
                self.outcome = "timeout"
            elif code in {"RATE_LIMITED", "RATE_LIMIT", "HTTP_429", "TOO_MANY_REQUESTS"}:
                self.outcome = "rate_limited"
            elif code in {"UNAUTHORIZED", "FORBIDDEN", "LOGIN_REQUIRED", "HTTP_401", "HTTP_403"}:
                self.outcome = "authentication_failed"
            elif code in {"CAPTCHA_REQUIRED", "HUMAN_ACTION_REQUIRED"}:
                self.outcome = "human_action_required"
            else:
                self.outcome = "api_failed"
        self.failure_category = self.failure_category or (
            self.outcome if self.outcome not in {"success_with_results", "success_empty"} else None
        )
        return self
