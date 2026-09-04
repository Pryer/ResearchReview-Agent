"""Agent 相关 Schema。

定义意图识别、槽位抽取、Agent 执行步骤和输出结构。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.common_schema import TaskStatus


class IntentType(str, Enum):
    """支持的用户意图类型。"""

    SEARCH_PAPERS = "search_papers"
    READ_PAPER = "read_paper"
    GENERATE_REVIEW = "generate_review"
    GENERATE_RELATED_WORK = "generate_related_work"
    GENERATE_INTRODUCTION = "generate_introduction"
    COMPARE_PAPERS = "compare_papers"
    GENERATE_REFERENCES = "generate_references"
    EXTRACT_PAPER_CARD = "extract_paper_card"
    FIND_DATASETS = "find_datasets"
    FIND_TRENDS = "find_trends"
    GENERAL_QA = "general_qa"


class IntentResult(BaseModel):
    """意图识别结果。"""

    intent: str = Field(default="", description="识别到的意图")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="置信度")
    reason: str = Field(default="", description="判断依据")
    slots: Dict[str, Any] = Field(
        default_factory=dict, description="从查询中抽取的槽位"
    )


class TopicScope(BaseModel):
    """歧义主题的一种可执行研究范围。"""

    scope_id: str = Field(..., min_length=1, description="稳定的范围标识")
    label: str = Field(..., min_length=1, description="用户可读名称")
    description: str = Field(default="", description="范围说明")
    include_terms: List[str] = Field(default_factory=list, description="建议纳入的概念")
    exclude_terms: List[str] = Field(default_factory=list, description="需要排除的相邻含义")
    seed_queries: List[str] = Field(default_factory=list, description="该范围的种子检索式")
    research_mode: str = Field(default="mixed", description="研究模式，如 technical, empirical, mixed")


class TopicAmbiguityResult(BaseModel):
    """主题消歧结果。"""

    ambiguous: bool = Field(default=False, description="是否存在会显著改变语料范围的歧义")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="")
    recommended_strategy: str = Field(
        default="single_scope",
        description="single_scope / ask_user / multi_branch",
    )
    default_scope_id: Optional[str] = Field(default=None)
    scopes: List[TopicScope] = Field(default_factory=list)
    question: Optional[str] = Field(default=None)


class SlotResult(BaseModel):
    """槽位抽取结果。"""

    topic: Optional[str] = Field(default=None, description="研究主题")
    start_year: Optional[int] = Field(default=None, description="起始年份")
    end_year: Optional[int] = Field(default=None, description="结束年份")
    max_papers: int = Field(default=30, description="论文数量")
    required_reference_count: int = Field(
        default=30,
        description="用户要求最终综述至少使用的唯一参考文献数量（不是论文被引次数）",
    )
    retrieval_target: int = Field(default=30, description="检索和排序阶段的候选保留目标")
    generation_limit: int = Field(default=30, description="生成综述时最多使用的论文数")
    year_range_explicit: bool = Field(default=False, description="年份范围是否由用户明确指定")
    strict_year_range: bool = Field(default=False, description="用户是否要求严格限制年份范围")
    max_papers_explicit: bool = Field(default=False, description="论文数量是否由用户明确指定")
    requested_sections: List[str] = Field(
        default_factory=lambda: ["related_work"],
        description="用户要求生成的正文部分，如 background/research_status/related_work",
    )
    language: str = Field(default="zh", description="综述语言")
    citation_style: str = Field(default="gbt7714", description="引用格式")


class AgentStep(BaseModel):
    """Agent 单步执行记录。

    用于前端展示 Agent 执行过程，并便于调试。
    """

    step_name: str = Field(..., description="步骤名称")
    tool_name: Optional[str] = Field(default=None, description="调用的工具名")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="输入数据摘要")
    output_data: Optional[Dict[str, Any]] = Field(
        default=None, description="输出数据摘要"
    )
    status: str = Field(default=TaskStatus.PENDING.value, description="执行状态")
    error: Optional[str] = Field(default=None, description="错误信息")
    duration_ms: Optional[int] = Field(default=None, description="执行耗时 (ms)")


class AgentRequest(BaseModel):
    """Agent 自然语言请求。"""

    user_query: str = Field(..., description="用户自然语言请求", min_length=1, max_length=20_000)
    session_id: Optional[str] = Field(default=None, max_length=128, description="会话 ID（可选）")
    clarification_answer: Optional[str] = Field(
        default=None,
        max_length=10_000,
        description="对上一轮主题澄清问题的回答，可传 scope_id、序号或选项名称",
    )
    state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="可选的初始状态，用于传入本文工作信息、研究背景等写作所需字段"
    )

    @field_validator("state")
    @classmethod
    def _restrict_client_state(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """只允许客户端提供公开写作上下文，禁止注入 Agent 内部控制状态。"""
        if value is None:
            return None
        allowed = {
            "our_work",
            "background",
            "existing_limitations",
            "verified_results",
            "target_length",
            "required_reference_count",
            "max_papers",
            "language",
            "citation_style",
            "requested_sections",
        }
        unexpected = sorted(set(value) - allowed)
        if unexpected:
            raise ValueError(f"state 包含不允许由客户端设置的字段: {', '.join(unexpected)}")
        return value


class ResearchRevisionRequest(BaseModel):
    """对已完成研究结果执行论文排除和增量重生成。"""

    session_id: str = Field(..., min_length=1, max_length=128, description="待修订的研究会话 ID")
    excluded_paper_ids: List[str] = Field(
        default_factory=list,
        max_length=500,
        description="本轮要排除的稳定论文 ID",
    )
    instruction: Optional[str] = Field(
        default=None,
        max_length=10_000,
        description="用户的自然语言修订说明，作为会话记忆保存",
    )


class AgentOutput(BaseModel):
    """Agent 最终输出。"""

    answer: str = Field(default="", description="最终回复文本")
    steps: List[AgentStep] = Field(
        default_factory=list, description="执行步骤列表"
    )
    references: Optional[List[str]] = Field(
        default=None, description="参考文献列表"
    )
    paper_cards: List[dict] = Field(
        default_factory=list, description="涉及的 PaperCard"
    )
    review: Optional[LiteratureReviewData] = Field(
        default=None, description="生成的综述"
    )
    intent: Optional[str] = Field(default=None, description="识别的意图")
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    status: Optional[str] = Field(
        default=None,
        description="running/needs_clarification/completed/partial/blocked/failed",
    )
    clarification: Optional[Dict[str, Any]] = Field(default=None, description="待用户确认的主题范围")
    claim_verification: Optional[Dict[str, Any]] = Field(
        default=None,
        description="句子级主张—证据验证结果",
    )
    core_deliverables: List[str] = Field(default_factory=list)
    deliverable_readiness: List[Dict[str, Any]] = Field(default_factory=list)
    deliverable_downgrades: List[Dict[str, Any]] = Field(default_factory=list)
    writing_plans: List[Dict[str, Any]] = Field(default_factory=list)
    citation_allocation_plans: List[Dict[str, Any]] = Field(default_factory=list)
    deliverable_validation: List[Dict[str, Any]] = Field(default_factory=list)
    writer_diagnostics: List[Dict[str, Any]] = Field(default_factory=list)
    writer_section_diagnostics: List[Dict[str, Any]] = Field(default_factory=list)
    generation_readiness: Optional[Dict[str, Any]] = Field(default=None)
    quality_gate: Optional[Dict[str, Any]] = Field(default=None)
    final_review_integrity: Optional[Dict[str, Any]] = Field(default=None)
    generation_blocked: bool = Field(default=False)


class LiteratureReviewData(BaseModel):
    """简化版综述数据（嵌入 AgentOutput）。"""

    topic: str = Field(default="")
    full_text: str = Field(default="")
    sections: List[dict] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
