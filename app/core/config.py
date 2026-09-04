"""应用配置模块。

使用 pydantic-settings 从环境变量或 .env 文件读取配置，
提供全局单例 ``get_settings`` 供各模块引用。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, List, Literal

from pydantic import Field, field_validator


@dataclass(frozen=True)
class ReviewThresholdPolicy:
    """集中管理路线、主张和综合门禁阈值。"""

    version: str = "review-thresholds-v1"
    route_min_core_evidence: int = 3
    route_drop_ratio_guard: float = 0.60
    route_min_keep_rate: float = 0.40
    route_min_splittable_core: int = 6
    route_oversized_share_factor: float = 1.2
    route_max_sub_routes: int = 3
    # 路线级补检索目标口径。route_min_core_evidence 仍只作判定阈值；
    # 下面这组值决定"要补到几篇"，按交付物类型派生。
    route_recovery_target_min: int = 3
    route_recovery_target_max: int = 12
    route_recovery_status_share: float = 0.85
    route_recovery_competing_work_bonus: int = 2
    route_recovery_diversity_min_years: int = 2
    claim_established_min_evidence: int = 7
    claim_established_min_independent_teams: int = 3
    claim_support_similarity: float = 0.35
    claim_min_text_length: int = 15
    verify_supported_overlap: float = 0.18
    verify_partial_overlap: float = 0.07
    verify_synthesis_partial_overlap: float = 0.04
    verify_entailment_confidence: float = 0.60
    synthesis_abstract_dominance: float = 0.70
    synthesis_abstract_support_rate: float = 0.70
    synthesis_fulltext_support_rate: float = 0.80
    # 正文引用必须由匹配主张授权；低于该一致率视为阻断，达到但仍有错配则降级为警告。
    claim_citation_consistency_rate: float = 0.80

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def get_review_threshold_policy() -> ReviewThresholdPolicy:
    settings = get_settings()
    return ReviewThresholdPolicy(
        route_min_core_evidence=settings.route_validator_min_core_evidence,
        route_drop_ratio_guard=settings.route_validator_drop_ratio_guard,
        route_min_keep_rate=settings.route_validator_min_keep_rate,
        route_min_splittable_core=settings.route_validator_min_splittable_core,
        route_oversized_share_factor=settings.route_validator_oversized_share_factor,
        route_max_sub_routes=settings.route_validator_max_sub_routes,
        route_recovery_target_min=settings.route_recovery_target_min,
        route_recovery_target_max=settings.route_recovery_target_max,
        route_recovery_status_share=settings.route_recovery_status_share,
        route_recovery_competing_work_bonus=settings.route_recovery_competing_work_bonus,
        route_recovery_diversity_min_years=settings.route_recovery_diversity_min_years,
        claim_established_min_evidence=settings.claim_established_min_evidence,
        claim_established_min_independent_teams=settings.claim_established_min_independent_teams,
        claim_support_similarity=settings.claim_support_similarity,
        claim_min_text_length=settings.claim_min_text_length,
        verify_supported_overlap=settings.verify_supported_overlap,
        verify_partial_overlap=settings.verify_partial_overlap,
        verify_synthesis_partial_overlap=settings.verify_synthesis_partial_overlap,
        verify_entailment_confidence=settings.verify_entailment_confidence,
        synthesis_abstract_dominance=settings.synthesis_abstract_dominance,
        synthesis_abstract_support_rate=settings.synthesis_abstract_support_rate,
        synthesis_fulltext_support_rate=settings.synthesis_fulltext_support_rate,
        claim_citation_consistency_rate=settings.claim_citation_consistency_rate,
    )
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- 应用基础 ----------
    app_name: str = "ResearchReview-Agent"
    app_debug: bool = False
    # 安全默认值：开发环境仅监听回环地址。若显式监听外部网卡，启动时会
    # 要求同时配置 APP_API_KEY（见 core.security）。
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    app_api_key: str = ""
    cors_allowed_origins: str = "http://127.0.0.1:8501,http://localhost:8501"
    frontend_api_base_url: str = "http://127.0.0.1:8000/api"

    # ---------- LLM ----------
    llm_provider: str = "deepseek"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    # DeepSeek V4 是混合推理模型且默认开启思考。综述写作需要稳定产出正文，
    # 因此默认显式关闭 thinking，避免输出预算被 reasoning_content 消耗。
    llm_thinking_enabled: bool = False
    # 最终正文单次覆盖为 thinking=True 时使用低强度推理，并直接提供足够
    # 的 completion 预算，避免先以 8K 把 token 全耗在 reasoning 后再重试。
    llm_thinking_effort: Literal["low", "high", "max"] = "low"
    llm_thinking_max_tokens: int = 32768
    llm_temperature: float = 0.3
    # 最终成文预算；控制面结构化任务使用独立的较小预算。
    llm_max_tokens: int = 8192
    llm_control_plane_max_tokens: int = 4096
    llm_request_timeout: int = 120
    # 意图、语义、消歧和检索规划属于可降级控制面。复杂的结构化检索规划
    # 实测可能超过 30 秒，因此给足一次完整生成时间（默认 120s）。主用调用
    # 失败（超时/连接错误/429/5xx）时由 LLMService 自动切换到备用提供商再试
    # 一次（同样 120s）；主备都失败后仍由各模块使用确定性策略降级。
    llm_control_plane_timeout: int = 120
    # 单个逻辑请求跨主/备用提供商的总 deadline，避免两边各等待完整 120 秒。
    llm_failover_total_timeout: int = 180

    # 备用 LLM 提供商：主用失败后自动切换，各 120s。
    # 留空则不启用备用，行为退化为「只用主用」。
    llm_backup_provider: str = ""
    llm_backup_api_key: str = ""
    llm_backup_base_url: str = ""
    llm_backup_model: str = ""

    # ---------- 外部 API ----------
    semantic_scholar_api_key: str = ""
    semantic_scholar_min_interval_seconds: float = 1.1
    semantic_scholar_max_retries: int = 3
    semantic_scholar_max_retry_wait_seconds: float = 15.0
    semantic_scholar_cooldown_seconds: float = 60.0
    crossref_mailto: str = ""
    # OpenAlex 429 处理：未配置 crossref_mailto 时只能用 common pool，
    # 客户端会自动降速；重试用尽后进入进程级冷却而不计入熔断失败。
    openalex_max_retries: int = 3
    openalex_max_retry_wait_seconds: float = 15.0
    openalex_cooldown_seconds: float = 90.0
    # 冷却时长上限。OpenAlex 日配额耗尽时会返回指向次日零点的 Retry-After
    # （实测 41559 秒），无条件采纳会让整天不再访问该源；定期探测成本极低，
    # 故钳制在此上限内。
    openalex_max_cooldown_seconds: float = 900.0
    # CNKI（知网）Selenium 客户端
    cnki_chromedriver_path: str = ""  # 留空则交给 Selenium Manager 自动下载
    cnki_headless: bool = False
    cnki_home_wait_seconds: float = 3.0
    # 对本轮 CNKI 结果进行详情页增强，提取作者、摘要、关键词、来源和 DOI。
    # 连续失败保护仍会在达到阈值后停止详情访问并保留结果页元数据。
    cnki_detail_enrichment_limit: int = 60
    cnki_max_consecutive_detail_failures: int = 2
    cnki_page_load_timeout_seconds: int = 30
    # 详情页增强是知网检索的主要耗时项，额度有限。开启后先翻页取全部结果页
    # 元数据，再按检索式相关度挑出额度内的记录开详情页，使有限额度落在最相关
    # 的文献上，而非结果页靠前（即最新）但未必相关的条目上。
    cnki_detail_rank_before_enrichment: bool = True
    # 额度按下游需求自适应：自适应翻页会让结果池远大于 detail 额度，固定 60
    # 会使多数文献只有标题年份（access_level=metadata_only），无法进入写作
    # 证据池。额度改为 max_results × 倍数，并受下面的上限与时长预算约束。
    # 倍数 >1 是因为粗排选中的记录与最终排序入选集不完全重合。
    cnki_detail_enrichment_demand_factor: float = 1.6
    cnki_detail_enrichment_max: int = 120
    # 详情增强专用时长预算，与翻页预算分开计量；触顶即停止增强并交付
    # 已获得的结果页元数据，不影响已抓取的文献总量。
    cnki_detail_time_budget_seconds: float = 420.0
    # 知网结果页默认按发表时间倒序。开启自适应翻页后，不再按 max_results 换算
    # 固定页数，而是持续翻页到年份窗口下界，让"近三年"这类请求拿全窗口内文献。
    cnki_adaptive_year_paging: bool = True
    # 连续多少整页均早于 start_year 才判定越过窗口下界。取 2 而非 1，容忍
    # 知网排序偶发抖动与年份解析失败。
    cnki_year_boundary_pages: int = 2
    # 自适应翻页的硬上限：页数、总条数与总时长，任一触顶即停止。
    cnki_max_pages: int = 15
    cnki_max_results_ceiling: int = 300
    cnki_paging_time_budget_seconds: float = 600.0

    # ---------- 数据库 ----------
    database_url: str = "sqlite:///./data/research_review.db"

    # ---------- 存储路径 ----------
    pdf_save_dir: str = "./data/pdfs"
    parsed_save_dir: str = "./data/parsed"
    review_save_dir: str = "./data/reviews"
    # 本地 PDF 导入采用 inbox 模式，API 不允许读取该目录之外的服务器文件。
    library_import_dir: str = "./data/imports"

    # ---------- Agent 参数 ----------
    default_max_papers: int = 30
    default_year_lookback: int = 3
    search_year_extension: int = 1
    max_year_lookback: int = 5
    default_search_sources: str = "arxiv,semantic_scholar,openalex,crossref,cnki"
    agent_request_timeout: int = 60
    frontend_request_timeout: int = 1800  # 30 分钟，适应 CNKI 爬取时间
    # 综述写作默认尝试获取开放全文；下载失败会逐篇降级到摘要证据，
    # 但不能再用“快速模式”静默跳过整条全文链路。
    enable_pdf_pipeline: bool = True
    enable_llm_card_extraction: bool = False
    enable_llm_clustering: bool = True
    enable_claim_verification: bool = True
    # 引用分配已有主题均衡的确定性算法；默认不再为这一控制面等待
    # 主备 LLM 超时，避免生成正文前额外阻塞数分钟。
    enable_llm_citation_planning: bool = False
    max_search_keywords: int = 4
    max_results_per_keyword: int = 60
    # 论文重排参数：可按部署预算和来源覆盖率调整。
    rerank_cnki_quota: int = 8
    rerank_candidate_min: int = 60
    rerank_candidate_max: int = 120
    rerank_batch_size: int = 12
    search_source_max_workers: int = 4
    # 关键词级并发派发限额（与源级并发叠加时注意 API 限速）。
    search_keyword_max_workers: int = 4
    research_job_max_workers: int = 2
    research_job_max_pending: int = 20
    # 中英文双分支排名：启用后中文/英文论文各自独立过滤、评分、重排后按配额合并。
    # 英文配额 = 1 - zh_ratio，由 branch_merge 内部推导，无需单独配置。
    language_branch_enabled: bool = True
    language_branch_zh_ratio: float = 0.40
    language_branch_min_zh: int = 8
    language_branch_min_en: int = 12
    # 主题语言倾向自适应配额：语义解析给出 zh_dominant/balanced/en_dominant
    # 判断，代码映射为下列比例，并始终钳制在 [min, max] 区间内，保证任一
    # 语种分支都不会被压到失效。
    language_branch_affinity_enabled: bool = True
    language_branch_zh_ratio_zh_dominant: float = 0.55
    language_branch_zh_ratio_en_dominant: float = 0.28
    language_branch_zh_ratio_min: float = 0.25
    language_branch_zh_ratio_max: float = 0.65
    detail_fetch_max_workers: int = 8
    pdf_download_max_workers: int = 4
    card_extraction_max_workers: int = 5
    pdf_download_retries: int = 3
    pdf_download_backoff_seconds: float = 0.8
    pdf_download_connect_timeout: int = 15
    pdf_download_max_mb: int = 80
    # 证据池在最低引用数之上保留一定余量，用于抵消后续主题/证据筛除；
    # 不再默认把 generation_limit 全部制作成 Paper Card。
    evidence_pool_reserve_ratio: float = 0.5
    taxonomy_induction_limit: int = 24
    taxonomy_induction_timeout: int = 120
    # 检索精化循环（search -> rank -> refine_search -> search）的最大重试轮数
    search_refinement_max_rounds: int = 2
    # 路线验证后的证据恢复闭环。LLM 只诊断和生成查询，以下预算由代码强制执行。
    enable_evidence_recovery: bool = True
    evidence_recovery_max_rounds: int = 2
    evidence_recovery_max_route_attempts: int = 2
    evidence_recovery_max_scope_revisions: int = 1
    evidence_recovery_min_new_evidence: int = 2
    evidence_recovery_min_coverage_gain: float = 0.05
    evidence_recovery_min_query_novelty: float = 0.35
    evidence_recovery_scope_gap_ratio: float = 0.75
    # Route Validator v2 使用离散特征规则而不是预先压成加权相似度。
    route_validator_min_core_evidence: int = 3
    route_validator_drop_ratio_guard: float = 0.60
    route_validator_min_keep_rate: float = 0.40
    route_validator_min_splittable_core: int = 6
    route_validator_oversized_share_factor: float = 1.2
    route_validator_max_sub_routes: int = 3
    # 路线级补检索目标：决定"某个方向要补到几篇"，与判定阈值
    # route_validator_min_core_evidence 分离，按交付物类型派生。
    route_recovery_target_min: int = 3
    route_recovery_target_max: int = 12
    route_recovery_status_share: float = 0.85
    route_recovery_competing_work_bonus: int = 2
    route_recovery_diversity_min_years: int = 2
    claim_established_min_evidence: int = 7
    claim_established_min_independent_teams: int = 3
    claim_support_similarity: float = 0.35
    claim_min_text_length: int = 15
    verify_supported_overlap: float = 0.18
    verify_partial_overlap: float = 0.07
    verify_synthesis_partial_overlap: float = 0.04
    verify_entailment_confidence: float = 0.60
    synthesis_abstract_dominance: float = 0.70
    synthesis_abstract_support_rate: float = 0.70
    synthesis_fulltext_support_rate: float = 0.80
    # 主张—引用一致率下限：正文每句的引用应由其匹配主张的证据授权。
    claim_citation_consistency_rate: float = 0.80
    # ---------- 全局证据门（Global Evidence Gate）----------
    # 综述级证据充分性评估：只测量与推荐，不执行恢复动作（v1 冻结范围）。
    enable_global_evidence_gate: bool = True
    # 年份窗口内论文占比下限；仅显式年份范围时阻断。
    global_gate_min_recency_ratio: float = 0.9
    # 路线均衡下限：最少论文路线 / 平均论文数，低于则判定 KEEP 路线失衡。
    global_gate_route_balance_min_ratio: float = 0.25
    # 同行评审论文占比下限；仅用户显式要求同行评审/期刊/SCI/EI 时阻断。
    global_gate_peer_review_ratio: float = 0.8
    # 单个关键词检索的最少结果数下限（即使 retrieval_target 更小也会请求这么多）
    min_results_per_keyword: int = 30

    @field_validator("default_search_sources")
    @classmethod
    def _split_sources(cls, v: str) -> str:
        """允许逗号分隔的 sources 字符串。"""
        return v

    @field_validator(
        "search_refinement_max_rounds",
        "evidence_recovery_max_rounds",
        "evidence_recovery_max_route_attempts",
        "evidence_recovery_max_scope_revisions",
        "evidence_recovery_min_new_evidence",
        "route_validator_min_core_evidence",
        "route_recovery_target_min",
        "route_recovery_target_max",
        "route_recovery_competing_work_bonus",
        "route_recovery_diversity_min_years",
    )
    @classmethod
    def _non_negative_agent_limits(cls, v: int) -> int:
        return max(0, int(v))

    @field_validator(
        "evidence_recovery_min_coverage_gain",
        "evidence_recovery_min_query_novelty",
        "evidence_recovery_scope_gap_ratio",
        "route_validator_drop_ratio_guard",
        "route_validator_min_keep_rate",
        "route_recovery_status_share",
        "global_gate_min_recency_ratio",
        "global_gate_route_balance_min_ratio",
        "global_gate_peer_review_ratio",
    )
    @classmethod
    def _unit_interval_thresholds(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    # ---------- 派生属性 ----------
    @property
    def search_sources_list(self) -> List[str]:
        """将逗号分隔的 sources 解析为列表。"""
        return [s.strip() for s in self.default_search_sources.split(",") if s.strip()]

    @property
    def cors_allowed_origins_list(self) -> List[str]:
        """返回显式允许的前端来源，禁止凭证模式下使用通配符。"""
        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip() and origin.strip() != "*"
        ]

    @property
    def llm_backup_enabled(self) -> bool:
        """是否配置了可用的备用 LLM 提供商。"""
        return bool(
            self.llm_backup_api_key
            and self.llm_backup_base_url
            and self.llm_backup_model
        )

    def ensure_dirs(self) -> None:
        """确保所有存储目录存在。"""
        for path in (
            self.pdf_save_dir,
            self.parsed_save_dir,
            self.review_save_dir,
            self.library_import_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例。"""
    settings = Settings()
    settings.ensure_dirs()
    return settings
