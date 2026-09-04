# ResearchReview-Agent 项目架构说明

> 面向多学科研究主题的检索、证据结构化与引用可验证综述智能体。
> 用户输入自然语言研究请求，系统自动检索开放论文、生成结构化论文卡片、归纳研究脉络，
> 最终产出带可验证引用的学术综述（四类交付物）。
>
> 历史实现设计稿见 `docs/ResearchReview-Agent-architecture.md`（仅供追溯，以本文为准）。

---

## 1. 技术栈

| 层 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn（`app/main.py`） |
| 前端 | Streamlit（`app/frontend/chat_app.py`，对话式界面） |
| 数据库 | SQLite + SQLAlchemy（`app/database/`） |
| LLM | 统一经 `app/services/llm_service.py` 调用（主备双模型） |
| 检索源 | arXiv / Semantic Scholar / OpenAlex / Crossref（HTTP）+ CNKI（Selenium 浏览器，headless 可配，默认关闭） |
| PDF 解析 | PyMuPDF（全文/分段提取） |
| 测试 | pytest（`tests/`，64 个测试文件） |

## 2. 目录结构

```text
ResearchReview-Agent/
├── ARCHITECTURE.md            # 本文档
├── README.md                  # 项目介绍与快速开始
├── AGENTS.md                    # 面向 Coding Agent 的仓库说明
├── run_api.py                 # FastAPI 启动入口（uvicorn app.main:app）
├── run_chat_frontend.py       # Streamlit 前端启动入口
├── requirements.txt / environment.yml / pyproject.toml
│
├── app/                       # 应用主体（分层架构）
│   ├── main.py                # FastAPI 实例：路由注册、CORS、异常处理、lifespan 建表与任务恢复
│   ├── agent/                 # 智能体核心（见 §3）
│   │   ├── graph.py           # 顺序编排：节点顺序、条件分支、三个入口（run/continue/regenerate）
│   │   ├── execution.py       # 执行原语：AgentCancelledError、节点边界检查、LLM 工厂
│   │   ├── retrieval_loop.py  # ReAct 检索精化循环（从 graph 下沉的策略层）
│   │   ├── recovery_loop.py   # 路线证据恢复状态机（从 graph 下沉的策略层）
│   │   └── nodes/             # 流程节点，按阶段分组：planning / retrieval / extraction / synthesis / verification
│   ├── api/                   # HTTP 路由层
│   │   ├── routes_review.py   # 研究任务：提交/查询/取消/修订/会话记忆/历史综述
│   │   ├── routes_paper.py    # 论文检索与详情、论文卡片
│   │   └── routes_library.py  # 本地论文库：PDF 导入、搜索
│   ├── services/              # 业务服务层
│   │   ├── research_job_service.py        # 后台任务队列、重启恢复、取消
│   │   ├── research_conversation_service.py  # 多轮会话（澄清、论文集合、版本与修订历史）
│   │   ├── llm_service.py / paper_service.py / review_service.py
│   │   └── citation_service.py / library_service.py
│   ├── clients/               # 外部检索源客户端：arxiv / semantic_scholar / openalex / crossref / cnki
│   ├── tools/                 # 原子工具：检索调度、过滤打分（rank_papers + paper_matching /
│   │                          #   paper_rerank / venue_tiers）、跨语言分支合并（branch_merge /
│   │                          #   language_router / language_filter）、写作分发（write_deliverable）、
│   │                          #   引用生成与校验、PDF 下载解析等
│   ├── deliverables/          # 四类交付物规格与渲染器（背景/现状/相关工作/叙述性综述）+ few-shot 蓝本
│   ├── schemas/               # Pydantic 模型（agent / paper / review / verification / taxonomy 等）
│   ├── database/              # db.py（建表）/ models.py（ORM）/ repositories.py（仓储）
│   ├── core/                  # 横切能力：config、安全（API Key）、熔断器、限流、指标、引用语法、文本质量
│   ├── frontend/              # Streamlit 界面（chat_app.py 主界面、query_utils.py）
│   ├── prompt/ + prompt_catalog.py  # 提示词与懒加载目录
│   └── utils/                 # 通用工具（日期、去重、文件、PDF、文本清洗）
│
├── scripts/                   # 运维/测试脚本
│   ├── check_llm_api.py       # OpenAI 兼容 LLM 接口连通性检查
│   ├── submit_research.py     # 命令行提交研究任务
│   ├── monitor_job.py         # 任务监控（结果落盘 data/artifacts/）
│   ├── run_agent_tests.py / run_classroom_behavior_e2e.py  # 测试运行器
│   ├── cnki_selenium_smoke.py / test_cnki_headless.py      # CNKI 冒烟测试
│   ├── migrate_db_v1.3.0.py / show_metrics.py
│   └── build_claim_verifier_dataset.py / export_claim_verification_data.py
│
├── tests/                     # pytest 用例（熔断、并发、引用、意图、质量门控等）
├── docs/                      # 设计与运维文档（CNKI 排障、前端指南、意图识别设计、证据验证等）
├── data/                      # 运行数据（不入库）
│   ├── research_review.db     # 主数据库（任务、会话、论文、综述）
│   ├── pdfs/ parsed/ imports/ reviews/   # PDF 缓存与导入
│   ├── eval_bundles/          # 每次运行的诊断评估包（graph.py 自动导出）
│   └── artifacts/             # 命令行脚本产出的结果快照
├── logs/                      # 运行日志
└── chromedriver-win64/        # CNKI Selenium 专用 chromedriver
```

## 3. 核心数据流（`app/agent/graph.py: run_research_agent`）

MVP 采用顺序节点编排（无 LangGraph 依赖），节点函数在 `app/agent/nodes/`，原子能力在 `app/tools/`：

```text
用户请求（POST /api/reviews/jobs，后台异步执行）
  │
  0. 门禁   unsupported_task_guard（规划前能力边界检查，越界请求直接阻断并说明）
  1. 规划   plan_node（意图/槽位/主题消歧 → 研究计划，概念组双语门禁）
  │         provisional_route_node（研究现状类任务：搜索前候选路线）
  │         related_work 检索前就绪检查（缺 our_work 时阻断并返回澄清问题）
  2. 检索   search_rank_with_refinement（retrieval_loop 闭环：
  │           search_node 多源检索[arXiv/S2/OpenAlex/Crossref/CNKI；
  │             英文查询屏蔽 CNKI，中文查询屏蔽 arXiv/S2]
  │           → rank_node 中英双分支过滤打分 + 百分位归一化配额合并
  │           → 覆盖度不足时 refine_search 精化循环，≤2 轮）
  │         → expand_search_year；全部失败/零结果则提前终止
  3. 详情   fetch_detail_node → download_pdf_node → parse_pdf_node
  4. 抽取   extract_card_node（PaperCard：研究问题/方法/数据/指标）
  5. 路线   validate_routes_node（双语 Anchor → feature matrix → Validity/Sufficiency；
  │           仅 taxonomy 类交付物且存在候选路线时生效，否则跳过）
  │         → recovery_loop（有界证据恢复：LLM 诊断 + 确定性预算的增量补搜；
  │           仅 research_status 类任务具备 provisional_framework 时触发）
  │         → cluster_node（无证据支撑路线时的证据驱动聚类回退）
  │         → global_evidence_gate_node（综述级充分性，只测量不执行）
  6. 写作   claim_plan_node → claim_evidence_gate_node
  │         → generate_deliverables_node（write_deliverable → deliverables/renderers，
  │            四类交付物单一写作路径，含引用配额分配与逐交付物校验；
  │            异常降级为 quality_gate 阻断而非任务失败）
  │         （paper_details 为空 → retrieval_shortfall_node 按交付物给出说明）
  7. 验证   verify_claims_node / citation_check_node（以 DOI/S2/OpenAlex/arXiv ID
  │           追踪证据，双向匹配防幻觉；正文渲染为顺序编码或作者—年份引用）
  │         → claim_alignment（写作后越权主张检查）与 claim_citation_consistency
  │         → citation_gap_repair（成文引用数低于用户显式要求时增量补检索并重写，
  │            引用数与支持率同时退化才回滚）
  8. 收尾   final_answer_node（质量门禁 + 答案组装）→ 自动导出评估包到 data/eval_bundles/
```

协作式取消在每个节点边界检查（`agent/execution.py: checkpoint`，抛 `AgentCancelledError`）；
任务持久化在 SQLite，服务重启后 `ResearchJobService.recover_after_restart()` 自动恢复队列。
编辑论文集合后的增量修订（`POST /api/reviews/jobs/revise`）只重做路线验证、写作与引用验证。

## 4. 关键机制

- **后台任务模型**：长任务经 job 服务异步执行，前端轮询 `GET /jobs/{job_id}` 获取步骤与进度；节点边界支持协作式取消（`agent/execution.py`）。
- **多轮会话**：`research_conversation_service` 保存澄清问答、论文集合、生成版本；支持按序号/标题排除论文。
- **节点契约**：节点经 `@node/@requires/@provides`（`agent/decorators.py`）声明输入输出；必需输入缺失写入 `contract_violations` 并随任务结果导出供审计。
- **跨语言公平**：中英论文分支内独立过滤打分后按百分位归一化 + 软配额合并（`tools/branch_merge.py`）；规划端对概念组执行双语对齐门禁（缺失主题语言的组被显式丢弃并记录）。
- **可观测性**：`core/metrics.py` 指标、`core/circuit_breaker.py` 熔断、`core/rate_limiter.py` 限流（针对各检索源）、`agent/diagnostics.py` 诊断导出。
- **安全**：`core/security.py` 提供 API Key 校验（`X-API-Key`）与部署安全检查。
- **时间口径**：相对时间（如"近三年"）按滚动年份边界近似，不为凑篇数自动扩展；详见 README。

## 5. 启动方式

```bash
# 后端（默认 8000 端口）
python run_api.py            # 或 uvicorn app.main:app --port 8000

# 前端（8501 端口）
python run_chat_frontend.py

# 命令行提交任务
python scripts/submit_research.py

# 测试
pytest tests/
```
