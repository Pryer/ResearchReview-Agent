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
| 测试 | pytest（`tests/`，本地回归与真实外部验收分开记录） |

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
│   │   ├── graph.py           # run/continue/regenerate 入口、动作处理器适配与写作质量边界
│   │   ├── state.py           # ResearchAgentState TypedDict 定义
│   │   ├── main_loop.py / main_policy.py # 五字段单动作决策循环、原生工具与 JSON 回退
│   │   ├── action_registry.py # 注册动作、参数 schema 与可用性
│   │   ├── controller.py      # 授权、任务生命周期、版本复核与提交
│   │   ├── action_contracts.py / task_context.py # 动作读写白名单、类型校验与输入投影
│   │   ├── result_merger.py   # 结果身份、指纹复核及受限补丁合并
│   │   ├── execution_budget.py / orchestration.py # 共享预算与旧会话模式规范
│   │   ├── context_builder.py # 由完整状态构建五字段主 Agent 上下文
│   │   ├── context_compaction.py # 保留硬约束和阻塞问题的确定性压缩
│   │   ├── subagents/         # Search / Analysis / Writing 专业 Agent 适配器
│   │   ├── execution.py       # 执行原语：AgentCancelledError、节点边界检查、LLM 工厂
│   │   ├── retrieval_loop.py  # ReAct 检索精化循环（从 graph 下沉的策略层）
│   │   ├── recovery_loop.py   # 路线证据恢复状态机（从 graph 下沉的策略层）
│   │   ├── planner.py / slot_extractor.py / intent.py  # 规划、槽位提取与意图识别
│   │   ├── writing_plan.py / claim_plan.py  # 写作计划构建与声明计划
│   │   ├── route_validator.py / route_targets.py / provisional_routes.py  # 路线验证、目标推导与候选路线
│   │   ├── evidence_recovery.py / generation_recovery.py  # 证据恢复与生成恢复
│   │   ├── global_evidence_gate.py / focus_coverage.py  # 全局证据门禁与焦点覆盖
│   │   ├── deliverable_router.py / evidence_roles.py  # 交付物路由与证据角色
│   │   ├── diagnostics.py / state_invariants.py / exceptions.py  # 诊断导出、状态不变量与异常
│   │   ├── research_semantic_parser.py / topic_disambiguation.py  # 研究语义分析与主题消歧
│   │   ├── search_plan_builder.py / semantic_consistency.py  # 搜索计划与语义一致性
│   │   ├── tool_registry.py / pipeline_stages.py / research_plan.py  # 工具映射、流水线阶段与研究计划
│   │   └── nodes/             # 流程节点，按阶段分组：planning / retrieval / extraction / synthesis / verification
│   ├── api/                   # HTTP 路由层
│   │   ├── routes_review.py   # 研究任务：提交/查询/取消/修订/会话记忆/历史综述
│   │   ├── routes_paper.py    # 论文检索与详情、论文卡片
│   │   └── routes_library.py  # 本地论文库：PDF 导入、搜索
│   ├── services/              # 业务服务层
│   │   ├── research_job_service.py        # 后台任务队列、重启恢复、取消
│   │   ├── research_conversation_service.py  # 多轮会话（澄清、论文集合、版本与修订历史）
│   │   ├── research_execution_service.py / durable_execution_service.py # 资料、预算和租约作用域
│   │   ├── research_artifact_service.py # 会话级不可变资料、分片与恢复
│   │   ├── research_memory_service.py # 截断前历史归档、迁移与增量摘要游标
│   │   ├── llm_service.py / paper_service.py / review_service.py
│   │   └── library_service.py
│   ├── clients/               # 外部检索源客户端：arxiv / semantic_scholar / openalex / crossref / cnki
│   ├── tools/                 # 原子工具：检索调度、过滤打分（rank_papers + paper_matching /
│   │                          #   paper_rerank / venue_tiers）、跨语言分支合并（branch_merge /
│   │                          #   language_router / language_filter）、写作分发（write_deliverable）、
│   │                          #   引用生成与校验、PDF 下载解析等
│   ├── deliverables/          # 四类交付物规格与渲染器（背景/现状/相关工作/叙述性综述）+ few-shot 蓝本
│   ├── schemas/               # Pydantic 模型（agent / paper / review / verification / taxonomy 等）
│   ├── database/              # db.py / models.py / repositories.py；runtime_repository.py 管理 CAS/账本/快照
│   ├── core/                  # 横切能力：config、logger、exceptions、安全（API Key）、熔断器、限流、
│   │                          #   指标、引用语法、引用密度、文本质量、检索源能力声明（source_capabilities）
│   ├── frontend/              # Streamlit 界面（chat_app.py 主界面、query_utils.py、progress_labels.py 进度标签）
│   ├── prompt/ + prompt_catalog.py  # 提示词与懒加载目录；prompt/writing/ 含章节、交付物、引言、相关工作等写作提示
│   └── utils/                 # 通用工具（日期、去重、PDF、文本清洗、标题清洗 title_cleaner）
│
├── scripts/                   # 运维/测试脚本
│   ├── check_llm_api.py       # OpenAI 兼容 LLM 接口连通性检查
│   ├── migrate_agent_runtime.py # 幂等添加执行、任务、请求、检查点、事件和摘要游标表
│   ├── submit_research.py     # 命令行提交研究任务
│   ├── monitor_job.py         # 任务监控（结果落盘 data/artifacts/）
│   ├── run_agent_tests.py / run_classroom_behavior_e2e.py  # 测试运行器
│   ├── cnki_selenium_smoke.py / test_cnki_headless.py      # CNKI 冒烟测试
│   ├── migrate_db_v1.3.0.py / show_metrics.py
│   ├── inspect_eval_bundle.py # 评估包内容检查
│   └── build_claim_verifier_dataset.py / export_claim_verification_data.py
│
├── tests/                     # pytest 用例（熔断、并发、引用、意图、质量门控等）
├── examples/                  # 标准场景（叙述性综述/相关工作/不支持请求）与对应的预期状态
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

当前只采用主 Agent 单动作决策循环；无 LangGraph 依赖。
节点函数在 `app/agent/nodes/`，原子能力在 `app/tools/`。

主 Agent 基于五字段上下文每轮选择一个已注册动作；动作权限、预算、取消、提交
版本与质量门禁仍由确定性代码裁决。历史会话（未记录模式或标记为 `legacy` 的旧
状态）在执行入口规范为同一调度：保留原研究目标、显式约束、论文与证据卡、恢复
历史与预算消耗，只重算与旧调度或旧证据版本绑定的派生结果。模型可见动态上下文只包含 `goal`、
`state`、`key_evidence`、`decisions`、`open_questions`；完整论文、证据和历史保留在
权威状态与会话级 artifact 中。检索、分析和写作阶段通过带输入指纹的专业 Agent
任务契约按动作白名单投影输入，再在隔离工作副本中执行；输出键、字段删除和类型
双重校验，提交前再次检查身份、指纹和版本，旧任务不会合并到新证据状态。

### 3.1 autonomous 控制循环

```text
请求 → 需求解析 → 构建五字段 → 主 Agent 选择一个动作
                                  ↓
                     Controller 授权与预算检查
                                  ↓
                     Search / Analysis / Writing
                                  ↓
           结果校验 → 数据库 CAS 提交 → 重建五字段 → 下一轮

申请交付 → 确定性质量门禁 → 完成 / 明确降级 / 拒绝交付
```

数据库提交保证由会话服务入口提供；直接调用 graph 的脚本仍是内存执行模式。

### 3.2 研究阶段与节点

下图描述各阶段的职责和典型依赖；这些节点被组成注册动作，由主 Agent 在权限和
前置条件允许时选择，不固定逐行执行此图。

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
  │         → global_evidence_gate_node（输出综述级充分性诊断，供恢复及交付门禁消费）
  6. 写作   claim_plan_node → claim_evidence_gate_node
  │         → generate_deliverables_node（write_deliverable → deliverables/renderers，
  │            四类交付物单一写作路径，含引用配额分配与逐交付物校验；
  │            异常降级为 quality_gate 阻断而非任务失败）
  │         （paper_details 为空 → retrieval_shortfall_node 按交付物给出说明）
  7. 验证   verify_claims_node / citation_check_node（以 DOI/S2/OpenAlex/arXiv ID
  │           追踪证据，双向匹配防幻觉；正文渲染为顺序编码或作者—年份引用）
  │         → claim_alignment（写作后越权主张检查）与 claim_citation_consistency
  │         → citation_gap_repair（成文引用数低于用户显式要求时增量补检索并重写，
  │            候选质量向量退化则保留原已验证版本）
  8. 收尾   final_answer_node（质量门禁 + 答案组装）→ 自动导出评估包到 data/eval_bundles/
```

协作式取消在每个节点边界检查（`agent/execution.py: checkpoint`，抛 `AgentCancelledError`）；
后台 job 与执行账本持久化在 SQLite。重启恢复会重新排队 queued job，并将中断的
running job 标为失败、完成待取消 job 的取消；它不自动重放在途研究动作。
中断研究通过 `resume_from_checkpoint=true` 显式恢复，保留累计预算和原约束，
崩溃租约到期后才能接管。待澄清与已完成会话分别使用澄清、修订入口。
编辑论文集合后的增量修订（`POST /api/reviews/jobs/revise`）只重做路线验证、写作与引用验证。

## 4. 关键机制

- **后台任务模型**：长任务经 job 服务异步执行，前端轮询 `GET /jobs/{job_id}` 获取步骤与进度；节点边界支持协作式取消（`agent/execution.py`）。
- **多轮会话**：`research_conversation_service` 保存澄清问答、论文集合、生成版本；支持按序号/标题排除论文。
- **节点契约**：节点经 `@node/@requires/@provides`（`agent/decorators.py`）声明输入输出；必需输入缺失写入 `contract_violations` 并随任务结果导出供审计。
- **动作契约**：`action_contracts.py` 定义十个专业动作的输入和输出字段；未知新增字段默认不可见、不可提交，任务不能修改用户硬约束或共享预算。
- **原子提交与取消**：会话租约与数据库 CAS 将任务结果、资料、检查点、事件和预算一起提交或回滚；后台领取也使用 queued→running CAS。取消先提交时拒绝迟到结果。
- **用量账本**：每次真实请求先预留、响应后结算，解析失败/空内容重试/备用请求均记账；崩溃未决请求保守计为 estimated/uncertain，旧执行不能覆盖新账本。
- **历史与恢复**：展示历史截断前归档，摘要只读取游标之后的新事件；读取时优先最新已提交检查点，旧版本已丢失历史不伪造恢复。
- **提示词与复用**：固定规则和输出结构前置、动态材料后置；原生工具目录只经 tools 传入。章节指纹覆盖真实授权、原稿和章节依赖，复用后仍运行完整验证链。
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

升级后重启服务，`init_db()` 会添加执行表；也可先运行
`python scripts/migrate_agent_runtime.py`，迁移不会删除既有资料。

## 6. 文档与验证边界

分层职责见 [docs/architecture.md](docs/architecture.md)，五字段上下文见
[上下文架构](docs/agent-context-architecture.md)，CAS、预算和恢复细节见
[运行机制](docs/agent-runtime-and-cache.md)。

2026-09-22 运行时与缓存修复验收的全量回归为 1224 项通过（后续检查见变更日志）；真实备用 LLM + CNKI 3 篇检索、
提交和数据库重开恢复已通过。40 篇完整综述场景因请求预留超过剩余预算而阻断，
尚未完成交付验收，未据此宣称确定节费比例。详见
[验收记录](docs/validation/2026-09-22-runtime-cache.md)。
