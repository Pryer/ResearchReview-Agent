# ResearchReview-Agent 📚

> 面向多学科研究主题的检索、证据结构化与引用可验证综述智能体

根据用户自然语言指令识别研究约束与主题歧义，自动检索开放论文、生成结构化论文卡片、按当前证据归纳研究脉络，最终生成带可验证引用的学术综述。

---

## ✨ 核心能力

| 能力 | 说明 |
|------|------|
| 研究请求理解 | LLM开放语义解析 + 规则校验，抽取主题、年份、篇数、范围与四类交付物 |
| 多轮研究会话 | 保存澄清、论文集合、生成版本和修订历史；支持按序号/标题排除论文 |
| 后台任务控制 | 长任务异步执行，可查询步骤与进度，并在节点边界协作式取消 |
| 增量重生成 | 编辑论文集合后只重做聚类、写作和引用验证，不重复检索与详情补全 |
| 多源检索 | arXiv + Semantic Scholar + OpenAlex + Crossref + CNKI（Selenium） |
| 失败诊断 | 区分"真空结果"与"检索失败"，避免误判关键词不合适 |
| 论文排序 | 相关性 + 质量综合打分，自动去重 |
| 按源引用量 | 记录 `citation_count_by_source`，避免跨源口径混淆 |
| PDF 解析 | PyMuPDF 全文/分段提取 |
| PaperCard | 结构化抽取研究问题、方法、数据、指标 |
| 文献矩阵 | 保存作者、完整题名、DOI、文献类型、证据等级、研究设计、样本规模、模态、结果与局限 |
| 文献聚类 | 从论文证据归纳理论、实证、方法、应用等研究脉络 |
| **📝 四类交付物** | 研究背景、研究现状、论文相关工作、叙述性综述初稿 |
| 引用校验 | 双向匹配降低论文幻觉，支持 claim_evidence_map 逐句核验（可选） |

时间范围说明：相对时间按滚动年份边界近似。例如系统在 2026 年处理“近三年”时，
检索年份元数据为 2023—2026 年，当前年度只纳入检索时已正式出版或在线公开的记录。
用户明确给出的时间范围不会为凑足篇数而自动向前扩展；数量不足时系统会报告实际篇数。

正文生成与内部检索标识分离：主张验证阶段使用 DOI、Semantic Scholar、OpenAlex 或
arXiv ID 追踪证据，最终论文正文统一渲染为顺序编码或作者—年份引用，完整元数据进入参考文献表。

---

## 🚀 快速开始

### 1. 创建 conda 环境（推荐）

```bash
conda env create -f environment.yml
conda activate rragent
```

如果环境已存在，只需更新依赖：

```bash
conda activate rragent
python -m pip install -r requirements.txt
python -m pip install pytest==7.4.0
```

### 2. 配置环境变量

```bash
copy .env.example .env
# 编辑 .env，填入 LLM_API_KEY 等配置
```

可在启动完整服务前验证 OpenAI 兼容接口配置：

```bash
python scripts/check_llm_api.py --target primary
```

### 3. 启动 API 服务

```bash
conda activate rragent
python run_api.py
```

访问 http://127.0.0.1:8000/docs 查看 Swagger UI。

### 4. 启动前端（可选）

```bash
conda activate rragent
python run_chat_frontend.py
```

访问 http://127.0.0.1:8501 使用前端。

### pip 安装方式（备选）

```bash
pip install -r requirements.txt
```

---

## 📡 API 使用示例

### 后台任务、取消与增量修订

前端默认使用后台任务接口，提交后会立即获得 `job_id`：

```bash
curl -X POST http://localhost:8000/api/reviews/jobs \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-001","user_query":"调研近三年某主题论文并生成综述"}'
```

使用 `GET /api/reviews/jobs/{job_id}` 查询状态，或调用
`POST /api/reviews/jobs/{job_id}/cancel` 请求取消。取消为节点边界协作式取消：
正在执行的单次外部 API/LLM 请求结束后停止，不会继续进入下一个步骤。

完成后可以按稳定 ID 排除论文并增量重生成：

```bash
curl -X POST http://localhost:8000/api/reviews/jobs/revise \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-001","excluded_paper_ids":["doi:example"],"instruction":"该论文与主题不直接相关"}'
```

也可只传自然语言指令，例如“删除第 2、5 篇后重新生成”。会话记忆可通过
`GET /api/reviews/sessions/{session_id}` 查询。

### 自然语言 Agent 请求

Streamlit 会自动创建 `session_id`。直接调用 API 时，传入稳定的 `session_id` 即可启用主题澄清与续跑：

```bash
curl -X POST http://localhost:8000/api/reviews/jobs \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-001","user_query":"调研近三年课堂行为分析论文，引用不少于40篇，并生成研究背景和研究现状"}'
```

轮询任务后若结果为 `status=needs_clarification`，使用同一会话提交自由文本回答：

```bash
curl -X POST http://localhost:8000/api/reviews/jobs \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-001","user_query":"偏向技术识别与教育解释的交叉研究","clarification_answer":"偏向技术识别与教育解释的交叉研究"}'
```

#### 生成叙述性综述初稿
```bash
curl -X POST http://localhost:8000/api/reviews/jobs \
  -H "Content-Type: application/json" \
  -d '{"user_query": "帮我调研近五年课堂行为识别相关论文，并生成中文文献综述，引用不少于10篇。"}'
```

#### 生成相关工作章节
```bash
curl -X POST http://localhost:8000/api/reviews/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "生成少样本动作识别的相关工作章节",
    "state": {
      "our_work": {
        "research_problem": "低样本条件下的动作类别泛化",
        "method_name": "TempFSL",
        "method_summary": "时序原型网络结合跨域迁移",
        "innovations": ["动态时间规整模块", "跨域自适应策略"]
      }
    }
  }'
```

### 结构化检索

```bash
curl -X POST http://localhost:8000/api/papers/search \
  -H "Content-Type: application/json" \
  -d '{"query": "vision transformer", "start_year": 2021, "end_year": 2025, "max_results": 20}'
```

---

## 🧪 运行测试

```bash
conda activate rragent
pytest
```

---

## 📁 项目结构

```
ResearchReview-Agent/
├── app/
│   ├── agent/        # 意图识别、槽位抽取、规划、路由、节点、图编排
│   │   ├── prompt_templates/  # 写作 Prompt 契约（review/related_work/introduction）
│   │   ├── graph.py           # 四类交付物的主工作流入口（run/continue/regenerate）
│   │   ├── nodes/             # 按阶段分组的节点包：planning/retrieval/extraction/synthesis/verification
│   │   ├── retrieval_loop.py  # ReAct 检索精化循环
│   │   ├── recovery_loop.py   # 路线证据恢复状态机
│   │   ├── intent.py          # 兼容意图识别；能力边界由 Guard 控制
│   │   └── state.py           # 状态定义（含新字段）
│   ├── api/          # FastAPI 路由
│   ├── clients/      # arXiv / Semantic Scholar / OpenAlex / Crossref / CNKI 客户端
│   ├── core/         # 配置、日志、异常
│   ├── database/     # SQLAlchemy 模型与仓储（支持 citation_count_by_source）
│   ├── frontend/     # Streamlit 前端
│   ├── schemas/      # Pydantic 数据模型（含 SourceDiagnostic）
│   ├── services/     # 业务服务层
│   ├── tools/        # 检索、排序、PDF、聚类、写作分发、引用工具
│   │   ├── write_deliverable.py     # 四类交付物统一写作分发
│   │   ├── verify_claims.py         # 句子级 Claim–Evidence 校验
│   │   └── fetch_metadata.py        # 元数据补全（支持按源记录引用量）
│   ├── deliverables/ # 四类交付物规格与渲染器（背景/现状/相关工作/叙述性综述）
│   ├── prompt/       # 提示词与懒加载目录（prompt_catalog.py）
│   ├── utils/        # 文本清洗、去重、文件工具
│   └── main.py       # FastAPI 实例：路由注册、CORS、lifespan 建表与任务恢复
├── tests/            # pytest 测试
├── data/             # 运行时产生的 PDF / 解析结果 / 综述 / 评估包
├── environment.yml   # conda 环境配置
├── run_api.py            # API 启动入口
├── run_chat_frontend.py  # Streamlit 前端启动入口
└── requirements.txt  # 依赖清单
```

---

## 🔧 配置说明

编辑 `.env` 文件：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | LLM 提供商 | `deepseek` |
| `LLM_API_KEY` | API 密钥 | 必填 |
| `LLM_BASE_URL` | API 地址 | DeepSeek |
| `LLM_MODEL` | 模型名称 | `deepseek-v4-flash` |
| `LLM_THINKING_ENABLED` | DeepSeek V4 思考模式；关闭时显式发送 `thinking.type=disabled` | `false` |
| `LLM_THINKING_EFFORT` | 最终写作开启思考时的推理强度 | `low` |
| `LLM_THINKING_MAX_TOKENS` | 最终思考写作的单次输出预算 | `32768` |
| `LLM_MAX_TOKENS` | 最终成文输出预算 | `8192` |
| `LLM_CONTROL_PLANE_MAX_TOKENS` | 语义解析、筛选、聚类和验证等结构化任务预算 | `4096` |
| `LLM_FAILOVER_TOTAL_TIMEOUT` | 单个逻辑请求跨主/备用模型的总超时（秒） | `180` |
| `DATABASE_URL` | 数据库地址 | SQLite |
| `SEMANTIC_SCHOLAR_API_KEY` | S2 可选密钥 | 空 |
| `APP_API_KEY` | 共享部署时保护业务接口的可选密钥 | 空（仅建议本地） |
| `CORS_ALLOWED_ORIGINS` | 允许访问API的前端来源，逗号分隔 | 本机8501端口 |

---

## 📋 实现路线图

- [x] 阶段 1：项目骨架 ✅
- [x] 阶段 2：意图识别和槽位抽取（规则 + LLM 双路径完成）✅
- [x] 阶段 3：论文检索（五源客户端完成，支持失败诊断）✅
- [x] 阶段 4：论文解析和 PaperCard ✅
- [x] 阶段 5：综述生成 ✅
- [x] 阶段 6：Agent 串联 + Streamlit 联调 ✅
- [x] **阶段 7：四类交付物写作能力（背景、现状、相关工作、叙述性综述）✅**
- [x] **阶段 8：引用量按源记录与失败诊断 ✅**
- [x] **阶段 9：Evidence Card 与句子级 Claim–Evidence 引用验证 ✅**

## 📚 文档与案例

- [AGENTS.md](AGENTS.md)：Coding Agent 必须遵守的规则、验证要求和完成标准。
- [docs/architecture.md](docs/architecture.md)：系统分层、模块边界和运行组件。
- [docs/research-workflow.md](docs/research-workflow.md)：Agent 工作流、阶段契约和分支保证。
- [docs/evidence-model.md](docs/evidence-model.md)：PaperMetadata、Evidence、Claim、Citation 的关系。
- [docs/quality-gates.md](docs/quality-gates.md)：能力、路线、主张、交付物和生成质量门禁。
- [docs/lessons-learned.md](docs/lessons-learned.md)：历史缺陷、根因和防回归经验。
- [docs/change-log.md](docs/change-log.md)：每次完成修改后的时间戳变更记录。
- [examples/](examples/)：不依赖真实外部服务的标准请求案例。

### 可验证综述生成

当前 Agent 会在论文详情获取后生成带原文位置的 Evidence Card，并在综述生成后逐句检查：

- 事实性主张是否有引用；
- 引用论文是否存在；
- 证据片段是否支持主张；
- 数字、数据集与强措辞是否得到证据支持；
- 验证失败时给出降级措辞或补充证据建议。

Agent API 的 `claim_verification` 字段提供完整报告，Streamlit 的“引用验证”页可查看
原句、证据片段、章节/页码和修改建议。设计与 QLoRA 数据导出说明见
[`docs/evidence_verification.md`](docs/evidence_verification.md)。

### 当前实现状态

**新增写作能力：**
- ✅ 相关工作章节生成（经 `write_deliverable` → `deliverables/renderers/related_work_renderer.py`）
- ✅ 研究背景、研究现状和叙述性综述生成（经 `deliverables/renderers/`）
- ✅ 支持 `claim_evidence_map` 逐句核验（可选）

**数据质量提升：**
- ✅ `citation_count_by_source` 按数据源分别记录引用量
- ✅ `SourceDiagnostic` 区分真空结果与检索失败

**意图识别增强：**
- ✅ 相关工作与背景类请求统一路由到四类核心交付物
- ✅ 关键词优先级调整（"related work" 不再误判为 search_papers）

> 注：早期版本的独立工具 `generate_review.py` / `generate_related_work.py` /
> `generate_introduction.py` 已在后续重构中删除，四类交付物统一走
> `write_deliverable.py` 与 `app/deliverables/` 渲染器单一写作路径。

---

## 📜 合规约束

本 Agent 遵守以下原则：

1. 只检索公开论文元数据
2. 只下载开放获取 PDF，不绕过付费墙
3. 不编造论文、作者、年份、引用量
4. 综述引用必须来自已检索到的论文
5. 证据来源明确标注（abstract / full_text）
