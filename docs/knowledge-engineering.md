# 三模块研究工作流与可选持久化知识索引

## 1. 多源文献发现与文档处理

已有主题理解、范围约束和查询细化负责发现论文；来源客户端聚合元数据，论文级排序后再处理
入选论文 PDF。保留标题/摘要/关键词的第一层混合检索，不为所有候选论文预先全文向量化。

`parse_pdf.py` 输出逐页的既有清洗文本，新增 PDF 文件 SHA-256、解析证据基准指纹、
parser profile 和最小诊断：原始总页数、空文本页身份、文字字符数、低文本 OCR 候选页、
识别到的章节。空页在过滤前计数，非空正文页保持原页码。表格数未知时为 None，阅读顺序
未验证，低文本不是扫描或解析失败的证明。本轮没有 OCR、表格结构恢复或替换解析器。

证据坐标仍相对于 `clean_parsed_text()` 后的页面字符串，不冒充 PDF 字节、图像或新 Markdown
的坐标。`document_ingestion.py` 校验每块的论文身份、唯一编号和 `[start,end)` 回读；重复
页面、改写文本或错位均拒绝持久化。当前检索文本与来源文本相同，暂不实现派生文本重写。

## 2. 分层混合检索与结构化抽取

第一层按论文排序；第二层在单篇论文范围内按七类字段执行 BM25、dense、RRF 和可选重排。
`ChunkVectorIndex.search()` 统一 NumPy 与 pgvector 的 dense 返回契约，其他排名及抽取流程
不重写。默认使用 pgvector，NumPy 保留为显式本地后端；业务不再提供独立 sections 模式。

可选 PostgreSQL 使用独立连接、连接池、连接和语句超时；只在请求内连接，无 import-time
建表或迁移。固定维度由显式迁移生成 `vector(D)`；当前默认维度为 1024，不兼容维度拒绝。
只建立来源/快照/块的普通索引，不建立 ANN。精确查询按 tenant、论文、当前来源 revision、
active snapshot、模型与处理 profile 过滤，再以 cosine 距离和候选 `chunk_order` 稳定排序。
次级排序保留原 NumPy 候选列表顺序，不能简单改成原始页顺序。

快照覆盖与抽取覆盖分开：200 块正文、计划索引 128 块、成功写入 128 块时，计划可发布但
`index_coverage.status=partial`，`index_truncated=true`；计划内仅写入 100 块时禁止发布。
八次 LLM 抽取额度仍独立约束实际阅读，Top-K、选块计划与处理状态不能互相代替。

### 生命周期与来源变更

来源登记按 `(tenant,paper_id)` 维护单调 revision；首次登记和相同内容可直接复用。不同
来源不能自动把自己标为“更新”：必须携带调用方已观察到的 `expected_revision` 做 CAS。
更新入口是 `register-source --expected-revision`，或服务层 `register_source()`；未经登记的
内容变化会明确拒绝，不能静默用旧快照。这是防止恢复的旧任务覆盖新来源的保守边界。

来源、解析器/切块器规则及候选次序、Embedding provider/端点/模型/维度/缓存版本共同决定
快照身份。相同 active 快照复用不重新编码文档；新快照构建可利用既有 VectorCache。
新来源或新 profile 的构建失败不修改旧快照，但旧来源不能用于当前请求。

building 快照含 builder ID、递增 fencing token 和数据库时间租约。外部 Embedding 不持有
数据库事务；批次前续租、批次写入及发布时再次检查来源和所有权。租约过期后可接管并重建
staging；旧 builder 无权续租、写入或发布。计划数量、连续次序和原文映射确认后，短事务
切换 active。取消/预算传播，未能终结的 building 留待过期接管，不继续烧费补建。

## 3. 研究 Agent 编排与证据质量控制

沿用主控循环、动作契约、预算、检查点、有界检索恢复、主张授权、引用验证和草稿隔离。
持久化命中只是候选；回读仍对照本次解析原文，逐字摘录检查和字段门禁继续执行。
数据库不可用必须停止，不再以 sections 降级替代持久化检索。来源/模型/权限/
所有权错误直接拒绝，不允许变成规则抽取成功。两类覆盖状态随运行卡片及检查点 JSON 保留。

本轮 tenant 是受信任部署配置的隔离作用域，不是面向外部请求的 RBAC/用户认证。尚未实现
组织权限、撤销访问推送或跨论文自由问答；不要将单篇过滤宣传为完整企业权限治理。

## 原生 PostgreSQL 部署、迁移与回滚

```powershell
python -m pip install -r requirements-vectorstore.txt
# 将开发口令放在本地私有环境变量，不提交到仓库。
docker compose -f compose.vectorstore.yml up -d
```

在本地私有配置设置独立 `CARD_VECTOR_DATABASE_URL`，数据库为 compose 的 research_vectors。
业务 `DATABASE_URL` 保持原值。迁移任务需扩展/DDL 权限，业务运行账号只需相应表读写权限。

```powershell
python scripts/manage_vectorstore.py migrate
python scripts/manage_vectorstore.py health
```

业务默认使用 LLM 抽取、Chunk Hybrid 与 pgvector；仍需配置模型与独立 DSN、显式迁移。
只填写 DSN 不会触发模型或自动迁移。启动前运行 `python scripts/check_rag_runtime.py`。

PDF 确认更新后，显式使用已观察到的 revision 登记，再执行研究任务：

```powershell
python scripts/manage_vectorstore.py register-source --paper-id PAPER --pdf INPUT.pdf --expected-revision 1
```

可显式切换到 numpy 进行本地检索，该操作不会删除向量表。若确实需要删除向量表及数据，
使用下列显式命令；保留 vector 扩展，不触及业务 SQLite：

```powershell
python scripts/manage_vectorstore.py rollback --confirm-vector-data-loss
```

## 测试与边界

普通 pytest 不依赖 PostgreSQL，DB 用例在未提供测试 DSN 时明确 skipped。只对专用测试库设置
`RRA_VECTOR_TEST_DATABASE_URL` 后运行 `tests/test_pgvector_store.py`；用例清理随机测试租户。
原生 PostgreSQL 设置 `RRA_VECTOR_TEST_NATIVE=1` 才执行多连接竞争用例。禁止对生产库运行。

无原生服务器的 Windows 环境可用官方 PGlite 做 PostgreSQL 协议、持久化和扩展 SQL 验证：

```powershell
npm.cmd install --prefix data/vector_test_runtime --ignore-scripts --no-audit --no-fund --legacy-peer-deps @electric-sql/pglite@0.5.12 @electric-sql/pglite-pgvector@0.0.13 @electric-sql/pglite-socket@0.2.15
node scripts/vectorstore_wasm.cjs
```

WASM 服务器仅绑定本机、单连接、不带业务账号鉴权，限开发测试。输入 stop 可优雅关闭。
该协议代理不具备原生会话隔离；其测试 fixture 禁用自动预备语句，原生测试保留开启。
通过 CLI 访问 WASM 时另设置 `CARD_VECTOR_PREPARED_STATEMENTS=false`；原生服务器默认 true。
这不是原生 PostgreSQL 的并发或生产部署证明。向量/LLM 都使用确定性替身；真实语义准确率、
字段完整率、费用收益及最终综述成功需要另行预算和人工审查，本轮未执行。
