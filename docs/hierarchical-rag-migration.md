# 分层 RAG 默认链路与旧代码清理（2026-10-28）

研究任务统一使用论文级 Hybrid、全文/摘要 LLM 抽取和字段级 Chunk Hybrid。
默认向量后端为独立 pgvector；连接、驱动、迁移或维度不满足时明确停止。
配置启用不代表真实模型已验收，本轮未新增 Provider 运行。

## 当前业务路径

```text
研究主题 → 多源论文发现
→ 论文级 BM25 + dense + RRF + Rerank
→ 独立逐篇语义准入及元数据复核
→ 合法开放 PDF 获取、解析
→ 章节/页面感知 Chunk → pgvector 索引
→ 字段级 BM25 + dense + RRF + Rerank，与章节覆盖共同选块
→ 有界 LLM 原文摘录 → PaperCard → 来源/位置及字段质量门禁
→ 主张规划 → 写作 → 主张/引用核验 → 发布门禁
```

无全文时允许明确标注的摘要级 LLM 抽取；它不具备全文覆盖记录，也不能宣传为
全文分层检索已经执行。无合法内容时只保留 invalid 元数据卡。普通模型失败不生成
规则内容卡；块级失败保留此前可核对的摘录和 partial 覆盖。取消、预算、租约、
来源完整性和向量库错误继续传播，不能转为抽取成功。

## 已删除与保留

已删除：

- _extract_rule_based 及其三处失败回退，包括批量 worker 的再次规则补卡。
- 规则专用的句子查找、指标/数据集词表、样本量/模态/研究设计的自动字段补全。
- rank_node 的 rules 与 hybrid_shadow 业务分支、分支诊断辅助代码。
- 检索循环末尾旧规则榜单的全量/增量 LLM 再重排。
- ENABLE_LLM_CARD_EXTRACTION 开关，独立 sections 业务模式和数据库失败后 sections 回退。

仍保留：

- 确定性的年份、题名、范围、身份及文献类型硬过滤。
- Hybrid 的 BM25、RRF、章节覆盖、模型故障诊断与所有证据质量门禁。
- rank_papers 等离线评分工具，供冻结候选池对照，不作为研究任务的规则排序入口。
- 显式 NumPy 向量后端，用于本地/离线检索；选择它不代表 pgvector 已运行。
- 历史卡片、报告的读取兼容与旧 profile 迁移，不重新引入旧执行分支。

旧规则/影子会话恢复时撤销旧排序与派生主张授权，将旧榜单回交候选池。原始证据、
用户约束、年份/篇数和累计预算保留。流水线版本进入已验证草稿指纹，旧通过标记不能
直接用于当前执行的 finish 授权。迁移不修改已归档真实实验结论。

## 安装与配置

安装 requirements.txt，现已包含 BM25、分词以及 psycopg/pgvector 驱动。
requirements-retrieval.txt 保留兼容安装入口；requirements-vectorstore.txt 可独立用于数据库测试。

私有 .env 的非敏感功能值：

```dotenv
RETRIEVAL_RANKING_MODE=hybrid
ENABLE_PDF_PIPELINE=true
CARD_CHUNK_RETRIEVAL_MODE=hybrid
CARD_CHUNK_ENABLE_RERANK=true
CARD_VECTOR_BACKEND=pgvector
CARD_VECTOR_DB_FAILURE_POLICY=fail
```

另外配置真实 LLM、Embedding、Reranker 端点及凭据，以及独立 CARD_VECTOR_DATABASE_URL。
不要复用业务 SQLite 的 DATABASE_URL。旧配置中的 rules、hybrid_shadow、sections 会被拒绝；
删除 ENABLE_LLM_CARD_EXTRACTION，内容抽取现在始终需要模型。

数据库部署、显式迁移与回滚步骤见 [知识工程文档](knowledge-engineering.md)。

```powershell
python scripts/manage_vectorstore.py migrate
python scripts/check_rag_runtime.py
```

启动检查为只读，不调用模型或自动建表。检查扩展可访问性、迁移版本、向量维度和运行表。
--config-only 只看配置，不能证明数据库已连通。ready=true 也不证明 Provider 端点可用、
模型字段准确或综述成功。进程中的 Settings 有缓存，修改配置后重启服务再执行新任务。

## 验收与实际环境边界

离线测试的模型是显式替身；语义准入、来源回读、质量门禁和取消/预算仍运行。
验证器用例直接构造已知来源卡片，避免依赖已删除的抽取算法。只针对旧执行模式的测试
随接口删除，保留对应硬过滤、混合检索、准入失败、上下文、增量池和草稿隔离测试。

启动检查及迁移用例见 tests/test_rag_runtime_migration.py；数据库只读就绪及维度回归见
tests/test_pgvector_store.py。本机复用隔离的 PGlite PostgreSQL 协议环境测试真实 SQL 和
pgvector；原生多连接竞争仍需 RRA_VECTOR_TEST_NATIVE=1 的原生服务器，不能由此替代。

最终回归为 1782 passed、1 skipped、11 条第三方弃用警告，72.40 秒。
新增 17 项启动/迁移用例与 1 项数据库就绪/维度用例。首次运行、继续、重生成和直接恢复
均在旧状态迁移前检查取消；已取消任务不能修改历史排序或授权。隔离测试服务器在验收后关闭。

上述代码迁移验收时尚未提供独立数据库连接，因此当时启动检查明确失败。
测试服务器没有被写入业务 .env，测试用专用数据目录和租户，不触及业务 SQLite。
真实模型验收仍需另定预算；上一轮新增尝试 2/2 已耗尽。本轮不宣称真实全文综述成功，
也不宣称字段准确率、幻觉率或 Token 成本已有测量改善。

2026-10-28 后续安装：rragent 中原生 PostgreSQL 16.15＋pgvector 0.8.6 已安装，
独立应用库已创建并迁移，私有连接已配置，启动检查 ready=true。
原生多连接及启动回归 34 项通过。见 [Conda 原生验收](validation/2026-10-28-conda-native-pgvector.md)；
该结果仍不代表真实 Provider E2E 完成。
