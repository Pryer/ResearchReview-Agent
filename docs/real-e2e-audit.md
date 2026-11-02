# 分层 RAG 真实验收与审查导出

## 运行前

Runner 使用新版 Hybrid 配置，独立 SQLite、唯一 `e2e_` 向量租户及独立缓存/PDF/解析/输出目录。
向量连接沿用部署侧私有配置，实验租户与日常租户隔离；不会隐式建向量表或修改迁移版本。
需先按向量数据库部署文档完成迁移。共享数据库内的租户隔离不是独立数据库，也不是用户权限认证。

默认模式仅检查配置和数据库，不访问学术来源、不调用 LLM、Embedding 或 Reranker：

```powershell
conda activate rragent
python scripts/run_real_e2e.py --task T01 --precheck-only --output-dir data/real_e2e/precheck
```

省略 `--precheck-only` 也只执行预检。预检成功状态为 `precheck_passed`，不会计作真实研究成功，
`primary_pre_registered_run=false`。RAG 预检会检查配置、实际向量库迁移及版本；模型配置存在不代表端点可用。
预检必须通过才进入业务数据库初始化和研究流程。NumPy 不能代替本 Runner 的 pgvector 验收要求。

真实执行需显式传 `--execute`、新的单次 Token 预算和时限。例如下面是调用格式，
不表示已获新的测试额度或已经执行：

```text
python scripts/run_real_e2e.py --task <task_id> --execute --token-budget <new_budget> --deadline-seconds <new_deadline> --output-dir data/real_e2e/new_round
```

旧一轮的日期/次数上限仍有效；本次代码改造不追加真实 Provider 额度。Runner 不自动安排重跑或累计扩大预算。
Token 账本、真实 Provider 上报用量及货币费用是不同口径；没有定价信息不能据 Token 数宣称精确费用。

## 版本与实验身份

在 Settings 加载前先写最小 `run.json`，即使配置/应用导入失败，也保留阶段、异常类型和失败终态。
不导出异常原文，防止校验错误和数据库错误回显密钥。JSON 更新采用同目录临时文件原子替换。

首次真实执行在第一笔模型调用前冻结任务、目标题名/强身份、来源、预算、模型身份、质量阈值、
Python/主要依赖版本、端点指纹、Pipeline Version 及运行前 RAG 检查结果。
代码身份包含 Git SHA、相对 HEAD 的受跟踪差异哈希，以及 app/scripts 全部 Python 源码清单和内容哈希；
清单包含尚未跟踪的代码。依赖清单及公开配置模板也计算哈希，不读取私有 .env 内容进入清单。
运行后再次检查源码是否改变，身份不稳定或导出不完整时返回非零退出码。

所有尝试使用新目录，不覆盖前次产物。重跑使用 `--rerun-of <previous_execute_run_directory>`：
必须保持原任务完全一致，保存原运行 ID、实验 ID 和父报告哈希。预检不作为主运行，
重跑不能放宽原请求；更换约束应另立实验并明确说明。首次主运行是调用方声明的新实验，
本脚本没有跨机器的全局实验注册中心，不能据此证明所有历史尝试都已被收集。

## 产物与口径

| 产物 | 内容 |
| --- | --- |
| run.json | 版本、冻结配置、预检、运行终态、实际观察、用量账本、导出完整性 |
| result.json / research.db | 公开业务结果与独立的完整运行状态、检查点 |
| observations.jsonl | 本运行进程实际观察到的排序、准入、模型操作和向量查询 |
| paper_retrieval.json | 预设目标在候选、排序窗口及准入阶段的命中，逐轮保留 |
| chunk_retrieval.json | 每篇实际检索诊断、字段 Top-K、快照、降级及抽取覆盖 |
| claim_evidence.csv | 原子主张、逐篇引用、选定证据、来源回读和待人工判定列 |
| manual_constraints.csv | 冻结的硬约束与独立人工检查列 |
| review_template.json | 原始审查样本，用于检测自动字段/样本被修改 |
| state_evidence.json / root_cause_chain.json | 有状态来源定位的观察条件；未证实的因果边为空 |
| evaluation_summary.json | 明确分子分母、人工覆盖率、未知数量与基线比较资格 |

`rag_runtime_precheck` 证明运行前条件；`rag_runtime_observed` 来自实际调用观察。
快照创建和后端配置不自动算一次查询成功。向量查询失败、返回空结果和查询未执行分别记录。
模型逻辑操作、返回的 HTTP 响应数及预算账本 HTTP 尝试数分别记录，不能混用；
纸级/块级上下文分别标记，缓存或索引复用可能使模型调用次数为零。报告不会把这些零调用解释为失败。
观察包装只安装在独立 Runner 进程内，并在退出时恢复；不改变检索结果、门禁或异常处理策略。

目标论文放在任务 JSONL 的 `target_papers`，在执行前冻结。T01 提供 arXiv 目标身份。
存在强标识冲突时禁止用题名匹配；缺少共同强标识时可以记录题名+年份弱命中，并明确匹配依据。
排名是**准入前实际检查窗口**中的位置，未进入窗口时不伪造完整候选排名。
Known-item Hit@5/Reciprocal Rank 只描述预设目标，不是主题整体 Recall@K、nDCG 或总体准确率。

证据回读使用当前持久化解析页或清洗后的摘要，检查区间、原文及可用的来源/文本哈希。
`true` 表示原文可定位，`false` 表示位置/版本不匹配或来源已判不可信，`unknown` 表示缺原文或位置。
这不是重新下载的出版社真伪认证，也不等于语义支持。未关联的引用、证据 ID、未发布草稿均保留可见标记。
自动主张核验结果与人工结论分列；`completed` 本身不会签署人工验收。

## 人工填写与指标

`claim_evidence.csv` 每行是一个原子主张与一篇引用的关联。若多段证据被选择，同一行保留其全部原文和回读状态。
无引用的事实主张也保留一行，但不冒充引用关联分母，单独报告 `uncited_atomic_claims`。

只填写 `evidence_support`、`final_judgement`、`reviewer`、`reviewed_at`、`notes`。
允许的支持状态为 supported / partially_supported / unsupported / contradicted / uncertain / not_reviewed。
`evidence_support` 是该引用的判断；`final_judgement` 是考虑全部引用后的原子主张判断。
一个主张的多行应填写同一个最终判断，可在 notes 说明共同支持关系；不要求每篇单独证明整个主张。
已审查的最终判断必须有 reviewer 和 reviewed_at。不确定填写 uncertain，不得用自动核验替代人工阅读。

`manual_constraints.csv` 只填写 judgement、reviewer、reviewed_at、evidence；
judgement 允许 passed / correctly_blocked / failed / uncertain / not_applicable / not_reviewed。
已判定项必须记录独立检查依据，不能只因为返回 blocked 就填写 correctly_blocked。

```text
python scripts/summarize_real_e2e.py --run-dir <local_run_directory>
```

汇总命令校验冻结样本，拒绝删除失败行、增加样本或修改自动回读结果，不重写原运行记录。
人工 CSV 的内容哈希随更新后的汇总保存；此流程不是密码学签名或防恶意篡改系统。

- 自动引用可回读率：全部可回读的 Claim–Citation 关联 / 全部导出的有效引用关联；未知和失败都保留在分母。
- 人工支持率：supported 原子主张 / 已完成人工判定的原子主张；uncertain 计入已审查分母但不计支持，另报其数量。
- 人工覆盖率：已判定原子主张 / 全部待审原子主张。联合引用不会重复计算同一主张。
- 硬约束遵循率：passed 或正确阻断 / 全部适用的冻结约束；尚未审完时值为 N/A，不缩小分母。
- 没有事实主张或对应分母为零时值为 null（N/A）；人工审查错误单列，独立验收保持 pending，脚本不代人签署。

基线通过 `--baseline-run <historical_run_directory>` 指定。检查任务、目标、Pipeline、源码、
配置、实际候选池指纹及审计完整性；条件缺失、不同或仅预检时不能直接比较。
即使相同也只属于严格同条件运行记录比较，不是算法改造的因果效果证明。
需要比较不同 Chunk/预算/模型时，应先独立设计受控实验，不能靠放宽该资格检查得到“性能提升”。

## 本轮边界

这次补齐的是可执行入口和审计导出契约。离线替身的脚本测试、只读数据库预检与真实模型 E2E 分开记录。
真实论文效果、成功正文交付、独立人工审查及冻结集合检索指标仍须后续按单独预算完成。

2026-10-29 本轮验证：全量 1807 passed、17 skipped，73.88 秒；之后补充排序前停止的
候选检查点用例，最终新增 Runner/审查契约定向 23 passed，16.70 秒。
17 项跳过项为未启用专用连接的原生数据库集成测试，本轮未重复执行。
rragent 原生 PostgreSQL 的最终只读预检通过，检查到真实数据库版本并生成完整审查文件；
观察日志为空、模型操作和向量检索次数均为零，未创建实验业务库。
人工支持率和硬约束遵循率保持 N/A，没有研究交付或人工验收成功声明。
