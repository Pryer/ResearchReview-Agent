# Check Agent 与 Main 反馈闭环：实现及真实验证

日期：2026-11-10（新加坡时间）。本次已完成本地架构修改：Main 规划和决策，Search/Analysis/Writing/Check 执行并返回结构化结果，Controller 管理权限、状态提交和发布。

离线回归：**1950 passed，17 skipped**。跳过项为显式启用的原生数据库集成；真实 E2E 另行使用 pgvector。

最终真实运行状态为 **blocked**，公开正文未发布，有效引用 0 篇。

## 实现内容

- 独立 Check：`validate_result` 从 Analysis 转交 Check；复用原核验能力，强制只读候选与证据，返回版本绑定的检查报告。
- 统一结果：复用 `AgentTaskResult` 的生命周期、业务结果及通信状态，新增类型化 issues、建议动作和检查报告引用；检查执行失败与内容不合格分别记录。
- Main 决策：子任务结束后返回 Main，普通执行错误保留原因；已提交操作的汇报失败不重放工具、不增加汇报重试；取消、预算和余额不足等控制停止仍传播。
- 任务说明到执行模型：Main 的操作目标、参数和选定摘要通过角色隔离的临时执行上下文传给实际模型；子 Agent 的工具理由不能替换 Main 目标，执行结束或异常后清理上下文。
- 核验与改稿分离：Writing 完成候选及引用排版；Check 不自动删除或弱化正文。Main 决定局部修改、分析已有材料或定向补检索。
- 版本和发布：稿件、证据、授权或约束变化会使旧检查失效。Main 申请交付，Controller 复核当前检查及门禁；普通成功与明确允许的 partial 均通过该入口。
- 候选回滚：只恢复依赖仍匹配的已检查旧稿，同时恢复缺省字段和旧版本；被拒候选的报告随恢复记录保留，引用在检查点恢复后仍可读取。
- 恢复归属：新协议任务的恢复决策统一留在 Main；会话层不在 Main 终止后另起自动恢复队列。旧会话与用户显式再次恢复保持兼容。
- 预算：总模型请求上限 1024，保留独立检索限制、Token/时间/恢复上限及每篇 8 块抽取预算。

## 三轮真实运行

任务固定为：2020—2023 年 RAG 在开放域问答中的研究背景，中文约 600 字，至少 3 篇有效唯一引用，禁止自动放宽年份和篇数。

每次使用独立 SQLite、缓存、PDF 目录与 pgvector tenant；复跑通过 rerun_of 关联，未覆盖或挑选性删除失败运行。三轮均在运行结束前保持源码不变。

| 运行 | 终态 | 用时 | 模型请求 | PaperCard | pgvector 成功/尝试 | 原文位置回读 | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `T06_20261110_184916_ca0ccff5f9` | failed | 441.7s | 168/1024 | 7 | 35/35 | 238/238 | 执行结果已提交后，子Agent汇报的工具协议错误导致任务失败 |
| `T06_20261110_185914_d066949a6a` | failed | 634.1s | 168/1024 | 5 | 14/14 | 115/115 | 助手在确认任务说明未下发、外层重复恢复后停止验证；不是用户取消 |
| `T06_20261110_191155_1d57e97057` | blocked | 406.9s | 175/1024 | 5 | 35/35 | 163/163 | 最终实现的真实复跑 |

第一轮证明两次写作、两次 Check 和定向补检索能够实际执行，但暴露了汇报协议异常的返回缺口，随后修复。

第二轮记录到 Main 停止后外层继续恢复，以及 Main 任务说明未进入实际执行模型。助手在确认问题后发起技术取消；运行器记录的原始终态如表所示，不能改写为用户主动结束或正常验收。

第三轮使用最终实现。最终动作序列：

`search_and_rank → fetch_metadata → extract_paper_cards → plan_claims → generate_deliverables → validate_result → diagnose_evidence_gaps → extract_paper_cards → plan_claims → generate_deliverables → validate_result → rewrite_sections`

最终检查：execution_status=`none`，verdict=`none`；候选正文 64 字符，公开正文状态与候选状态分别记录。

Main 返回的停止理由（自动诊断记录）：写作环节连续三轮（generate_deliverables ×2、rewrite_sections ×1）均返回 blocked：候选正文始终为 64 字符、有效引用 0/3 篇，主题漂移且缺 3—5 个连续自然段，生成器未能产出可用正文。当前 allowed_actions 仅剩 validate_result 与 report_blocked，rewrite_sections/generate_deliverables 已不可用，无法再触发正文重写；而 validate_result 对同一不可用候选稿重复核验不会产生新进展（前两次核验结论一致：needs_revision）。证据侧已尽力：5 篇 2020—2023 年论文元数据核验完成、5 张证据卡抽取完成（补抽未新增）、3 条段落级主张已授权 3 篇唯一论文（s2:659bf9ce…、s2:6fcdad7b…、arXiv:2009.08553），证据本身足以支撑约 600 字背景，阻塞点在写作生成器而非证据缺口。按用户约束不得放宽年份（2020—2023）与篇数（≥3），故不通过降低要求换取交付。建议：恢复写作动作权限后，以现有 3 篇授权论文与 e036—e045 段落级证据重写 research_background（3—5 段、约 600 字、GB/T 7714 顺序编码），再交独立核验。。

## 本地资料与复核

- [架构及协议说明](../check-agent-and-feedback.md)
- [最终运行摘要](../../data/e2e_runs/20261110_check_agent/exported/T06_20261110_191155_1d57e97057/summary.json)
- [最终原文位置回读](../../data/e2e_runs/20261110_check_agent/exported/T06_20261110_191155_1d57e97057/source_readback.csv)
- [最终证据卡列表](../../data/e2e_runs/20261110_check_agent/exported/T06_20261110_191155_1d57e97057/paper_cards.csv)
- [离线回归日志](../../data/e2e_runs/20261110_check_agent/offline-regression.log)
- [源码与导出资料校验](../../data/e2e_runs/20261110_check_agent/artifact_checks.json)

原始运行目录位于仓库 `data/e2e_runs/20261110_check_agent/execute/`。本报告区分程序执行、来源位置校验和正文交付，不将其等同于独立人工语义审查。
