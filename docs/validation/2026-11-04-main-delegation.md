# 2026-11-04｜Main 规划与 Subagent 执行交接验证

本次将专业 Agent 的后续动作与跨角色恢复交回 Main，并加入模型上下文选择协议。
本报告记录源码与离线行为验证，不代表真实 Provider 端到端成功交付。

## 行为变化

1. Main 每轮输出一个注册动作，可携带最多六步的 `plan`（action、objective）。计划只作建议，下一轮按真实观察修订，不授予未来权限。
2. 委派可携带 `context_selection`，选择当前状态栏中的 `evidence_ids`、`decision_ids`、`question_ids`、`memory_ids`。未知 ID 拒绝；通信元数据在执行参数校验前移除。JSON 和原生工具模式均支持。
3. 子 Agent 的模型视图按本次操作合同构造，保留五字段载荷；Main 的选择只补充摘要，不替代工具输入合同，也不允许任选原始状态字段、原始对话或兄弟私有轨迹。省略选择兼容旧调用；显式空选择不附加摘要。目标、硬约束、门禁、失败原因和预算强制保留。
4. 子 Agent 每次只可执行本次授权研究动作，随后返回 completed、needs_help 或 blocked。回查和有限的协议纠错保留。结果和建议不能直接授权下一个角色。
5. Analysis 可诊断证据缺口、返回受限查询建议或执行已允许的框架修订；Search 的单次补检索只提交候选与恢复状态。`pending_evidence_recovery` 跨检查点保留，Main 逐次委派详情、证据提取、路线核验及主张授权。每路线新增证据与边际收益在分析后结算，未提高原预算。
6. 引用补搜保留检索角色的互补查询精化，不再在 Search 内嵌分析与生成。显式现有证据刷新和研究重点恢复进入 Main 必做队列。
7. Writing 只提交候选稿并标记 `pending_result_validation`；公开交付、再次写作和 finish 均不能绕过独立核验。Main 另行委派 Analysis 的 `validate_result`。
8. 旧稿比较快照分片保存为仅 service 可读资料；只在证据及授权未变时恢复退化前版本。旧验证指纹匹配才可复用，安全回滚不被误判为无效状态循环。

## 主要文件

- `app/schemas/agent_decision_schema.py`、`app/agent/main_policy.py`：计划与上下文选择协议、双传输校验。
- `app/agent/delegation_context.py`、`app/agent/subagents/loop.py`：操作级上下文、单动作委派、结果汇报与恢复。
- `app/agent/graph.py`、`app/agent/delegation_recovery.py`：跨角色交接、补搜及独立核验。
- `app/agent/action_registry.py`、`action_contracts.py`、`controller.py`：调度与执行两层权限；Search 不具备提交主张或正文的权限。
- `app/prompt/agent_system.py`：Main 全局规划与上下文选择、Subagent 执行和状态汇报提示词。
- `state.py`、`context_builder.py`、`evidence_invalidation.py`、资料注册表与服务：状态、版本、私有比较快照。
- 新增 `tests/test_main_delegation_context.py`（9项）与 `tests/test_delegation_recovery.py`（6项），并更新旧编排回归的阶段预期。

## 验证

- 最终命令：`python -X utf8 -m pytest -q -p no:cacheprovider --basetemp <独立临时目录> --tb=short --show-capture=no`。
- 结果：**1859 passed、17 skipped、11 warnings**，79.48秒。17项跳过为原生 pgvector 集成；11项为既有依赖的弃用提示。
- 外部模型和论文来源使用测试替身；业务数据库使用内存 SQLite，文件日志和 pytest 临时资料隔离。
- 定向验证包括：JSON/原生元数据、无效上下文 ID、约束不能被空选择删除、跨角色隔离、活动委派恢复、私有快照分片、Search 越权拒绝、恢复累计预算、每路线收益、候选稿禁止提前发布、独立验证与退化回滚。
- `git -c core.autocrlf=false diff --check` 通过；26份改动的 Python 文件语法解析通过。
- 最终最小行为复核完成：去除不再调用的复合引用修复及直接专业执行入口；保留既有证据门禁、动作预算、取消、租约与状态版本校验，不增加刷新/重试兜底。

## 验证边界与运行说明

- 未执行真实 LLM、真实检索源、原生 PostgreSQL/pgvector 集成或人工字段验收；不据此宣称真实综述成功率、字段准确率、成本或延迟改善。
- 委派仍同步执行，不代表多个 Agent 并行自治。更细的委派可能增加控制请求，累计请求上限与证据抽取预算保持原设置。
- 角色提示词在进程中加载；已有服务需重启后使用新版本。旧活动委派继续保留已提交结果，但不再从旧历史获得同角色后续动作权限。
- 此次仅修改本地工作树，未提交或推送远端。原有未跟踪文件保留；原文件在临时备份及本地变更备份包保存。
