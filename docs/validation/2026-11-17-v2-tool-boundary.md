# v2 工具执行边界与持久回执

日期：2026-11-17，Asia/Singapore。对应 plan.md 的 P4 及 S1—S3 工具接入。

## 执行路径

```text
ActionExecutor
  → Controller 创建研究 Task
  → Handler 进入受限 ToolContext
  → 已注册客户端 / PDF / 模型 / 向量能力
  → ToolExecutor 校验 Action 能力
  → Repository 复核活动 Grant、Invocation、Task 与业务版本
  → durable tool_started → 再查取消/租约 → 实际能力
  → tool_finished → 原 Action 输出校验与提交
```

`ActionSpec.tool_dependencies` 是动作工具能力的声明源。`ToolRegistry` 保存副作用与重试语义，不维护第二份角色权限；角色仍由 Action 的 owner 派生。组合根校验全部专业动作及工具依赖，未知工具名不能静默接入。

## 本批纳管范围

| 边界 | 入口与保留行为 |
| --- | --- |
| 多源搜索 | arXiv、Semantic Scholar、OpenAlex、Crossref、CNKI 公共搜索入口；来源访问政策、限流和诊断不变 |
| 元数据 | 三个详情客户端；arXiv 继续使用原按 ID 反查入口 |
| PDF | 开放下载与结构化解析；保留公网出口、OA 授权、原文位置及解析诊断 |
| LLM | budgeted_create，主备与重试均经过原预留/结算 |
| Embedding/Reranker | 原生 HTTP 模型请求，原有逐 HTTP 尝试预算继续生效 |
| 向量业务 | source/register、claim、renew、write、activate、load、search；保留 Building/Active、版本与 fencing |
| 向量管理 | migrate/delete 注册为管理能力，专业 Action 无此权限；独立运维入口保持原行为 |

向量 fail 属于原构建失败清理；预算结算、检查点、心跳和授权签发属于运行基础设施，不递归包装为业务工具。

## 回执、费用与恢复

- ToolContext 只保存服务端身份与版本，不向 Handler/Worker提供共享账本或 Session。
- 请求前的 start 回执使用既有 research_events 短事务持久化；不改 ResearchWorkVersion，也不推进快照版本。无需增加第二套业务提交表。
- 回执保存工具类别、代码入口、参数摘要哈希、执行身份、终态与耗时；不保存参数正文、响应正文、凭证或提示词。
- 模型费用仍只由既有 ResearchAttempt 和执行账本结算。attempt_started 关联 tool_call_id，工具回执不再预留或累计一次费用。
- Embedding/Reranker 的一次工具调用可包含多个 HTTP 重试；工具回执数和模型 HTTP 请求数分开解释。
- completed/failed 描述边界调用是否返回/抛错，不代表检索有有效产出、证据正确、核验通过，也不证明失败请求没有产生费用。
- 新 Owner 领取执行时，将没有终态回执的 start 标为 uncertain。模型未知消费继续按原持久预留结算；工具层不自动重放、不承诺外部 exactly-once。
- 同 Owner 取消后的费用/回执结算可完成，研究结果提交仍受取消门禁约束；失去 Owner 的迟到回执不写入新执行。
- 回执失败时保留原始控制或业务异常，并记录失败类型；未决 start 供恢复核对。此路径兼容项目要求的 Python 3.10。

Worker 通过现有 PersistenceWriter 使用线程内 Session，短事务串行，确认返回后才继续外部调用。没有新增常驻队列、第二套预算锁或跨线程 ORM Session。

## 根因与验证

元数据补全、下载/解析节点原先可能把取消或 RuntimeConflict 当作普通来源失败降级。现保留原普通失败降级，仅控制异常立即上抛；工具权限拒绝属于必须停止的控制异常。

回归覆盖：外部调用前独立连接读到 start 和费用预留；Action 业务版本仅提交时推进；授权撤销/版本改变/取消后无请求；Analysis 不可读元数据；Check 不可删除向量数据；1/4/8 Worker 各100次模拟模型调用且不重复记账；重复回执幂等；回执故障保留原异常；真实 os._exit 后的工具 uncertain 与模型预留恢复。静态依赖检查保护已纳管入口，并禁止 Handler 组合根引入裸 HTTP/Provider 客户端。

验证结果：相关专项39 passed；全量 `python -m pytest -q --tb=short` 得到 **2272 passed、17 skipped、11 warnings，187.24秒**；最后的Python 3.10异常处理兼容及入口标识调整后，工具专项17 passed。全量日志保存在本地 `data/architecture_v2_tool_boundary_tests.txt`，`git diff --check` 通过。

收敛审查：工具层只保留一个 ToolExecutor；装饰器复用它，不新增业务提交器或费用累计器。真实客户端接入点由依赖测试保护；原来源降级只对普通异常继续生效，控制异常上抛。工具失败回执不会覆盖原异常，已确认回执也不会被重复结算改写。费用、研究业务版本和运行快照版本仍分别管理。

## 后续范围

后续批次已接入Artifact/候选读写、输入资源清单与跨Action提交lineage，详见[Artifact与候选记录](2026-11-17-v2-artifact-lineage.md)。精确论文/文件级Grant Scope仍待完成；现有会话、角色、字段与版本约束继续执行。当前能力约束面向可信仓库代码，不提供不可信 Python 插件的进程级隔离。

报告/API 的工具回执展示、100 种竞态调度、真实 Provider/原生 pgvector 和独立质量验收仍待实施。默认入口保持 v1，不以本地替身回归提前切换生产默认。
