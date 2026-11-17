# v2 Artifact 资源授权与候选 lineage

日期：2026-11-17，Asia/Singapore。接续工具执行边界，对应 S1—S3 的资料与候选接入。

## 已接入的路径

```text
ActionExecutor → Controller.begin_task
  → 绑定实际Task/Invocation
  → 从当前权威状态与Action输入契约冻结Artifact清单
  → Handler → ToolExecutor → Artifact Repository
  → 临时产物与审计先持久化
  → 输出/证据/独立Check协议校验
  → 业务提交、结果、lineage、委派游标、检查点同事务
```

Artifact 读写和候选读写加入工具目录。`ResearchArtifactService` 在 v2 Handler 内使用实际执行身份路由；旧运行时、API服务及检查点等基础设施保持原存取方式。资料访问复用现有 Artifact 校验、同一 Controller 提交内核和 PersistenceWriter。

## 三种资源记录

| 记录 | 产生时机 | 权限与用途 |
| --- | --- | --- |
| action_resources_bound | 与研究Task创建同事务 | 以Action输入字段、角色可读类型和实际业务版本绑定读取范围；只存引用与身份，不存输入正文 |
| artifact_staged | 与业务Artifact写入同一短事务 | 同一Attempt可读取自己新产生的合法输出；保留候选审计，不授予证据或发布资格 |
| action_artifacts_committed | 与Action业务提交同事务 | 记录实际提交的输出清单、输入scope、Invocation、Task与业务版本；后续Action从当前权威状态及自身输入契约绑定新范围 |

搜索新增的论文产物可成为同委派下一步 fetch_metadata 的合法输入。每步读取仍需当前字段契约和角色许可；不会因为同属某个委派就读取全部历史资料。

## 读取、写入与候选边界

- 输入清单从权威业务字段重新生成，不信任过期 manifest、模型提交的引用或 Handler 传入的替代 state。
- 实际读取使用 Grant 角色；传入 `role=service` 不能提升权限。会话、引用、类型、冻结版本及内容指纹分别校验。
- 写入类型由 Action 输出契约与 Artifact 注册表共同约束；Check 可提交核验产物，不能写主张授权或新候选。
- 写入的 v2 lineage 身份由服务端覆盖；显式 source_refs 必须已在当前输入范围或同Attempt合法产物中，不能通过伪造父引用授权其他资料。
- 已有候选按项目既有 task_identity 规则绑定研究身份；其他研究的候选不会进入当前Artifact输入清单。旧候选数据保留在父状态和审计中。
- 候选审计存储、正文提交、独立 Check 和正式/草稿发布继续分别裁决。候选持久化成功只证明审计资料已保存。
- 失败Attempt的暂存资料不会自动扩展新Attempt的读取范围；Controller明确保留到当前审计状态的候选，可在下次冻结输入时重新受控绑定。

资料读写会在真正数据库事务内重新检查 Grant、租约和业务版本，覆盖“工具start确认后、实际访问前”发生撤销的窗口。

## Session、恢复与事务

Worker 经 PersistenceWriter 在线程内创建与关闭 Session。业务Artifact载荷和暂存lineage同事务持久化，移除v2候选路径对父Session的额外flush；回执和费用仍使用原工具/预算链路。

输入scope在Task开始时固定，按Attempt隔离，恢复的新Attempt重新绑定其合法输入。已提交Action的派生清单和业务版本一起落盘；检查点提交失败会整体回滚，审计暂存和正式业务提交不会互相冒充。

全部记录复用现有 research_events 与不可变Artifact，不增加另一套业务数据库或第二个提交器。现有Schema版本与旧读取格式保持兼容。

## 验证与后续

新增回归覆盖正常输入读取、同会话猜测引用、跨会话、错误版本/类型、service角色伪装、伪造source_refs、Search→fetch_metadata、Writer并发候选回读、重开后的Writer→Check、暂存后中断、lineage事务回滚、撤销窗口和旧研究候选过滤。

相关专项54 passed；全量 `python -m pytest -q --tb=short` 为 **2287 passed、17 skipped、11 warnings，189.30秒**，本地日志为 `data/architecture_v2_artifact_resources_tests.txt`；`git diff --check` 通过。验证使用受控替身和真实本地SQLite/线程/恢复路径，没有进行真实Provider或独立人工质量验收。

收敛审查完成：资料类型权限派生自既有Action输入/输出契约；工具start与真正资料事务共用同一授权核验；临时Artifact写入、输入scope和正式提交lineage的生命周期分别明确。仅资料Repository内部关闭递归资料工具包装，不关闭其他工具校验。未新增业务提交器、共享Worker Session或另一个费用账本，已保留原候选审计和质量门禁。

剩余工作：论文/文件级Grant scope、PDF/向量资源参数的进一步收窄、100种竞态、完整API/报告/前端展示、历史活动子Agent迁移、真实Provider/原生pgvector验证及独立质量审查。当前能力面向可信仓库代码，不提供不可信Python插件沙箱。默认入口保持v1。
