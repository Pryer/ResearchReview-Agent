# v2唯一生产入口与旧运行器退役

后续的运行库升级、原生 pgvector、API 重启与真实任务验证见[真实环境验收记录](2026-11-18-v2-live-closeout.md)。下文的 2710 项测试记录早期退役批次。最新全量回归通过 2769 项，正式综述质量与人工审查仍待验收。

## 生产实现

默认配置为AGENT_ARCHITECTURE_VERSION=v2、AGENT_V2_LOCAL_STRATEGY=deterministic，即计划中的v2-B。TaskMainLoop→DelegationRunner→ActionExecutor→Controller是唯一生产调度与提交路径。ReAct与Plan-and-Act保持显式选项，Writing/Check保持direct。没有以策略收益尚未确认作为继续保留旧生产引擎的理由，也没有默认开启未经效果评测的自主策略。

已从app/移除main_loop.py、main_policy.py、subagents/loop.py和delegation_context.py。共享业务进展缓存移至runtime/progress.py，决策异常移至runtime/errors.py，可回读摘要身份目录独立为context_selection.py。旧Main/Subagent提示移出生产前缀，Main角色使用当前任务协议。E2E观察脚本记录当前TaskMainPolicy的协议失败，去掉旧context_selection拦截器。

新的指定恢复/强制动作使用服务端single_action授权，动作及参数精确绑定；模型不能自行请求这一内部Profile。legacy_single_v1仅用于已知历史记录迁移/回读，授权实现复用同一精确单动作规则，不保留旧调度器。

## 入口与恢复

会话服务已有durable_scope时，graph入口直接复用。直接Python调用自动建立持久化作用域和Artifact Store，预算、Action提交、工具回执和恢复使用同一个内核。需求解析也处在该作用域内。报告在租约释放、Session关闭前冻结真实来源，导出失败不覆盖原异常。

公开结果的research_state补齐architecture_version、active_delegation_id、delegation_control、task_main_control及历史迁移标记，避免结果保存覆盖已完成的v2恢复游标。无状态标记但数据库已有活动委派时，新Main继续该委派。

历史配置只升级执行架构，保留模型、交付策略、年份、篇数和累计费用。先验证全部参数，再更新快照和指纹；失败时不留下半升级状态。已知活动委派仍须通过目标/来源指纹、任务提交证明和CAS迁移；未知格式明确拒绝。

## 历史参考与数据清点

旧协议参考保存在tests/reference_v1，manifest列出来源摘要和22个历史协议测试模块。保留其业务断言，生产和脚本禁止导入参考代码。旧检查点测试在生成历史输入时使用参考观察配对，实际迁移仍走生产迁移器。该组不作为新运行时效果或真实A/B/C收益证据。用途和删除条件见该目录README。

只读脚本：python -m scripts.audit_runtime_migration。输出仅含聚合计数和结构类别，不含会话身份、查询、正文或预算明细，不执行迁移或修复。本地清点为11条历史/未启动快照，未发现活跃租约或活动子委派，1条running动作仍保留原状；当前本地库尚无v2表，服务正常启动或直接入口会通过既有加法迁移安装新表。没有自动改写或清除用户研究数据。

## 验证

- 新增8项退役/默认配置/直接入口/游标/只读清点回归。
- 运行器、迁移、业务、角色模型、产品等专项110 passed，65.13秒；单动作授权标签收敛后83项专项通过，28.68秒。
- 旧入口守卫改为验证新Main的持久化要求及数据库活动委派恢复；旧context_selection观察测试改为当前任务协议错误观察，保留异常透传和公开诊断边界。
- 原故障矩阵、真实进程强退、取消/接管/ACK竞态、费用连续性、正文与引用门禁及四角色冻结业务链继续进入全量回归。

最终全量：**2710 passed、17 skipped、11 warnings，405.25秒**，日志data/v2_retirement_full_tests.txt。新增8项入口/清点专项最终通过；single_action收敛后的83项专项通过。Python编译检查与git diff --check通过，静态回归确认app/及scripts/没有退役模块/测试参考依赖。运行态清点文件及测试原始输出保存在本地data目录。

## 收敛和部署边界

生产没有旧引擎路由或参考目录依赖，不增加第二套事务提交、费用账本或发布门禁。临时重构脚本和退役模块编译缓存已清理；旧研究数据和无关工作区文件未删除。兼容字段只承担历史读取、迁移证明和报告用途。

旧.env若显式配置AGENT_ARCHITECTURE_VERSION=v1，应删除该项或改为v2。部署前停写并保存一致备份，部署后重启服务，使既有加法迁移生效；运行中的旧进程不会因源码变化自动升级。v2在途任务由当前运行时恢复。二进制回退需要兼容备份，并核对升级期间产生的外部费用及未决调用，不能把切换Feature Flag视为逆向迁移。

既有数据库副本迁移/恢复演练继续通过自动测试，本批没有在用户运行库实施恢复覆盖。真实来源/Provider、原生pgvector、独立质量审查、策略收益试验和最终部署回退仍需分别验收；未推送GitHub，也未重启用户服务。
