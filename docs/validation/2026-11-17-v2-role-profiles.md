# v2角色模型与上下文身份

日期：2026-11-17，Asia/Singapore。覆盖计划§6.3及§9的角色配置、规划/执行简报与恢复身份。

## 使用配置

`AGENT_ROLE_MODEL_PROFILES`为JSON对象，支持main、search、analysis、writing、check。默认`{}`沿用既有全局配置。示例：

```json
{
  "analysis": {"model": "analysis-model", "control_plane_max_tokens": 2048},
  "writing": {"provider_slot": "backup", "model": "writer-model", "fallback": "disabled"},
  "check": {"model": "check-model", "fallback_model": "check-backup", "context_window_tokens": 65536}
}
```

模型名称需由已配置的提供商实际支持；本例不声明这些名称可用。通道与凭证使用既有LLM主备环境配置，角色JSON不接受api_key、base_url等额外字段。

| 字段 | 语义 |
| --- | --- |
| provider_slot | primary或backup，选择已有传输通道，默认primary |
| model | 覆盖所选通道模型；省略时使用通道现有模型 |
| fallback | inherit、disabled、primary或backup；显式fallback必须与首选通道不同 |
| fallback_model | 覆盖后备模型；fallback禁用时不能指定 |
| temperature | 文本生成的默认温度；原生动作决策继续按既有策略使用0 |
| max_tokens / control_plane_max_tokens | 生成与控制调用的默认输出额度；业务调用的显式参数继续生效 |
| context_window_tokens | 最终请求容量上限，应配置为主备模型可用窗口的较小值；省略时继承全局窗口 |
| native_tools_enabled / thinking_enabled / thinking_effort | 覆盖角色协议和思考参数；厂商适配与既有操作级推理策略继续执行 |

inherit在首选primary时使用已有backup；首选backup时使用已有primary。对端未配置时不新增隐含通道；显式选择不可用通道会拒绝配置绑定。

## 冻结和恢复

角色JSON纳入现有effective_config，模型服务进入任务后绑定冻结设置。旧任务快照缺少该字段时按空配置解释，不追溯采用进程中新加的角色配置。

每个委派保存execution_profile：请求模型/Provider、后备模型、窗口和包含提示/工具目录/连接地址身份的指纹。地址只参与哈希，不进入记录。恢复时若当前身份与已保存身份不同，拒绝继续该委派，保留既有提交；恢复原配置后可以继续未完成动作。

配置摘要描述请求策略；真实远程调用的Provider/model与费用仍以已有响应记录和用量账本为准。模型层沿用既有budgeted_create，不新增计费层。Check配置进入verifier_profile和核验输入指纹；Search调参不单独撤销已有Check结论。

## 三层视图和请求容量

Main继续读取五字段研究状态栏；局部PlanningContext只包含目标、约束和有界产物摘要，并携带最新业务版本与来源指纹；Action工作状态继续按输入契约和paper_scope投影。

规划简报绑定委派、Profile和Grant版本；执行简报区分delegation_id、invocation_id、attempt_id和task_id，附带来源版本及输入引用。ContextVar按作用域恢复，其他角色不能读取该简报。增加标识不扩大资料读取或写入权限。

角色模型的每一次实际请求都在调用Provider前检查完整消息、角色提示、任务简报、工具Schema与最终输出额度。备用切换、格式降级、空回复扩额重试都经过同一检查，避免只校验首轮。窗口超限作为ContextBudgetExceeded传播，不能被格式纠错、模板降级或备用切换吞掉；已经发生的首次调用费用保留。

容量估算沿用UTF-8保守上界，属于本地守卫，不是厂商tokenizer实测。该批未新增自动裁剪证据或更改既有轨迹压缩算法；小窗口可能更早交权或阻断，应根据真实模型窗口与输出额度配置。

## 验证与收敛

新增22项专项通过；模型/缓存/局部协议/停止恢复相关86项通过，图编排与专项88项通过。最终全量`python -m pytest -q --tb=short`：**2345 passed、17 skipped、11 warnings，225.09秒**；本地日志为`data/architecture_v2_role_profiles_tests.txt`。`git diff --check`通过。

覆盖独立角色参数、从角色视图重新绑定其他角色时回到全局传输、实际备用模型回执、禁用备用、任务冻结与旧快照、Check失效、消息/工具容量、扩额重试保留费用、三类策略超限原样传播、上下文身份、真实Runner读取递增业务版本、配置变化拒绝恢复及原配置恢复。

收敛审查：模型绑定集中在LLMService；最终容量检查集中在实际请求出口，移除探索时的外层重复检查；复用现有费用账本和停止异常分类。写作Handler使用组合根传入的角色服务，修复绕回全局工厂的配置断点。旧测试替身补齐已有llm参数，保留原研究断言；配置变化拒绝恢复的回归确认已提交Search不重放。

后续批次已补入[章节计算缓存、强退恢复与修订时限](2026-11-18-v2-section-recovery.md)。完整业务冻结场景、更多历史检查点格式及在途会话清点、100种竞态、真实Provider/原生pgvector、独立质量审查及默认切换与旧路径退役继续实施。默认入口仍为v1；没有运行付费Provider/CNKI端到端或人工质量验收。
