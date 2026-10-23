# ResearchReview-Agent：按《深入理解 AI Agent》设计的 Agent Harness

> 设计基线（非全部已完成的功能）。参考 [《深入理解 AI Agent》](https://github.com/bojieli/ai-agent-book) 第 1–10 章，重点以第 1、2、3、4、5、7、10 章为依据。\
> 代码实施进度：已接入稳定前缀、追加式多轮轨迹、末尾状态栏、输入预算预检、L1/L2/L3 分层控制及轨迹检查点恢复；可选 Sidecar 保持默认关闭。离线回归已执行，真实提供商缓存与科研端到端效果仍待测量。

## 1. 目标与边界

- **Model**：理解研究目标，根据观察决定检索、证据分析、补检索、写作或停止。
- **Harness**：管理 Context、Tool Calling、Controller、权限、验证、纠正、恢复和可观测性。
- **Environment**：数据库、研究文献、PDF、Artifact、执行结果与外部检索服务。
- **最终交付**：模型可提出完成，但不能自行批准交付；论文真实性、Claim–Evidence 与引用门禁由独立的验证机制决定。

主流程是 **Context Build → ReAct (Think → Act → Observe) → Verification / Feedback → Delivery**。不要额外建立一条同功能的决策循环。

## 2. Context：以 API 消息轨迹为核心

对齐第 1–2 章：完整请求包含稳定前缀与动态轨迹；状态栏属于 Harness 注入的运行信息，而不是让模型重读全部数据库。

```text
Stable Prefix
  system: 身份、引用规则、权限和交付规则（跨轮次稳定）
  tools: 标准工具定义（顺序稳定；Controller 始终是权限最终裁判）
Dynamic Trajectory
  user: 原始任务、明确约束以及后续修订
  assistant: 工具调用意图和可传输的结构化调用
  tool: 真实执行结果、失败原因、来源引用
Dynamic Agent Status Bar（末尾）
  当前阶段 / TODO / 已核实进度 / 可用动作 / 预算 / 阻塞项
```

**当前实现：** 每轮由权威状态生成五字段状态栏并追加到持久轨迹末尾，已发送的历史消息在同一 Epoch 内不重写。原生通道保留 assistant/tool 配对；JSON 通道记录结构化动作与明确标记的观察。

### Prefix / Prompt Cache / KV Cache

- System Prompt 不插入时间戳或动态研究计数；固定工具定义的顺序。若因权限必须改变工具可见性，优先安全隔离，并评估缓存代价。
- 原始用户约束与已执行的 Tool Call / Tool Result 原样保留于同一 Context Epoch；随后只追加新交互。
- 动态状态栏置于末尾，优先用代码从权威状态生成。当前选择保留历史状态栏，优先保持先前输入的完整前缀；阈值压缩时才开启新 Epoch。成本仍须实测。
- 使用 SDK/API 的规范消息角色与原生 `tool_call_id` 配对，不把工具交互拼成单条伪对话文本。
- KV Cache 是推理执行中的键值缓存，Prompt Cache 是跨请求对共同前缀的复用；**缓存命中取决于实际 token 前缀，而不是数据结构字段数量**。
- 每个服务商的缓存计费、可缓存最短前缀、思维链回传规范均须单独验证。

### Progressive Disclosure 与 Agentic RAG

- 主轨迹只保留执行结论、少量关键摘要、证据 ID 和受控 Artifact 引用。
- 文献全文、解析结果、Evidence Card 等留在外部权威存储；仅在检索、分析或写作任务需要时通过有权限的读取接口加载。
- 检索不必按固定单向管道：主 Agent 可以根据证据缺口决定补检索、元数据补全与复核。
- 外部网页、PDF 和工具文本都是不可信数据，不能提升为系统指令或动作授权。

## 3. ReAct Loop：真实 Tool Result 参与下一轮判断

```text
Build Context -> Main Agent 选择合法动作
                    |
        [可选 Sidecar 对高风险动作作只读分类]
                    |
            Controller 确定性授权
                    |
          Search / Analysis / Writing
                    |
     校验结果 + CAS/版本复核 + 有界提交
                    |
       Append Tool Result / Error Observation
                    |
    更新末尾状态栏 -> 下一轮模型决策
                    |
      请求完成 -> 独立验证 -> 交付或修复
```

工具返回必须包含「做了什么、确实发生了什么、失败/证据缺口、Artifact 引用」的简洁结构。长结果不直接充满主 Agent 上下文。

## 4. 超长上下文：三层分级压缩

分层压缩应遵守书中第 2 章「最小修改、保留来源、临近阈值集中处理」的原则。

1. **L1 工具结果限额（执行时）**：保存完整原始输出到 Artifact Store；模型只看到不可变的有界工具反馈及来源引用。不要在每轮重写此前可见的工具消息。
2. **L2 轨迹批量压缩（阈值触发）**：当本轮估算输入 Token + 输出预留接近真实可用模型上下文窗口的指定比例（建议实验从 80% 起）时，将完整轨迹归档，然后在 Checkpoint 处开启新的 Context Epoch：稳定前缀 + 原始要求 + 结构化工作摘要 + 可回溯证据引用 + 必要的最近完整工具交互 + 最新状态栏。
3. **L3 预算熔断**：压缩后仍超限，或连续压缩失败，就明确进入 blocked / waiting_user，而不是删除用户硬约束、吞掉引用或无限重复尝试压缩。

**压缩保留集：** 用户原始要求与修改记录、已核实完成事项、待办与阻塞项、关键决策和失败原因、Claim–Evidence 的 ID 与来源引用、最近必要的完整 Tool Call / Tool Result 配对。

**缓存取舍：** 压缩历史会从最早改动处破坏前缀缓存，因此不能每轮做；Checkpoint 带来的缓存重建是有意识的容量换成本。

**真实性界限：** 已实现轨迹持久化、完整消息协议及 Epoch 压缩。输入采用 UTF-8 字节上界加消息开销的保守估算，而非声称精确 tokenizer 计数；真实 Token 与缓存用量来自服务商 usage。`main_context_*` 字符限制继续约束可重建状态栏，整次请求由独立 Token 预算预检。详见[实现与配置](agent-context-architecture.md#react-epochs-and-context-capacity)。

## 5. Sidecar、验证与纠正

- 第 4 章的 **Sidecar** 是对 *工具调用行为* 的独立只读风险分类，不替代 Controller，也不负责科研结论正确性。
- Sidecar 只读动作名称与结构化参数，不接收主 Agent 的隐含推理和整份研究状态；危险操作需审查放行后才执行。
- 审查服务普通故障仍升级为人工确认；预算耗尽、用户取消、租约失效、账本冲突及模型余额不足必须透传给执行边界，不能改记为等待用户。
- 工具执行后由 Controller 与业务校验检查结果，终稿由已有的 Claim–Evidence、引用一致性和质量门禁裁决。失败应作为新观察反馈给主循环。
- 当前分支中的 Sidecar 是可选、默认关闭的**同步**审查，不是书中的流式并行版本。运行时开关如下：

```dotenv
AGENT_SIDECAR_ENABLED=false
AGENT_SIDECAR_REVIEW_ACTIONS=generate_deliverables,rewrite_sections
```

## 6. 实施顺序

| 阶段 | 改造事项 | 验收标准 |
| --- | --- | --- |
| 已实现、离线验证 | Sidecar 审查及有界任务反馈 | 单测及保留集；真实误拒绝率待测 |
| 已实现、离线验证 | 标准 API 多轮消息、原生工具配对与轨迹持久化 | 前缀稳定、配对与恢复回归通过 |
| 已实现、离线验证 | Agent Status Bar、Token 预算及受控长输出 | 预算预检、真实 usage、跨会话隔离回归通过 |
| 已实现、离线验证 | 阈值触发 L1/L2/L3 压缩与 Context Epoch | 原始要求、归档引用、完整调用与超限阻断回归通过 |
| P3 | 与当前版本对照评测 | 任务完成率、引用正确性、重复动作率、缓存命中、成本、延迟 |

### 本地回归建议

```bash
pytest -q tests/test_action_sidecar.py tests/test_agent_context.py tests/test_main_agent_loop.py tests/test_main_agent_policy.py
pytest -q
```

第 7 章要求区分边界集和保留集。新增 `tests/test_react_context.py` 与真实 SQLite 事务恢复回归覆盖这些边界；公开验证边界见 [变更摘要](../CHANGELOG.md)；详细开发流水仅本地保存。离线测试通过不能等同于真实缓存收益或科研任务成功率已经验证。

## 7. 不做的事情

- 不为了技术名词额外创建 Main Agent 之外的规划循环。
- 不把全部论文全文、全部历史、完整数据库状态塞入 Context。
- 不让 Sidecar 或 LLM 自审替代确定性引用门禁。
- 不为提高 Prompt Cache 命中而牺牲工具权限隔离。
- 不在缺乏评测时宣称 Token / 成功率提升。

**核心原则：Context 负责让模型看到必要观察；ReAct 负责选择动作并接收反馈；Controller 负责确定性执行；验证器负责判定是否完成。**
