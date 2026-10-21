# 项目介绍与源码证据

更新：2026-10-21。这里是面试准备材料，能力陈述不等于真实综述 E2E 已通过。
测试和运行边界以 [本轮验收记录](validation/2026-10-20-project-acceptance.md) 为准。

## 90 秒介绍草稿

ResearchReview-Agent 是面向学术综述的证据研究 Agent。它把研究范围理解、检索、
元数据核验、证据提取、主张对齐、写作和最终验证分成可检查的阶段。
主控根据当前状态选择动作，Search、Analysis、Writing 分别运行独立的 ReAct 循环；
Controller 用确定性的权限、字段契约、版本和预算约束执行。当前委派是同步的。

我的重点是让研究过程可追溯、可停止、可恢复。论文进入候选集合后还要经过证据
准入，排序分数不能直接授予引用权限。正文必须使用获准证据；不满足要求时继续
有界补检索、报告缺口，或在明确授权且其他门禁允许时交付标注限制的部分结果。

长任务通过后台 Job 执行，SQLite 保存租约、状态版本、预算和检查点。当前模型是
单机受控并发，恢复基于已提交检查点。真实服务仍可能遇到限流、模型决策不合规
和证据不足；这些结果需要如实记录，不能用单测数量代替端到端成功率。

## 四条可用于简历的技术事实

- 构建 Main 与 Search / Analysis / Writing 专业 Agent 的同步委派机制，使用独立
  ReAct 上下文、结构化汇报和 Controller 动作契约约束权限与状态提交。
- 实现多源学术检索、候选去重与证据准入，支持候选论文上的可配置 BM25、Dense、
  Family-aware RRF 与 Reranker；保留来源失败、限流和降级诊断。
- 实现有界上下文压缩、历史观察归档与回查，以及当前研究会话内的词项记忆检索、
  失效管理和使用反馈；效果提升须另行做受控对照。
- 以 SQLite 条件更新、Owner / Version / Lease 校验和原子检查点保护长任务提交，
  保留取消、已付费响应结算、未决费用估算和显式恢复边界。

## 5 分钟展开顺序

| 时间 | 内容 | 可定位的源码 |
| --- | --- | --- |
| 0—1 分钟 | 用户目标、年份、唯一引用要求与证据不足 | `app/agent/slot_extractor.py`、`app/agent/nodes/planning.py` |
| 1—2 分钟 | 主控决策、专业循环、动作权限 | `app/agent/main_loop.py`、`app/agent/subagents/loop.py`、`app/agent/controller.py` |
| 2—3 分钟 | 候选检索、排序、证据准入与引用授权 | `app/tools/search_papers.py`、`app/services/retrieval_ranking_service.py`、`app/agent/action_contracts.py` |
| 3—4 分钟 | Context、Research Memory 与证据的不同职责 | `app/agent/react_context.py`、`app/services/research_memory_retrieval.py` |
| 4—5 分钟 | Job、租约、原子提交、停止与恢复 | `app/services/research_job_service.py`、`app/database/runtime_repository.py` |

## 20 分钟源码深挖

第一条调用链：`ResearchConversationService.handle` → 执行作用域与预算 →
主控 `MainAgentPolicy.decide` → `SubagentLoop` → `AgentController.execute` →
工具真实观察 → Runtime Snapshot → `report_to_main` → 主控继续决策。

第二条调用链：`search_papers` → 候选去重与排序 → 元数据和语义准入 →
PaperCard → 主张与证据授权 → 写作计划和交付物渲染 → 引用/主张/完整性验证 →
补检索、隔离候选、明确部分交付或报告未满足要求。具体动作并非固定只能调用一次。

## 可复现故障案例

| 案例 | 根因与改动 | 回归证据 | 不能宣称 |
| --- | --- | --- | --- |
| 过期租约被续活 | 续约遗漏有效期条件；SQLite 等锁还可能让 Python 采样时间过时。改为执行时检查有效期，辅助账本原子校验 Owner | `tests/test_durable_runtime.py` 的过期、写锁等待、接管竞态和已付费响应测试 | 已解决跨主机时钟、远端请求恰好一次执行 |
| 明确单年变成默认近三年 | 规则未识别 `2020年`。补充单年和带“年”的闭区间解析 | `tests/test_slot_extractor.py`；真实运行中修复后保持 2020—2020 | 所有时间自然语言均已支持 |
| 子 Agent 汇报连续被拒绝 | Schema 允许任意建议字符串，执行端只接受注册动作名。统一枚举、提示和纠错信息 | `tests/test_subagent_react.py` 的原生/JSON 契约与越权边界测试 | 建议动作已自动获得执行权 |
| 最多 3 篇被要求凑满 3 篇 | 纯检索的数量上限被复用为最低引用数。统一主控视图和交付判定，去重后裁剪返回上限，保留同时存在的下限 | `tests/test_main_agent_loop.py` 的单篇命中、去重上限与上下限测试 | 综述引用硬门禁已放宽 |
| 不生成综述却触发写作 | 关键词规则没有识别否定，纯检索误入写作门禁；同时指定题名只作为软相关性处理。修正意图并在发布时检查题名硬约束 | `tests/test_intent.py` 的预检和组合意图测试；`tests/test_main_agent_loop.py` 的精确题名与空命中测试 | 仅凭 completed 就证明研究目标满足 |

2026-09-27 的共享 Session 心跳问题在源码注释中有开发事故记录；当前心跳使用
独立短生命周期 Session。面试时可解释竞争机制和既有测试，但没有重新核验当日
原始运行日志，不应编造事故频率或生产收益。

## 需本人完成的准备

- 脱稿练习 90 秒、5 分钟和 20 分钟版本，并按真实问答调整措辞。
- 逐条确认自己实际承担的工作，简历不得把本材料自动等同于个人经历证明。
- 人工抽检论文与 Claim–Evidence，填写独立人工结论。
- 确认毕业届别、简历版本和投递记录，按面试通知安排时间。
