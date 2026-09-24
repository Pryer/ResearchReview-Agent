# 架构审核记录（2026-09-23，进行中）

## 当前判断

整体分层方向合理：`ResearchAgentState` 保存权威证据和决策，五字段上下文只作为可重建的模型视图；专业任务通过字段契约与提交校验隔离；检索、证据、写作和最终质量门禁仍可分别验证。持久化执行、预算与取消边界能阻止无界重试和过期结果提交。

主要风险集中在跨层契约，而非缺少一个新的编排框架。2026-09-23 审核时同时支持 `legacy` 与 `autonomous`；用户随后决定只保留自主编排，迁移施工见根目录 `plan.md` 和 2026-09-24 的 change-log。`graph.py`、`research_conversation_service.py` 与若干大节点文件同时承载多种入口，容易出现“恢复决策已选定，但专业任务看不到该决策”一类遗漏。`ResearchAgentState` 的全部字段可选，新增状态字段需要同步核对专业任务契约、持久化、恢复和最终门禁。

本次发现的实例：重点证据缺口在旧编排中经过检索精化，但默认自主编排可能选择路线 `targeted_search`；`search_and_rank` 的隔离输入又缺少 `active_quality_recovery`。此外，候选论文元数据命中重点词，不等于证据卡满足直接证据门禁。三处边界已按当前生产路径修复，见同日 change-log。

## 已逐文件核对

| 文件 | 结论与处理 |
| --- | --- |
| `app/agent/graph.py` | 核对初次、增量、自主和旧编排入口；自主模式下先履行已选定的重点检索，再进入主循环。 |
| `app/agent/generation_recovery.py` | 核对诊断、终态、动作记录；从活动质量恢复提取缺失重点，硬问题代码消失计为进展。 |
| `app/agent/action_contracts.py` | 核对专业任务输入/输出边界；只读传入活动质量恢复，仍禁止检索任务改写该决策。 |
| `app/agent/retrieval_loop.py` | 核对首轮检索、精化和停止条件；首轮预留重点查询，不再依赖候选层覆盖触发下一轮。 |
| `app/agent/focus_coverage.py` | 核对查询生成与覆盖计算；复用已有的语义帧查询生成器，未改匹配阈值。 |
| `app/agent/nodes/retrieval.py`、`app/agent/nodes/base.py` | 核对首批检索词筛选、双语排序、重点 refine、显式年份边界和详情补全；`targeted_recovery` 有预留名额，显式时间范围不会被扩展。本轮未改。 |
| `app/agent/provisional_routes.py` | 确认旧评分器及其私有辅助函数在生产和测试中均无调用后删除；保留新验证器仍使用的路线命名与候选路线函数。 |
| `app/agent/route_validator.py` | 核对路线验证及 SPLIT 的独占归属、证据报告和写作输入；修复子路线沿用父路线充分性报告的问题。子簇未分别满足核心证据阈值时保留父路线，成功拆分时各自重算充分性报告。真实验证入口夹具已覆盖父路线从所有论文归属中消失。 |
| `app/agent/recovery_loop.py`、`app/agent/evidence_recovery.py` | 核对共享恢复预算、来源不可用、路线轮次/单路线次数上限、查询新颖度、边际收益停止与最终门禁快照重算；既有边界可阻止已耗尽路线再次请求模型。修复同一查询服务多条路线时，仅首条路线获得审计分配的遗漏。 |
| `app/services/research_conversation_service.py` | 核对质量恢复入口、最终最佳努力和用户澄清；重点缺口显示缺失重点与可执行选项，用户选补充检索后重新授权并进入重点专项检索。继续审查发现用户选重新分类时只设置 `force_taxonomy_remediation`，旧分类随后会被再生成入口清除；入口现保证先执行聚类，再重建主张并写作。篇数缺口问句原提供“纳入更多文献类型”，但只写无人读取的 `include_preprints`，而检索原本已允许会议论文和预印本；新问句已去除此选项，旧会话回答仍按原范围补检索。 |
| `app/agent/task_context.py`、`app/agent/subagents/base.py`、`app/agent/result_merger.py` | 核对任务输入投影、隔离执行、补丁权限及提交前版本校验；专业任务不能直接修改权威状态。 |
| `app/agent/controller.py`、`app/agent/execution_budget.py` | 核对预算预留与提交、幂等复用、运行时原子快照和失败状态；提交预算只记账，不会在结果合并后再抛出额度异常。 |
| `app/agent/main_loop.py`、`app/agent/action_registry.py`、`app/agent/orchestration.py` | 2026-09-23 核对自主模式动作选择、无进展停止和历史会话固定模式；随后按 `plan.md` 将旧会话规范到唯一自主路径。 |
| `app/agent/main_policy.py`、`app/services/llm_service.py` | 核对主 Agent 的原生工具调用契约；修复向模型提供本轮不可执行工具的问题，仍保留返回值的服务端动作校验。 |
| `app/services/research_execution_service.py`、`app/services/durable_execution_service.py`、`app/database/runtime_repository.py`、`app/database/repositories.py` | 核对执行作用域、租约、任务原子快照、公开会话保存和恢复；离线故障注入已有对应测试。 |
| `app/agent/nodes/synthesis.py`、`app/agent/global_evidence_gate.py` | 核对路线拆分在写作主题和全局证据体量代理中的消费；修复跨交付物引用回填把未知、失效和未确认引用算作有效并集的问题，检索结果的可见文本不再附带内部论文 ID。最终门禁主要分支已核对，其余合成细节仍待逐段审查。 |
| `app/agent/claim_plan.py` | 核对路线主张、补充引用授权、写作前证据门禁和引用一致性；修复同篇多片段冒充多篇支撑，以及无可读证据片段时绕过语义蕴含验证。其余主张聚合与正文句级校验仍待继续。 |
| `app/schemas/recovery_schema.py`、`app/agent/state.py` | 核对活动决策与进展向量字段；本轮未改 schema。 |

## 后续审核次序

1. 会话服务与用户决策：继续核对多轮恢复次数、持久化结果和用户再次答复。
2. 主 Agent 与专业任务：继续核对模型响应失败、专业任务异常和运行时状态保存的交界。
3. 检索与路线：SPLIT 完整验证入口夹具和共享查询分配回归已补；继续核对补检索新证据的跨阶段归属。
4. 证据与写作：继续逐段审查 `claim_plan.py`、`nodes/synthesis.py`、renderer、引用校验和质量门禁。
5. API、前端、文档与清理：只删除经全仓调用核对为零引用的旧实现；历史会话标记是迁移输入，公开 API 和其他旧数据字段仍按各自兼容边界审核。

离线测试通过不等于真实 CNKI/LLM 召回和完整综述验收。此记录只说明已检查的文件，不宣称全仓审核结束。

## 2026-09-24 四项跨层建议的施工结果

1. **会话预检与规划解析：** 会话层的意图、槽位和语义帧在同一原文进入 `plan_node` 时直接复用；规划层仍负责检索策略。工作查询或澄清内容变化后，语义帧按新查询重新解析，避免沿用旧范围。预检结果只作一次性交接，不成为另一份长期权威状态。
2. **只找论文的交付门禁：** `request_finish` 按去重后的论文数、用户明确指定的篇数和研究重点覆盖度验收；不足但有论文时标为 `partial` 并列出缺口，零论文时标为 `blocked`。论文列表和计数使用同一去重口径，检索结果不套用正文质量门禁。
3. **主 Agent 往复空转：** 进展指纹忽略列表顺序抖动，短窗口识别 A→B→A 的重复状态并显式阻断；连续相同状态仍按原有无进展阈值处理。只以实际研究产物计算指纹，不以决策轮数伪装进展。
4. **句级验证成本：** 原有每批至多 12 个原子主张、同指纹缓存和局部重验继续生效；新增每轮模型批次数、失败批次数和提交主张数，便于量化重复调用。未改为事实主张抽样，证据验证门禁不降低。

这些是离线行为修复与观测数据；真实提供方的 token 用量由 LLM 服务指标记录，未用批次数推断实际计费节省。
