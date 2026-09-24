# 变更记录

## 2026-09-23 22:17:07 +08:00 — 单一自主编排迁移收尾（plan.md 步骤 4、5）

### 背景与范围

按 `plan.md` 完成「研究执行只保留一套生产调度」的迁移。本轮之前，步骤 1（行为
基线）、步骤 2（先迁能力）、步骤 3（统一会话迁移）与步骤 4 的主体代码改动已在
工作区完成：固定顺序的 `legacy` 执行分支已从 `app/agent/graph.py` 删除，三个公共
入口（`run_research_agent` / `continue_research_agent` / `regenerate_research_agent`）
与会话澄清、检查点恢复统一进入 `_run_autonomous_pipeline`，`app/core/config.py` 已
不再持有编排模式设置。

本轮完成剩余的步骤 4 清理与全部步骤 5（可执行契约与文档）。

### 修改文件

| 文件 | 改动 |
| --- | --- |
| `.env.example` | 删除 `AGENT_ORCHESTRATION_MODE` 发布开关及其注释 |
| `app/agent/orchestration.py` | `orchestration_mode` 改名 `normalize_orchestration_mode`，删除死参数 `existing` |
| `app/agent/graph.py` | 4 处调用点与 import 跟随改名 |
| `app/services/research_conversation_service.py` | 4 处调用点与 import 跟随改名；`_run_and_persist` 去掉一次多余的 `self.repo.get(session_id)` 读取 |
| `tests/test_global_evidence_gate.py` | 固定流程夹具改为经单一生产入口的注册动作序列驱动 |
| `tests/test_agent_completion_regressions.py` | 会话模式迁移回归跟随改名 |
| `README.md` | 删除开关表格行；「当前架构」改为单一调度与历史会话迁移语义 |
| `ARCHITECTURE.md` | `graph.py`/`orchestration.py` 职责描述、§3 数据流与 §3.2 阶段图说明去掉双模式表述 |
| `docs/agent-context-architecture.md` | 发布开关段落改为单一调度与幂等迁移；配置清单删除该键 |
| `docs/change-log.md` | 追加本次条目 |

### 根因

1. **配置开关已无对应代码路径。** `AGENT_ORCHESTRATION_MODE` 在 `config.py` 中已被
   摘除，但 `.env.example`、README 与两份架构文档仍在宣传它，读者会据此以为可以
   回滚到固定流程。
2. **`existing` 参数已成死参数。** 双模式下它用于区分「新会话取默认」与「旧会话
   按标记恢复」；单模式下函数无条件写回 `autonomous`，参数不再参与判定。它仍迫使
   `_run_and_persist` 为了计算这个被忽略的值多做一次数据库读取，且函数名
   `orchestration_mode` 读起来像一个可返回多种模式的取值器，掩盖了「只做规范」的
   真实职责。
3. **门禁集成测试仍驱动已删除的固定流程。** `tests/test_global_evidence_gate.py` 的
   `_install_graph_fakes` monkeypatch `app.agent.graph.expand_search_year_node` —— 该
   节点随 `legacy` 分支一起删除，导致 3 项测试以 `AttributeError` 失败；同时它把
   `_get_llm` 打成 `None`，无法驱动需要决策模型的五字段主循环。

### 行为变化

- 生产不再存在编排模式开关。`agent_orchestration_mode` 保留为持久化审计字段，
  不再参与路由；缺失、`legacy`、`autonomous` 三种历史取值在执行边界一律规范为
  `autonomous`，迁移幂等，非法取值仍显式抛错。
- `_run_and_persist` 每次研究执行少一次会话表读取（该读取的结果此前被丢弃）。
- 全局证据门的 3 项集成回归现在经真实生产入口执行：主循环按
  `search_and_rank → fetch_metadata → extract_paper_cards → validate_routes →
  plan_claims → generate_deliverables → validate_result → request_finish`
  的注册动作序列推进，终态仍由代码的质量门禁裁决，测试不再直接调用固定顺序节点。
- 未放宽任何证据验证、显式约束或门禁；未改变研究目标、证据卡、恢复历史与预算
  消耗的保留语义。

### 测试与验证

- **变异校验（确认测试非空过）：**
  - 打乱夹具动作次序（`plan_claims` 移到 `generate_deliverables` 之后）→ 2 项 smoke 失败。
  - 在生产 `_claims` 处理器内把全局门禁移到 `claim_plan_node` 之前 → `test_gate_runs_after_claim_plan_and_reports_real_claim_source` 失败，`claim_support_source` 退回 `route_evidence_volume_fallback`，即 2026-08-29 的原始缺陷。两次变异后均已完整还原生产代码。
- **定向：** `test_global_evidence_gate.py` 26 passed；`test_agent_completion_regressions.py` + `test_research_conversation.py` + `test_durable_runtime.py` + `test_agent_graph.py` + `test_autonomous_recovery_handoff.py` 161 passed。
- **全量离线：** 1303 passed，0 failed，0 errors（隔离 `--basetemp`）。迁移前基线为 1300 passed + 3 failed。
- **静态：** `python -m compileall app scripts tests run_api.py run_chat_frontend.py` 退出 0；`git diff --check` 无空白错误。
- **旧函数名残留：** 全仓 grep `orchestration_mode(` 仅余 `normalize_orchestration_mode`。

### 收敛复核

- 根因修复覆盖目标行为：门禁次序回归现由生产入口的真实动作序列证明，而非已删除的节点名。
- 与既有模式一致：`_SequencedDecisionLLM` 沿用 `tests/test_agent_graph.py::WorkflowLLM` 的原生工具替身写法；`_get_llm` 改为返回共享单实例，因为主循环、写作与验证各自调用它，逐次新建会重置动作序列。
- 已清理探索期残留：删除 `fake_expand_year` 及其 patch、死参数 `existing`、多余的数据库读取、以及原生工具路径下不可达的 `complete`/`complete_messages` 回退。
- 只清理本轮自己产生的临时目录（`.pytest_progress_1`、`.pytest_gate_*`、`.pytest_rename_1`、`.pytest_baseline_full`、`.pytest_final_full`、`.pytest_verify_final` 与 `/tmp` 备份）；工作区原有的 `.pytest_arch_full_*`、`.pytest_single_mode_*`、`.pytest_full_audit_*` 未动。
- 非显然约束保留中文 `WHY` 注释：`orchestration.py` 说明历史标记只代表原调度算法；测试夹具说明为何必须共享 LLM 实例、为何不能再按固定顺序节点驱动。
- 按 `plan.md` 第 9 行，未批量删除其他同名 `legacy` 兼容语义（旧事件导入 `import_legacy_history`、旧单数字段、旧计数器迁移、`test_legacy_section_failure_without_ccc_rebuilds_claim_authorization` 指旧会话缺 CCC 审计），均按各自调用链保留。

### 已知限制与未验证项

- **真实 CNKI/LLM 端到端未执行。** 全部验证均为离线模拟，不能证明生产 provider 的原生工具兼容性、真实检索质量、成本与延迟。
- 旧会话的实际数据库迁移未在生产库上跑过；`scripts/migrate_agent_runtime.py` 本轮未执行。规范发生在执行边界（读取时），因此未恢复过的历史会话其落盘标记仍可能是 `legacy` 或缺失，直到下一次执行。
- `docs/change-log.md` 第 1248、1518 行的历史条目仍描述双编排时期的决策；按追加式日志规则未改写。
- 本轮未重启现有服务，未创建 Git commit。
