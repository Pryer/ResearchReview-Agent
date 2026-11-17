# v2章节计算恢复与修订时限

日期：2026-11-18，Asia/Singapore。对应计划§4.5、S3的章节恢复与超时边界。

## 已修复的中断窗口

原section_checkpoints与section_candidate_checkpoints要等整篇渲染、合并及Controller快照后才能持久化。即使第一节已经生成，第二节抛出取消异常或进程直接退出时，前面章节也可能需要重新调用模型。

现在v2分节写作在每节通过局部结构/引用标签检查后，通过section_cache_write独立持久化规范化文本。缓存内容保存为计算用途Artifact，引用与描述符在同一短事务登记；确认返回后才算已保存。它不改review、不增加研究业务版本、不产生Check通过报告，也不发布不完整正文。

恢复时按会话、研究任务和完整指纹查询；指纹包含目标约束、完整WritingPlan、基稿、修订目标、章节证据/主张授权、实际生成提示及写手模型策略。写手或提示源码/蓝图变化同样使既有章节检查点失效，旧指纹schema不自动升级。

命中后仍执行局部结构与引用检查，拼装后的完整候选走既有交付物验证和独立Check。局部合格只说明计算结果可重用，不授予事实支持或正式发布资格。

## 权限与存储

缓存只向Writing的generate_deliverables和rewrite_sections开放；修订缓存的读写限制在签发的section_ids内。Search与Check不能写计算缓存。读取还要验证当前Invocation、Task、业务版本、Grant和存活租约。

Worker沿用PersistenceWriter和线程内Session，不接触父Session，不创建第二套费用账本。缓存键不可变；同键重复写保留首个持久结果。Artifact内容哈希损坏时终止复用，不把破损缓存降级为已通过验证。

章节文本只进入内部Artifact；工具回执保存参数哈希，常规进度与报告不纳入计算缓存事件正文。缓存不会加入活跃研究状态的大列表。

## 修订期限

`AGENT_SECTION_REVISION_TIMEOUT_SECONDS`默认300秒，纳入既有任务配置冻结。它限定每次rewrite_sections的Action尝试，覆盖Sidecar审查、业务Handler及Worker；实际期限取本次时限和剩余全局执行期限的较小值。已有单次最多3章的授权规则继续生效。

HTTP/模型调用沿用现有预算读取，因此请求timeout也受Action剩余时间限制。Controller在合并后、正式快照前再次检查期限；超时丢弃本轮候选补丁，保留原正文、已有缓存和已发生费用。退出Action作用域恢复全局期限，后续收尾不会永久继承局部期限。

这是协作式期限，不提供线程强杀。外部库若忽略超时，可能晚返回，但晚返回的候选仍不能提交。重新恢复会建立新尝试的局部时限，全局累计消耗和期限仍继承；本批未新增独立的委派累计时间账本。

## 验证和收敛

新增16项专项通过，章节/专业运行器/停止恢复/Artifact专项56 passed。最终全量`python -m pytest -q --tb=short`：**2361 passed、17 skipped、11 warnings，221.26秒**；既有章节恢复、工具及写作边界回归均通过。本地日志为`data/architecture_v2_section_recovery_tests.txt`；`git diff --check`通过。

覆盖第二节取消后第一节零重算、首次费用保留、真实os._exit后跨进程恢复、基稿/授权/证据/年份/提示/模型变更、缓存不可变与损坏、Search和越界章节拒绝、Worker继承期限、HTTP剩余timeout、全局期限更严格、Sidecar不重置计时，以及超时候选不进入研究提交。

收敛审查：复用既有Artifact、ToolExecutor、PersistenceWriter及Controller提交；没有章节Stage/Submit发布协议。移除Action作用域退出后的重复终止检查，避免已提交后才报告本次超时；正式提交前的检查集中在Controller。模型/提示依赖进入已有章节指纹，缓存命中后再次局部检查；权威业务版本只随正式Action提交增加。

后续批次已补齐[三类各100条竞态组合及租约时钟修复](2026-11-18-v2-race-matrix.md)。默认入口仍为v1。真实Provider/CNKI/原生pgvector、独立人工质量验收、完整业务冻结场景、更多历史会话清点、最终默认切换和旧路径退役继续实施。本次未运行付费模型端到端或人工审查，不将模拟调用成本外推为实际节省比例。
