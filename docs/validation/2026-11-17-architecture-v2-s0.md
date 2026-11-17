# 全架构重构执行记录：S0

日期：2026-11-17，Asia/Singapore。
状态：S0 进行中，未完成整体重构，也未切换生产默认入口。

## 基线

- Git HEAD：259c90b840a1b91651fd485387c99bee7ef6d4fa。
- 执行前工作区保留已有下载、核验身份与报告脱敏修复，未将其回退或覆盖。
- 本地源文件哈希清单：data/architecture_v2/baseline.json；不包含配置秘密。
- 本轮重新运行基线：2174 passed、17 skipped、11 warnings，110.20 秒。17 项为需要原生 pgvector 环境的测试。
- 基线未产生实际 Provider 费用或真实研究交付指标；相应量保持 unknown。

## 本批实现

1. 新增严格委派请求、服务端 Grant、Invocation、委派结果契约。
2. 注册四角色并从 Action owner 派生能力；新增八个静态 Profile 与注册一致性验证函数。
3. AuthorizationPolicy 复用现有动作准入、mandatory/recovery 与参数校验；增加目标身份、章节范围和兼容 single 约束。
4. 新增 direct 参数构造器与 legacy_single_v1 精确动作/不可变参数迁移器。
5. 新增委派、Invocation、业务版本表；通过现有 Base.metadata.create_all 加法建表，不修改旧表字段。专门迁移版本登记、升级/回退副本演练仍待实施。
6. ActionExecutor 通过独立 invocation_scope 调用现有 Controller；未创建第二套业务提交器。
7. 在现有 begin_task/snapshot 事务中关联 Invocation，并原子提交结果、步骤游标与业务版本。观察消费游标独立推进。
8. 稳定 Invocation 重投读取已提交结果、不应用旧 patch；失败重试产生新的 Task 尝试键。旧路径幂等键保持原构造方式。
9. 提交端再次检查撤销与业务版本；取消/失权沿用现有运行时边界。

## 已验证场景

- 真实 SQLite 上搜索与详情两个 Action 连续独立提交。
- Durable 结果存在而 Agent 尚未消费时恢复；消费游标与提交游标分离。
- 数据库重开后读取原结果，不重放已提交 Handler。
- 提交事务故障同时回滚业务、结果、游标和版本。
- failed Attempt 在同一 Invocation 下可创建新 Task。
- Grant 撤销、回调取消与外部业务版本推进后拒绝提交。
- mandatory 与结果同事务消费；下一 mandatory 改变后仍可读取历史提交结果。
- direct 参数构造与历史参数深拷贝，跨角色兼容迁移拒绝。
- 运行快照和预算记账不推进新增业务版本；现有 Runtime.version 继续承担运行检查点 CAS。

上述为单元/组件测试和 SQLite 故障替身，尚未完成计划的全部 16 格故障矩阵、竞态调度及多进程退出测试。

## 后续必须完成

- Scope/资源 lineage 全量检查、版本化验收器、Sidecar 接入和配置冻结。
- 统一 PersistenceWriter 线程归属与持久预留/ack/结算协议，故障与背压测试。
- 迁移登记、数据库副本升级/回退演练、全部依赖和行为冻结场景。
- 三类 Strategy、四角色运行时、Main 任务级入口与全部 ToolExecutor 接入。
- API/Job、报告、前端、历史完整会话迁移和默认切换。
- 实际 Provider 预算、阶段策略评测和独立字段/引用审查。

新增模块为尚未对默认入口开放的重构内核。调用现有 Controller 仅是迁移适配，不代表 ActionExecutor/CommitCoordinator 拆分或整个 S0 已验收。

## 本批最终验证

- 全量离线回归：2188 passed、17 skipped、11 warnings，177.42 秒。
- 新增 v2 行为测试 14 项；包含两步提交、重开恢复、失败重试、提交故障、撤销/取消/业务版本冲突、mandatory、direct 与 legacy single。
- git diff --check 通过；计划归档和执行记录链接存在。
- 收敛检查：旧路径幂等键保持原身份；新结果缓存不再重放 patch；读取已提交结果不受下一 mandatory 的前置条件影响；Grant 读取强制刷新，提交再次校验撤销；兼容 Action 同样推进已建立的业务版本。
- 未增加第二套业务提交事务，移除无实现的策略占位目录；默认主控入口尚未切换。
- 本批未运行真实 LLM/CNKI/pgvector，未进行 GitHub 推送。

## 后续批次：线程归属与Search运行器

- 新增同步 PersistenceWriter 接口：共享执行锁，Worker 在自身线程创建短生命周期 Session，预留方法返回即提交确认；不建立常驻队列线程。默认主线程仍使用原 Session 与提交器。
- 选择依据：在父 Session 的 execute/get 注入线程归属断言，并发路径经新接口通过；先以1/4/8 Worker各16次完成集成，最终扩大至各100次模拟调用并全部通过；完整竞态/进程故障矩阵仍待完成。
- 请求前持久预留，Provider失败保守结算；预留确认后再次执行取消/期限检查。Worker没有数据库对象或研究提交接口权限。
- 锁顺序保持预算锁→执行锁；同步短事务内不调用预算代码、不等待网络/Worker，避免引入队列ack的反向等待。
- 新增 Search DelegationRunner 与阶段验收：同一委派执行搜索→详情，消费已提交观察，持久化委派汇总；任务级结果包含全部Action引用。
- 默认确定性策略与可注入Search ReAct策略共用执行器；Native/JSON替身均验证相同授权行为。规划只传递有限的Search摘要。
- 失败后读取原pending Invocation继续，不重放已提交搜索；委派汇总落库后再次调用读取原汇总。
- 空结果不会被当作阶段成功；候选元数据就绪不等于最终唯一引用数或引用语义验收通过。
- v2执行端保留配置开启时的Sidecar前审查；Policy拒绝不可提权，未提供已配置Sidecar时拒绝执行。

该运行器目前为内部集成入口，尚未迁入默认Main/API生产路径。S0的完整迁移/回退、故障矩阵、lineage和业务验收仍未全部完成；S1的真实Handler与端到端、模型协议有界纠错和完整轨迹恢复仍待接入。

## 本批收敛检查

- Invocation绑定grant_revision，执行开始和提交均拒绝审查期间或执行期间的授权更新；新测试覆盖执行中Grant revision改变。
- 恢复比较规范模型而非JSON字段顺序，避免等价参数顺序造成身份误拒绝。
- Search进展仅由实际成功/空结果的查询账本键及业务内容计算；running/failed、时间戳和尝试次数不构成进展。
- Worker写入接口只允许check/reserve/settle/persist，父Session访问与研究提交明确拒绝。
- 临时无效测试调用已修正为项目已有submit_with_context接口；未扩展线程帮助函数签名。
- 最后一次定向验证：27 passed（新运行器、并发持久写入和委派恢复），12.96秒。

## 本批最终结果（线程写入与Search接入）

- 全量回归：2201 passed、17 skipped、11 warnings，138.11秒。
- 最终新增Sidecar拒绝测试后，v2委派/运行器/线程接口定向29 passed（15.01秒）。
- 扩大并发负载后，1/4/8 Worker各100次模拟请求的线程归属、持久预留、Token精确结算，以及失败与权限测试共5 passed（12.80秒）。后两项为全量之后的专项验证，不把它们当作额外独立全量通过数相加。
- 数据库与模型调用均为本地SQLite/替身；未执行真实LLM、CNKI或原生pgvector。
- 本批仍未完成全架构重构或默认生产入口切换。下一优先项是S0迁移/回退及故障矩阵，随后将Search接到真实Main/Handler，再完成Analysis/Writing/Check。

## 第三批：迁移、真实进程强退与共享Handler入口

### 已实现

- architecture_migrations：版本与Schema哈希登记，列名/类型/空值/主键/Invocation步骤唯一约束校验；未知版本拒绝。
- SQLite升级采用BEGIN IMMEDIATE串行检查/建表/登记，可重复启动，DDL失败可回滚。不会自动删除旧字段或旧表。
- 数据库副本使用SQLite backup验证旧记录保留、重复升级、备份回退；两个并发启动仅写入一份版本记录。
- build_research_actions抽出现有graph Handler和finish validator，保持一份业务实现；旧pipeline仍调用同一组合根，LLM实例复用。
- run_search_delegation任务级服务入口接入该组合根，真实fetch_detail_node消费模拟外部元数据；尚未替换Main/API默认调度。
- 授权角色/范围与已有委派身份先校验，再初始化Handler/运行快照。每会话只允许一个活动的可写委派。
- 数据库为活动委派权威来源；create之后状态快照之前的中断可找回记录；已完成汇总清理marker失败后可读取原汇总。

### 故障证据

- single/multi各八个故障窗口共16格：Invocation持久化前后、Handler开始、事务提交前、提交后、消费后、租约接管和取消。
- 目标变化的单独负例：第一步已提交，用户改变主题后旧Grant不能继续第二步。
- 两个真实子进程os._exit(17)用例：single/multi在模拟Provider调用处强退，不运行finally；请求前预留已落库，恢复将未知消费保守入账，原Invocation可新Attempt重试，已提交前置动作不重放。
- 子进程死亡后仅在测试数据库推进租约至过期，未冒用仍有效的新owner；生产仍依赖真实lease/fencing。
- 真实Handler组合的检索响应和详情API使用替身；其余详情节点与字段契约实际执行。不是对真实外部来源的验收。

### 尚待实施

- 100种受控竞态调度/故障序列、业务依赖声明/lineage与完整Profile验收仍未完成。
- 默认Main任务目录、Analysis/Writing/Check运行器、ToolExecutor全量接入、API/报告/前端切换仍待完成。
- 数据库备份回退当前在测试副本验证；未对用户生产库执行降级或恢复操作。
- 第三批核心定向26 passed（15.25秒）；旧graph/脚本回归74 passed，其他组合验证亦通过。最终全量另行记录。

## 第三批收敛与兼容保护

- 非法角色/范围和恢复请求先验证，拒绝请求不会先构建Handler或推进初始化检查点。
- 活动委派只维护一份数据库权威记录，清理内存marker必须等待最终快照成功。
- 原Main入口拒绝architecture_version=2或数据库中活动v2记录，防止恢复时掉回单Action调度。新任务级服务保持显式接入，尚未作为完整Main/API默认实现发布。
- SQLite迁移的列类型/长度/nullable、主键、唯一约束均校验；BEGIN IMMEDIATE串行启动并确保异常DDL回滚。畸形Schema测试确认不遗留半迁移表。
- 旧Main与新任务服务共用同一Handler/finish validator组合根，未创建复制的详情/写作/核验业务实现，LLM实例只创建一次。
- 故障测试只撤销自身commit注入，不撤销pytest自动配置的离线Provider替身。
- 全量2229 passed/17 skipped/11 warnings（167.82秒）；最后兼容保护专项73 passed，迁移专项4 passed。最终全量确认结果另记。

## 第三批最终验证

最终完整离线回归：**2231 passed、17 skipped、11 warnings，196.36秒**。17项原生pgvector测试因环境条件跳过；警告为第三方依赖弃用。真实LLM/CNKI/外部检索Provider和人工语义审查未执行。

已验证副本升级/回退、并发启动与错误DDL回滚；16格异常窗口与两个os._exit子进程恢复；共享生产Handler组合与真实详情节点；旧Main禁止接管v2活动任务。未完成100种竞态调度、完整lineage/全部Profile验收、其他角色与默认Main/API/前端切换，因此S0/S1整体验收仍保持未勾选。

后续进展：四角色与任务级Main已显式接入；最终本地回归2255 passed/17 skipped。新阶段范围及剩余项见[四角色与任务级Main](2026-11-17-v2-role-main.md)。
