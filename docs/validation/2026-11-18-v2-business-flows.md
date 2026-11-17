# v2冻结业务闭环与发布恢复

日期：2026-11-18，Asia/Singapore。对应S1—S4的跨角色流程、服务暂停恢复及草稿/正式门禁。

## 冻结输入和执行边界

材料见[合成业务输入](../../examples/architecture-v2/frozen-business-input.json)。三个记录的题名、来源标识和原文均为明确标注的离线合成数据；标识采用系统既有格式以测试引用协议，不代表存在对应的在线论文。脚本核验标签不是人工金标。

冻结检索、详情、抽取和主张规划的产物。检索结果继续绑定实际内容/范围指纹及准入字段，不关闭Hybrid准入。写作仍使用实际write_deliverable/renderer、候选审计和validate_deliverable；Check使用生产Handler、逐句核验工具、引用校验、授权一致性和最终质量门。模型文本及蕴含标签由离线脚本返回，不手写passed CheckReport或替换finish_validator。

测试通过真实TaskMain、DelegationRunner、Controller、SQLite和Artifact Store运行。Native/JSON两通道与deterministic/react/plan_and_act三种局部策略组合成六条正常路径；Writing/Check保持direct。

## 业务场景

| 场景 | 验收行为 |
| --- | --- |
| 四角色正式交付，6个组合 | Search→详情→Analysis→Writing→Check→finish；3篇有效唯一引用，正文无内部标识 |
| 脚本语义结果为矛盾 | 实际Check与最终门禁阻止正文公开 |
| 候选未经过Check即请求finish | 拒绝结束，候选保持私有 |
| 要求5篇，实际3篇 | 最终引用门禁阻断，年份和最低篇数保持原要求 |
| 已通过报告对应的正文被改动 | 旧报告失效，finish拒绝，新正文不可公开 |
| 单章节修订 | 新写作版本再次经过独立Check后才允许交付 |
| 主控澄清与用户回答 | 服务保存可恢复问题；回答更新年份并保留40篇要求 |
| 发布预检暂时拒绝 | 返回rejected观察，主控可继续规划与重试 |
| 收尾额度触发 | 暴露目录前停止新增检索委派，保留收尾额度 |
| 新候选Check中取消 | 保留旧发布草稿及取消状态，新候选不替换公开正文 |
| 发布快照提交前/后中断 | 恢复没有未配对publish_draft调用，不重复生成发布版本 |
| 发布观察不预写归档 | 控制观察和发布版本只在同一快照持久化 |

共18项新增回归，另与既有TaskMain/草稿策略组合完成68项专项。

## 修复内容

1. v2 request_clarification原先仅返回question，服务层按kind=main_agent及needed识别暂停，导致可恢复澄清状态丢失。现与原主控协议对齐，回答后的显式约束继续生效。
2. publish_draft的来源/版本预检ValueError原先直接中断v2循环。现将业务拒绝作为观察交回Main；预算、上下文、取消和失权等控制异常原样传播。
3. v2循环原先未执行已有update_finalization和prepare_memory_context。现每轮在目录构造前执行，继续使用原预算与已有记忆摘要，不新增额度或证据授权。
4. 原发布快照保存了版本，却可能留下待配对的publish_draft。现Controller在候选副本中完成小型控制观察的内存配对，与发布记录同快照提交；不在CAS前另写成功归档。TaskMain不再重复构造成功观察，finish自动发布也不会错误消费request_finish调用。
5. 正常返回时同步result_status；显式report_blocked采用稳定的main_agent_blocked代码并保留原原因，便于服务与界面识别。

## 验证和剩余范围

最终专项：**68 passed，52.76秒**，日志`data/v2_business_final_targeted_tests.txt`。最终全量`python -m pytest -q --tb=short`：**2682 passed、17 skipped、11 warnings，488.44秒**，日志`data/architecture_v2_business_final_tests.txt`；`git diff --check`通过。

发布提交窗口使用确定点SystemExit注入，真实进程强退由既有费用和章节恢复测试覆盖；本批不将两类证据混为同一种测试。

收敛审查：复用实际发布预检、既有收尾预算/记忆准备和单一Controller快照；成功发布观察由Controller统一配对，异常拒绝由主控反馈，未增加备用提交路径。测试材料补齐既有准入、验证列表及引用格式，不修改或关闭质量门禁；Artifact Store负责长轨迹压缩与恢复，未通过扩大上下文窗口让测试通过。

本批证明离线控制流与门禁行为，不证明检索效果、模型语义准确率或成本收益。真实来源/Provider、原生pgvector、策略对照和独立人工审查仍未验收；完整API/前端/报告核对、历史任务清点、最终回退、默认切换及旧路径退役继续推进。默认入口仍为v1。
