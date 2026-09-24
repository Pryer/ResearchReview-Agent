# 多轮研究会话与恢复

更新：2026-09-22。正式入口为 `POST /api/reviews/jobs`，后台任务经
`ResearchConversationService` 进入持久化执行范围。主 Agent 每轮只读五字段，
按一个动作推进；原始证据、任务补丁、历史事件和预算保存在服务端。

## 状态与数据

`research_sessions` 保存公开会话及研究状态；runtime、task、attempt、checkpoint、
event 和 memory-cursor 表保存执行提交、用量、检查点与增量历史。

后台任务从 `queued` 进入 `running`，可结束为 `completed`、`partial`、`blocked`、
`needs_clarification`、`failed` 或 `cancelled`。`cancel_requested` 表示取消尚在处理。
会话状态与后台任务状态分开保存；等待澄清的后台任务已经结束，用户回答会创建新任务。
`partial` 是明确降级交付，`blocked` 表示要求尚未满足，两者都不能显示为完整成功。

## 首次请求与澄清

向 `POST /api/reviews/jobs` 提交：

```json
{
  "session_id": "example-research-001",
  "user_query": "生成近三年的课堂行为识别研究背景和研究现状，引用不少于40篇"
}
```

API 返回的 `data` 中包含 `job_id`、`session_id`；使用
`GET /api/reviews/jobs/{job_id}` 查询进度和 `result`。首次可省略 `session_id`，
但后续必须使用返回的真实会话 ID。配置了 `APP_API_KEY` 时，请求携带 `X-API-Key`。

只有会实质影响研究范围的歧义才触发澄清。结果为 `needs_clarification` 时，
读取 `result.clarification` 的问题与选项，然后再次提交同一会话：

```json
{
  "session_id": "example-research-001",
  "user_query": "重点研究视觉与多模态方法对师生课堂行为的自动识别",
  "clarification_answer": "重点研究视觉与多模态方法对师生课堂行为的自动识别"
}
```

范围选项支持 `scope_id`、从 1 开始的序号、完整名称或自然语言说明。
澄清保留原始主题、显式年份、章节和最终引用要求；“不少于40篇”指正文最终使用的
唯一有效参考文献，不能用检索记录数或每篇论文的被引次数代替。

## 中断后的检查点恢复

对已有已提交检查点、状态为 `running/failed/cancelled/blocked` 的会话，提交：

```json
{
  "session_id": "example-research-001",
  "user_query": "继续上次研究",
  "resume_from_checkpoint": true
}
```

这是显式的新操作：恢复原约束、证据和累计预算，由主 Agent 重新决定未完成工作。
没有检查点会报错；已有活动任务或未到期的执行租约不会被抢占。恢复不会清零或提高
预算，也不保证一次恢复即可交付。请求正文不能注入内部控制状态。

新配置默认额度不会覆盖旧账本。需要调整原会话额度时，先使用
`scripts/update_session_budget.py` 显式更新空闲会话预算，再提交恢复请求；
累计消耗保持不变，命令与并发限制见 [预算说明](agent-runtime-and-cache.md)。

服务重启会重新排队 `queued` 任务，将中断的 `running` 任务标记失败，并完成已请求
的取消；不会自动重放未知是否已计费的外部请求。待澄清会话应回答澄清，已完成结果
应走修订；它们与 `resume_from_checkpoint` 是不同入口。

## 修订、取消与历史

| 操作 | API 与语义 |
| --- | --- |
| 排除论文并增量重生成 | `POST /api/reviews/jobs/revise`；按 `ResearchRevisionRequest` 传原会话、排除集合及可选说明，仍执行质量门禁 |
| 请求取消 | `POST /api/reviews/jobs/{job_id}/cancel`；取消先提交时，迟到研究结果不得再提交，已发生用量仍结算 |
| 查询会话 | `GET /api/reviews/sessions/{session_id}`；返回公开历史、修订记录与当前结果，不暴露完整内部状态 |

会话历史先逐条归档，再保留最近 50 条展示窗口；公开历史 API 不等于完整事件导出。
摘要按持久化游标增量更新。旧版本已截断的历史无法补造，缺失会被记录。

前端目前没有检查点恢复按钮或数据库历史会话自动重建功能，详见
[前端指南](CHAT_FRONTEND_GUIDE.md)。完整契约见
[执行提交、恢复与缓存边界](agent-runtime-and-cache.md)，示例见
[检查点恢复请求](../examples/checkpoint_resume_request.json)。
真实验收进展见 [2026-09-22 验收记录](validation/2026-09-22-runtime-cache.md)。
