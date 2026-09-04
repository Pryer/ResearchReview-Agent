# 多轮研究会话 MVP

## 目标

多轮能力用于解决会实质改变检索语料的主题歧义，而不是对所有请求机械追问。会话状态保存在 `research_sessions` 表中，至少跨轮保留：

- 原始用户请求；
- 年份、最低引用数、语言、引用格式和指定章节；
- 候选研究范围；
- 用户确认的范围及纳入、排除概念；
- 完成后选中的论文 ID。

## 状态流转

```text
created
  → needs_clarification
  → running
  → completed
       ↘ failed
```

主题边界明确或用户已经明确要求跨分支综合时，可直接从 `created` 进入 `running`。

## API

首次请求传入客户端生成的稳定 `session_id`：

```json
{
  "session_id": "session-001",
  "user_query": "调研近三年某主题论文，引用不少于40篇"
}
```

需要澄清时返回：

```json
{
  "status": "needs_clarification",
  "session_id": "session-001",
  "research_request": {},
  "clarification": {
    "question": "请选择研究范围",
    "scopes": [
      {
        "scope_id": "scope_a",
        "label": "范围名称",
        "description": "范围说明",
        "include_terms": [],
        "exclude_terms": [],
        "seed_queries": []
      }
    ]
  }
}
```

续跑请求：

```json
{
  "session_id": "session-001",
  "user_query": "继续",
  "clarification_answer": "scope_a"
}
```

`clarification_answer` 支持 `scope_id`、从 1 开始的序号和完整选项名称。

## 当前边界

这是第一阶段的澄清型多轮能力。当前已经支持主题范围确认和原始约束恢复；后续阶段再增加完成后的自然语言修改、分类覆盖审计、检索结果复用和局部段落重写。
