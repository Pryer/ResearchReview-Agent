# ReAct 检索流程完整示例

本文档用一个模拟请求说明当前 ResearchReview-Agent 的完整运行过程。

示例请求：

```text
帮我调研近五年少样本动作识别相关论文，引用不少于50篇，并生成相关工作
```

目标是说明：

- planner 如何生成关键词、核心概念组和排除词
- search 如何检索论文
- rank 如何判断论文是否相关
- 检索不足时如何把反馈交给 LLM
- LLM 如何重新生成下一轮关键词
- 最终为什么生成综述或停止生成

## 1. 用户请求进入 Agent

入口函数：

```python
run_research_agent(user_query)
```

初始状态：

```json
{
  "user_query": "帮我调研近五年少样本动作识别相关论文，引用不少于50篇，并生成相关工作",
  "steps": [],
  "errors": []
}
```

## 2. 意图识别

节点：

```python
plan_node()
```

规则或 LLM 会识别出用户想生成综述：

```json
{
  "intent": "generate_review",
  "confidence": 0.9
}
```

原因是用户请求里包含：

```text
调研
生成相关工作
引用不少于50篇
```

## 3. 槽位抽取

从原始请求中抽取结构化参数：

```json
{
  "topic": "少样本动作识别",
  "start_year": 2022,
  "end_year": 2026,
  "max_papers": 50,
  "year_range_explicit": true,
  "max_papers_explicit": true,
  "language": "zh",
  "citation_style": "gbt7714"
}
```

其中：

- 近五年按当前年份 2026 解析为 2022-2026
- 不少于50篇解析为 `max_papers=50`
- 生成相关工作解析为中文综述

## 4. Planner 第一次生成检索策略

当前设计下，正常情况不再直接使用固定兜底词表，而是让 LLM 生成检索策略。

LLM 输入包含：

```text
用户请求：帮我调研近五年少样本动作识别相关论文，引用不少于50篇，并生成相关工作
研究主题：少样本动作识别
```

LLM 需要返回 JSON：

```json
{
  "keywords": [
    "少样本动作识别",
    "few-shot action recognition",
    "few-shot video action recognition",
    "few-shot human activity recognition"
  ],
  "required_concepts": [
    {
      "concept": "少样本学习",
      "terms": ["few-shot", "few shot", "one-shot", "low-shot", "少样本", "小样本"]
    },
    {
      "concept": "动作识别",
      "terms": ["action recognition", "activity recognition", "human action", "video action", "动作识别", "行为识别"]
    }
  ],
  "excluded_title_terms": [
    "zero-shot action recognition",
    "image classification",
    "object recognition",
    "text recognition"
  ]
}
```

Agent 状态更新为：

```json
{
  "topic": "少样本动作识别",
  "keywords": [
    "少样本动作识别",
    "few-shot action recognition",
    "few-shot video action recognition",
    "few-shot human activity recognition"
  ],
  "required_concepts": [
    ["few-shot", "few shot", "one-shot", "low-shot", "少样本", "小样本"],
    ["action recognition", "activity recognition", "human action", "video action", "动作识别", "行为识别"]
  ],
  "excluded_title_terms": [
    "zero-shot action recognition",
    "image classification",
    "object recognition",
    "text recognition"
  ]
}
```

日志中会出现：

```text
AGENT_STEP_DEBUG {"step_name": "plan", ...}
```

## 5. 第一轮选择实际检索关键词

配置项：

```text
MAX_SEARCH_KEYWORDS=3
```

因此 search 节点不会把所有关键词都搜一遍，而是选前三个高价值检索词。

由于中文关键词容易在 Crossref/OpenAlex 中引入中文期刊噪声，如果已有英文关键词，会优先搜索英文关键词。

第一轮实际检索词可能是：

```json
[
  "few-shot action recognition",
  "few-shot video action recognition",
  "few-shot human activity recognition"
]
```

每个英文关键词会路由到：

```json
["arxiv", "semantic_scholar", "openalex"]
```

不会路由到 Crossref。

## 6. 第一轮检索

节点：

```python
search_node()
```

模拟检索结果：

```json
{
  "few-shot action recognition": {
    "arxiv": 22,
    "semantic_scholar": 0,
    "openalex": 50
  },
  "few-shot video action recognition": {
    "arxiv": 15,
    "semantic_scholar": 0,
    "openalex": 40
  },
  "few-shot human activity recognition": {
    "arxiv": 8,
    "semantic_scholar": 0,
    "openalex": 25
  }
}
```

如果 Semantic Scholar 被限流，日志可能出现：

```text
Semantic Scholar search failed: 429 Client Error
```

这不会直接中断流程，因为 arXiv 和 OpenAlex 仍然可用。

search 节点输出示例：

```json
{
  "count": 95,
  "new_count": 160,
  "per_keyword": [
    {
      "keyword": "few-shot action recognition",
      "returned": 72,
      "sample": [
        {
          "title": "A Comprehensive Review of Few-shot Action Recognition",
          "year": 2024,
          "source": "arxiv"
        }
      ]
    }
  ]
}
```

这里的 `new_count` 是原始新增结果数量，`count` 是按 DOI/arXiv ID/paper_id 初步合并后的候选数量。

## 7. 第一轮相关性筛选

节点：

```python
rank_node()
```

rank 会先去重，再调用：

```python
evaluate_topic_filter()
```

当前匹配逻辑：

1. `excluded_title_terms` 只匹配标题，命中则排除。
2. 如果有 `required_concepts`，必须每个概念组都命中。
3. 有摘要时，概念组可在 title + abstract + venue + source 中命中。
4. 但如果有两个以上概念组，标题至少要命中一个核心概念。
5. 没有摘要时，只看标题。

### 通过示例

论文：

```json
{
  "title": "MVP-Shot: Multi-Velocity Progressive-Alignment Framework for Few-Shot Action Recognition",
  "abstract": "Recent few-shot action recognition methods perform semantic matching..."
}
```

匹配情况：

```text
少样本概念组：title 命中 Few-Shot
动作识别概念组：title 命中 Action Recognition
标题核心概念：命中
```

结果：

```json
{
  "passed": true,
  "reason": "通过概念组 (2/2, 全字段)"
}
```

### 拒绝示例 1：只有动作识别，没有少样本

论文：

```json
{
  "title": "MMG-Ego4D: Multi-Modal Generalization in Egocentric Action Recognition",
  "abstract": "We study multimodal generalization in egocentric action recognition..."
}
```

匹配情况：

```text
少样本概念组：未命中
动作识别概念组：命中 action recognition
```

结果：

```json
{
  "passed": false,
  "reason": "概念组命中不足 (1/2, 全字段): few-shot / few shot / one-shot"
}
```

### 拒绝示例 2：摘要提到主题，但标题完全不相关

论文：

```json
{
  "title": "Vision Transformer Recognition Tasks: A Survey",
  "abstract": "The survey briefly discusses few-shot action recognition."
}
```

匹配情况：

```text
摘要命中 few-shot action recognition
标题没有命中 few-shot / action recognition / activity recognition 等核心概念
```

结果：

```json
{
  "passed": false,
  "reason": "标题未命中核心概念"
}
```

这样可以避免只在摘要里顺带提一句的泛化论文混入。

第一轮 rank 输出示例：

```json
{
  "ranked": 18,
  "accepted_before_dedup": 24,
  "filter_reasons": {
    "概念组命中不足 (1/2, 全字段): few-shot / few shot / one-shot": 21,
    "标题未命中核心概念": 9,
    "命中排除词: zero-shot action recognition": 2
  },
  "ranked_sample": [
    {
      "title": "A Comprehensive Review of Few-shot Action Recognition",
      "year": 2024
    },
    {
      "title": "MVP-Shot: Multi-Velocity Progressive-Alignment Framework for Few-Shot Action Recognition",
      "year": 2025
    }
  ]
}
```

用户要求 50 篇，但第一轮只筛出 18 篇，因此进入 ReAct 修正。

## 8. ReAct 反馈给 LLM

节点：

```python
refine_search_node()
```

Agent 会把上一轮检索和筛选反馈整理为 JSON，传给 LLM。

反馈示例：

```json
{
  "target": 50,
  "candidate_count": 95,
  "ranked_count": 18,
  "searched_keywords": [
    "few-shot action recognition",
    "few-shot video action recognition",
    "few-shot human activity recognition"
  ],
  "search_output": {
    "per_keyword": [
      {
        "keyword": "few-shot action recognition",
        "returned": 72
      },
      {
        "keyword": "few-shot video action recognition",
        "returned": 55
      }
    ]
  },
  "rank_output": {
    "filter_reasons": {
      "概念组命中不足 (1/2, 全字段): few-shot / few shot / one-shot": 21,
      "标题未命中核心概念": 9
    },
    "filtered_examples": [
      {
        "title": "Generic Action Recognition",
        "reason": "概念组命中不足 (1/2, 全字段): few-shot / few shot / one-shot"
      }
    ],
    "ranked_sample": [
      {
        "title": "MVP-Shot: Multi-Velocity Progressive-Alignment Framework for Few-Shot Action Recognition"
      }
    ]
  }
}
```

LLM 会看到：

- 当前目标是 50 篇
- 已经搜过哪些关键词
- 哪些关键词召回多
- 哪些论文被过滤
- 过滤失败的主要原因是什么
- 已通过论文标题中出现了哪些领域术语

## 9. LLM 生成第二轮检索策略

LLM 可能发现：

```text
FSAR 是 Few-Shot Action Recognition 的常用缩写。
上一轮没有搜索 FSAR。
```

于是返回：

```json
{
  "keywords": [
    "少样本动作识别",
    "FSAR",
    "few-shot action recognition",
    "few-shot skeleton action recognition",
    "cross-domain few-shot action recognition"
  ],
  "required_concepts": [
    {
      "concept": "少样本",
      "terms": ["few-shot", "few shot", "one-shot", "low-shot", "FSAR", "少样本"]
    },
    {
      "concept": "动作识别",
      "terms": ["action recognition", "activity recognition", "human action", "video action", "动作识别"]
    }
  ],
  "excluded_title_terms": [
    "zero-shot action recognition",
    "image classification",
    "object detection"
  ]
}
```

Agent 状态更新：

```json
{
  "keywords": [
    "少样本动作识别",
    "FSAR",
    "few-shot action recognition",
    "few-shot skeleton action recognition",
    "cross-domain few-shot action recognition"
  ],
  "search_refinement_count": 1
}
```

日志中会出现：

```text
AGENT_STEP_DEBUG {"step_name": "refine_search", "status": "success", ...}
```

## 10. 第二轮检索

search 节点会跳过已经搜过的关键词，只检索新关键词。

第二轮实际检索词可能是：

```json
[
  "FSAR",
  "few-shot skeleton action recognition",
  "cross-domain few-shot action recognition"
]
```

模拟 arXiv 召回：

```json
{
  "FSAR": 26,
  "few-shot skeleton action recognition": 12,
  "cross-domain few-shot action recognition": 9
}
```

第二轮 search 会累计候选，不覆盖第一轮：

```json
{
  "previous_candidates": 95,
  "new_count": 47,
  "count": 130
}
```

## 11. 第二轮 rank

rank 再次对累计候选做去重和相关性过滤。

注意：此前已修复标题去重逻辑。

旧逻辑是字符集合相似度，容易把大量 FSAR 论文误合并。

现在是词集合相似度，不会因为标题都包含：

```text
Few-Shot Action Recognition
```

就把不同方法论文合并掉。

第二轮 rank 可能输出：

```json
{
  "ranked": 35,
  "accepted_before_dedup": 43,
  "filter_reasons": {
    "概念组命中不足 (1/2, 全字段): few-shot / few shot / one-shot": 15,
    "标题未命中核心概念": 6
  }
}
```

如果仍然不足 50，Agent 最多还会再执行一次：

```text
refine_search -> search -> rank
```

当前默认最多修正 2 轮，避免无限循环。

## 12. 是否生成相关工作

最终判断：

```python
hard_shortfall = (
    max_papers_explicit
    and not retrieval_requirement_met
)
```

如果用户明确要求不少于 50 篇，而最终只有 35 篇：

```json
{
  "target": 50,
  "actual": 35,
  "retrieval_requirement_met": false
}
```

系统会停止生成正文，输出：

```markdown
## 未生成相关工作

用户要求至少引用 50 篇论文，但在 2022-2026 年范围内，经多关键词检索、去重和主题相关性筛选后仅获得 35 篇。为避免用无关论文凑数，系统已停止生成相关工作。请扩大年份范围或降低最低篇数后重试。

## 已筛选候选文献

[1] ...
[2] ...
```

如果最终达到 50 篇：

```json
{
  "target": 50,
  "actual": 50,
  "retrieval_requirement_met": true
}
```

则进入：

```python
generate_review_node()
```

并基于筛选出的论文标题、摘要、年份、venue 等生成相关工作。

## 13. 完整步骤序列

一次典型 ReAct 运行的步骤顺序如下：

```text
plan
search
rank
refine_search
search
rank
refine_search
search
rank
expand_year
fetch_detail
download_pdf
generate_review 或 retrieval_shortfall
citation_check
final_answer
```

其中 `refine_search` 只在结果不足时出现。

如果第一轮已经满足数量要求，则流程会是：

```text
plan
search
rank
expand_year
fetch_detail
download_pdf
generate_review
citation_check
final_answer
```

## 14. 对应日志怎么看

关键日志都写入：

```text
logs/app.log
```

搜索：

```text
AGENT_STEP_DEBUG
```

重点看这些字段：

```json
{
  "step_name": "search",
  "input_data": {
    "keywords": ["FSAR", "few-shot action recognition"]
  },
  "output_data": {
    "count": 130,
    "new_count": 47,
    "per_keyword": []
  }
}
```

```json
{
  "step_name": "rank",
  "output_data": {
    "ranked": 35,
    "filter_reasons": {},
    "filtered_examples": [],
    "ranked_sample": []
  }
}
```

```json
{
  "step_name": "refine_search",
  "input_data": {
    "feedback": {
      "target": 50,
      "ranked_count": 18,
      "filter_reasons": {}
    }
  },
  "output_data": {
    "keywords": ["少样本动作识别", "FSAR", "few-shot action recognition"],
    "refinement_count": 1
  }
}
```

## 15. 当前设计总结

当前不是固定词表驱动，而是：

```text
LLM 生成检索策略
-> 检索
-> 过滤
-> 把失败原因反馈给 LLM
-> LLM 修正关键词
-> 再检索
```

本地 fallback 只在 LLM 超时、空输出、没有英文检索词等异常情况下使用。

相关性过滤不是简单关键词匹配，而是：

```text
排除词：只看 title
核心概念组：title + abstract + venue + source
额外约束：title 至少命中一个核心概念
```

这样做的目的：

- 允许摘要参与匹配，提高召回
- 要求标题有主题锚点，降低无关论文混入
- 排除词只看标题，避免摘要里提到对比方法导致误杀
