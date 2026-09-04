"""主题消歧与用户范围回答解析 Prompt。"""

TOPIC_DISAMBIGUATION_PROMPT = """\
你是跨学科学术检索系统的主题消歧模块。请判断用户主题是否存在会显著改变检索语料、纳入标准和综述结构的多种成熟含义。

用户请求：{user_query}
已抽取主题：{topic}
结构化约束：{constraints_json}

判断原则：
- 只有当同一主题在不同学科、研究对象、分析单位或研究范式中具有实质不同的常用含义时，才判定 ambiguous=true。
- 普通子方向、同一任务的不同方法、不同数据集或不同应用场景不属于必须追问的主题歧义。
- 用户已明确限定学科、研究对象、方法范围或排除项时，应尊重限定，通常不再追问。
- 用户明确要求全面、综合或跨学科覆盖时，优先推荐 multi_branch，不要重复追问。
- 用户未限定范围且不同解释会导致明显不同的论文集合时，推荐 ask_user。
- 所有范围名称、纳入词、排除词和种子查询必须从当前主题推导，不得套用固定领域示例。
- ask_user 时给出 2 到 4 个互斥或层次清楚的范围选项；必要时可增加一个跨分支综合选项。
- question 必须是一个简洁、自然的疑问句，在句子中概括由当前主题动态识别出的主要范围差异；不要要求用户回复编号或固定选项。
- 每次只追问一个最影响检索范围的问题，不要在同一轮连续提出多个问题。
- scope_id 使用简短稳定的英文 snake_case；seed_queries 应包含适合学术数据库的中英文标准表达。
- include_terms 和 exclude_terms 必须同时提供该范围关键概念的中文与英文表达，确保后续能对中英文题名和摘要执行同一范围约束。

严格返回 JSON：
{{
  "ambiguous": true,
  "confidence": 0.0,
  "reason": "为什么不同解释会改变检索范围",
  "recommended_strategy": "single_scope | ask_user | multi_branch",
  "default_scope_id": "默认范围ID或null",
  "question": "需要追问用户的问题；无需追问时为null",
  "scopes": [
    {{
      "scope_id": "scope_id",
      "label": "范围名称",
      "description": "该范围研究什么以及与其他范围的区别",
      "include_terms": ["中文纳入概念", "English inclusion concept"],
      "exclude_terms": ["中文排除概念", "English exclusion concept"],
      "seed_queries": ["标准检索式"]
    }}
  ]
}}

除 JSON 外不得输出其他内容。
"""

SCOPE_ANSWER_RESOLUTION_PROMPT = """\
你是学术研究 Agent 的多轮范围理解模块。请结合候选研究范围，理解用户对上一轮问题的自由文本回答。

上一轮问题：{question}
候选范围：{scopes_json}
用户回答：{answer}

规则：
- 判断用户表达更符合哪个候选范围，不要求用户复述候选名称。
- 可以根据研究对象、数据、方法、目标和学科视角进行语义匹配。
- 如果用户明确要求同时覆盖多个范围，返回多个 matched_scope_ids。
- 如果现有信息仍不足以可靠确定范围，needs_clarification=true，并只生成一个简洁、自然的后续疑问句。
- 不得创造候选列表之外的新范围 ID。
- question 只在需要继续澄清时填写，否则为 null。

严格返回 JSON：
{{
  "matched_scope_ids": ["scope_id"],
  "needs_clarification": false,
  "question": null,
  "reason": "判断依据"
}}

除 JSON 外不得输出其他内容。
"""

__all__ = ("TOPIC_DISAMBIGUATION_PROMPT", "SCOPE_ANSWER_RESOLUTION_PROMPT")
