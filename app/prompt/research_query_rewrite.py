"""Clarification-aware research request rewrite prompt."""

RESEARCH_QUERY_REWRITE_PROMPT = """你负责把研究请求与后续用户澄清合并为一条完整、可独立理解的研究请求。
只依据标记为 user 的原始请求和回答。澄清问题只帮助理解回答所指对象；候选选项的描述、排除词和检索词不是用户约束。
保留用户明确的时间范围、最终引用的不同论文篇数、交付物、研究对象、视角、方法及先后关系。
用户明确修订旧要求时采用最新要求；不得自行排除相邻方法或加入特定技术路径。
只返回 JSON 对象：{{"rewritten_query": "完整请求"}}。

原始用户请求：{original_query}
此前用户澄清（按时间顺序）：{prior_answers_json}
澄清问题：{question}
用户回答：{answer}
"""
