"""基础参数提取 Prompt。"""

SLOT_EXTRACTION_PROMPT = """\
从用户请求中提取结构化参数。严格返回 JSON：
{{
  "topic": "研究主题",
  "start_year": 整数或 null,
  "end_year": 整数或 null,
  "max_papers": 整数,
  "language": "zh 或 en",
  "citation_style": "gbt7714 / apa / ieee / bibtex"
}}

规则：
- "近五年" 按 current_year={current_year} 计算
- "中文综述" → language="zh"
- "引用不少于 N 篇" → max_papers=N
- 未提及字段默认 null

用户请求：{user_query}
当前年份：{current_year}
"""

__all__ = ("SLOT_EXTRACTION_PROMPT",)
