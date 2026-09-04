"""引用一致性检查 Prompt。"""

CITATION_CHECK_PROMPT = """\
请检查以下文献综述中的引用与参考文献列表是否一致。

综述正文：
{review_text}

参考文献列表：
{references_json}

返回 JSON：
{{
  "valid": true/false,
  "missing_citations": ["正文引用但不在参考文献表的 paper_id"],
  "unused_references": ["参考文献表中未在正文引用的条目"],
  "suggestions": ["建议"]
}}
"""

__all__ = ("CITATION_CHECK_PROMPT",)
