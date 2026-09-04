"""论文证据卡片抽取 Prompt。"""

# 保留历史拼写，避免破坏现有 API；后续可在独立兼容周期中更名。
PAPER_CARD_EXTRACTION_PROCTION_PROMPT = """\
请根据以下论文内容，提取结构化论文卡片。
必须严格返回 JSON，不要编造任何论文中没有的信息。
论文可能属于任意学科，也可能采用理论论证、定性研究、定量研究、实验研究、系统设计或混合方法。请按原文实际研究范式理解字段，不得强行套用算法论文结构。

{evidence_label}

论文标题：{title}
{full_text_or_json}

返回 JSON 结构：
{{
  "paper_id": "{paper_id}",
  "title": "...",
  "authors": ["作者"],
  "year": 整数或 null,
  "venue": "...",
  "doi": "...或null",
  "url": "稳定链接或null",
  "publication_type": "journal_article / conference_paper / conference_short_paper / preprint / systematic_review / meta_analysis / unknown",
  "peer_review_status": "peer_reviewed / likely_peer_reviewed / not_peer_reviewed / unknown",
  "evidence_level": "meta_analysis / systematic_review / peer_reviewed_article / conference_paper / preprint / unknown",
  "research_problem": "研究问题",
  "study_design": "研究设计或证据类型",
  "sample_size": "原文明确报告的样本或数据规模（可 null）",
  "data_modalities": ["视频/语音/文本/姿态/问卷等"],
  "behavior_categories": ["原文明示的行为或状态类别"],
  "method": "核心理论、研究设计、分析方法或技术方案",
  "dataset": "数据、样本、语料、材料或研究对象（可 null）",
  "metrics": ["评价指标、分析维度或判定标准"],
  "results": "主要发现、论证结论或实验结果（可 null）",
  "contributions": ["贡献1"],
  "limitations": ["局限1"],
  "relevance_reason": "与主题的相关性说明",
  "evidence_source": "{evidence_source}"
}}

重要：
- 如果论文全文为空或过短，相应字段留空字符串，不要推测。
- 不适用于该论文研究范式的字段应留空，不得为了填满结构而虚构数据集、指标或实验。
- 如果当前仅提供摘要，只能提取摘要明确陈述的研究问题、概括性方法、数据/样本/指标和作者明确报告的结果；limitations 必须返回空数组，不得补充详细模型结构、数据划分、实验设置、消融结论或公平基线比较。
- 即使标签写有“全文”，也只能使用输入中实际出现的章节；看不到的章节和细节必须留空。
- results 中的数字、比较和强结论必须逐字存在于输入内容中；不得将“更高”改写为“显著优于”。
- evidence_source 必须设为 "{evidence_source}"。
"""

__all__ = ("PAPER_CARD_EXTRACTION_PROCTION_PROMPT",)
