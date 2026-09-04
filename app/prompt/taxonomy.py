"""论文分类轴归纳与主题分配 Prompt。"""

AXIS_INDUCTION_PROMPT = """\
以下是一个研究请求及其最具代表性的论文样本。请把研究范围作为约束，结合提供的分类指导策略，为该领域的文献编译一个最有解释力的综述分类体系（仅生成主题定义，不需要分配论文）。

分类策略指导：
{strategy_instruction}
补充约束（可能为空）：
{strategy_examples}

分类要求：
- 每篇论文只进入一个主类；本阶段给出的主题定义必须尽量互斥，避免同一论文按同一证据同时满足多个主类。
- 请严格按照指导策略和样本论文提取最合适的分类维度。
- 主题数量应适中：建议生成 3 到 8 个主题，使得类别之间具有实质区别，且内部具有共同特征。
- 给出每个主题的纳入与排除标准，该标准必须具体、可操作，后续将依此将百篇论文自动分拣入库。
- 不要用“其他相关研究”掩饰无法归类的情况。
- 仅需基于样本确定分类骨架，不包含 assignments 数组。
- 分类轴必须由当前研究请求和样本证据决定。不要默认采用技术路线、理论视角、应用场景、数据类型或年份中的任何一种；先判断哪种维度最能解释论文之间的实质差异。

论文样本 JSON：
{paper_cards_json}

严格输出 JSON 格式，如下所示：
{{
  "organizing_principle": "本次分类采用的主要组织原则",
  "rationale": "为什么该原则适合当前主题、范围和论文集合",
  "themes": [
    {{
      "theme_id": "T1",
      "name": "由当前论文归纳出的类别名称",
      "description": "分类依据和该类别的共同特征",
      "inclusion_criteria": ["具体判断条件"],
      "exclusion_criteria": ["应排除的情况"],
      "representative_papers": ["paper_id_代表"]
    }}
  ]
}}
"""

ASSIGNMENT_PROMPT = """\
请根据已确定的分类体系，将以下批次的论文分配到最匹配的主题类别中。

分类体系：
{taxonomy_themes_json}

待分配论文批次：
{paper_cards_json}

分配要求：
- 严格遵循每个主题的 inclusion_criteria 和 exclusion_criteria 进行判定。
- 每篇待分配论文必须有且仅有一个 primary_theme_id，且必须来自上文指定的分类体系。
- 不得编造、修改、或遗漏输入批次中的任何 paper_id，也不能分配输入批次以外的论文。
- 如果论文确实不属于任何已知主题，必须将其分配到最接近的兜底类别，不得留空。

严格输出 JSON 格式，如下所示：
{{
  "assignments": [
    {{
      "paper_id": "paper_id_1",
      "primary_theme_id": "T1",
      "secondary_theme_ids": ["T2"],
      "confidence": 0.90,
      "rationale": "基于论文研究方法匹配 T1 纳入标准",
      "evidence_fields": ["research_problem", "method"]
    }}
  ]
}}
"""

__all__ = ("AXIS_INDUCTION_PROMPT", "ASSIGNMENT_PROMPT")
