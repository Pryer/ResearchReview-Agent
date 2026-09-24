"""章节改写：固定规则在前，章节资料在后。"""
import json
from typing import Any
from app.schemas.deliverable_schema import CoreDeliverableType
from app.deliverables.few_shot_blueprints import get_section_blueprint


def _heading(title, level):
    return "#" * level + " " + title


def _section_rewrite_prompt(
    *,
    deliverable_type: CoreDeliverableType,
    section_id: str,
    title: str,
    heading_level: int = 2,
    topic: str,
    original: str,
    required_ids: list[str],
    purpose: str = "",
    target_word_count: int | None = None,
    research_focus: str = "",
    survey_papers: list[dict[str, Any]] | None = None,
    comparison_dimensions: list[str] | None = None,
    claim_constraints: str = "",
    require_cross_route_synthesis: bool = False,
) -> str:
    blueprint = get_section_blueprint(deliverable_type, section_id)
    survey_papers_json = json.dumps(survey_papers or [], ensure_ascii=False)
    dimensions = [str(value) for value in comparison_dimensions or [] if str(value).strip()]
    # 末条研究路线承担跨路线综合：该要求原先只写在 purpose 里，与"避免空泛
    # 固定结尾"的要求冲突而常被模型忽略，导致结构校验报"末段缺少跨路线综合"。
    cross_route_requirement = (
        "\n20. **本节是最后一条研究路线，必须以一个独立末段完成跨路线综合**："
        "概括各路线的共同进展、彼此差异与有证据支持的共性不足，"
        "并显式使用“综合”“总体”“共同”“差异”等表述；"
        "该段只综合本正文各路线已写明的判断，不得引入新事实，"
        "也不得另设“研究空白”“未来方向”之类的小标题。"
        if require_cross_route_synthesis
        else ""
    )
    if dimensions:
        comparison_instruction = (
            "仅在真实证据能够形成比较时，围绕 WritingPlan 动态给出的维度 "
            f"{json.dumps(dimensions, ensure_ascii=False)} 归纳差异、取舍或演进；"
            "比较结论必须由本节引用证据直接支持，不得套用预设方法分类。"
        )
    else:
        comparison_instruction = (
            "WritingPlan 未要求固定比较维度；根据章节任务和真实证据决定是否比较，"
            "不得为了形成趋势而发明方法类别、演进方向或适用条件。"
        )
    return f"""你是中文学术综述的章节编辑。请只改写下面这一个章节。

硬性要求：
1. 第一行必须且只能是下文给定的“规定标题行”，不得增加其他标题、前言或修改说明。
2. 把英文证据忠实转述为自然、严谨的中文；模型名、缩写、数据集名可保留英文，但不得保留完整英文句子。
3. 只使用草稿已有事实，不补充常识、数字、结论或推测；摘要证据只按摘要可见范围表述。
4. 按共同研究问题、方法或发现进行综合，不逐篇列举，不使用“论文明确报告”“从其他纳入证据看”等机械句式。
5. 引用必须紧跟所支持的中文主张，并严格使用半角 ASCII 方括号 [paper_id]；禁止改成〔paper_id〕或其他括号。可把支持同一综合判断的多篇文献并列引用，但不能遗漏、新增或改写任何引用编号。
6. 删除残缺的英文片段、摘要页眉和关键词串；若片段无法形成完整事实，只保留其引用并与同节已有、确有证据的综合判断合并。
7. 避免空泛的固定结尾，避免与其他章节可能重复的通用句。
8. 只模仿少样本的组织方式；不得输出其中的〈占位内容〉、〔证据A〕或任何示例事实。
9. 每个“现有研究、多项工作、普遍、共同、形成趋势”类综合判断都必须紧跟支持它的引用，不能作为无引用的过渡句。
10. 必须按“用户确认的分析重点”区分感知或识别输出、结构化编码产物、指定分析方法与下游解释；不得用相邻阶段的证据替代当前章节要求的证据角色。
11. 对“该领域快速发展”“获得持续关注”“研究热点”“呈现X格局”等宏观判断，只能引用下文“综述论文清单”中的综述/调研类论文；不得用单篇方法论文支撑领域级断言。若名单为空，把宏观断言收窄到具体技术路线或子领域层面。
12. 与研究主题或当前任务阶段只有场景邻接关系的陈述不得作为方法证据；低相关论文可以不写，不得为了凑引用数量强行拼接。
13. 禁止连续句号、句号与逗号叠加等异常标点。
14. 严格完成”章节任务”规定的段落数量和组织方式，并在证据允许时接近目标字数；若要求连续自然段，不得自行增加内部小标题。
15. **动态比较要求**：执行下文“本节比较要求”。
16. **证据强度决定语言强度**（按草稿中支持同一主张的独立 paper_id 数量）：
    - 仅 1 篇 → 只能写"有研究尝试…""一项工作提出…""X 等人报告…"
    - 2–3 篇 → 可写"部分研究采用…""若干工作探索…""已有证据显示…"
    - 4–6 篇 → 可写"多项研究…""形成了较为明确的…""在…方面取得了可验证的进展"
    - 7+ 篇且有综述支撑 → 才可写"已成为重要研究方向""该领域形成了…格局"
    违反此映射表的"趋势""已成为""共同面临""普遍认为""主流"等宏观断言将被视为无引用支撑。
17. **授权主张清单**：只能写以下清单中的主张，使用对应措辞强度，引用对应证据ID。不能创造新的趋势判断、领域空白、性能声明或方法优劣评价。过渡句和结构连接可以自由生成，但不能包含新的事实性内容。
18. **引用密度与点名引用**：单处引用建议 1~2 篇，最多不得超过 3 篇。绝对严禁在段落首尾一次性倾倒大段连排引用（如 [p1..p20]）。每次引用[paper_id]时，必须在同句中写明该论文的方法/模型名称缩写或第一作者姓；禁止"有研究提出[id]""有工作尝试[id]""另有研究指出[id]"等匿名引用句式。正确写法示例："Author 等提出的 Method 模型[paper_id]通过特定机制优化了核心任务表现"。
19. **章节文体与定位解耦**：
    - 若当前为【研究背景】：重点阐明现实应用痛点、理论与技术驱动力、数据与场景约束以及研究范式的现实需求，不罗列具体算法细节。
    - 若当前为【国内外研究现状】：直接聚焦于各方法学流派的核心网络机制、代表模型（包含创新点）、基准评测表现与技术边界；严禁在开头或子节中重复复述背景定义（如“本研究领域旨在...”）。

研究主题：{topic}
用户确认的分析重点：{research_focus or topic}
章节标题：{title}
章节任务：{purpose}
目标字数：{target_word_count or '按证据充分程度合理展开'}
必须原样保留且每个至少出现一次的引用编号：
{json.dumps(required_ids, ensure_ascii=False)}

【脱敏写作少样本——只示范修辞结构，不是事实证据】
建议修辞步骤：{json.dumps(blueprint["moves"], ensure_ascii=False)}
写法示例：{blueprint["example"]}

规定标题行：{_heading(title, heading_level)}
综述论文清单：{survey_papers_json}
本节比较要求：{comparison_instruction}
{claim_constraints}
{cross_route_requirement}

【真实证据草稿——正文事实与引用的唯一来源】
{original}
"""
