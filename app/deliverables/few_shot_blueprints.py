"""四类交付物的脱敏写作少样本。

示例只表达修辞结构，不包含真实论文事实、作者、数据集、数值或引用。
真实证据与写作示例必须在提示词中保持物理分区。
"""

from __future__ import annotations

import re
from typing import Any

from app.schemas.deliverable_schema import CoreDeliverableType, WritingPlan


_BLUEPRINTS: dict[CoreDeliverableType, dict[str, dict[str, Any]]] = {
    CoreDeliverableType.RESEARCH_BACKGROUND: {
        "problem_context": {
            "moves": ["从应用场景收敛到可研究问题", "用多篇证据界定问题边界"],
            "example": (
                "围绕〈示例主题〉，现有研究共同关注〈核心现象〉，但不同场景对"
                "研究对象和分析粒度的界定并不一致〔证据A〕〔证据B〕。因此，本节"
                "先说明问题发生的条件，再界定本文所讨论的具体范围。"
            ),
        },
        "importance": {
            "moves": ["先说明证据支持的影响", "再分别落到实践价值与理论价值"],
            "example": (
                "已有证据表明，〈核心现象〉与〈理论或实践结果〉存在稳定关联"
                "〔证据A〕〔证据B〕；另一组研究则从〈理论机制〉解释其研究价值"
                "〔证据C〕。这些发现共同说明该问题值得持续研究，但不能据此扩大"
                "为未经验证的因果结论。"
            ),
        },
        "existing_approaches": {
            "moves": ["归纳共同路线", "比较适用条件", "指出证据明确报告的困难"],
            "example": (
                "现有工作主要形成两类处理思路：一类侧重〈路线一〉，适用于"
                "〈条件一〉〔证据A〕〔证据B〕；另一类采用〈路线二〉，更关注"
                "〈条件二〉〔证据C〕。两类方法的差异主要体现在数据、分析粒度"
                "与评价目标，而不是简单的性能高低。"
            ),
        },
        "research_need": {
            "moves": ["从已证实困难推导研究需要", "控制推断强度"],
            "example": (
                "综合已有证据，后续研究需要在〈已报告困难〉与〈目标需求〉之间"
                "建立更清晰的联系〔证据A〕〔证据B〕。这一判断来自文献共同报告的问题，"
                "不意味着现有路线已经失效，也不预设某一种方法必然更优。"
            ),
        },
    },
    CoreDeliverableType.RESEARCH_STATUS: {
        "scope_definition": {
            "moves": ["说明主题边界", "交代相邻概念的纳入条件"],
            "example": (
                "本文将〈示例主题〉限定为对〈研究对象〉的描述、识别或解释；"
                "相邻概念只有在直接服务于该对象时才纳入讨论。"
            ),
        },
        "theme_*": {
            "moves": ["先给路线综合判断", "再写共同方法与差异", "最后归纳适用条件"],
            "example": (
                "在〈示例路线〉中，多项研究围绕〈共同问题〉形成了相近的问题设定"
                "〔证据A〕〔证据B〕。方法上，部分工作强调〈方法侧重一〉，另一些"
                "工作关注〈方法侧重二〉〔证据C〕〔证据D〕；这种差异反映的是"
                "研究目标与数据条件不同，不能仅凭单项指标作统一排序。"
            ),
        },
        "cross_route_comparison": {
            "moves": ["使用一致维度横向比较", "避免路线优劣榜"],
            "example": (
                "横向比较可见，〈路线一〉与〈路线二〉在研究目标、数据来源和评价"
                "单位上各有侧重〔证据A〕〔证据B〕；〈路线三〉则补充了前两者较少"
                "涉及的解释维度〔证据C〕。因此，各路线更适合被视为互补证据。"
            ),
        },
        "research_gaps": {
            "moves": ["区分作者报告局限与跨论文推断", "给出支持来源"],
            "example": (
                "作者直接报告的局限主要集中在〈局限一〉〔证据A〕；跨论文比较还"
                "显示，〈局限二〉在多条路线中反复出现〔证据B〕〔证据C〕。后者属于"
                "综合推断，应使用审慎措辞，而不能改写成单篇论文的明确结论。"
            ),
        },
    },
    CoreDeliverableType.RELATED_WORK: {
        "theme_*": {
            "moves": ["只选择与用户问题直接相关的路线", "围绕同一维度比较已有工作"],
            "example": (
                "与〈用户研究问题〉直接相关的工作主要沿〈示例路线〉展开。已有研究"
                "分别从〈比较维度一〉和〈比较维度二〉处理该问题〔证据A〕〔证据B〕，"
                "差异在于问题假设和适用条件，而非笼统的先进或落后。"
            ),
        },
        "gap_and_positioning": {
            "moves": ["先归纳有证据的不足", "再依据用户资料定位", "不虚构贡献"],
            "example": (
                "现有工作在〈已有能力〉方面提供了基础，但对〈已报告不足〉的处理"
                "仍有限〔证据A〕〔证据B〕。依据用户提供的研究资料，拟开展的工作"
                "关注〈用户方法或方向〉；这里仅说明研究位置，不预先声称性能优势。"
            ),
        },
    },
    CoreDeliverableType.NARRATIVE_REVIEW: {
        "abstract": {
            "moves": ["交代范围与证据基础", "概括主要认识", "披露证据限制"],
            "example": (
                "本文围绕〈示例主题〉梳理〈时间范围〉内的研究，按〈组织原则〉综合"
                "主要路线。现有证据呈现〈总体认识〉，但结论受全文可访问性和研究"
                "异质性限制。"
            ),
        },
        "introduction": {
            "moves": ["说明问题价值", "明确叙述性综述目标与边界"],
            "example": (
                "〈示例主题〉连接了〈问题场景〉与〈研究目标〉。本文旨在综合已有"
                "证据的主要路线与争议，不声称执行了系统综述或偏倚风险评价。"
            ),
        },
        "search_scope": {
            "moves": ["只报告真实执行步骤", "说明未执行事项"],
            "example": (
                "检索覆盖〈实际年份〉和〈实际来源〉，经过去重与相关性筛选形成写作"
                "证据池。该过程未包含〈未执行流程〉，因此结果属于叙述性综合。"
            ),
        },
        "scope_definition": {
            "moves": ["界定概念", "说明纳入和排除边界"],
            "example": (
                "本文把〈示例主题〉限定为〈核心对象〉，并仅在相邻研究直接解释该"
                "对象时予以纳入。"
            ),
        },
        "theme_*": {
            "moves": ["综合路线", "比较问题、方法与发现", "保留证据等级边界"],
            "example": (
                "〈示例路线〉主要处理〈共同问题〉〔证据A〕〔证据B〕。研究在"
                "〈方法差异〉上形成不同选择〔证据C〕，这些工作共同推进了"
                "〈已有认识〉，但摘要证据不足以支持更细的实验比较。"
            ),
        },
        "cross_route_comparison": {
            "moves": ["统一比较维度", "归纳互补关系与共性问题"],
            "example": (
                "不同路线在目标、数据和评价单位上并不等价〔证据A〕〔证据B〕；"
                "统一比较后可见，它们在〈共性问题〉上相互补充，而非形成单一排名。"
            ),
        },
        "future_directions": {
            "moves": ["从已识别挑战推导方向", "避免凭空提出趋势"],
            "example": (
                "针对前述〈证据支持的挑战〉，后续工作可优先检验〈可验证方向〉"
                "〔证据A〕〔证据B〕。这一方向是对已有不足的谨慎推导，而非既成结论。"
            ),
        },
        "conclusion": {
            "moves": ["总结主要认识", "不新增事实或引用"],
            "example": (
                "总体而言，现有研究形成了若干互补路线，其差异来自问题目标、数据"
                "条件和解释层级。结论应结合证据可访问性理解。"
            ),
        },
        "evidence_statement": {
            "moves": ["披露全文、摘要和元数据分布", "明确不可支持的判断"],
            "example": (
                "本次综合依据实际获得的全文和摘要；仅有摘要的论文不用于支持详细"
                "结构、消融实验或作者未报告的局限判断。"
            ),
        },
    },
}


_PLACEHOLDER_RE = re.compile(
    r"〈[^〉]+〉|〔证据[A-Z0-9]+〕|示例主题|示例路线|示例文献|"
    r"SAMPLE_(?:TOPIC|CITATION|FACT)",
    re.I,
)


def get_section_blueprint(
    deliverable_type: CoreDeliverableType | str,
    section_id: str,
) -> dict[str, Any]:
    """返回章节对应的脱敏蓝图；动态主题统一匹配 ``theme_*``。"""
    dtype = CoreDeliverableType(deliverable_type)
    values = _BLUEPRINTS.get(dtype, {})
    key = "theme_*" if str(section_id).startswith("theme_") else str(section_id)
    blueprint = values.get(key) or {
        "moves": ["先提出综合判断", "再用真实证据支持", "最后归纳适用条件"],
        "example": (
            "围绕〈示例主题〉，现有证据从不同角度支持〈综合判断〉"
            "〔证据A〕〔证据B〕；结论强度应与可访问证据保持一致。"
        ),
    }
    return {
        "section_id": str(section_id),
        "moves": list(blueprint["moves"]),
        "example": str(blueprint["example"]),
        "evidence_role": "style_only_not_evidence",
    }


def get_plan_blueprints(plan: WritingPlan) -> list[dict[str, Any]]:
    return [
        get_section_blueprint(plan.deliverable_type, section.id)
        for section in plan.sections
    ]


def detect_blueprint_leakage(text: str) -> list[str]:
    """检测模型是否把蓝图占位符当成正文事实或引用输出。"""
    return list(dict.fromkeys(
        match.group(0) for match in _PLACEHOLDER_RE.finditer(str(text or ""))
    ))
