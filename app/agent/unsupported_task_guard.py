"""检索前能力门禁：拒绝四类交付物之外的生成任务。"""

from __future__ import annotations

import re
from typing import Iterable

from app.agent.deliverable_router import resolve_core_deliverables
from app.agent.intent import recognize_intent
from app.agent.slot_extractor import extract_requested_sections
from app.schemas.agent_schema import IntentType
from app.schemas.deliverable_schema import UnsupportedTaskGuardResult


_UNSUPPORTED_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"系统综述|systematic\s+review|PRISMA", "系统综述或PRISMA流程"),
    (r"元分析|meta[-\s]?analysis", "元分析"),
    (r"(?:写|生成|撰写|输出|单独写|只写).{0,8}(?:论文)?(?:引言|introduction)|(?:引言|introduction).{0,8}(?:章节|部分)", "论文引言"),
    (r"方法章节|研究方法部分|方法设计建议|设计研究方案|帮我设计.{0,12}方法|methodology\s+section", "研究方法或方案设计"),
    (r"实验章节|实验设计|消融实验设计|帮我设计.{0,12}实验|experiment\s+section", "实验章节或实验设计"),
    (r"(?:生成|撰写|写).{0,8}(?:论文)?摘要|abstract\s+section", "独立论文摘要"),
    (r"(?:生成|撰写|写).{0,8}(?:论文)?结论|conclusion\s+section", "独立论文结论"),
    (r"开题报告|研究计划书|项目申请书|基金申请|proposal", "开题报告或研究计划书"),
    (r"完整论文|论文全文|毕业论文|学位论文|可直接投稿", "完整论文或可投稿稿件"),
    (r"幻灯片|演示文稿|PPT|答辩稿|演讲稿", "演示文稿或答辩材料"),
    (r"代码实现|编写代码|程序实现|code\s+implementation", "代码或程序实现"),
)

_INTENT_LABELS = {
    IntentType.READ_PAPER.value: "单篇论文解析",
    IntentType.COMPARE_PAPERS.value: "论文对比",
    IntentType.GENERATE_REFERENCES.value: "独立参考文献格式生成",
    IntentType.EXTRACT_PAPER_CARD.value: "独立论文卡片抽取",
    IntentType.FIND_DATASETS.value: "数据集查找",
    IntentType.FIND_TRENDS.value: "独立趋势分析",
    IntentType.GENERAL_QA.value: "其他学术问答或生成任务",
}

_SUPPORTED_NAMES = "研究背景、研究现状、论文相关工作、叙述性综述初稿"
_EXPLICIT_CORE_TASK_PATTERN = re.compile(
    r"研究背景|背景与意义|研究现状|相关工作|related\s+work|"
    r"叙述性综述|文献综述|综述初稿|narrative\s+review",
    re.IGNORECASE,
)
_STANDALONE_SEARCH_PATTERN = re.compile(
    r"(?:找|搜索|检索|推荐|查找).{0,40}(?:论文|文献|papers?)|"
    r"(?:论文|文献).{0,20}(?:搜索|检索|推荐)",
    re.IGNORECASE,
)


def check_unsupported_task(
    user_query: str,
    intent: str | None = None,
    requested_sections: Iterable[str] | None = None,
) -> UnsupportedTaskGuardResult:
    """返回门禁结果；检索仅允许作为四类写作任务的内部步骤。"""
    query = str(user_query or "").strip()
    resolved_intent = intent or recognize_intent(query, llm=None).intent
    sections = list(requested_sections) if requested_sections is not None else extract_requested_sections(
        query, resolved_intent
    )
    deliverables = resolve_core_deliverables(resolved_intent, sections)

    unsupported: list[str] = []
    for pattern, label in _UNSUPPORTED_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            unsupported.append(label)

    # 独立论文检索是受支持的任务：graph 的 search_papers 意图会在检索
    # 排序后提前返回论文列表。“找某领域论文”的自然表达按检索意图放行，
    # 仅当请求还叠加了其他不支持任务时才被上面的模式拦截。
    search_only = (
        resolved_intent == IntentType.SEARCH_PAPERS.value
        or (
            _STANDALONE_SEARCH_PATTERN.search(query)
            and not _EXPLICIT_CORE_TASK_PATTERN.search(query)
        )
    )

    if resolved_intent in _INTENT_LABELS:
        unsupported.append(_INTENT_LABELS[resolved_intent])

    # generate_introduction 的旧意图仅在用户明确说“研究背景”时兼容；真正的引言已由上面拒绝。
    if resolved_intent == IntentType.GENERATE_INTRODUCTION.value and not re.search(
        r"研究背景|背景与意义|背景和意义|研究意义", query
    ):
        unsupported.append("论文引言")

    unsupported = list(dict.fromkeys(unsupported))
    if unsupported:
        requested = "、".join(unsupported)
        return UnsupportedTaskGuardResult(
            allowed=False,
            supported_deliverables=deliverables,
            unsupported_requests=unsupported,
            message=(
                f"目前我没有生成“{requested}”的能力。"
                f"当前支持四类学术写作任务（{_SUPPORTED_NAMES}）与独立论文检索。"
                "请把需求改写为其中一种后再提交；本次没有执行论文检索。"
            ),
        )

    if search_only:
        # 检索意图经 graph 的 search_papers 分支返回论文列表
        return UnsupportedTaskGuardResult(
            allowed=True,
            supported_deliverables=deliverables,
        )

    if deliverables:
        return UnsupportedTaskGuardResult(
            allowed=True,
            supported_deliverables=deliverables,
        )

    return UnsupportedTaskGuardResult(
        allowed=False,
        unsupported_requests=["无法映射到当前四类交付物的任务"],
        message=(
            f"目前我没有处理该任务的能力。当前支持四类学术写作任务（{_SUPPORTED_NAMES}"
            "）与独立论文检索。请重新说明希望生成的交付物；本次没有执行论文检索。"
        ),
    )
