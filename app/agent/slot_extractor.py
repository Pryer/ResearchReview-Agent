"""槽位抽取模块。

从用户请求中提取结构化参数（主题、年份范围、论文数量等）。
策略与意图识别类似：规则优先，LLM 兜底。
"""

from __future__ import annotations

import math
import re
from typing import Optional

from app.core.json_utils import parse_json_object as _parse_json_object_robust

from app.core.exceptions import SlotExtractionError
from app.core.logger import get_logger
from app.schemas.agent_schema import IntentType, SlotResult
from app.utils.date_utils import current_year as get_current_year

logger = get_logger(__name__)


def extract_slots(
    user_query: str,
    intent: IntentType | str,
    llm=None,
    current_year: int | None = None,
) -> SlotResult:
    """统一抽取槽位。

    Args:
        user_query: 用户请求。
        intent: 识别到的意图。
        llm: 可选的 LLM 客户端。
        current_year: 当前年份，用于解析 "近五年" 等相对时间。

    Returns:
        抽取的槽位结果。
    """
    intent_str = intent.value if isinstance(intent, IntentType) else intent
    current_year = current_year or get_current_year()

    year_range = extract_year_range(user_query, current_year)
    explicit_max_papers = _extract_explicit_max_papers(user_query)
    required_reference_count = explicit_max_papers or 30
    retrieval_target, generation_limit = _derive_count_targets(
        required_reference_count,
        explicit=explicit_max_papers is not None,
    )

    # 规则抽取
    slot = SlotResult(
        topic=extract_topic(user_query),
        start_year=year_range[0] if year_range else None,
        end_year=year_range[1] if year_range else None,
        max_papers=required_reference_count,
        required_reference_count=required_reference_count,
        retrieval_target=retrieval_target,
        generation_limit=generation_limit,
        year_range_explicit=year_range is not None,
        strict_year_range=bool(
            year_range
            and re.search(r"(?:仅限|只限|严格限定|不要超出|不得早于)", user_query)
        ),
        max_papers_explicit=explicit_max_papers is not None,
        requested_sections=extract_requested_sections(user_query, intent_str),
        language=extract_language(user_query),
        citation_style=extract_citation_style(user_query),
    )

    # LLM 只补充规则无法稳定提取的主题。年份和篇数必须来自用户原文，
    # 未明确提供时统一采用系统默认值，避免模型自行猜测参数。
    if llm and not slot.topic:
        try:
            llm_slots = llm_extract_slots(user_query, intent_str, llm, current_year)
            if not slot.topic:
                llm_topic = str(llm_slots.get("topic") or "").strip()
                # LLM 偶尔把"近三年综述"这类纯时间+交付物组合当主题；
                # 无领域内容的主题拒收，让下游回退到 user_query。
                if llm_topic and has_topic_content(llm_topic):
                    slot.topic = llm_topic
        except Exception as e:
            logger.warning("LLM slot extraction failed: %s", e)

    return slot


def extract_requested_sections(user_query: str, intent: str = "generate_review") -> list[str]:
    """从请求中提取用户真正需要的论文正文部分，并保持原文顺序。"""
    patterns = {
        "background": r"研究背景|背景与意义|背景和意义|研究意义",
        "research_status": r"研究现状|相关研究现状|国内外研究现状|国内外现状",
        "related_work": r"相关工作|related\s+work",
        "narrative_review": r"叙述性综述|全篇综述|完整综述|narrative\s+review|统一综述|综合综述",
    }
    matches: list[tuple[int, str]] = []
    for section, pattern in patterns.items():
        match = re.search(pattern, user_query, re.IGNORECASE)
        if match:
            matches.append((match.start(), section))

    if matches:
        sections = [section for _, section in sorted(matches)]
        # narrative_review 自包含全篇结构，显式命中时优先于分节请求
        if "narrative_review" in sections:
            return ["narrative_review"]
        return sections
    if intent == "generate_related_work":
        return ["related_work"]
    if intent == "generate_review":
        # 用户说“生成综述”但未明确指定分节方式时，默认生成
        # 研究背景 + 研究现状两段，而非 taxonomy 维度分节的
        # narrative_review。后者仍需用户显式要求才能触发。
        return ["background", "research_status"]
    return []


def extract_topic(user_query: str) -> Optional[str]:
    """抽取研究主题。

    策略：找到「关于/针对/调研/检索 ... 相关」中的主题部分。
    未匹配时返回 None 让 LLM 补充。
    """
    patterns = [
        r"(?:关于|针对)([^，。,.\n]{2,40}?)(?:的)?(?:文献综述|综述|survey|review)",
        r"(?:找几篇|找|推荐一些|推荐)(?:关于|有关)?([^，。,.\n]{2,40}?)(?:相关)?(?:的)?(?:论文|paper|papers)",
        # “生成X研究现状/研究背景/相关工作”句式：交付物名词之前的内容才是主题
        r"^(?:请)?(?:帮我|帮忙)?(?:生成|写|撰写|输出)(?:一篇|一个)?(?:关于|针对)?"
        r"([^，。,.\n]{2,40}?)(?:的)?"
        r"(?:研究背景|背景与意义|研究现状|相关工作|文献综述|综述|survey|review|背景|现状)",
        # 动词引导句式必须锚定句首：否则“识别【研究】现状”中间的“研究”
        # 会被当作引导动词，把主题截成“现状”这类残片。
        r"^(?:请)?(?:帮我|帮忙)?(?:关于|针对|调研|研究|检索|搜索|推荐)"
        r"([^，。,.\n]{2,30})(?:相关|方面|方向|论文)?",
        r"近[一二三四五六七八九十\d]+年(.{2,30})(?:相关|方面|方向)?",
        r"(?:请)?(?:帮我)?(?:生成|写|撰写|输出)(?:一篇|一个)?(?:关于|针对)?([^，。,.\n]{2,40}?)(?:的)?(?:文献综述|综述|survey|review)",
    ]
    for pat in patterns:
        m = re.search(pat, user_query, re.IGNORECASE)
        if m:
            topic = _clean_topic(m.group(1))
            # 去除尾部冗余词（长交付物名词在前，避免被“研究”抢先截断）
            topic = re.sub(
                r"(的)?(研究背景|背景与意义|研究现状|相关工作|文献综述|综述|"
                r"相关|方面|方向|领域|论文|研究)+$",
                "",
                topic,
            )
            # "近三年综述"式捕获没有任何领域内容：放弃该模式，继续尝试
            # 后续模式与兜底路径，而不是把无内容残片当主题。
            if topic and has_topic_content(topic):
                return topic

    # 裸主题兜底，例如用户只输入“时序动作定位”。
    candidate = _clean_topic(user_query)
    if (
        candidate
        and len(candidate) <= 40
        and not re.search(r"[，。,.;；!?！？\n]", candidate)
        and has_topic_content(candidate)
    ):
        return candidate
    return None


def _clean_topic(topic: str) -> str:
    """清理主题中的时间、数量和动作词噪声。"""
    topic = topic.strip()
    topic = re.sub(r"^(?:请)?(?:帮我|帮忙)", "", topic)
    topic = re.sub(r"^(?:最近?|近)[一二三四五六七八九十\d]+年", "", topic)
    topic = re.sub(
        r"^(?:19|20)\d{2}\s*[-~—到至]\s*(?:19|20)\d{2}年?",
        "",
        topic,
    )
    topic = re.sub(r"^(?:19|20)\d{2}年?(?:以后|起|以来|之后)", "", topic)
    topic = re.sub(r"^(?:本年|今年|近年|近年来)", "", topic)
    topic = re.sub(r"^(?:的|关于|针对|有关|围绕)", "", topic)
    # 时间表达也可能出现在主题中段/尾部（"X近三年"），全文剥离
    topic = re.sub(
        r"(?:最近?|近)[一二三四五六七八九十\d]+\s*年(?:的)?",
        "",
        topic,
    )
    topic = re.sub(
        r"(?:的)?(?:研究背景|背景与意义|研究现状|相关工作|文献综述|综述|survey|review)$",
        "",
        topic,
        flags=re.IGNORECASE,
    )
    topic = re.sub(r"(?:并)?(?:生成|写|输出).*$", "", topic)
    # 主题不含并列子句：“综述并引用30篇”类捕获的“并…”部分是请求参数
    topic = re.sub(r"并.*$", "", topic)
    topic = re.sub(
        r"(?:论文|文献)?\s*\d+\s*(?:[-~—到至]\s*\d+)?\s*篇.*$",
        "",
        topic,
    )
    topic = re.sub(r"(?:，|,|。|；|;).*$", "", topic)
    return topic.strip()


# 组成"主题"时没有任何领域信息量的词：时间、交付物、通用学术词、动词。
# 全部剥除后为空的主题（如"近三年综述"）不是研究主题。
_CONTENTLESS_TOPIC_RE = re.compile(
    r"(?:最近?|近)[一二三四五六七八九十\d]+\s*年"
    r"|最新|近年来|近年|今年"
    r"|研究背景|背景与意义|研究现状|相关工作|文献综述|综述|背景|现状"
    r"|论文|文献|研究|调研|检索|搜索|生成|撰写|输出|查找|推荐|分析|总结"
    r"|引用|参考文献"
    r"|survey|review|research|paper|papers"
    r"|[\s，。、,.;；:：的的了呢吧与和及对关于按把并]",
    re.IGNORECASE,
)


def has_topic_content(topic: str) -> bool:
    """判定主题是否含领域内容：剥除时间/交付物/通用词后仍有实质内容。"""
    text = str(topic or "").strip()
    if not text:
        return False
    residual = _CONTENTLESS_TOPIC_RE.sub("", text)
    return len(re.sub(r"[\s0-9]+", "", residual)) >= 2


def extract_year_range(
    user_query: str,
    current_year: int,
) -> Optional[tuple[int, int]]:
    """抽取年份范围。

    支持格式：
    - "近五年" / "近3年" / "最近十年"
    - "2021-2025" / "2021~2025" / "2021年到2025年"
    - "2023年起" / "2020年以后"
    """
    # "近N年"
    m = re.search(r"近[最]?([一二三四五六七八九十\d]+)年", user_query)
    if m:
        num_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                   "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
                   "十一": 11, "十二": 12, "二十": 20}
        num_str = m.group(1)
        if num_str in num_map:
            n = num_map[num_str]
        else:
            try:
                n = int(num_str)
            except ValueError:
                n = 5
        # 论文数据库通常只能稳定按发表年份过滤，无法精确执行滚动 N×12 个月。
        # 因此默认采用“含当前年在内的 N 个自然年度”，例如 2026 年的
        # “近三年”明确编译为 2024—2026，而不是含四个年份的 2023—2026。
        return (current_year - n + 1), current_year

    # "YYYY-YYYY"
    m = re.search(r"(20\d{2})\s*[-~—到]\s*(20\d{2})", user_query)
    if m:
        return int(m.group(1)), int(m.group(2))

    # "YYYY年以后/起"
    m = re.search(r"(20\d{2})年?(以后|起|以来|之后)", user_query)
    if m:
        return int(m.group(1)), current_year

    return None


def _extract_explicit_max_papers(user_query: str) -> Optional[int]:
    """提取用户明确给出的论文数量；数量范围取上限。"""
    range_match = re.search(
        r"(?:引用|不少于|至少|需要|检索|找|推荐)?\s*"
        r"(\d+)\s*(?:[-~—到至]|篇\s*(?:到|至))\s*(\d+)\s*篇",
        user_query,
    )
    if range_match:
        return max(int(range_match.group(1)), int(range_match.group(2)))

    patterns = (
        r"(?:引用|不少于|至少|需要|检索|找|推荐)\s*(\d+)\s*篇",
        r"(\d+)\s*篇(?:论文|文献)?(?:左右|以上|以内|即可)?",
        r"(?:论文|文献)\s*(\d+)\s*篇",
    )
    for pattern in patterns:
        match = re.search(pattern, user_query)
        if match:
            return int(match.group(1))
    return None


def extract_max_papers(user_query: str) -> int:
    """抽取论文数量要求；用户未指定时默认 30 篇。"""
    return _extract_explicit_max_papers(user_query) or 30


def _derive_count_targets(required_reference_count: int, explicit: bool) -> tuple[int, int]:
    """按引用要求加安全余量派生检索目标和生成输入上限。

    检索循环是否停止还要同时满足证据覆盖；这里的目标仅提供去重、筛选和
    重排损耗所需的候选余量，不能把显式引用要求机械放大为三倍。

    余量比例由实测损耗定标：文献形态复核、元数据核验、证据卡片降级与引用
    分配逐级筛除。2026-09-01 实测 45 篇证据卡片 → 32 篇可用 → 25 篇被引用，
    端到端成品率约 0.556，即 40 篇引用要求需要约 72 篇候选池。0.60 的余量
    只到 64，恰好在天花板处夹住池目标，显式引用要求系统性不达标。
    ``min(40, ...)`` 上限保证大额请求不被无节制放大（200 篇仍为 240）。
    """
    required_reference_count = max(1, int(required_reference_count or 30))
    margin_ratio = 0.80 if explicit else 0.40
    safety_margin = max(5, min(40, math.ceil(required_reference_count * margin_ratio)))
    retrieval_target = required_reference_count + safety_margin
    generation_limit = retrieval_target
    return retrieval_target, generation_limit


def extract_language(user_query: str) -> str:
    """抽取综述语言。"""
    # ``en``/``zh`` 只能作为独立语言代码匹配，不能命中 engagement、student 等检索词。
    if re.search(
        r"英文综述|英文|(?<![a-z])english(?![a-z])|(?<![a-z])en(?![a-z])",
        user_query,
        re.IGNORECASE,
    ):
        return "en"
    if re.search(
        r"中文综述|中文|(?<![a-z])chinese(?![a-z])|(?<![a-z])zh(?![a-z])",
        user_query,
        re.IGNORECASE,
    ):
        return "zh"
    return "zh"  # 默认中文


def extract_citation_style(user_query: str) -> str:
    """抽取引用格式。"""
    style_map = {
        "gbt7714": ["gbt7714", "gb/t", "国标"],
        "apa": ["apa"],
        "ieee": ["ieee"],
        "bibtex": ["bibtex", "bib"],
    }
    query_lower = user_query.lower()
    for style, keywords in style_map.items():
        if any(kw in query_lower for kw in keywords):
            return style
    return "gbt7714"


# 同行评审/期刊来源要求关键词。单独出现"期刊"也视为要求（如"尽量都是期刊论文"）。
_PEER_REVIEW_PATTERNS = (
    r"同行评审|同行审议|同行评议|同行审阅",
    r"peer[- ]?reviewed|peer[- ]?review\b",
    r"期刊论文|学术期刊|核心期刊|南大核心|北大核心",
    r"期刊",
    r"\bSCI\b|\bSCIE\b|\bSSCI\b|\bCSSCI\b|\bEI\b",
    r"收录于(?:SCI|EI|核心)",
)


def extract_peer_review_requirement(user_query: str) -> bool:
    """判断用户是否显式要求同行评审/期刊/SCI/EI 来源的论文。

    纯规则判定（与 extract_citation_style 同风格）；返回 False 表示未显式
    要求，全局证据门只测量不阻断。不会调用 LLM。
    """
    query = str(user_query or "")
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in _PEER_REVIEW_PATTERNS)


def llm_extract_slots(
    user_query: str,
    intent: str,
    llm,
    current_year: int = 2025,
) -> dict:
    """规则无法稳定提取时调用 LLM。

    Returns:
        包含 topic / start_year / end_year / max_papers 的字典。
    """
    from app.prompt.slots import SLOT_EXTRACTION_PROMPT

    prompt = SLOT_EXTRACTION_PROMPT.format(
        user_query=user_query,
        current_year=current_year,
    )
    try:
        response = llm.complete(prompt, response_format="json")
        return _safe_parse_json(response)
    except Exception as e:
        raise SlotExtractionError(f"LLM 槽位抽取失败: {e}")


from app.core.json_utils import parse_json_object as _safe_parse_json  # noqa: E402
