"""跨写作工具与渲染器共享的正文质量规则。"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Agent 过程语言与证据元评价正则
AGENT_PROCESS_LANGUAGE_RE = re.compile(
    r"证据边界|摘要可见范围|引用可核验性|当前可访问证据|"
    r"本节仅保留|保留章节框架供后续补充|未能完成可靠中文综合|"
    r"该路线中已获得的证据无法|检索证据不足以支撑|"
    r"摘要证据尚不足以|现有摘要证据|证据(?:也|仍)?尚?不足以|"
    r"作者局限性说明|"
    r"该字段缺失|字段缺失时|需(?:要)?结合具体应用场景",
    re.I,
)

# 编辑提示与修改说明泄漏正则
EDITORIAL_LEAKAGE_RE = re.compile(
    r"(?:〔|【|\(|（)?\s*注[：:]\s*(?:此处|请|见)|"
    r"此处需(?:要)?(?:改为|修改|补充|删除)|请改为(?:半角|全角)|"
    r"见说明|修改说明|机器检查结果|上一次输出及机器检查结果",
    re.I,
)

_EVIDENCE_META_CLAUSE_PATTERNS = [
    r"摘要证据尚不足以[^。；;\n]{0,60}[。；;\n]?",
    r"现有摘要证据[^。；;\n]{0,60}[。；;\n]?",
    r"仅凭(摘要|标题)[^。；;\n]{0,40}(不足|无法|难以)[^。；;\n]{0,40}[。；;\n]?",
    r"(?:证据|信息)(?:仍)?尚不足以[^。；;\n]{0,40}[。；;\n]?",
    r"(?:现有)?证据(?:也|仍)?尚?不足以[^。；;\n]{0,50}[。；;\n]?",
    r"(?:不足以|难以)(?:判断|确定|支撑|支持)[^。；;\n]{0,40}[。；;\n]?",
    r"(?:需|需要)(?:要)?结合具体应用场景[^。；;\n]{0,40}[。；;\n]?",
    r"该字段缺失[^。；;\n]{0,30}[。；;\n]?",
    r"字段缺失[^。；;\n]{0,30}[。；;\n]?",
]
_EVIDENCE_META_PATTERN = re.compile("|".join(_EVIDENCE_META_CLAUSE_PATTERNS))


# 引用标记整体不参与断句：DOI 形如 ``[doi:10.1142/s0219843625400080]``，
# 其中的小数点会被句末标点规则切开，后半截 "1142/s0219843625400080]。"
# 变成一个独立"句子"。同一篇论文被引两次就必然产生两个相同的残句，
# 交付物结构检查因此恒报"正文存在重复或高度相似句子"（2026-08-29 实测）。
_CITATION_SPAN_RE = re.compile(r"\[[^\]\n]*\]|〔[^〕\n]*〕|【[^】\n]*】")
_MASK_RE = re.compile("\x00(\\d+)\x00")

# 句末标点后紧跟数字时不断句：这是小数或版本号的内部点号
# （"准确率提升 1.77%"、"YOLOv8.2"），不是句子边界。中文句末标点
# （。！？）不受此限制，它们后面不会紧跟数字构成小数。
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])(?!\d)\s*|\n+")


def content_sentences(text: str) -> list[str]:
    """提取适合做语言和重复检查的正文句子，忽略 Markdown 标题。

    引用标记（``[...]``/``〔...〕``/``【...】``）在断句前整体掩蔽、断句后还原，
    因此其内部的点号、换行和标点都不会被误当作句子边界。
    """
    body_lines = [
        line.strip()
        for line in str(text or "").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    joined = "\n".join(body_lines)
    spans: list[str] = []

    def _mask(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"\x00{len(spans) - 1}\x00"

    def _restore(value: str) -> str:
        return _MASK_RE.sub(lambda item: spans[int(item.group(1))], value)

    return [
        restored
        for sentence in _SENTENCE_SPLIT_RE.split(_CITATION_SPAN_RE.sub(_mask, joined))
        if (restored := _restore(sentence or "").strip())
    ]


def detect_english_sentences(text: str) -> list[str]:
    """返回疑似完整英文句子；技术名词、模型名和缩写不会单独触发。"""
    # paper_id（尤其 CNKI 的长随机 ID）包含大量拉丁字母和连字符，若直接
    # 计词会把“84%[cnki:...]”误判为英文整句。语言检测前先移除引用标记，
    # 引用合法性由独立的 citation validator 负责。
    cleaned = re.sub(
        r"\[[^\]\n]+\]|〔[^〕\n]+〕|【[^】\n]+】",
        "",
        str(text or ""),
    )
    english: list[str] = []
    for sentence in content_sentences(cleaned):
        latin_words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", sentence)
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", sentence)
        if len(latin_words) >= 7 and len(cjk_chars) <= 3:
            english.append(sentence)
    return english


def detect_incomplete_fragments(text: str) -> list[str]:
    """检测章节末尾明显被模型截断的半句话。"""
    matches = list(re.finditer(r"^#{2,4}\s+(.+?)\s*$", str(text or ""), re.M))
    bodies: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text or "")
        bodies[match.group(1).strip()] = str(text or "")[match.end():end].strip()

    fragments: list[str] = []
    for title, body in bodies.items():
        content = re.sub(r"^>.*$", "", body, flags=re.M).strip()
        if not content:
            fragments.append(title)
            continue
        last_paragraph = next(
            (part.strip() for part in reversed(re.split(r"\n\s*\n", content)) if part.strip()),
            "",
        )
        # 允许句号、问叹号、分号或句末引用；汉字/字母/数字裸结尾通常是
        # max_tokens 截断或模型提前停止，例如“这一证据使课堂”。
        if last_paragraph and not re.search(
            r"(?:[。！？.!?；;][”’）》】）)]*|\][”’）》】）)]*)$",
            last_paragraph,
        ):
            fragments.append(f"{title}: {last_paragraph[-60:]}")
    return fragments


def strip_evidence_meta_language(review: str) -> str:
    """最多保留一次证据边界元评价，删除后续重复套话。"""
    if not review:
        return review
    matches = list(_EVIDENCE_META_PATTERN.finditer(review))
    if len(matches) <= 1:
        return review
    parts: list[str] = []
    last_end = 0
    for index, match in enumerate(matches):
        if index == 0:
            continue
        parts.append(review[last_end:match.start()])
        last_end = match.end()
    parts.append(review[last_end:])
    cleaned = "".join(parts)
    cleaned = re.sub(r"[，,]{2,}", "，", cleaned)
    cleaned = re.sub(r"[。]{2,}", "。", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    logger.info("Evidence meta-language stripped: %d clause(s)", len(matches) - 1)
    return cleaned


# 兼容历史私有别名
_AGENT_PROCESS_LANGUAGE_RE = AGENT_PROCESS_LANGUAGE_RE
_EDITORIAL_LEAKAGE_RE = EDITORIAL_LEAKAGE_RE
_english_sentences = detect_english_sentences
_incomplete_section_fragments = detect_incomplete_fragments
_content_sentences = content_sentences

__all__ = (
    "AGENT_PROCESS_LANGUAGE_RE",
    "EDITORIAL_LEAKAGE_RE",
    "content_sentences",
    "detect_english_sentences",
    "detect_incomplete_fragments",
    "strip_evidence_meta_language",
    "_AGENT_PROCESS_LANGUAGE_RE",
    "_EDITORIAL_LEAKAGE_RE",
    "_english_sentences",
    "_incomplete_section_fragments",
    "_content_sentences",
)
