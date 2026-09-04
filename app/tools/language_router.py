"""论文语言检测与中英文分支拆分。

提供 ``detect_paper_language`` 判断单篇论文语言，
以及 ``split_papers_by_language`` 将候选池拆分为中/英文两个分支。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.core.logger import get_logger

logger = get_logger(__name__)

# CJK 统一码区间（基本汉字 + 扩展 A）
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")

# 日文假名（平假名 U+3040-309F + 片假名 U+30A0-30FF）：出现即为强非中文信号
_KANA_RE = re.compile(r"[ぁ-ヿ]")


@dataclass(frozen=True)
class PaperLanguageDetection:
    language: str
    confidence: float
    reason: str


def calculate_cjk_ratio(text: str) -> float:
    """计算文本中 CJK 字符的占比（0.0 ~ 1.0）。"""
    if not text:
        return 0.0
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    cjk_count = sum(1 for ch in chars if _CJK_RE.match(ch))
    return cjk_count / len(chars)


def detect_term_language(term: str) -> str:
    """判断单个术语的语言，返回 ``"zh"`` | ``"en"`` | ``"mixed"``。

    用于 ``terms`` fallback 时按语言过滤。
    """
    if not term:
        return "en"
    ratio = calculate_cjk_ratio(term)
    if ratio >= 0.5:
        return "zh"
    if ratio > 0.0:
        return "mixed"
    return "en"


def detect_paper_language(
    paper: Dict[str, Any],
    *,
    title_weight: float = 0.6,
    abstract_weight: float = 0.3,
    metadata_weight: float = 0.1,
) -> str:
    """纯函数式判断单篇论文语言，返回 ``"zh"`` 或 ``"en"``。

    判断优先级（加权多信号）：
    1. 显式 ``_language_branch_override``
    2. metadata ``language`` 字段
    3. 标题 + 摘要的 CJK 字符加权占比 ≥ 阈值
    4. ``source == "cnki"`` 兜底
    5. 默认 ``"en"``

    不修改输入。需要置信度和诊断原因时使用
    :func:`detect_paper_language_result`。
    """
    return detect_paper_language_result(
        paper,
        title_weight=title_weight,
        abstract_weight=abstract_weight,
        metadata_weight=metadata_weight,
    ).language


def detect_paper_language_result(
    paper: Dict[str, Any],
    *,
    title_weight: float = 0.6,
    abstract_weight: float = 0.3,
    metadata_weight: float = 0.1,
) -> PaperLanguageDetection:
    """返回语言、置信度和判定原因，不修改输入论文。"""
    # 0. 显式 override
    override = paper.get("_language_branch_override")
    if override in ("zh", "en"):
        return PaperLanguageDetection(override, 1.0, "explicit_override")

    # 1. metadata language 字段（高置信）
    metadata_lang = str(paper.get("language", "")).strip().lower()
    if metadata_lang in {"zh", "zh-cn", "zh-tw", "chinese", "chi"}:
        return PaperLanguageDetection("zh", 0.95, f"metadata:{metadata_lang}")
    if metadata_lang in {"en", "eng", "english"}:
        return PaperLanguageDetection("en", 0.95, f"metadata:{metadata_lang}")

    # 2. 标题 + 摘要加权 CJK 占比
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")

    # 日文文献标题常以汉字为主，仅按 CJK 占比会把日文论文误送进 CNKI
    # 中文支线；假名是强非中文信号，本项目只有中/英两条检索分支，
    # 日文归入英文支线。
    if _KANA_RE.search(title) or _KANA_RE.search(abstract):
        return PaperLanguageDetection("en", 0.85, "kana_detected_japanese")

    title_ratio = calculate_cjk_ratio(title)
    abstract_ratio = calculate_cjk_ratio(abstract)

    # 标题权重更高：中文标题 + 英文摘要 → 中文分支
    weighted_ratio = (
        title_ratio * title_weight
        + abstract_ratio * abstract_weight
    )
    # 未使用的权重分配给先验（0.5 = 不确定）
    prior = 0.5
    used_weight = title_weight + abstract_weight
    if used_weight < 1.0:
        prior_weight = 1.0 - used_weight
    else:
        prior_weight = 0.0
    adjusted_ratio = weighted_ratio + prior * prior_weight

    # 置信度：标题和摘要的 CJK 信号一致时更高
    if title and abstract:
        if title_ratio >= 0.5 and abstract_ratio >= 0.3:
            confidence = 0.90
            reason = "title_abstract_both_cjk"
        elif title_ratio >= 0.5:
            confidence = 0.75
            reason = "title_cjk_abstract_en"
        elif abstract_ratio >= 0.5:
            confidence = 0.60
            reason = "abstract_cjk_title_en"
        elif title_ratio == 0.0 and abstract_ratio == 0.0:
            confidence = 0.85
            reason = "no_cjk_detected"
        else:
            confidence = 0.55
            reason = f"mixed_cjk_title={title_ratio:.2f}_abs={abstract_ratio:.2f}"
    elif title:
        confidence = 0.80 if title_ratio >= 0.5 else 0.70
        reason = f"title_only_cjk={title_ratio:.2f}"
    else:
        confidence = 0.50
        reason = "no_text_available"

    # 3. CNKI 来源兜底
    source = str(paper.get("source") or "").strip().lower()
    if source == "cnki" and adjusted_ratio < 0.50:
        adjusted_ratio = 0.5  # 提升到中文阈值
        reason = f"cnki_source_override({reason})"

    # 中文判定：标题或摘要以 CJK 为主（≥0.5），或加权占比+先验达到 0.5。
    # 旧阈值 0.15 过低：英文论文先验即有 0.05，标题混入少量汉字
    # （如中文术语引用、日文汉字）就会被误判成中文、误入 CNKI 支线。
    # 混排中文标题（如“基于BERT的文本分类”，CJK 占比 ~0.6）由
    # title_ratio ≥ 0.5 单独覆盖。
    if title_ratio >= 0.5 or abstract_ratio >= 0.5 or adjusted_ratio >= 0.5:
        return PaperLanguageDetection("zh", round(confidence, 2), reason)

    # 4. 默认英文
    return PaperLanguageDetection("en", round(confidence, 2), reason)


def split_papers_by_language(
    papers: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """将论文列表按语言拆分为中文和英文两个分支。

    返回输入论文的浅拷贝，并在拷贝上标记 ``_language_branch`` 字段，
    以及 ``_detected_language``、``_language_confidence``、
    ``_language_detection_reason`` 诊断字段。

    Returns:
        ``(zh_papers, en_papers)`` 元组。
    """
    zh_papers: List[Dict[str, Any]] = []
    en_papers: List[Dict[str, Any]] = []

    for source_paper in papers:
        paper = dict(source_paper)
        detection = detect_paper_language_result(paper)
        paper["_detected_language"] = detection.language
        paper["_language_confidence"] = detection.confidence
        paper["_language_detection_reason"] = detection.reason
        paper["_language_branch"] = paper.get("_language_branch_override") or detection.language
        if paper["_language_branch"] == "zh":
            zh_papers.append(paper)
        else:
            en_papers.append(paper)

    logger.info(
        "Language split: %d total → zh=%d en=%d",
        len(papers), len(zh_papers), len(en_papers),
    )
    return zh_papers, en_papers
