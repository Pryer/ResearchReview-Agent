"""检索源的确定性能力过滤，不承担召回策略或来源优先级规划。"""

from __future__ import annotations

import re
from typing import Any


def is_chinese_query(query: str) -> bool:
    return bool(re.search(r"[一-鿿]", str(query or "")))


# 整条检索词等于这些通用词时没有任何区分力（“现状”能命中一切也等于
# 什么都没命中），只会浪费一轮检索与配额。作为主题本身时除外。
_GENERIC_SEARCH_KEYWORDS = {
    "现状", "综述", "研究", "背景", "意义", "方法", "分析", "进展", "趋势",
    "应用", "论文", "文献", "相关工作", "调研", "检索", "总结",
    "survey", "review", "research", "overview", "background", "method",
    "analysis", "paper", "papers", "introduction", "related work", "study",
    "trend", "trends", "progress",
}


def is_generic_search_keyword(query: str) -> bool:
    """判定整条检索词是否为无区分力的通用词（如“现状”/“survey”）。"""
    return str(query or "").strip().lower() in _GENERIC_SEARCH_KEYWORDS


_TIME_PREFIX_RE = re.compile(r"^(?:近[一二三四五六七八九十\d]+\s*年|最新|近年来|最近)\s*")
_VERB_PREFIX_RE = re.compile(r"^(?:调研|检索|搜索|查询|查找|研究)\s*")


def sanitize_search_keyword(query: str) -> str | None:
    """中英混杂检索词的确定性清洗：拆出中文段，拆不出则丢弃。

    refine/恢复 LLM 偶尔生成"survey 近三年少样本动作识别研究综述"这类
    中英混杂查询：路由层因含英文词把它送给国际源，而国际源拿中文长句
    基本返回 0 或噪声，白白浪费一轮检索。这里提取最长连续中文段（并
    剥掉"近三年"式时间前缀与"调研"式动词前缀）作为替代查询；中文段
    不足 4 字时返回 None 表示该词应丢弃。纯中文、纯英文的查询原样返回。
    """
    text = re.sub(r"\s+", " ", str(query or "")).strip()
    if not text:
        return None
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_latin_word = bool(re.search(r"[A-Za-z]{3,}", text))
    if not (has_cjk and has_latin_word):
        cleaned = _VERB_PREFIX_RE.sub("", _TIME_PREFIX_RE.sub("", text))
        return cleaned if len(cleaned) >= 2 else None

    cjk_runs = re.findall(r"[\u4e00-\u9fff][\u4e00-\u9fff\s、，,]*", text)
    cleaned_runs = [re.sub(r"\s+", "", run) for run in cjk_runs]
    best = max(cleaned_runs, key=len, default="")
    best = _VERB_PREFIX_RE.sub("", _TIME_PREFIX_RE.sub("", best))
    return best if len(best) >= 4 else None


def compatible_sources(query: str, configured_sources: list[str]) -> list[str]:
    """只移除明确不兼容的来源，并保持调用方配置顺序。

    确定性语言兼容边界：CNKI 不接受非中文检索式；arXiv 与
    Semantic Scholar 不接受中文检索式（中文查询只会返回 0/噪声）。
    其余来源是否值得检索应由 Agent 的检索计划、预算和运行时诊断
    决定，不能由工具层硬编码优先级。
    """
    chinese = is_chinese_query(query)
    return [
        source for source in configured_sources
        if (source != "cnki" or chinese)
        and (source != "arxiv" or not chinese)
        and (source != "semantic_scholar" or not chinese)
    ]


def source_name(record: dict[str, Any]) -> str:
    """从标准来源字段、稳定标识或落地页识别来源，不推断内容语义。"""
    explicit = str(record.get("source") or "").strip().lower()
    if explicit:
        return explicit
    paper_id = str(record.get("paper_id") or "").strip().lower()
    if ":" in paper_id:
        return paper_id.split(":", 1)[0]
    urls = (
        record.get("pdf_url"), record.get("url"), record.get("landing_page_url")
    )
    if any("cnki.net" in str(url or "").lower() for url in urls):
        return "cnki"
    return ""


def source_allows_full_text(record: dict[str, Any]) -> bool:
    """执行来源许可边界；这类安全策略不交给 LLM 判断。"""
    return source_name(record) != "cnki"
