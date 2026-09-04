"""正文引用标记的统一规范化与提取。

内部写作协议只接受 ``[paper_id]``。模型偶尔会模仿中文少样本，把引用
改写为 ``〔paper_id〕``，或输出未闭合的 ``[cn`` 代码块语言标记。本模块
在验证、主张核验和最终渲染前统一处理这些情况，并确保引用匹配不跨行。
"""

from __future__ import annotations

import re
from collections.abc import Iterable


_ASCII_GROUP_RE = re.compile(r"\[([^\]\r\n]+)\]")
_FULLWIDTH_GROUP_RE = re.compile(r"〔([^〕\r\n]+)〕")
_GROUP_SPLIT_RE = re.compile(r"[;；,，、\s]+")
_CODE_FENCE_LANGUAGE_RE = re.compile(
    r"\[(?:cn|zh|zh-cn|en|markdown|md)\s*$",
    re.I | re.M,
)
_STABLE_ID_RE = re.compile(
    r"^(?:openalex|s2|arxiv|cnki|doi):[^\s\[\]〔〕,，;；]+$",
    re.I,
)
_SOURCE_LIKE_ID_RE = re.compile(
    r"^[a-z][a-z0-9_-]{1,20}:[^\s\[\]〔〕,，;；]+$",
    re.I,
)


def split_citation_group(raw: str) -> list[str]:
    """拆分一个复合引用组，并保持首次出现顺序。"""
    result: list[str] = []
    for value in _GROUP_SPLIT_RE.split(str(raw or "")):
        value = value.strip()
        if value and value not in result:
            result.append(value)
    return result


def normalize_citation_syntax(
    text: str,
    valid_ids: Iterable[str] | None = None,
) -> str:
    """把可确认的全角引用转换为内部格式，并清除 ``[cn`` 等残缺标记。

    只有组内全部条目都像论文 ID 或数字引用时才转换，
    因此少样本里的 ``〔证据A〕`` 仍会保留并由蓝图泄漏检查拦截。
    未知来源前缀（如误写的 ``openex:``）也会转为内部方括号，
    让后续验证器将其明确报告为无法映射，而不是静默混入正文。
    """
    rendered = str(text or "")
    allowed = {str(value) for value in (valid_ids or []) if str(value)}

    def replace_fullwidth(match: re.Match[str]) -> str:
        values = split_citation_group(match.group(1))
        if not values:
            return match.group(0)
        if all(_is_citation_id(value, allowed) for value in values):
            return "[" + "; ".join(values) + "]"
        return match.group(0)

    rendered = _FULLWIDTH_GROUP_RE.sub(replace_fullwidth, rendered)
    # 常见的模型代码块语言残片会被误当成一个未闭合引用。直接移除该标记，
    # 避免后续引用渲染器把下一章节之前的全部文本吞进一个引用组。
    rendered = _CODE_FENCE_LANGUAGE_RE.sub("", rendered)
    return rendered


def extract_citation_ids(
    text: str,
    *,
    normalize_fullwidth: bool = True,
    valid_ids: Iterable[str] | None = None,
) -> list[str]:
    """提取单引和复合引用，且永远不跨越换行符。"""
    rendered = (
        normalize_citation_syntax(text, valid_ids)
        if normalize_fullwidth
        else str(text or "")
    )
    result: list[str] = []
    for match in _ASCII_GROUP_RE.finditer(rendered):
        for value in split_citation_group(match.group(1)):
            if value not in result:
                result.append(value)
    return result


def malformed_citation_fragments(text: str) -> list[str]:
    """返回疑似未闭合的内部引用片段，用于确定性质量门禁。"""
    fragments: list[str] = []
    for line in str(text or "").splitlines():
        scrubbed = _ASCII_GROUP_RE.sub("", line)
        for match in re.finditer(r"\[([^\[\]]*)$", scrubbed):
            value = match.group(1).strip()
            if value:
                fragments.append(value)
    return fragments


def _is_citation_id(value: str, allowed: set[str]) -> bool:
    return (
        value in allowed
        or value.isdigit()
        or bool(_STABLE_ID_RE.fullmatch(value))
        or bool(_SOURCE_LIKE_ID_RE.fullmatch(value))
    )
