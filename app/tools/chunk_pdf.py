"""有界字符切块；位置始终相对于未重新清洗的页面文本或 full_text。"""

from __future__ import annotations

import re
from typing import Any


_HEADING = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?"
    r"(abstract|introduction|(?:materials?\s+and\s+)?methods?|methodology|approach|model|proposed|"
    r"experimental\s+(?:results?|setup)|experiments?|evaluation|results?|discussion|"
    r"conclusions?(?:\s+and\s+future\s+work)?|summary|limitations?|references|bibliography|"
    r"appendi(?:x|ces)(?:[ \t]+[A-Z])?|supplementary[ \t]+(?:materials?|information|methods?)|"
    r"摘要|引言|绪论|方法|研究方法|研究设计|模型|实验结果|实验设置|实验|评估|评价|结果分析|结果|讨论|"
    r"结论与展望|总结与展望|结论|局限性|参考文献|附录|补充材料)\s*[:：]?\s*$"
)


def section_family(name):
    name = name.lower()
    if name.startswith(("appendix", "appendices", "supplementary", "附录", "补充材料")):
        return "appendix"
    for family, terms in (
        ("method", ("method", "approach", "setup", "model", "proposed", "方法", "设置", "研究设计", "模型")),
        ("results", ("result", "experiment", "evaluation", "实验", "结果", "评估", "评价")),
        ("limitations", ("limitation", "局限")),
        ("discussion", ("discussion", "讨论")),
        ("conclusion", ("conclusion", "summary", "结论", "总结")),
        ("abstract", ("abstract", "摘要")),
        ("introduction", ("introduction", "引言", "绪论")),
        ("references", ("references", "bibliography", "参考文献")),
    ):
        if any(term in name for term in terms):
            return family
    return name


def section_headings(text, *, initial_section="full_text"):
    """解析与切块共享标题位置，防止两条路径对同一章节产生不同身份。"""
    headings = []
    section = initial_section
    for match in _HEADING.finditer(text):
        family = section_family(match.group(1))
        # WHY: 文献题名换行后可能恰好是“model”；参考文献只能由明确附录标题结束。
        if section == "references" and family not in {"references", "appendix"}:
            continue
        headings.append((match.start(), family))
        section = family
    return headings


def chunk_pdf(parsed: dict[str, Any], *, max_chars: int = 6000,
              overlap_chars: int = 400, paper_id: str | None = None) -> list[dict[str, Any]]:
    """优先页面、章节和段落边界；跨页不拼接，保留可直接回读的坐标。"""
    if max_chars < 1 or not 0 <= overlap_chars < max_chars:
        raise ValueError("要求 max_chars > overlap_chars >= 0")
    pages = parsed.get("pages") or []
    sources = [(item.get("page"), str(item.get("text") or "")) for item in pages]
    if not any(text.strip() for _, text in sources):
        sources = [(None, str(parsed.get("full_text") or ""))]
    chunks = []
    section = "full_text"
    for page, text in sources:
        boundaries = [(0, section)]
        boundaries.extend(section_headings(text, initial_section=section))
        for i, (start, name) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            section = name
            cursor = start
            while cursor < end:
                stop = min(cursor + max_chars, end)
                if stop < end:
                    # WHY: 不重新清洗文本，段落边界调整不能改变来源坐标。
                    split = text.rfind("\n", cursor + max_chars // 2, stop)
                    if split >= 0:
                        stop = split + 1
                if text[cursor:stop].strip():
                    trivial = len(text[cursor:stop].strip()) < 2
                    chunks.append(dict(
                        paper_id=paper_id or parsed.get("paper_id"),
                        chunk_id=len(chunks), text=text[cursor:stop], page=page,
                        section=name, char_start=cursor, char_end=stop,
                        offset_basis="page" if page is not None else "full_text",
                        element_type="noise" if trivial else "text",
                        excluded=trivial or name == "references",
                    ))
                if stop == end:
                    break
                cursor = max(cursor + 1, stop - overlap_chars)
    return chunks
