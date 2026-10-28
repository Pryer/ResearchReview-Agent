"""有界字符切块；位置始终相对于未重新清洗的页面文本或 full_text。"""

from __future__ import annotations

import re
from typing import Any


_HEADING = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?"
    r"(abstract|introduction|(?:materials?\s+and\s+)?methods?|methodology|approach|model|proposed|"
    r"experimental\s+(?:results?|setup)|experiments?|evaluation|results?|discussion|"
    r"conclusions?(?:\s+and\s+future\s+work)?|summary|limitations?|references|bibliography|appendix|"
    r"摘要|引言|绪论|方法|研究方法|研究设计|模型|实验结果|实验设置|实验|评估|评价|结果分析|结果|讨论|"
    r"结论与展望|总结与展望|结论|局限性|参考文献|附录)\s*[:：]?\s*$"
)


def section_family(name):
    name = name.lower()
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


def section_headings(text):
    """解析与切块共享标题位置，防止两条路径对同一章节产生不同身份。"""
    return [(m.start(), section_family(m.group(1))) for m in _HEADING.finditer(text)]


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
        boundaries.extend(section_headings(text))
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
                    chunks.append(dict(
                        paper_id=paper_id or parsed.get("paper_id"),
                        chunk_id=len(chunks), text=text[cursor:stop], page=page,
                        section=name, char_start=cursor, char_end=stop,
                        offset_basis="page" if page is not None else "full_text",
                        element_type="text",
                        excluded=name in {"references", "bibliography", "参考文献"},
                    ))
                if stop == end:
                    break
                cursor = max(cursor + 1, stop - overlap_chars)
    return chunks
