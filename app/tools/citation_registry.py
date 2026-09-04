"""从最终正文构造唯一论文、引用出现与章节分配注册表。"""

from __future__ import annotations

import re
from typing import Any

from app.core.citation_syntax import extract_citation_ids
from app.schemas.deliverable_schema import CitationOccurrence, CitationRegistry


def build_citation_registry(
    text: str,
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    papers_by_id = {
        str(paper.get("paper_id") or ""): paper
        for paper in papers
        if paper.get("paper_id")
    }
    occurrences: list[CitationOccurrence] = []
    allocations: dict[str, list[str]] = {}
    detailed: dict[str, int] = {}
    current_section = ""
    paragraph_index = 0
    occurrence_index = 0

    blocks = re.split(r"\n\s*\n", str(text or ""))
    for block in blocks:
        heading = re.match(r"^#{1,4}\s+(.+?)\s*$", block.strip())
        if heading:
            current_section = heading.group(1).strip()
            paragraph_index = 0
            continue
        paragraph_index += 1
        ids = extract_citation_ids(block)
        unique_ids = list(dict.fromkeys(ids))
        if current_section:
            allocated = allocations.setdefault(current_section, [])
            for paper_id in unique_ids:
                if paper_id in papers_by_id and paper_id not in allocated:
                    allocated.append(paper_id)
        # 单篇论文主导且文本较长，视为一次详细介绍；后续验证可据此限制重复。
        plain = re.sub(r"\[[^\]]+\]|\s+", "", block)
        if len(unique_ids) == 1 and len(plain) >= 80:
            detailed[unique_ids[0]] = detailed.get(unique_ids[0], 0) + 1
        for paper_id in ids:
            if paper_id not in papers_by_id:
                continue
            occurrence_index += 1
            occurrences.append(CitationOccurrence(
                paper_id=paper_id,
                section_title=current_section,
                paragraph_index=paragraph_index,
                occurrence_index=occurrence_index,
            ))

    cited_ids = list(dict.fromkeys(item.paper_id for item in occurrences))
    registry = CitationRegistry(
        unique_papers={paper_id: papers_by_id[paper_id] for paper_id in cited_ids},
        citation_occurrences=occurrences,
        section_allocations=allocations,
        detailed_introductions=detailed,
    )
    return registry.model_dump(mode="json")
