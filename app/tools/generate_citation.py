"""参考文献生成与引用校验工具。

支持 gbt7714 / apa / ieee / bibtex 格式。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.core.citation_syntax import (
    extract_citation_ids as extract_normalized_citation_ids,
    normalize_citation_syntax,
    split_citation_group,
)
from app.core.exceptions import CitationGenerationError, CitationValidationError
from app.core.logger import get_logger
from app.tools.venue_tiers import is_platform_placeholder_venue
from app.utils.title_cleaner import clean_venue

logger = get_logger(__name__)

_SUPPORTED_CITATION_STYLES = {"gbt7714", "apa", "ieee", "bibtex"}


def _clean_venue(value: Any) -> str:
    """移除检索页面控件文案与院校标签，避免其进入正式参考文献。"""
    return clean_venue(value)


# 数据源自有的、可回溯到该源检索页的记录标识前缀。DOI / arXiv ID 是跨源
# 通用标识，此处补的是"没有 DOI 但仍可溯源"的情形：CNKI 中文期刊论文普遍
# 不注册 DOI，其 paper_id 取自详情页 URL 的 v= 参数，能定位到唯一记录。
# 此前判据只认 openalex/s2/arxiv/doi 四类，导致 9 条正常的 CNKI 中文文献
# 被报"尚未获得稳定标识"（2026-08-29 实测），而它们的题名、作者、年份、
# 刊名齐全且 evidence_state 已是 source_verified。
_SOURCE_SCOPED_ID_PREFIXES = ("openalex:", "s2:", "arxiv:", "doi:", "cnki:")


def normalize_doi(value: Any) -> str:
    """归一 DOI：去掉 URL/`doi:` 前缀并统一小写。

    WHY: 重复检测与一致性比较必须在同一空间进行，否则
    ``https://doi.org/10.X`` 与 ``10.x`` 会被当成两条不同记录。
    """
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):]
            break
    return lowered.strip().strip("/")


def _has_stable_identity(paper: Dict[str, Any], doi: str) -> bool:
    """论文是否具备可回溯的稳定标识。

    判据是"能否唯一定位到一条源记录"：跨源标识（DOI / arXiv ID）优先，
    否则接受带数据源前缀的记录标识。裸标识（无前缀、无 DOI）仍判未核验。
    """
    if doi or paper.get("arxiv_id"):
        return True
    paper_id = str(paper.get("paper_id") or "")
    if not paper_id.startswith(_SOURCE_SCOPED_ID_PREFIXES):
        return False
    # 前缀后必须有实际标识内容，"cnki:" 这样的空壳不算可溯源。
    return bool(paper_id.split(":", 1)[1].strip())


def extract_citation_ids(text: str) -> list[str]:
    """提取单引和复合引用中的独立 ID，并保持首次出现顺序。"""
    return extract_normalized_citation_ids(text)


def generate_reference(paper: Dict[str, Any], citation_style: str) -> str:
    """生成单条参考文献。

    Args:
        paper: 论文元数据字典。
        citation_style: gbt7714 / apa / ieee / bibtex。

    Returns:
        格式化后的参考文献字符串。
    """
    citation_style = str(citation_style or "").strip().lower()
    if citation_style not in _SUPPORTED_CITATION_STYLES:
        raise CitationGenerationError(
            f"Unsupported citation style: {citation_style or '<empty>'}"
        )
    title = paper.get("title", "").strip().rstrip(".")
    authors = paper.get("authors", [])
    year = paper.get("year", "")
    venue = _clean_venue(paper.get("venue", ""))
    doi = paper.get("doi", "")
    url = paper.get("url", "")

    # arXiv 客户端对每条记录硬编码 venue="arXiv"，但同一记录可能已带出版方
    # DOI（正式发表后回填）。此时 "arXiv" 是错误刊名——渲染成 "[J]. arXiv"
    # 等于断言该文发表在名为 arXiv 的期刊上。有可解析的出版方 DOI 兜底出处，
    # 宁可略去 venue，也不写一个已知错误的刊名。真正的预印本（arXiv 自有 DOI
    # 或无 DOI）不受影响，仍保留 "arXiv"。
    if is_platform_placeholder_venue(venue=venue, doi=doi):
        venue = ""

    # 作者格式化
    if isinstance(authors, list) and authors:
        if citation_style == "apa":
            author_str = ", ".join(authors[:6])
            if len(authors) > 6:
                author_str += ", et al."
        elif citation_style == "ieee":
            author_str = ", ".join(authors[:6])
            if len(authors) > 6:
                author_str += ", et al."
        else:
            author_str = ", ".join(
                str(author).strip(" ,.;，。")
                for author in authors[:3]
                if str(author).strip(" ,.;，。")
            )
            if len(authors) > 3:
                author_str += ", et al."
    else:
        author_str = ""

    # 按格式生成
    if citation_style == "bibtex":
        key = generate_bibtex_key(paper)
        bibtex_authors = " and ".join(
            str(author).strip(" ,.;，。") for author in authors if str(author).strip(" ,.;，。")
        ) if isinstance(authors, list) else str(authors or "").strip()
        entry = f"@article{{{key},\n  title={{{title}}},\n  author={{{bibtex_authors}}},\n  year={{{year}}},"
        if venue:
            entry += f"\n  journal={{{venue}}},"
        if doi:
            entry += f"\n  doi={{{doi}}},"
        entry += "\n}"
        return entry

    if citation_style == "apa":
        ref = f"{author_str} ({year}). {title}."
        if venue:
            ref += f" {venue}."
        if doi:
            ref += f" https://doi.org/{doi}"
        return ref

    if citation_style == "ieee":
        ref = f"{author_str}, \"{title},\""
        if venue:
            ref += f" {venue},"
        if year:
            ref += f" {year}."
        return ref

    # gbt7714（默认）
    publication_type = str(paper.get("publication_type") or "").lower()
    if not publication_type:
        source = str(paper.get("source") or "").lower()
        publication_type = "preprint" if source == "arxiv" or paper.get("arxiv_id") else "journal_article"
    type_mark = {
        "conference_paper": "[C]",
        "conference_short_paper": "[C]",
        "preprint": "[EB/OL]",
        "systematic_review": "[J]",
        "meta_analysis": "[J]",
        "journal_article": "[J]",
        "thesis": "[D]",
        "dissertation": "[D]",
        "master_thesis": "[D]",
        "doctoral_thesis": "[D]",
    }.get(publication_type, "[Z]")
    # 作者串可能以 et al. 结尾，模板不再额外制造双句点。
    ref = f"{author_str.rstrip('.')}. {title}{type_mark}."
    if venue:
        ref += f" {venue},"
    if year:
        ref += f" {year}."
    if doi:
        ref += f" DOI: {doi}."
    return ref


def generate_references(papers: List[Dict[str, Any]], citation_style: str) -> List[str]:
    """批量生成参考文献；失败时保持强失败语义，不静默改变条目数量。"""
    refs: List[str] = []
    key_counts: dict[str, int] = {}
    for paper in papers:
        try:
            ref = generate_reference(paper, citation_style)
            if str(citation_style).strip().lower() == "bibtex":
                base_key = generate_bibtex_key(paper)
                count = key_counts.get(base_key, 0)
                key_counts[base_key] = count + 1
                if count:
                    suffix = chr(ord("a") + count - 1) if count <= 26 else str(count)
                    ref = ref.replace(
                        f"@article{{{base_key},",
                        f"@article{{{base_key}{suffix},",
                        1,
                    )
            if ref:
                refs.append(ref)
        except Exception as e:
            raise CitationGenerationError(
                f"Failed to generate reference for {paper.get('paper_id') or 'unknown'}: {e}"
            ) from e
    return refs


def generate_in_text_citation(paper: Dict[str, Any], citation_style: str) -> str:
    """生成正文引用标记。"""
    if citation_style in ("apa",):
        authors = paper.get("authors", [])
        year = paper.get("year", "")
        first_author = authors[0].split()[-1] if authors else "Unknown"
        return f"({first_author}, {year})"

    if citation_style in ("ieee", "gbt7714"):
        return f"[{paper.get('paper_id', '')}]"

    return f"[{paper.get('paper_id', '')}]"


def validate_citations(
    review_text: str,
    references: List[str],
    paper_cards: Optional[List[Dict[str, Any]]] = None,
    reference_papers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """检查正文引用与参考文献是否匹配。

    Returns:
        valid / missing_citations / unused_references / suggestions。
    """
    # 提取正文中所有 [xxx] 引用
    cited_ids = set(extract_citation_ids(review_text))

    ref_count = len(references)

    # 检查 paper_id 是否在参考文献范围内
    missing: list[str] = []
    if paper_cards:
        valid_ids = {str(c.get("paper_id") or "") for c in paper_cards}
        for cid in cited_ids:
            if cid in valid_ids:
                continue
            if cid.isdigit():
                # 纯数字引用按证据卡片位次解析；越界编号如实报告缺失，
                # 不能借 isdigit 豁免静默漏报。
                if not 1 <= int(cid) <= len(paper_cards):
                    missing.append(cid)
                continue
            missing.append(cid)

    # 未被引用的参考文献
    unused: list[str] = []
    external_cited = {cid for cid in cited_ids if not cid.isdigit()}
    effective_reference_papers = reference_papers
    if effective_reference_papers is None and external_cited and paper_cards:
        effective_reference_papers = paper_cards[:ref_count]
    if external_cited and effective_reference_papers is not None:
        reference_ids = [str(p.get("paper_id") or "") for p in effective_reference_papers]
        missing.extend(cid for cid in external_cited if cid not in set(reference_ids) and cid not in missing)
        unused.extend(f"[{pid}]" for pid in reference_ids if pid and pid not in external_cited)
    else:
        # 数字引用 [1] [2] ...
        numeric_cited = {int(x) for x in cited_ids if x.isdigit()}
        for i in range(1, ref_count + 1):
            if i not in numeric_cited:
                unused.append(f"[{i}]")

    effective_papers = reference_papers or []
    incomplete_metadata: list[str] = []
    unverified_metadata: list[str] = []
    unknown_publication_status: list[str] = []
    retracted_papers: list[str] = []
    seen_dois: dict[str, str] = {}
    duplicate_dois: list[str] = []
    for paper in effective_papers:
        paper_id = str(paper.get("paper_id") or "unknown")
        if not all((paper.get("title"), paper.get("authors"), paper.get("year"), paper.get("venue"))):
            incomplete_metadata.append(paper_id)
        doi = normalize_doi(paper.get("doi"))
        if doi:
            if doi in seen_dois:
                duplicate_dois.append(doi)
            else:
                seen_dois[doi] = paper_id
        source = str(paper.get("source") or "").lower()
        if source in {"", "unknown"} or not _has_stable_identity(paper, doi):
            unverified_metadata.append(paper_id)
        status = str(
            getattr(paper.get("publication_status"), "value", paper.get("publication_status")) or "unknown"
        ).strip().lower()
        if status in {"retracted", "withdrawn"}:
            retracted_papers.append(paper_id)
        elif status in {"", "unknown"}:
            unknown_publication_status.append(paper_id)

    # WHY: 撤稿/撤回文献不能作为普通事实证据交付；出版状态未知只作提示，
    # 因为外部数据库确实可能暂时查不到正式出版记录。
    structurally_valid = (
        len(missing) == 0
        and not incomplete_metadata
        and not duplicate_dois
        and not retracted_papers
    )
    metadata_quality_valid = (
        not incomplete_metadata
        and not unverified_metadata
        and not duplicate_dois
        and not retracted_papers
    )

    suggestions: list[str] = []
    if missing:
        suggestions.append(f"正文引用了未在参考文献表中的条目：{missing}")
    if unused:
        suggestions.append(f"参考文献中有 {len(unused)} 条未被正文引用")
    if incomplete_metadata:
        suggestions.append(f"有 {len(incomplete_metadata)} 条参考文献缺少作者、年份、题名或来源")
    if unverified_metadata:
        suggestions.append(f"有 {len(unverified_metadata)} 条参考文献尚未获得稳定标识或可靠来源核验")
    if duplicate_dois:
        suggestions.append(f"参考文献存在重复 DOI：{sorted(set(duplicate_dois))}")
    if retracted_papers:
        suggestions.append(
            f"有 {len(retracted_papers)} 条参考文献已撤稿或撤回，不能作为普通事实证据"
        )
    if unknown_publication_status:
        suggestions.append(
            f"有 {len(unknown_publication_status)} 条参考文献的出版状态未确认，不能声明为已正式发表"
        )

    return {
        "valid": structurally_valid,
        "structurally_valid": structurally_valid,
        "metadata_quality_valid": metadata_quality_valid,
        "cited_ids": sorted(cited_ids),
        "missing_citations": missing,
        "unused_references": unused,
        "incomplete_metadata": incomplete_metadata,
        "unverified_metadata": unverified_metadata,
        "unknown_publication_status": unknown_publication_status,
        "retracted_references": retracted_papers,
        "duplicate_dois": sorted(set(duplicate_dois)),
        "suggestions": suggestions,
    }


def generate_bibtex_key(paper: Dict[str, Any]) -> str:
    """生成 BibTeX key。

    格式：FirstAuthorYear，如 Vaswani2017。
    """
    authors = paper.get("authors", [])
    year = str(paper.get("year", ""))
    if authors:
        first = authors[0].split()[-1] if " " in authors[0] else authors[0]
        first = re.sub(r"[^a-zA-Z]", "", first)
        return f"{first}{year}"
    return f"paper{year}"


def sanitize_in_text_citations(text: str) -> str:
    """移除正文中残留的内部标记（如 [EN][CN] 等语言标签）和无效引用碎片。

    这些标记来自内部 paper_id 或元数据字段未被正确替换为数字编号，
    出现在最终交付物中属于引用格式错误。
    """
    # 独立出现的语言/区域标记：[EN] [CN] [ZH] [en] [cn] 等
    text = re.sub(r"\[(?:EN|CN|ZH|en|cn|zh)\](?!\w)", "", text)
    # 连续多个空括号或标记合并产生的碎片如 [EN][15] → [15]
    text = re.sub(r"\]\[", ", ", text)
    # 移除孤立的单/双字母标记 [A] [a] [Ab] 等（非数字非 paper_id）
    text = re.sub(r"\[([A-Za-z]{1,2})\](?!\w)", "", text)
    # 清理可能产生的多余空格
    text = re.sub(r"  +", " ", text)
    return text


def generate_and_validate_citations(
    review_text: str,
    paper_cards: List[Dict[str, Any]],
    citation_style: str = "gbt7714",
    llm=None,
) -> Dict[str, Any]:
    """一站式：生成参考文献 + 校验。

    Returns:
        {"references": [...], "validation": {...}}
    """
    valid_ids = {
        str(paper.get("paper_id") or "")
        for paper in paper_cards
        if paper.get("paper_id")
    }
    review_text = normalize_citation_syntax(review_text, valid_ids)
    review_text = sanitize_in_text_citations(review_text)
    cards_by_id = {
        str(paper.get("paper_id") or ""): paper
        for paper in paper_cards
        if paper.get("paper_id")
    }
    if any(cid.isdigit() for cid in extract_citation_ids(review_text)):
        # 纯数字引用按证据卡片位次解析为内部 ID，与显式 ID 路径统一；
        # 越界编号保留原样，由 validate_citations 如实报告缺失。
        def _resolve_numeric(match: re.Match[str]) -> str:
            values = split_citation_group(match.group(1))
            resolved = [
                str(paper_cards[int(value) - 1].get("paper_id") or value)
                if value.isdigit() and 1 <= int(value) <= len(paper_cards)
                else value
                for value in values
            ]
            return "[" + "; ".join(resolved) + "]"

        review_text = re.sub(r"\[([^\]\r\n]+)\]", _resolve_numeric, review_text)

    ordered_ids = extract_citation_ids(review_text)
    cited_ids = set(ordered_ids)
    # GB/T 7714 顺序编码制：编号必须按正文首现顺序分配，而不是按证据
    # 卡片列表顺序；参考文献表顺序与正文编号一一对应。
    reference_papers = [
        cards_by_id[cid]
        for cid in ordered_ids
        if cid in cards_by_id
    ]

    references = generate_references(reference_papers, citation_style)
    validation = validate_citations(
        review_text,
        references,
        paper_cards,
        reference_papers=reference_papers,
    )

    if llm and not validation["valid"]:
        try:
            from app.prompt.citation import CITATION_CHECK_PROMPT
            import json
            prompt = CITATION_CHECK_PROMPT.format(
                review_text=review_text[:3000],
                references_json=json.dumps(references, ensure_ascii=False),
            )
            response = llm.complete(prompt, response_format="json")
            llm_validation = _safe_parse_json(response)
            # 合并结果
            validation["llm_suggestions"] = llm_validation.get("suggestions", [])
        except Exception as e:
            logger.debug("LLM citation check failed: %s", e)

    citation_map = {
        str(paper.get("paper_id") or ""): index
        for index, paper in enumerate(reference_papers, 1)
        if paper.get("paper_id")
    }
    rendered_text = render_in_text_citations(
        review_text,
        reference_papers,
        citation_style,
    )
    return {
        "references": references,
        "validation": validation,
        "reference_papers": reference_papers,
        "citation_map": citation_map,
        "rendered_text": rendered_text,
    }


def render_in_text_citations(
    review_text: str,
    reference_papers: List[Dict[str, Any]],
    citation_style: str,
) -> str:
    """将内部 paper_id 引用转换为论文可读的编号或作者—年份格式。"""
    replacements: dict[str, str] = {}
    for index, paper in enumerate(reference_papers, 1):
        paper_id = str(paper.get("paper_id") or "")
        if not paper_id:
            continue
        if citation_style == "apa":
            replacements[paper_id] = generate_in_text_citation(paper, "apa")
        else:
            replacements[paper_id] = str(index)

    def replace_group(match: re.Match[str]) -> str:
        raw = match.group(1)
        values = split_citation_group(raw)
        if not values:
            return match.group(0)
        if citation_style == "apa":
            rendered_values = [
                replacements.get(value, f"[{value}]")
                for value in values
            ]
            return "; ".join(rendered_values)
        rendered_values = [replacements.get(value, value) for value in values]
        # 单处引用不超过 3 篇；超出部分删除而不是拆成连续方括号组——
        # 拆组只是把堆砌伪装到检测阈值之下，正文信息量并未增加。
        # 引用缺口由引用数量校验和最终质量门禁如实报告。
        max_per_group = 3
        kept = rendered_values[:max_per_group]
        return "[" + ", ".join(kept) + "]"

    return re.sub(r"\[([^\]\r\n]+)\]", replace_group, str(review_text or ""))


from app.core.json_utils import parse_json_object as _safe_parse_json  # noqa: E402
