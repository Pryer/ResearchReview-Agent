"""论文去重工具。"""

from __future__ import annotations

import re
from typing import Dict, List

from app.core.logger import get_logger
from app.utils.title_cleaner import clean_title

logger = get_logger(__name__)


def normalize_title(title: str) -> str:
    """标题归一化。"""
    if not title:
        return ""
    cleaned = clean_title(title)
    title = cleaned.lower()
    title = re.sub(r"[\-–—:;,.!?()[\]{}\"']", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def title_tokens(title: str) -> set[str]:
    """返回标题词集合，用于去重相似度。"""
    normalized = normalize_title(title)
    if not normalized:
        return set()
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized))


def title_similarity(a: str, b: str) -> float:
    """计算两个标题的 Jaccard 相似度（基于词集合）。

    Args:
        a: 标题 A。
        b: 标题 B。

    Returns:
        0.0 ~ 1.0 的相似度。
    """
    set_a = title_tokens(a)
    set_b = title_tokens(b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def strong_paper_identifier(paper: Dict) -> str | None:
    """提取高置信度论文标识符（DOI / arXiv ID / OpenAlex ID）。

    跨源检索合并与中英文分支去重的共用规则：只有强标识才允许在
    评分前判定两篇记录为同一论文；标题等弱标识留给合并后的第二阶段，
    避免评分阶段误删不同论文。
    """
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    arxiv_id = (paper.get("arxiv_id") or "").strip().lower()
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    paper_id = str(paper.get("paper_id") or "")
    if paper_id.startswith("openalex:"):
        return paper_id
    return None


def deduplicate_papers(
    papers: List[Dict],
    similarity_threshold: float = 0.85,
) -> List[Dict]:
    """按 DOI、arXiv ID、标题去重（标识符跨键闭包）。

    优先级：DOI > arXiv ID > 标题相似度；保留信息更完整的版本
    （见 ``_merge_paper``）。

    同一论文经不同数据源进入时可能一把记录带 DOI、另一把带 arXiv ID。
    两把键必须收敛到同一条存活记录（跨键闭包），并且合并后补全的
    DOI/arXiv 字段会让后续所有身份判定看到同一组键——否则同一论文会
    以不同键重复进入排序与写作池，占用配额并造成引用表双条目。

    Args:
        papers: 候选论文列表。
        similarity_threshold: 标题相似度阈值。

    Returns:
        去重后的论文列表。
    """
    if not papers:
        return []

    key_to_index: Dict[str, int] = {}          # "doi:x" / "arxiv:y" → 存活索引
    title_entries: list[tuple[str, int]] = []  # (normalized_title, 存活索引)
    result_indices: list[int] = []

    def _remap_owner(from_idx: int, to_idx: int) -> None:
        """把 from_idx 名下的全部键与标题变体改挂到 to_idx。"""
        for key, owner in key_to_index.items():
            if owner == from_idx:
                key_to_index[key] = to_idx
        for pos, (seen_title, owner) in enumerate(title_entries):
            if owner == from_idx:
                title_entries[pos] = (seen_title, to_idx)

    for i, paper in enumerate(papers):
        doi = (paper.get("doi") or "").strip().lower()
        arxiv_id = (paper.get("arxiv_id") or "").strip().lower()
        norm = normalize_title(paper.get("title", ""))
        keys = [
            *(f"doi:{value}" for value in [doi] if value),
            *(f"arxiv:{value}" for value in [arxiv_id] if value),
        ]

        owners = {key_to_index[key] for key in keys if key in key_to_index}
        if norm:
            for seen_title, idx in title_entries:
                if title_similarity(norm, seen_title) >= similarity_threshold:
                    owners.add(idx)
                    break

        if owners:
            keep_idx = min(owners)
            # 折叠互相冲突的旧存活记录（如 r0 只带 DOI、r1 只带 arXiv，
            # 本条同时命中两把键），并把它们的键全部改挂到最终胜者名下。
            for owner in sorted(owners - {keep_idx}):
                _merge_paper(papers, keep_idx, owner)
                _remap_owner(owner, keep_idx)
                result_indices.remove(owner)
            _merge_paper(papers, keep_idx, i)
            for key in keys:
                key_to_index[key] = keep_idx
            if norm:
                title_entries.append((norm, keep_idx))
            continue

        for key in keys:
            key_to_index[key] = i
        if norm:
            title_entries.append((norm, i))
        result_indices.append(i)

    result = [papers[i] for i in result_indices]
    logger.info(
        "Deduplication: %d → %d (threshold=%.2f)",
        len(papers), len(result), similarity_threshold,
    )
    return result


def _merge_paper(papers: List[Dict], keep_idx: int, drop_idx: int) -> None:
    """合并两篇论文的信息：保留信息更完整的版本。"""
    keep = papers[keep_idx]
    drop = papers[drop_idx]

    # 优先保留有 PDF URL 的版本
    if not keep.get("pdf_url") and drop.get("pdf_url"):
        keep["pdf_url"] = drop["pdf_url"]
        keep["is_open_access"] = drop.get("is_open_access", False)

    # 保留引用量更高的
    if (drop.get("citation_count") or 0) > (keep.get("citation_count") or 0):
        keep["citation_count"] = drop.get("citation_count")

    # 补全缺失字段。保留最初命中的稳定 paper_id，但不要丢掉其它来源带来的
    # 作者、链接和关键词，否则跨源去重反而会降低元数据完整度。
    for key in (
        "title", "authors", "year", "abstract", "venue", "doi", "arxiv_id",
        "url", "keywords",
    ):
        if not keep.get(key) and drop.get(key):
            keep[key] = drop[key]

    # 记录一篇论文来自哪些数据源/检索分支，供后续诊断与分支配额使用。
    sources = [
        *(keep.get("_sources") or []),
        keep.get("source"),
        *(drop.get("_sources") or []),
        drop.get("source"),
    ]
    keep["_sources"] = list(dict.fromkeys(str(x) for x in sources if x))
    branches = [
        *(keep.get("_search_branches") or []),
        *(drop.get("_search_branches") or []),
    ]
    if branches:
        keep["_search_branches"] = list(dict.fromkeys(str(x) for x in branches if x))

    citation_by_source = {
        **(keep.get("citation_count_by_source") or {}),
        **(drop.get("citation_count_by_source") or {}),
    }
    if citation_by_source:
        keep["citation_count_by_source"] = citation_by_source
