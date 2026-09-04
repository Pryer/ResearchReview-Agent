"""论文详情补全工具。

根据 DOI / arXiv ID / Semantic Scholar ID 补全引用量、PDF 链接等元数据。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, List

from app.core.config import get_settings
from app.core.logger import get_logger
from app.utils.date_utils import current_year

logger = get_logger(__name__)


IDENTITY_FIELDS = {
    "paper_id", "title", "authors", "year", "abstract",
    "doi", "arxiv_id", "source",
}


def _has_core_metadata(paper: Dict[str, Any]) -> bool:
    """判断检索结果是否已足够支持排序、引用和摘要级 Paper Card。"""
    return bool(
        paper.get("title")
        and paper.get("authors")
        and paper.get("year")
        and paper.get("venue")
        and paper.get("abstract")
    )


def _merge_enrichment(paper: Dict[str, Any], detail: Dict[str, Any]) -> Dict[str, Any]:
    """合并外部元数据，并用 DOI 完全匹配的 Crossref 记录校正作者。"""
    original_title = str(paper.get("title") or "")
    detail_title = str(detail.get("title") or "")
    if original_title and detail_title:
        from app.tools.rank_papers import title_similarity

        similarity = title_similarity(original_title, detail_title)
        if similarity < 0.65:
            paper["_metadata_mismatch"] = {
                "source": detail.get("source"),
                "returned_title": detail_title,
                "title_similarity": round(similarity, 3),
            }
            logger.warning(
                "Rejected metadata mismatch for %s: %s",
                original_title,
                detail_title,
            )
            return paper

    original_doi = str(paper.get("doi") or "").strip().lower()
    detail_doi = str(detail.get("doi") or "").strip().lower()
    if original_doi and detail_doi and original_doi != detail_doi:
        paper["_metadata_mismatch"] = {
            "source": detail.get("source"),
            "returned_doi": detail_doi,
            "expected_doi": original_doi,
        }
        logger.warning("Rejected DOI metadata mismatch for %s", original_doi)
        return paper

    authoritative_doi_match = bool(
        original_doi
        and detail_doi == original_doi
        and str(detail.get("source") or "").lower() == "crossref"
    )
    for key, value in detail.items():
        if value in (None, "", [], {}):
            continue
        if key in IDENTITY_FIELDS:
            # paper_id / DOI / source 仍保持本地稳定身份；但 DOI 完全一致时，
            # Crossref 的作者列表比聚合检索源更适合用于正式参考文献。
            if key == "authors" and authoritative_doi_match:
                if paper.get("authors") != value:
                    paper["_metadata_corrections"] = {
                        **(paper.get("_metadata_corrections") or {}),
                        "authors": {
                            "from": list(paper.get("authors") or []),
                            "to": list(value),
                            "source": "crossref",
                        },
                    }
                paper["authors"] = list(value)
            continue
        if key == "citation_count":
            # 新策略：记录按源分别存储，citation_count 保持为 max 聚合
            detail_source = detail.get("source", "unknown")
            if "citation_count_by_source" not in paper:
                paper["citation_count_by_source"] = {}
            current_by_source = paper.get("citation_count_by_source") or {}
            current_by_source[detail_source] = max(
                current_by_source.get(detail_source, 0), value or 0
            )
            paper["citation_count_by_source"] = current_by_source
            # citation_count 保持为所有源的 max
            paper["citation_count"] = max(current_by_source.values())
        elif not paper.get(key):
            paper[key] = value
    return paper


def fetch_paper_detail(paper: Dict[str, Any]) -> Dict[str, Any]:
    """根据已有标识补全论文详情。

    优先级：DOI (Crossref) > arXiv ID > Semantic Scholar。
    """
    paper = deepcopy(paper)
    doi = paper.get("doi")
    arxiv_id = paper.get("arxiv_id")

    # DOI 是正式引用的稳定身份。只要论文不是 Crossref 自身返回且不是
    # 知网内部 DOI，就核验一次 Crossref，避免聚合源中的作者拼写错误进入参考文献。
    doi_text = str(doi or "").lower()
    should_verify_doi = bool(
        doi
        and paper.get("source") != "crossref"
        and "d.cnki." not in doi_text
        and ".cnki." not in doi_text
    )
    if doi and (should_verify_doi or not _has_core_metadata(paper)):
        try:
            from app.clients.crossref_client import get_crossref_detail
            detail = get_crossref_detail(doi)
            if detail:
                paper = _merge_enrichment(paper, detail)
                return paper
        except Exception as e:
            logger.warning("Crossref detail fetch failed: %s", e)

    if arxiv_id:
        try:
            from app.clients.arxiv_client import search_arxiv
            # 用 arXiv ID 反查获取更完整元数据
            year = paper.get("year") or current_year()
            results = search_arxiv(arxiv_id, year - 1, year + 1, 1)
            if results:
                paper["arxiv_id"] = arxiv_id
                if not paper.get("pdf_url"):
                    paper["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}"
                    paper["is_open_access"] = True
        except Exception as e:
            logger.warning("arXiv detail fetch failed: %s", e)

    # 确保有 pdf_url
    if not paper.get("pdf_url") and arxiv_id:
        paper["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}"
        paper["is_open_access"] = True

    return paper


def enrich_with_citation_count(paper: Dict[str, Any]) -> Dict[str, Any]:
    """通过 Semantic Scholar 补充引用量。"""
    paper = deepcopy(paper)
    # Semantic Scholar / OpenAlex 检索结果一般已包含引用量，0 也是有效值。
    if paper.get("citation_count") is not None:
        return paper
    # 知网论文的 DOI 是知网内部前缀（10.27xxx/d.cnki.xxx），SS 查不到，直接跳过。
    # 用 source 判断最稳妥（覆盖所有知网 DOI 前缀，零漏判）。
    if paper.get("source") == "cnki":
        return paper
    doi = paper.get("doi")
    # 兜底：万一 source 没标 cnki，但 DOI 含知网特征子串，也跳过
    if doi and ("d.cnki." in doi or ".cnki." in doi):
        return paper
    try:
        from app.clients.semantic_scholar_client import get_semantic_scholar_detail
        if doi:
            detail = get_semantic_scholar_detail(f"DOI:{doi}")
            if detail and detail.get("citation_count") is not None:
                # 记录 SS 来源的引用量
                if "citation_count_by_source" not in paper:
                    paper["citation_count_by_source"] = {}
                current_by_source = paper.get("citation_count_by_source") or {}
                current_by_source["semantic_scholar"] = detail["citation_count"]
                paper["citation_count_by_source"] = current_by_source
                # citation_count 更新为所有源的 max
                paper["citation_count"] = max(current_by_source.values())
            if detail and detail.get("open_access_pdf") and not paper.get("pdf_url"):
                paper["pdf_url"] = detail["open_access_pdf"]
                paper["is_open_access"] = True
    except Exception as e:
        logger.debug("Citation enrichment failed: %s", e)
    return paper


def enrich_with_pdf_url(paper: Dict[str, Any]) -> Dict[str, Any]:
    """补充开放获取 PDF 链接。"""
    paper = deepcopy(paper)
    if paper.get("pdf_url"):
        return paper
    # 仅对来自 Semantic Scholar 的论文尝试用 paper_id 查 SS 详情
    # SS 只认自家原始 ID（不带 s2: 前缀）或 DOI
    if paper.get("source") != "semantic_scholar":
        return paper
    # S2 搜索接口与详情接口请求的是同一组字段。若摘要和引用量已经存在，
    # 缺少 pdf_url 通常表示搜索结果已确认没有开放 PDF，无需重复查询。
    if paper.get("abstract") and paper.get("citation_count") is not None:
        return paper
    try:
        from app.clients.semantic_scholar_client import get_semantic_scholar_detail
        paper_id = paper.get("paper_id", "")
        # 去掉可能的 s2: 前缀
        if paper_id.startswith("s2:"):
            paper_id = paper_id[3:]
        detail = get_semantic_scholar_detail(paper_id)
        if detail and detail.get("open_access_pdf"):
            paper["pdf_url"] = detail["open_access_pdf"]
            paper["is_open_access"] = True
    except Exception:
        pass
    return paper


def _enrich_one_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    """补全单篇论文；任一外部数据源失败时保留已有元数据。"""
    try:
        paper = fetch_paper_detail(paper)
        paper = enrich_with_citation_count(paper)
        paper = enrich_with_pdf_url(paper)
    except Exception as e:
        logger.warning("Failed to fetch detail for %s: %s", paper.get("paper_id"), e)
    return paper


def fetch_batch_details(
    papers: List[Dict[str, Any]],
    max_workers: int | None = None,
) -> List[Dict[str, Any]]:
    """并发补全论文详情，同时保持输入排序和单篇失败降级语义。"""
    if not papers:
        return []

    settings = get_settings()
    configured_workers = max_workers or settings.detail_fetch_max_workers
    worker_count = max(1, min(int(configured_workers), len(papers)))
    inputs = [deepcopy(paper) for paper in papers]

    if worker_count == 1:
        return [_enrich_one_paper(paper) for paper in inputs]

    results: List[Dict[str, Any] | None] = [None] * len(inputs)
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="paper-detail",
    ) as executor:
        future_to_index = {
            executor.submit(_enrich_one_paper, paper): index
            for index, paper in enumerate(inputs)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:  # pragma: no cover - _enrich_one_paper 已负责降级
                logger.warning(
                    "Unexpected detail worker failure for %s: %s",
                    inputs[index].get("paper_id"),
                    e,
                )
                results[index] = inputs[index]

    return [result if result is not None else inputs[index] for index, result in enumerate(results)]
