"""论文详情补全测试。"""

from __future__ import annotations

import threading
import time

from app.agent.nodes import fetch_detail_node
from app.tools.fetch_metadata import fetch_batch_details, fetch_paper_detail


def test_fetch_arxiv_detail_uses_paper_year_range(monkeypatch):
    captured = {}

    def fake_search_arxiv(query, start_year, end_year, max_results):
        captured.update(
            {
                "query": query,
                "start_year": start_year,
                "end_year": end_year,
                "max_results": max_results,
            }
        )
        return []

    monkeypatch.setattr("app.clients.arxiv_client.search_arxiv", fake_search_arxiv)

    result = fetch_paper_detail({"paper_id": "p1", "arxiv_id": "2401.00001", "year": 2024})

    assert result["arxiv_id"] == "2401.00001"
    assert captured["start_year"] == 2023
    assert captured["end_year"] == 2025


def test_crossref_mismatch_cannot_overwrite_paper_identity(monkeypatch):
    monkeypatch.setattr(
        "app.clients.crossref_client.get_crossref_detail",
        lambda doi: {
            "paper_id": "doi:wrong",
            "title": "Tackling Climate Change on the Local Level",
            "authors": ["Wrong Author"],
            "year": 2023,
            "doi": "10.1000/wrong",
            "citation_count": 99,
            "source": "crossref",
        },
    )
    paper = {
        "paper_id": "original",
        "title": "Few-Shot Action Recognition with Temporal Alignment",
        "authors": ["Original Author"],
        "year": 2024,
        "doi": "10.1000/original",
        "source": "openalex",
    }

    result = fetch_paper_detail(paper)

    assert result["paper_id"] == "original"
    assert result["title"] == "Few-Shot Action Recognition with Temporal Alignment"
    assert result["authors"] == ["Original Author"]
    assert result["doi"] == "10.1000/original"
    assert "_metadata_mismatch" in result
    assert "_metadata_mismatch" not in paper


def test_matching_crossref_detail_corrects_aggregator_authors(monkeypatch):
    monkeypatch.setattr(
        "app.clients.crossref_client.get_crossref_detail",
        lambda doi: {
            "paper_id": "doi:canonical",
            "title": "Few-Shot Action Recognition with Temporal Alignment",
            "authors": ["Crossref Author"],
            "year": 2024,
            "venue": "CVPR",
            "doi": doi,
            "citation_count": 42,
            "source": "crossref",
        },
    )
    paper = {
        "paper_id": "original",
        "title": "Few-Shot Action Recognition with Temporal Alignment",
        "authors": ["Original Author"],
        "year": 2024,
        "venue": None,
        "doi": "10.1000/original",
        "source": "openalex",
    }

    result = fetch_paper_detail(paper)

    assert result["paper_id"] == "original"
    assert result["authors"] == ["Crossref Author"]
    assert result["source"] == "openalex"
    assert result["venue"] == "CVPR"
    assert result["citation_count"] == 42
    assert result["_metadata_corrections"]["authors"]["source"] == "crossref"
    assert paper["authors"] == ["Original Author"]


def test_complete_doi_result_still_verifies_crossref_authors(monkeypatch):
    calls = {"crossref": 0, "semantic_scholar": 0}

    def fake_crossref(_doi):
        calls["crossref"] += 1
        return None

    def fake_semantic_scholar(_identifier):
        calls["semantic_scholar"] += 1
        return None

    monkeypatch.setattr("app.clients.crossref_client.get_crossref_detail", fake_crossref)
    monkeypatch.setattr(
        "app.clients.semantic_scholar_client.get_semantic_scholar_detail",
        fake_semantic_scholar,
    )
    paper = {
        "paper_id": "s2:complete",
        "title": "Complete Paper",
        "authors": ["Author"],
        "year": 2025,
        "venue": "TestConf",
        "abstract": "A complete abstract.",
        "doi": "10.1000/complete",
        "citation_count": 0,
        "source": "semantic_scholar",
    }

    result = fetch_batch_details([paper], max_workers=1)

    assert result[0]["paper_id"] == "s2:complete"
    assert calls == {"crossref": 1, "semantic_scholar": 0}


def test_crossref_can_correct_xin_ma_to_xuejian_ma(monkeypatch):
    monkeypatch.setattr(
        "app.clients.crossref_client.get_crossref_detail",
        lambda doi: {
            "paper_id": f"doi:{doi}",
            "title": "A WAD-YOLOv8-based method for classroom student behavior detection",
            "authors": ["First Author", "Xuejian Ma"],
            "year": 2025,
            "venue": "Scientific Reports",
            "doi": doi,
            "source": "crossref",
        },
    )
    paper = {
        "paper_id": "s2:nature-paper",
        "title": "A WAD-YOLOv8-based method for classroom student behavior detection",
        "authors": ["First Author", "Xin Ma"],
        "year": 2025,
        "venue": "Scientific Reports",
        "abstract": "A complete abstract.",
        "doi": "10.1038/s41598-025-87661-w",
        "source": "semantic_scholar",
    }

    result = fetch_paper_detail(paper)

    assert result["authors"] == ["First Author", "Xuejian Ma"]


def test_parallel_batch_preserves_ranked_order(monkeypatch):
    lock = threading.Lock()
    active = 0
    peak_active = 0

    def fake_fetch(paper):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        paper["enriched"] = True
        return paper

    monkeypatch.setattr("app.tools.fetch_metadata.fetch_paper_detail", fake_fetch)
    monkeypatch.setattr(
        "app.tools.fetch_metadata.enrich_with_citation_count", lambda paper: paper
    )
    monkeypatch.setattr("app.tools.fetch_metadata.enrich_with_pdf_url", lambda paper: paper)
    papers = [{"paper_id": f"p{index}"} for index in range(8)]

    results = fetch_batch_details(papers, max_workers=4)

    assert [paper["paper_id"] for paper in results] == [f"p{index}" for index in range(8)]
    assert all(paper["enriched"] for paper in results)
    assert peak_active > 1


def test_fetch_detail_node_only_enriches_generation_pool_when_sufficient(monkeypatch):
    batch_sizes = []

    def fake_batch(papers):
        batch_sizes.append(len(papers))
        return papers

    monkeypatch.setattr("app.tools.fetch_metadata.fetch_batch_details", fake_batch)
    monkeypatch.setattr("app.tools.rank_papers.passes_topic_filter", lambda *args, **kwargs: True)
    state = {
        "ranked_papers": [{"paper_id": f"p{index}"} for index in range(120)],
        "required_reference_count": 40,
        "max_papers": 40,
        "generation_limit": 80,
        "steps": [],
        "errors": [],
    }

    fetch_detail_node(state)

    assert batch_sizes == [60]
    assert len(state["paper_details"]) == 60
    step = state["steps"][-1]
    assert step["output_data"]["skipped_unneeded"] == 60
    assert step["input_data"]["evidence_pool_target"] == 60
