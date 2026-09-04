# -*- coding: utf-8 -*-
"""双通道混合检索（Dual-Channel Retrieval）单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.schemas.paper_schema import PaperMetadata, SourceDiagnostic
from app.tools.search_papers import search_papers, _RELEVANCE_RATIO, _RECENCY_RATIO


def _make_paper(title: str, arxiv_id: str | None = None, doi: str | None = None, source: str = "arxiv") -> PaperMetadata:
    return PaperMetadata(
        paper_id=f"test:{title[:20]}",
        title=title,
        authors=["Author A"],
        year=2025,
        source=source,
        arxiv_id=arxiv_id,
        doi=doi,
    )


# --- 固定回调：根据 sort_by 返回不同的论文集 ---
def _mock_arxiv(query, start_year, end_year, max_results, sort_by="relevance"):
    if sort_by == "relevance":
        return [
            _make_paper("Classic Paper A", arxiv_id="2401.00001"),
            _make_paper("Classic Paper B", arxiv_id="2401.00002"),
        ]
    else:
        return [
            _make_paper("Cutting-edge Paper X", arxiv_id="2512.11111"),
            _make_paper("Cutting-edge Paper Y", arxiv_id="2512.22222"),
            # 这篇与 relevance 通道重复，应被去重
            _make_paper("Classic Paper A", arxiv_id="2401.00001"),
        ]


def _mock_empty(query, start_year, end_year, max_results, sort_by="relevance"):
    return []


class TestDualChannelRetrieval:
    """双通道混合检索的核心逻辑测试。"""

    @patch("app.tools.search_papers._search_arxiv", side_effect=_mock_arxiv)
    @patch("app.tools.search_papers._search_semantic_scholar", side_effect=_mock_empty)
    @patch("app.tools.search_papers._search_openalex", side_effect=_mock_empty)
    @patch("app.tools.search_papers._search_crossref", side_effect=_mock_empty)
    def test_dual_channel_merges_relevance_and_date(self, *mocks):
        """双通道模式应同时返回 relevance 和 date 通道的论文并去重。"""
        papers = search_papers(
            query="few-shot action recognition",
            start_year=2022,
            end_year=2026,
            max_results=20,
            sources=["arxiv"],
            enable_dual_channel=True,
        )
        titles = {p.title for p in papers}
        # 应包含 relevance 通道的经典论文
        assert "Classic Paper A" in titles
        assert "Classic Paper B" in titles
        # 应包含 date 通道的前沿论文
        assert "Cutting-edge Paper X" in titles
        assert "Cutting-edge Paper Y" in titles
        # 去重后总数应为 4（Classic A 出现两次，去重为 1）
        assert len(papers) == 4

    @patch("app.tools.search_papers._search_arxiv", side_effect=_mock_arxiv)
    @patch("app.tools.search_papers._search_semantic_scholar", side_effect=_mock_empty)
    @patch("app.tools.search_papers._search_openalex", side_effect=_mock_empty)
    @patch("app.tools.search_papers._search_crossref", side_effect=_mock_empty)
    def test_dual_channel_calls_both_sort_modes(self, *mocks):
        """双通道模式应对同一数据源发两次请求（relevance + date）。"""
        search_papers(
            query="test",
            start_year=2022,
            end_year=2026,
            max_results=20,
            sources=["arxiv"],
            enable_dual_channel=True,
        )
        arxiv_mock = mocks[3]  # 最后一个 patch 是 _search_arxiv
        assert arxiv_mock.call_count == 2
        sort_by_args = [call.kwargs.get("sort_by") or call.args[4] for call in arxiv_mock.call_args_list]
        # 不用检查 kwargs，直接检查位置参数
        calls_sort = []
        for call in arxiv_mock.call_args_list:
            # sort_by 是第 5 个参数（index 4）
            if len(call.args) > 4:
                calls_sort.append(call.args[4])
            elif "sort_by" in call.kwargs:
                calls_sort.append(call.kwargs["sort_by"])
        assert "relevance" in calls_sort
        assert "date" in calls_sort

    @patch("app.tools.search_papers._search_arxiv", side_effect=_mock_arxiv)
    @patch("app.tools.search_papers._search_semantic_scholar", side_effect=_mock_empty)
    @patch("app.tools.search_papers._search_openalex", side_effect=_mock_empty)
    @patch("app.tools.search_papers._search_crossref", side_effect=_mock_empty)
    def test_single_channel_fallback(self, *mocks):
        """关闭双通道时，应只发一次 relevance 请求。"""
        papers = search_papers(
            query="test",
            start_year=2022,
            end_year=2026,
            max_results=20,
            sources=["arxiv"],
            enable_dual_channel=False,
        )
        arxiv_mock = mocks[3]
        assert arxiv_mock.call_count == 1

    @patch("app.tools.search_papers._search_arxiv", side_effect=_mock_arxiv)
    @patch("app.tools.search_papers._search_semantic_scholar", side_effect=_mock_empty)
    @patch("app.tools.search_papers._search_openalex", side_effect=_mock_empty)
    @patch("app.tools.search_papers._search_crossref", side_effect=_mock_empty)
    def test_diagnostics_aggregate_dual_channel(self, *mocks):
        """双通道模式的诊断信息应合并同源双通道的返回计数。"""
        diags: list[SourceDiagnostic] = []
        search_papers(
            query="test",
            start_year=2022,
            end_year=2026,
            max_results=20,
            sources=["arxiv"],
            diagnostics=diags,
            enable_dual_channel=True,
        )
        arxiv_diag = [d for d in diags if d.source == "arxiv"]
        assert len(arxiv_diag) == 1
        # relevance 返回 2 篇，date 返回 3 篇（含 1 篇重复），总计 5
        assert arxiv_diag[0].returned_count == 5
        assert arxiv_diag[0].status == "success"

    def test_ratio_constants(self):
        """双通道配额比例常量应合理。"""
        assert 0.5 <= _RELEVANCE_RATIO <= 0.9
        assert 0.1 <= _RECENCY_RATIO <= 0.5
        assert abs(_RELEVANCE_RATIO + _RECENCY_RATIO - 1.0) < 0.01
