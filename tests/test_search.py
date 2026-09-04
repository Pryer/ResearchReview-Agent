"""论文检索测试。

使用 mock 外部 API 进行测试，避免真实网络请求。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.agent.nodes import _select_batch_first_keywords, _select_branch_diverse_keywords
from app.schemas.paper_schema import PaperMetadata
from app.tools.search_papers import (
    filter_by_year,
    merge_search_results,
    search_papers,
    select_sources_by_language,
)


def test_english_screening_recovery_uses_only_english_sources(monkeypatch):
    from app.agent.retrieval_loop import recover_english_screening_shortfall

    calls = []

    def fake_search_papers(**kwargs):
        calls.append(kwargs)
        return [{
            "paper_id": "en-new",
            "title": "Classroom Teaching Behavior Analysis",
            "abstract": "classroom teaching behavior analysis",
            "source": "openalex",
        }]

    monkeypatch.setattr("app.tools.search_papers.search_papers", fake_search_papers)
    state = {
        "screening_report_low_pass_protection": {
            "quota_transfer_blocked": True,
        },
        "selected_scope": {"include_terms": ["classroom teaching behavior"]},
        "research_semantic_frame": {},
        "screening_protocol": {},
        "required_concepts": [["classroom teaching behavior"]],
        "topic_anchors": [["classroom teaching behavior"]],
        "search_branches": [],
        "excluded_title_terms": [],
        "topic": "classroom teaching behavior",
        "start_year": 2024,
        "end_year": 2026,
        "candidate_papers": [],
        "searched_keywords": [],
    }

    assert recover_english_screening_shortfall(state) is True
    assert calls
    assert all("cnki" not in call["sources"] for call in calls)
    assert "en-new" in {item["paper_id"] for item in state["candidate_papers"]}
    assert state["english_screening_recovery"]["attempted"] is True
    assert recover_english_screening_shortfall(state) is False


def test_branch_diverse_keyword_selection_reserves_each_branch():
    branches = [
        {"branch_type": "domain", "queries": ["domain query", "domain backup"]},
        {"branch_type": "method", "queries": ["method query"]},
        {"branch_type": "bridge", "queries": ["bridge query"]},
    ]
    selected = _select_branch_diverse_keywords(
        ["domain query", "domain backup", "method query", "bridge query", "其他主题"],
        branches,
        limit=3,
    )
    assert selected == ["domain query", "method query", "bridge query"]


def test_batch_first_selection_fires_exact_group_before_expansion():
    # 批次内容完全由关键词生成工具的 type 元数据决定，此处用课堂行为
    # 分析领域验证机制与任何具体领域词表无关。
    batches = [
        {"type": "exact", "keywords": ["课堂行为分析", "classroom behavior analysis"]},
        {"type": "broader", "keywords": ["行为分析"]},
        {"type": "variant", "keywords": ["教室行为分析"]},
    ]
    pool = [
        "课堂行为分析 研究现状",  # 未分批词（全局召回式）应排在批次之后
        "classroom behavior analysis",
        "教室行为分析",
        "课堂行为分析",
        "行为分析",
    ]

    picked = _select_batch_first_keywords(pool, batches, [], limit=4)

    # 首轮：exact 批先发，名额未满时按 broader→variant 顺序补足；
    # 未分批的全局召回式在批次耗尽前不占名额。
    assert set(picked) == {
        "课堂行为分析", "classroom behavior analysis", "行为分析", "教室行为分析",
    }
    assert "课堂行为分析 研究现状" not in picked


def test_batch_first_selection_advances_to_expansion_after_exact_exhausted():
    batches = [
        {"type": "exact", "keywords": ["课堂行为分析"]},
        {"type": "broader", "keywords": ["行为分析"]},
        {"type": "variant", "keywords": ["教室行为分析"]},
    ]
    # exact 已检索过（unseen 池中消失）→ 自动进入外扩批
    picked = _select_batch_first_keywords(
        ["行为分析", "教室行为分析"], batches, [], limit=2,
    )
    assert set(picked) == {"行为分析", "教室行为分析"}


def test_batch_first_selection_with_batches_keeps_unbatched_as_fallback():
    batches = [{"type": "exact", "keywords": ["课堂行为分析"]}]
    picked = _select_batch_first_keywords(
        ["课堂行为分析 研究现状", "课堂行为分析"],
        batches,
        [],
        2,
    )
    # exact 优先，未分批词在批次之后补足名额
    assert picked[0] == "课堂行为分析"
    assert "课堂行为分析 研究现状" in picked


def test_targeted_recovery_queries_are_reserved_before_batches():
    """定向补检索查询若排在全部批次之后，批次未跑完时会被整轮挤出。"""
    batches = [
        {"type": "exact", "keywords": ["课堂行为分析", "classroom behavior analysis"]},
        {"type": "broader", "keywords": ["行为分析"]},
    ]
    branches = [{
        "branch_type": "evidence_recovery_1",
        "constraint_level": "targeted_recovery",
        "queries": ["教师反馈 课堂互动 定向补检索"],
    }]
    pool = [
        "课堂行为分析",
        "classroom behavior analysis",
        "行为分析",
        "教师反馈 课堂互动 定向补检索",
    ]

    picked = _select_batch_first_keywords(pool, batches, branches, limit=2)

    assert "教师反馈 课堂互动 定向补检索" in picked
    assert len(picked) == 2


def test_targeted_recovery_reservation_never_takes_all_slots():
    """预留名额不超过一半，批次查询仍能进入本轮。"""
    batches = [{"type": "exact", "keywords": ["课堂行为分析"]}]
    branches = [{
        "branch_type": "evidence_recovery_1",
        "constraint_level": "targeted_recovery",
        "queries": ["定向查询一", "定向查询二", "定向查询三", "定向查询四"],
    }]
    pool = ["课堂行为分析", "定向查询一", "定向查询二", "定向查询三", "定向查询四"]

    picked = _select_batch_first_keywords(pool, batches, branches, limit=4)

    # 预留名额上限为 limit//2；其余名额仍按批次优先。
    assert picked[:2] == ["定向查询一", "定向查询二"]
    assert "课堂行为分析" in picked


def test_search_can_cancel_before_external_source_call():
    with pytest.raises(InterruptedError):
        search_papers(
            query="test",
            start_year=2024,
            end_year=2026,
            max_results=5,
            sources=["arxiv"],
            should_cancel=lambda: True,
        )


def _make_paper(**kwargs) -> PaperMetadata:
    """构造测试用 PaperMetadata。"""
    defaults = dict(
        paper_id="test:1",
        title="Test Paper",
        authors=["Author A"],
        year=2023,
        venue="CVPR",
        abstract="Test abstract",
        doi="10.1000/test",
        arxiv_id=None,
        url="https://example.com",
        pdf_url=None,
        citation_count=None,
        source="test",
    )
    defaults.update(kwargs)
    return PaperMetadata(**defaults)


class TestMergeSearchResults:
    """搜索结果合并测试。"""

    def test_deduplicate_by_doi(self):
        """相同 DOI 应去重。"""
        papers = [
            _make_paper(paper_id="a:1", doi="10.1000/same"),
            _make_paper(paper_id="a:2", doi="10.1000/same"),
        ]
        result = merge_search_results(papers)
        assert len(result) == 1

    def test_merge_complementary_info(self):
        """合并应补全缺失信息。"""
        papers = [
            _make_paper(paper_id="a:1", doi="10.1000/same", abstract=""),
            _make_paper(paper_id="a:2", doi="10.1000/same", abstract="Real abstract"),
        ]
        result = merge_search_results(papers)
        assert len(result) == 1

    def test_empty_input(self):
        assert merge_search_results([]) == []


def test_year_filter_is_strict_unless_unknown_years_are_explicitly_retained():
    papers = [
        _make_paper(paper_id="known", year=2020),
        _make_paper(paper_id="unknown", year=None),
        _make_paper(paper_id="old", year=2010),
    ]

    assert [paper.paper_id for paper in filter_by_year(papers, 2018, 2024)] == ["known"]
    assert [
        paper.paper_id
        for paper in filter_by_year(papers, 2018, 2024, retain_unknown=True)
    ] == ["known", "unknown"]


def test_source_filter_routes_by_query_language():
    configured = ["crossref", "arxiv", "semantic_scholar", "openalex", "cnki"]
    # 英文查询：CNKI 不参与
    assert select_sources_by_language("evidence synthesis", configured) == configured[:-1]
    # 中文查询：arXiv/Semantic Scholar 不参与，CNKI 参与
    assert select_sources_by_language("证据综合", configured) == [
        "crossref", "openalex", "cnki",
    ]


def test_diagnostic_outcomes_distinguish_empty_and_failures():
    from app.tools.search_papers import _classify_search_exception, _diagnostic_for_outcome
    import requests

    assert _diagnostic_for_outcome("openalex", "success_empty").status == "empty"
    assert _diagnostic_for_outcome("openalex", "success_empty").outcome == "success_empty"
    assert _diagnostic_for_outcome("openalex", "query_not_adapted").status == "skipped"
    assert _classify_search_exception(requests.Timeout("timed out")) == ("timeout", "TIMEOUT")
    assert _classify_search_exception(RuntimeError("HTTP 429 Retry-After: 2"))[0] == "rate_limited"
    assert _classify_search_exception(RuntimeError("HTTP 401 unauthorized"))[0] == "authentication_failed"


def test_merge_search_results_does_not_mutate_source_models():
    incomplete = _make_paper(paper_id="a", doi="10.1/same", abstract="")
    complete = _make_paper(paper_id="b", doi="10.1/same", abstract="complete")
    merge_search_results([incomplete, complete])
    assert incomplete.abstract == ""


@patch("app.tools.search_papers._search_arxiv")
@patch("app.tools.search_papers._search_semantic_scholar")
@patch("app.tools.search_papers._search_cnki")
def test_language_guards_skip_incompatible_sources_with_diagnostics(
    mock_cnki, mock_s2, mock_arxiv
):
    """arXiv/Semantic Scholar 不接收中文检索式，CNKI 不接收英文检索式。"""
    mock_arxiv.return_value = []
    mock_s2.return_value = []
    mock_cnki.return_value = []

    diags: list = []
    search_papers(
        query="少样本动作识别", start_year=2022, end_year=2026,
        max_results=10, sources=["arxiv", "semantic_scholar", "cnki"],
        diagnostics=diags, enable_dual_channel=False,
    )
    assert mock_arxiv.call_count == 0
    assert mock_s2.call_count == 0
    assert mock_cnki.call_count == 1
    skipped = [d for d in diags if d.status == "skipped"]
    assert [d.source for d in skipped] == ["arxiv", "semantic_scholar"]
    assert all(d.error_code == "INCOMPATIBLE_QUERY_LANGUAGE" for d in skipped)

    diags = []
    search_papers(
        query="few-shot action recognition", start_year=2022, end_year=2026,
        max_results=10, sources=["arxiv", "cnki"],
        diagnostics=diags, enable_dual_channel=False,
    )
    assert mock_cnki.call_count == 1  # 英文查询未再次触发 CNKI
    assert mock_arxiv.call_count == 1
    skipped = [d for d in diags if d.status == "skipped"]
    assert [d.source for d in skipped] == ["cnki"]


class TestSearchPapers:
    """搜索论文测试（mock）。"""

    @patch("app.tools.search_papers._search_arxiv")
    @patch("app.tools.search_papers._search_semantic_scholar")
    @patch("app.tools.search_papers._search_openalex")
    def test_search_all_sources(
        self, mock_ss, mock_s2, mock_arxiv
    ):
        """测试多源检索入口（不同 paper_id 的论文应全部保留）。"""
        mock_arxiv.return_value = [_make_paper(paper_id="arxiv:1", doi=None)]
        mock_s2.return_value = [_make_paper(paper_id="s2:1", doi=None)]
        mock_ss.return_value = [_make_paper(paper_id="openalex:1", doi=None)]

        result = search_papers(
            query="test", start_year=2020, end_year=2025,
            max_results=10, sources=["arxiv", "semantic_scholar", "openalex"],
            enable_dual_channel=False,
        )
        assert len(result) == 3

    @patch("app.tools.search_papers._search_arxiv")
    def test_search_handles_client_failure(self, mock_fn):
        """单个数据源失败不应中断整体检索。"""
        mock_fn.side_effect = RuntimeError("API down")

        result = search_papers(
            query="test", start_year=2020, end_year=2025,
            max_results=10, sources=["arxiv"],
            enable_dual_channel=False,
        )
        assert result == []
