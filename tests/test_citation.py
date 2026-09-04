"""引用生成与校验测试。"""

from __future__ import annotations

import pytest

from app.tools.generate_citation import (
    generate_reference,
    generate_references,
    validate_citations,
    generate_and_validate_citations,
    generate_bibtex_key,
)


def _make_paper(**kwargs) -> dict:
    defaults = {
        "paper_id": "test:1",
        "title": "Vision Transformer for Image Classification",
        "authors": ["Alice Bob", "Charlie Doe"],
        "year": 2023,
        "venue": "CVPR",
        "doi": "10.1000/test.2023",
        "url": "https://example.com",
    }
    defaults.update(kwargs)
    return defaults


class TestGenerateReference:
    @pytest.mark.parametrize("style", ["gbt7714", "apa", "ieee"])
    def test_all_text_styles(self, style):
        paper = _make_paper()
        ref = generate_reference(paper, style)
        assert isinstance(ref, str)
        assert "Vision Transformer" in ref
        assert "2023" in ref

    def test_bibtex_style(self):
        paper = _make_paper()
        ref = generate_reference(paper, "bibtex")
        assert "@article" in ref
        assert "Alice Bob and Charlie Doe" in ref

    def test_unknown_style_is_rejected(self):
        from app.core.exceptions import CitationGenerationError

        with pytest.raises(CitationGenerationError):
            generate_reference(_make_paper(), "made-up-style")

    def test_empty_paper(self):
        ref = generate_reference({}, "gbt7714")
        assert isinstance(ref, str)

    def test_cnki_page_chrome_is_removed_from_reference(self):
        paper = _make_paper(
            authors=["张三", "李四", "王五", "赵六"],
            venue="计算机技术与发展 . 2025 ,35 (12) : 58-66 查看该刊数据库收录来源",
            publication_type="journal_article",
        )
        ref = generate_reference(paper, "gbt7714")

        assert "查看该刊数据库收录来源" not in ref
        assert "et al.." not in ref

    def test_cnki_search_ui_is_not_treated_as_venue(self):
        paper = _make_paper(
            venue="检索 CNKI AI 出版来源 我的CNKI",
            publication_type="thesis",
        )
        ref = generate_reference(paper, "gbt7714")

        assert "我的CNKI" not in ref
        assert "[D]" in ref

    def test_cnki_page_noise_is_removed_from_venue(self):
        paper = _make_paper(
            venue="教育信息技术 . 2026 (Z1) : 13-17 下载 HTML阅读 CNKI AI阅读",
            publication_type="journal_article",
        )
        ref = generate_reference(paper, "gbt7714")

        assert "下载" not in ref
        assert "HTML阅读" not in ref
        assert "CNKI AI阅读" not in ref

    def test_institution_venue_keeps_full_name(self):
        """回归：省份标签正则曾把省名从机构名内部剥掉，参考文献里出现假刊名《大学》。"""
        paper = _make_paper(
            venue="浙江大学浙江省211工程院校985工程院校教育部直属院校一流大学",
            publication_type="journal_article",
        )
        ref = generate_reference(paper, "gbt7714")

        assert "浙江大学" in ref
        assert "教育部直属院校" not in ref
        assert "211工程" not in ref

    def test_ssrn_preprint_is_not_rendered_as_journal(self):
        """SSRN 是预印本平台；Crossref 的假刊名不得当期刊渲染。"""
        paper = _make_paper(
            venue="SSRN Electronic Journal",
            doi="10.2139/ssrn.5055942",
            publication_type="preprint",
        )
        ref = generate_reference(paper, "gbt7714")

        assert "[EB/OL]" in ref
        assert "SSRN," in ref
        assert "Electronic Journal" not in ref

    def test_volume_issue_tail_does_not_duplicate_the_year(self):
        """回归：CNKI 的年/卷/期/页残留在 venue 位，渲染出两个年份和错乱标点。"""
        paper = _make_paper(
            venue="智能物联技术 . 2026 ,58 (03) : 93-98 查看该刊数据库收录来源",
            year=2026,
            publication_type="journal_article",
        )
        ref = generate_reference(paper, "gbt7714")

        assert "智能物联技术, 2026." in ref
        assert "93-98" not in ref
        assert ref.count("2026") == 1

    def test_arxiv_placeholder_venue_is_suppressed_when_doi_resolves(self):
        """回归：venue 硬编码为 "arXiv" 的正式论文渲染成 "[C]. arXiv" 假刊名。

        arxiv_client 对每条记录硬编码 venue="arXiv"；正式发表后回填出版方 DOI。
        此时 "arXiv" 是已知错误刊名，宁可略去也不断言该文发表在 arXiv 期刊上。
        """
        cvpr = _make_paper(
            venue="arXiv",
            doi="10.1109/CVPR52688.2022.01932",
            source="arxiv",
            arxiv_id="2201.00000",
            publication_type="conference_paper",
        )
        ref = generate_reference(cvpr, "gbt7714")
        assert "[C]" in ref
        assert "arXiv" not in ref
        assert "DOI: 10.1109/CVPR52688.2022.01932" in ref

        ijcv = _make_paper(
            venue="arXiv",
            doi="10.1007/s11263-023-01917-4",
            source="arxiv",
            arxiv_id="2303.00001",
            publication_type="journal_article",
        )
        ref_j = generate_reference(ijcv, "gbt7714")
        assert "[J]" in ref_j
        assert "arXiv" not in ref_j

    def test_genuine_arxiv_preprint_keeps_arxiv_venue(self):
        """真预印本（arXiv 自有 DOI / 无出版方 DOI）仍保留 arXiv 出处。"""
        paper = _make_paper(
            venue="arXiv",
            doi="10.48550/arXiv.2411.11335",
            source="arxiv",
            arxiv_id="2411.11335",
            publication_type="preprint",
        )
        ref = generate_reference(paper, "gbt7714")
        assert "[EB/OL]" in ref
        assert "arXiv," in ref

        no_doi = _make_paper(
            venue="arXiv", doi="", source="arxiv", arxiv_id="2401.00001",
            publication_type="preprint",
        )
        assert "arXiv," in generate_reference(no_doi, "gbt7714")


class TestGenerateReferences:
    def test_batch(self):
        papers = [_make_paper(paper_id=str(i)) for i in range(3)]
        refs = generate_references(papers, "gbt7714")
        assert len(refs) == 3

    def test_empty(self):
        assert generate_references([], "gbt7714") == []

    def test_duplicate_bibtex_keys_are_disambiguated(self):
        papers = [
            _make_paper(paper_id="p1", authors=["Alice Bob"], year=2023),
            _make_paper(paper_id="p2", authors=["Alice Bob"], year=2023),
        ]
        refs = generate_references(papers, "bibtex")
        assert "@article{Bob2023," in refs[0]
        assert "@article{Bob2023a," in refs[1]


class TestValidateCitations:
    def test_valid_citations(self):
        review = "正如 [test:1] 所述，该方法效果显著。"
        references = ["[1] Ref one"]
        cards = [_make_paper(paper_id="test:1")]
        result = validate_citations(review, references, cards)
        assert isinstance(result["valid"], bool)

    def test_missing_citations_detected(self):
        review = "引用 [nonexistent:paper] 和 [test:1]"
        references = ["ref1", "ref2"]
        cards = [_make_paper(paper_id="test:1")]
        result = validate_citations(review, references, cards)
        assert "nonexistent:paper" in result["missing_citations"]

    def test_unused_references_detected(self):
        review = "仅引用 [1]"
        references = ["ref1", "ref2"]
        result = validate_citations(review, references)
        assert len(result["unused_references"]) >= 1

    def test_empty_review(self):
        result = validate_citations("", [], [])
        assert result["valid"] is True

    def test_duplicate_doi_and_incomplete_metadata_fail_quality_check(self):
        papers = [
            _make_paper(paper_id="p1", doi="10.1/same", source="crossref"),
            _make_paper(paper_id="p2", doi="10.1/same", source="openalex", authors=[]),
        ]
        result = validate_citations(
            "研究包括 [p1] 和 [p2]。",
            ["ref1", "ref2"],
            papers,
            reference_papers=papers,
        )
        assert result["valid"] is False
        assert result["duplicate_dois"] == ["10.1/same"]
        assert result["incomplete_metadata"] == ["p2"]

    def test_generate_only_references_cited_in_review(self):
        papers = [
            _make_paper(paper_id="p1", title="Cited Paper"),
            _make_paper(paper_id="p2", title="Unused Paper"),
        ]

        result = generate_and_validate_citations(
            review_text="相关研究包括 [p1]。",
            paper_cards=papers,
            citation_style="gbt7714",
        )

        assert len(result["references"]) == 1
        assert "Cited Paper" in result["references"][0]
        assert result["validation"]["unused_references"] == []
        assert result["rendered_text"] == "相关研究包括 [1]。"
        assert result["citation_map"] == {"p1": 1}

    def test_internal_database_ids_are_not_exposed_in_final_text(self):
        papers = [
            _make_paper(
                paper_id="s2:abcdef",
                title="Complete Paper Title",
                doi="10.1000/first",
            ),
            _make_paper(
                paper_id="openalex:W123",
                title="Another Complete Title",
                doi="10.1000/second",
            ),
        ]
        result = generate_and_validate_citations(
            review_text="已有研究 [s2:abcdef] 与 [openalex:W123]。",
            paper_cards=papers,
            citation_style="gbt7714",
        )
        assert result["rendered_text"] == "已有研究 [1] 与 [2]。"
        assert "s2:" not in result["rendered_text"]
        assert "openalex:" not in result["rendered_text"]

    def test_composite_internal_ids_are_split_validated_and_rendered(self):
        papers = [
            _make_paper(
                paper_id="s2:abcdef",
                title="Complete Paper Title",
                doi="10.1000/first",
            ),
            _make_paper(
                paper_id="openalex:W123",
                title="Another Complete Title",
                doi="10.1000/second",
            ),
        ]
        result = generate_and_validate_citations(
            review_text="多项研究形成共同判断[s2:abcdef, openalex:W123]。",
            paper_cards=papers,
            citation_style="gbt7714",
        )

        assert result["validation"]["valid"] is True
        assert result["validation"]["cited_ids"] == [
            "openalex:W123", "s2:abcdef",
        ]
        assert len(result["references"]) == 2
        assert result["rendered_text"] == "多项研究形成共同判断[1, 2]。"
        assert "s2:" not in result["rendered_text"]
        assert "openalex:" not in result["rendered_text"]

    def test_fullwidth_internal_ids_are_normalized_before_validation(self):
        papers = [
            _make_paper(
                paper_id="s2:abcdef",
                title="Complete Paper Title",
                doi="10.1000/first",
            ),
            _make_paper(
                paper_id="openalex:W123",
                title="Another Complete Title",
                doi="10.1000/second",
            ),
        ]

        result = generate_and_validate_citations(
            review_text="已有研究〔s2:abcdef〕与〔openalex:W123〕形成互补证据。",
            paper_cards=papers,
            citation_style="gbt7714",
        )

        assert result["validation"]["valid"] is True
        assert result["validation"]["cited_ids"] == [
            "openalex:W123", "s2:abcdef",
        ]
        assert result["rendered_text"] == "已有研究[1]与[2]形成互补证据。"
        assert len(result["references"]) == 2

    def test_unclosed_language_marker_cannot_swallow_following_section(self):
        papers = [
            _make_paper(
                paper_id="s2:abcdef",
                title="Complete Paper Title",
                doi="10.1000/first",
            ),
        ]
        result = generate_and_validate_citations(
            review_text=(
                "## 第一节\n\n本节结论[cn\n\n"
                "## 下一节\n\n后续研究提供了证据[s2:abcdef]。"
            ),
            paper_cards=papers,
            citation_style="gbt7714",
        )

        assert "[cn" not in result["rendered_text"]
        assert "## 下一节" in result["rendered_text"]
        assert "后续研究提供了证据[1]。" in result["rendered_text"]
        assert result["validation"]["missing_citations"] == []

    def test_misspelled_fullwidth_source_id_is_reported_not_silently_rendered(self):
        papers = [_make_paper(paper_id="openalex:W123")]

        result = generate_and_validate_citations(
            review_text="方法研究〔openex:W123〕。",
            paper_cards=papers,
            citation_style="gbt7714",
        )

        assert result["validation"]["valid"] is False
        assert result["validation"]["missing_citations"] == ["openex:W123"]
        assert "openex:W123" in result["rendered_text"]


class TestCitationOrdering:
    """GB/T 7714 顺序编码制：编号按正文首现顺序分配。"""

    def test_reference_order_follows_first_appearance_not_card_order(self):
        papers = [
            _make_paper(paper_id="p_a", title="Alpha Paper", doi=None),
            _make_paper(paper_id="p_b", title="Beta Paper", doi=None),
        ]
        result = generate_and_validate_citations(
            review_text="先看 [p_b] 的结论，再对比 [p_a]。",
            paper_cards=papers,
            citation_style="gbt7714",
        )
        assert result["rendered_text"] == "先看 [1] 的结论，再对比 [2]。"
        assert "Beta Paper" in result["references"][0]
        assert "Alpha Paper" in result["references"][1]
        assert result["citation_map"] == {"p_b": 1, "p_a": 2}
        assert result["validation"]["valid"] is True

    def test_numeric_citations_resolve_and_renumber_by_appearance(self):
        papers = [
            _make_paper(paper_id="p1", title="One"),
            _make_paper(paper_id="p2", title="Two"),
            _make_paper(paper_id="p3", title="Three"),
        ]
        result = generate_and_validate_citations(
            review_text="结论见 [3]，与 [1] 相互印证。",
            paper_cards=papers,
            citation_style="gbt7714",
        )
        # [3] 先出现 → 重编号为 1；[1] → 2；参考文献表同步按首现排序
        assert result["rendered_text"] == "结论见 [1]，与 [2] 相互印证。"
        assert "Three" in result["references"][0]
        assert "One" in result["references"][1]

    def test_out_of_range_numeric_citation_reported_missing(self):
        papers = [
            _make_paper(paper_id="p1", title="One"),
            _make_paper(paper_id="p2", title="Two"),
        ]
        result = generate_and_validate_citations(
            review_text="见 [1] 与 [99]。",
            paper_cards=papers,
            citation_style="gbt7714",
        )
        assert result["validation"]["valid"] is False
        assert "99" in result["validation"]["missing_citations"]

    def test_out_of_range_numeric_citation_detected_by_validator(self):
        cards = [_make_paper(paper_id="p1")]
        result = validate_citations("见 [5]。", ["ref"], cards)
        assert "5" in result["missing_citations"]


class TestGenerateBibtexKey:
    def test_standard_key(self):
        paper = _make_paper(authors=["Alice Bob"], year=2023)
        key = generate_bibtex_key(paper)
        assert key == "Bob2023"

    def test_no_authors(self):
        key = generate_bibtex_key({}, )
        assert isinstance(key, str)


class TestStableIdentityForSourceScopedIds:
    """无 DOI 但可回溯到源记录的文献不得被判"缺少稳定标识"。"""

    @staticmethod
    def _cnki_paper() -> dict:
        return {
            # paper_id 取自 CNKI 详情页 URL 的 v= 参数，可唯一定位该记录
            "paper_id": "cnki:tPTW3hFh4gcUtYAMm0jjjoq36vfODvttUqu4FOLuhPs",
            "title": "基于人工智能的智慧课堂行为分析系统研究",
            "authors": ["孟亚玲", "仇萍"],
            "year": 2026,
            "venue": "教育信息技术",
            "doi": None,
            "source": "cnki",
        }

    def test_cnki_paper_without_doi_is_treated_as_verified(self):
        """回归实测缺陷：中文期刊普遍不注册 DOI，此前 9 条被误报未核验。"""
        paper = self._cnki_paper()
        result = validate_citations(
            "结论见[cnki:tPTW3hFh4gcUtYAMm0jjjoq36vfODvttUqu4FOLuhPs]。",
            ["[1] 孟亚玲, 仇萍. 基于人工智能的智慧课堂行为分析系统研究[J]. 教育信息技术, 2026."],
            paper_cards=[paper],
            reference_papers=[paper],
        )

        assert result["unverified_metadata"] == []
        assert result["metadata_quality_valid"] is True

    def test_bare_id_without_source_prefix_stays_unverified(self):
        """没有数据源前缀也没有 DOI 的裸标识仍判未核验。"""
        paper = {**self._cnki_paper(), "paper_id": "local-42", "source": "manual"}
        result = validate_citations(
            "结论见[local-42]。",
            ["[1] 某作者. 某标题[J]. 某刊, 2026."],
            paper_cards=[paper],
            reference_papers=[paper],
        )

        assert result["unverified_metadata"] == ["local-42"]

    def test_empty_source_still_reported_even_with_prefixed_id(self):
        """source 缺失是独立缺陷：即便标识可溯源也照实报告。"""
        paper = {**self._cnki_paper(), "source": ""}
        result = validate_citations(
            "结论见[cnki:tPTW3hFh4gcUtYAMm0jjjoq36vfODvttUqu4FOLuhPs]。",
            ["[1] 孟亚玲. 标题[J]. 教育信息技术, 2026."],
            paper_cards=[paper],
            reference_papers=[paper],
        )

        assert result["unverified_metadata"] == [paper["paper_id"]]


class TestPublicationStatusAndDoiNormalization:
    """出版状态与 DOI 归一化必须进入最终参考文献验收。"""

    def _paper(self, **kwargs) -> dict:
        base = {
            "paper_id": "doi:10.1000/a",
            "title": "课堂行为分析研究",
            "authors": ["张三"],
            "year": 2025,
            "venue": "现代教育技术",
            "doi": "10.1000/a",
            "source": "crossref",
            "publication_status": "published",
        }
        base.update(kwargs)
        return base

    def test_doi_url_and_prefix_forms_are_detected_as_duplicates(self):
        first = self._paper(paper_id="doi:10.1000/a", doi="https://doi.org/10.1000/A")
        second = self._paper(paper_id="doi:10.1000/a-dup", doi="doi:10.1000/a")
        result = validate_citations(
            "结论见[doi:10.1000/a]和[doi:10.1000/a-dup]。",
            ["[1] ref one", "[2] ref two"],
            paper_cards=[first, second],
            reference_papers=[first, second],
        )

        assert result["duplicate_dois"] == ["10.1000/a"]
        assert result["metadata_quality_valid"] is False

    def test_unknown_publication_status_is_reported_without_blocking(self):
        paper = self._paper(publication_status="unknown")
        result = validate_citations(
            "结论见[doi:10.1000/a]。",
            ["[1] ref one"],
            paper_cards=[paper],
            reference_papers=[paper],
        )

        assert result["unknown_publication_status"] == ["doi:10.1000/a"]
        assert result["structurally_valid"] is True
        assert any("出版状态未确认" in item for item in result["suggestions"])

    def test_retracted_reference_blocks_delivery(self):
        paper = self._paper(publication_status="retracted")
        result = validate_citations(
            "结论见[doi:10.1000/a]。",
            ["[1] ref one"],
            paper_cards=[paper],
            reference_papers=[paper],
        )

        assert result["retracted_references"] == ["doi:10.1000/a"]
        assert result["structurally_valid"] is False
        assert result["metadata_quality_valid"] is False
