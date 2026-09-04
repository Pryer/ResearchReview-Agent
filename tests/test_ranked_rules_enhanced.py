# -*- coding: utf-8 -*-
"""增强型学术质量分层与标题/摘要相关性打分测试。"""

from __future__ import annotations

import pytest

from app.tools.rank_papers import (
    compute_quality_score,
    compute_relevance_score,
    rank_papers,
)
from app.tools.venue_tiers import classify_venue_tier as _classify_venue_tier


class TestVenueTieringAndQualityScore:
    """测试学术质量分层与时间/引用评分行为。"""

    def test_citation_count_does_not_change_quality_score(self):
        paper = {
            "title": "Same Paper", "venue": "CVPR", "abstract": "evidence",
            "authors": ["Author"], "year": 2026, "doi": "10.1/x",
        }
        low = compute_quality_score({**paper, "citation_count": 0}, current_year=2026)
        high = compute_quality_score({**paper, "citation_count": 100000}, current_year=2026)
        assert low == high

    def test_year_diversity_uses_explicit_request_window(self):
        papers = [
            {"paper_id": f"y{year}", "title": "Topic", "abstract": "Topic", "year": year}
            for year in [2020, 2021, 2022, 2023, 2024]
        ]
        ranked = rank_papers(
            papers, topic="Topic", top_k=3, start_year=2021, end_year=2023,
        )
        assert {paper["year"] for paper in ranked} == {2021, 2022, 2023}


    def test_top_tier_venues_classified_correctly(self):
        top_venues = [
            "IEEE Transactions on Pattern Analysis and Machine Intelligence",
            "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
            "International Conference on Computer Vision (ICCV)",
            "European Conference on Computer Vision (ECCV)",
            "International Journal of Computer Vision",
            "IEEE Transactions on Circuits and Systems for Video Technology",
            "Proceedings of the AAAI Conference on Artificial Intelligence",
            "NeurIPS",
            "ICML",
            "ACM Multimedia",
            "计算机学报",
            "软件学报",
        ]
        for v in top_venues:
            tier, score = _classify_venue_tier(v)
            assert tier == "top_tier_published", f"Failed for {v}"
            assert score == 0.35

    def test_ambiguous_journal_words_require_single_token_venue(self):
        """回归：Science/Nature/Cell 是普通英文词，含领域实义词的名称
        （Computer Science、Stem Cell Research）不得借单 token 命中被
        误判为顶刊 +0.35；裸刊名及带出版地注释的仍算顶刊。"""
        # 误判源：这些名称都含 nature/science/cell token，但不是顶刊本身
        for v in [
            "Computer Science",
            "Stem Cell Research",
            "Cell Biology and Functional Genomics",
            "Food Science and Technology",
            "Science Education",
            "Nature Conservation",
        ]:
            tier, score = _classify_venue_tier(v)
            assert tier == "standard_published", f"Failed for {v}"
            assert score == 0.25

        # 裸刊名 / 出版地结构词伴随：仍是顶刊
        for v in [
            "Science",
            "Nature",
            "Cell",
            "Science (New York, N.Y.)",
            "Nature (London)",
            # Lancet 家族与子刊由短语表覆盖，不受影响
            "Lancet",
            "The Lancet",
            "Nature Communications",
            "Science Advances",
        ]:
            tier, score = _classify_venue_tier(v)
            assert tier == "top_tier_published", f"Failed for {v}"
            assert score == 0.35

    def test_preprint_venues_classified_correctly(self):
        preprints = [
            ("arXiv", "arxiv"),
            ("bioRxiv", "biorxiv"),
            ("SSRN Electronic Journal", "crossref"),
            ("Research Square", "crossref"),
        ]
        for v, src in preprints:
            tier, score = _classify_venue_tier(v, source=src)
            assert tier == "preprint", f"Failed for {v}"
            assert score == 0.10

    def test_standard_published_venues_classified_correctly(self):
        published = [
            "IEEE International Conference on Image Processing (ICIP)",
            "Neurocomputing",
            "Applied Sciences",
            "2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)",
            "Future Internet",
        ]
        for v in published:
            tier, score = _classify_venue_tier(v)
            assert tier == "standard_published", f"Failed for {v}"
            assert score == 0.25


class TestDocumentFormPredicates:
    """文献形态判据：预印本 / 会议 / 学位论文。与分层打分共用同一份词表。"""

    def test_arxiv_id_alone_does_not_make_a_preprint(self):
        """正式论文普遍同时挂着 arXiv 预印本；有出版方 DOI 时不算预印本。"""
        from app.tools.venue_tiers import is_preprint_record

        assert is_preprint_record(
            venue="Computer Vision and Pattern Recognition",
            doi="10.1109/CVPR52729.2023.01727", source="s2", arxiv_id="2304.00415",
        ) is False
        # 平台自有 DOI 前缀是充分条件
        assert is_preprint_record(
            venue="arXiv.org", doi="10.48550/arXiv.2411.11335",
            source="arxiv", arxiv_id="2411.11335",
        ) is True
        assert is_preprint_record(
            venue="SSRN Electronic Journal", doi="10.2139/ssrn.5055942", source="crossref",
        ) is True
        # 无 DOI 的 arXiv 条目仍是预印本
        assert is_preprint_record(source="arxiv", arxiv_id="2401.00001") is True

    def test_platform_venue_does_not_override_publisher_doi(self):
        """回归：arxiv_client 对每条记录硬编码 venue="arXiv"，平台 venue 曾短路判定。

        上一版只在 arxiv_id 分支加了出版方 DOI 守卫，venue 分支仍排在最前，
        于是 arXiv 源的 CVPR / ICCV / AAAI / IJCV / TCSVT 正式论文照旧被标
        [EB/OL]（真实运行的 60 条参考文献里错了 7 条）。
        """
        from app.tools.venue_tiers import is_preprint_record

        for doi in [
            "10.1007/s11263-023-01917-4",        # IJCV
            "10.1109/CVPR52688.2022.01932",      # CVPR
            "10.1109/ICCV51070.2023.00963",      # ICCV
            "10.1609/AAAI.V37I3.25403",          # AAAI
            "10.1109/TCSVT.2023.3287201",        # TCSVT
            "10.1016/j.knosys.2024.112539",      # Knowledge-Based Systems
        ]:
            assert is_preprint_record(
                venue="arXiv", doi=doi, source="arxiv", arxiv_id="2304.00415",
            ) is False, f"Failed for {doi}"

        # 平台自有 DOI 仍压过一切：真预印本不受影响
        assert is_preprint_record(
            venue="arXiv", doi="10.48550/arXiv.2411.11335",
            source="arxiv", arxiv_id="2411.11335",
        ) is True

    def test_platform_placeholder_venue_needs_both_signals(self):
        """占位 venue 判据：venue 是纯平台名 **且** 已有出版方 DOI 才算错刊名。"""
        from app.tools.venue_tiers import is_platform_placeholder_venue

        for venue in ["arXiv", "arxiv", "arXiv.org", "ArXiv preprint", "bioRxiv"]:
            assert is_platform_placeholder_venue(
                venue=venue, doi="10.1109/CVPR52688.2022.01932",
            ) is True, f"Failed for {venue}"

        # 无出版方 DOI 时 "arXiv" 是正确出处，必须保留
        assert is_platform_placeholder_venue(venue="arXiv", doi="") is False
        assert is_platform_placeholder_venue(
            venue="arXiv", doi="10.48550/arXiv.2411.11335") is False
        # SSRN 预印本的 venue 已归一化成 "SSRN"，其自有 DOI 不构成占位
        assert is_platform_placeholder_venue(
            venue="SSRN", doi="10.2139/ssrn.5055942") is False
        # 正当刊名不得被平台名子串误伤
        for venue in ["Open Mind", "arXiv Journal of Robotics", "Nature"]:
            assert is_platform_placeholder_venue(
                venue=venue, doi="10.1109/CVPR52688.2022.01932",
            ) is False, f"Failed for {venue}"
        assert is_platform_placeholder_venue(venue="", doi="10.1/x") is False

    def test_bare_conference_names_are_recognized(self):
        """S2 / OpenAlex 给会议的 venue 常不含 conference 字面词。"""
        from app.tools.venue_tiers import is_conference_venue

        for venue in [
            "Computer Vision and Pattern Recognition",
            "ACM Multimedia",
            "AAAI Conference on Artificial Intelligence",
            "International Conference on Control, Robotics and Intelligent System",
        ]:
            assert is_conference_venue(venue=venue) is True, f"Failed for {venue}"

        # 期刊不得被会议名子串误伤：Pattern Recognition 不含完整 CVPR 名
        for venue in [
            "Pattern Recognition",
            "International Journal of Computer Vision",
            "Knowledge-Based Systems",
            "IEEE Access",
        ]:
            assert is_conference_venue(venue=venue) is False, f"Failed for {venue}"

    def test_degree_thesis_detection_separates_institutions_from_journals(self):
        """培养单位 venue 与 /d.cnki. DOI 判学位论文；大学学报是期刊。"""
        from app.tools.venue_tiers import is_degree_thesis

        assert is_degree_thesis(doi="10.27170/d.cnki.gjsuu.2022.001834") is True
        assert is_degree_thesis(venue="合肥工业大学安徽省211工程院校教育部直属院校") is True
        assert is_degree_thesis(venue="桂林电子科技大学广西壮族自治区") is True
        assert is_degree_thesis(venue="湖南理工学院湖南省") is True
        assert is_degree_thesis(venue="", title="基于图神经网络的动作识别（硕士学位论文）") is True

        # 标题里提到学位论文说的是主题，不是该文自身的形态
        assert is_degree_thesis(
            venue="学位与研究生教育", title="研究生学位论文质量评价体系研究",
            doi="10.1/x",
        ) is False

        # 刊名后缀优先：大学学报是期刊而不是培养单位
        assert is_degree_thesis(venue="南京邮电大学学报(自然科学版)") is False
        assert is_degree_thesis(venue="天津大学学报(自然科学与工程技术版)") is False
        assert is_degree_thesis(
            venue="IEEE Access", doi="10.1109/ACCESS.2024.3365448") is False
        assert is_degree_thesis(venue="检索 CNKI AI 出版来源 我的CNKI") is False

    def test_quality_score_strict_ordering(self):
        """严格验证：顶会顶刊 > 普通已发表 > 预印本 > 未知出处。"""
        current_year = 2026

        # 1. 顶会顶刊论文 (CVPR 2026)
        p_top = {
            "title": "Top Conference Paper",
            "venue": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
            "abstract": "A comprehensive study on few-shot learning.",
            "authors": ["Alice", "Bob"],
            "year": 2026,
            "doi": "10.1109/CVPR52688.2026.0001",
            "source": "crossref",
        }

        # 2. 普通已发表论文 (ICIP 2026)
        p_standard = {
            "title": "Standard Published Paper",
            "venue": "IEEE International Conference on Image Processing (ICIP)",
            "abstract": "A comprehensive study on few-shot learning.",
            "authors": ["Alice", "Bob"],
            "year": 2026,
            "doi": "10.1109/ICIP.2026.0001",
            "source": "crossref",
        }

        # 3. 预印本 (arXiv 2026)
        p_preprint = {
            "title": "Preprint Paper",
            "venue": "arXiv",
            "abstract": "A comprehensive study on few-shot learning.",
            "authors": ["Alice", "Bob"],
            "year": 2026,
            "arxiv_id": "2601.00001",
            "url": "https://arxiv.org/abs/2601.00001",
            "source": "arxiv",
        }

        # 4. 未知出处 (无 venue)
        p_unknown = {
            "title": "Unknown Source Paper",
            "venue": None,
            "abstract": "A comprehensive study on few-shot learning.",
            "authors": ["Alice", "Bob"],
            "year": 2026,
            "url": "https://example.com",
            "source": "unknown",
        }

        q_top = compute_quality_score(p_top, current_year)
        q_standard = compute_quality_score(p_standard, current_year)
        q_preprint = compute_quality_score(p_preprint, current_year)
        q_unknown = compute_quality_score(p_unknown, current_year)

        # 验证数值区间
        assert q_top >= 0.85, f"Top tier score {q_top} should be >= 0.85"
        assert 0.70 <= q_standard < q_top, f"Standard score {q_standard} should be in [0.70, {q_top})"
        assert 0.50 <= q_preprint < q_standard, f"Preprint score {q_preprint} should be in [0.50, {q_standard})"
        assert q_unknown < q_preprint, f"Unknown score {q_unknown} should be < {q_preprint}"

        # 严格单调递减
        assert q_top > q_standard > q_preprint > q_unknown


class TestRelevanceHierarchy:
    """测试相关性评分层级（标题完整关键词最高分，摘要完整关键词第二高分）。"""

    def test_title_match_scores_higher_than_abstract_match(self):
        topic = "few-shot action recognition"

        # 标题出现完整关键词
        p_title = {
            "title": "Few-Shot Action Recognition via Temporal Alignment",
            "abstract": "We propose a novel framework for video analysis.",
        }

        # 仅摘要出现完整关键词（标题无）
        p_abstract = {
            "title": "Temporal Alignment Network for Video Analysis",
            "abstract": "We propose a novel framework for few-shot action recognition.",
        }

        # 仅词袋部分重叠（无完整短语）
        p_partial = {
            "title": "Action Classification with Video Models",
            "abstract": "We study recognition with few labeled examples.",
        }

        score_title = compute_relevance_score(p_title, topic)
        score_abstract = compute_relevance_score(p_abstract, topic)
        score_partial = compute_relevance_score(p_partial, topic)

        # 标题出现完整关键词给最高分 (>= 0.65)
        assert score_title >= 0.65
        # 摘要出现完整关键词给第二高分
        assert 0.40 <= score_abstract < score_title
        # 仅部分词重叠给更低分
        assert score_partial < score_abstract

        assert score_title > score_abstract > score_partial

    def test_cross_lingual_title_match_scores_higher_than_abstract(self):
        topic = "少样本动作识别"
        synonyms = ["few-shot action recognition"]

        p_title_en = {
            "title": "Trokens: Relational Trajectory Tokens for Few-Shot Action Recognition",
            "abstract": "Video classification under data scarcity.",
        }

        p_abstract_en = {
            "title": "Trokens: Relational Trajectory Tokens for Video Analysis",
            "abstract": "We evaluate on few-shot action recognition benchmarks.",
        }

        score_title = compute_relevance_score(p_title_en, topic, topic_synonyms=synonyms)
        score_abstract = compute_relevance_score(p_abstract_en, topic, topic_synonyms=synonyms)

        assert score_title > score_abstract
        assert score_title >= 0.50
        assert score_abstract >= 0.25


class TestEndToEndRanking:
    """测试端到端综合排序对已发表顶会顶刊与标题命中论文的优先排列。"""

    def test_published_top_tier_outranks_preprint_with_similar_relevance(self):
        topic = "few-shot action recognition"

        papers = [
            {
                "paper_id": "p_preprint",
                "title": "Few-Shot Action Recognition with Transformers",
                "venue": "arXiv",
                "abstract": "We study few-shot action recognition using attention mechanisms.",
                "authors": ["Author A"],
                "year": 2025,
                "source": "arxiv",
            },
            {
                "paper_id": "p_top_published",
                "title": "Few-Shot Action Recognition with Transformers",
                "venue": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
                "abstract": "We study few-shot action recognition using attention mechanisms.",
                "authors": ["Author A"],
                "year": 2025,
                "doi": "10.1109/TPAMI.2025.0001",
                "source": "crossref",
            },
        ]

        ranked = rank_papers(papers, topic=topic, top_k=2)
        assert len(ranked) == 2
        # 已发表顶会顶刊综合得分显著高于预印本，排在第一名
        assert ranked[0]["paper_id"] == "p_top_published"
        assert ranked[1]["paper_id"] == "p_preprint"
        assert ranked[0]["_rank_score"] > ranked[1]["_rank_score"]
