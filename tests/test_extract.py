"""论文卡片抽取测试。"""

from __future__ import annotations

import pytest

from app.tools.extract_paper_card import (
    _is_clean_evidence_sentence,
    extract_paper_card,
    resolve_evidence_permissions,
    validate_paper_card,
)
from app.schemas.paper_schema import PaperCard


def _make_paper(**kwargs) -> dict:
    defaults = {
        "paper_id": "test:1",
        "title": "Test Paper on Vision Transformers",
        "authors": ["Alice", "Bob"],
        "year": 2023,
        "venue": "CVPR",
        "abstract": "This paper proposes a new vision transformer method for image classification.",
        "doi": "10.1000/test",
        "arxiv_id": None,
        "pdf_url": None,
        "citation_count": 50,
        "source": "test",
    }
    defaults.update(kwargs)
    return defaults


class TestDictToCardAuthoritativeFields:
    """LLM 输出不得覆盖权威元数据或自我升级证据等级。"""

    def _to_card(self, data: dict, paper: dict):
        from app.tools.extract_paper_card import _dict_to_card

        return _dict_to_card(
            data, paper, evidence_source="abstract", topic="vision"
        )

    def test_llm_cannot_override_authoritative_paper_id(self):
        card = self._to_card(
            {"paper_id": "hallucinated:999", "method": "transformer"},
            _make_paper(),
        )
        assert card.paper_id == "test:1"

    def test_source_year_takes_priority_over_llm_year(self):
        card = self._to_card(
            {"year": "2019年", "method": "x"},
            _make_paper(year=2023),
        )
        assert card.year == 2023

    def test_llm_year_is_not_used_when_source_year_is_missing(self):
        """AGENTS.md 第 9 条：未知元数据保持 None，不得用 LLM 猜测替代。"""
        card = self._to_card(
            {"year": "2021", "method": "x"},
            _make_paper(year=None),
        )
        assert card.year is None

    def test_llm_cannot_override_authoritative_bibliographic_fields(self):
        card = self._to_card(
            {
                "title": "Hallucinated Title",
                "authors": ["Feng"],
                "venue": "Fake Journal",
                "doi": "10.9999/fake",
                "url": "https://example.invalid/fake",
                "method": "x",
            },
            _make_paper(),
        )
        assert card.title == "Test Paper on Vision Transformers"
        assert card.authors == ["Alice", "Bob"]
        assert card.venue == "CVPR"
        assert card.doi == "10.1000/test"
        assert "example.invalid" not in str(card.url or "")

    def test_invalid_year_falls_back_to_none(self):
        card = self._to_card(
            {"year": "recent years", "method": "x"},
            _make_paper(year=None),
        )
        assert card.year is None

    def test_llm_cannot_self_upgrade_to_systematic_review(self):
        card = self._to_card(
            {
                "publication_type": "systematic_review",
                "evidence_level": "systematic_review",
                "peer_review_status": "peer_reviewed",
                "method": "x",
            },
            _make_paper(venue="IEEE Conference on Computer Vision"),
        )
        # 会议论文：类型与证据等级只由元数据规则决定
        assert card.publication_type == "conference_paper"
        assert card.evidence_level == "conference_paper"
        assert card.peer_review_status == "likely_peer_reviewed"

    def test_arxiv_preprint_stays_not_peer_reviewed(self):
        card = self._to_card(
            {"peer_review_status": "peer_reviewed", "method": "x"},
            # 真预印本：arXiv 源、无出版方 DOI。默认夹具的 doi="10.1000/test"
            # 是出版方 DOI，配 source=arxiv 表示正式发表后回填，那不是预印本。
            _make_paper(source="arxiv", venue="arXiv", doi=None, arxiv_id="2401.00001"),
        )
        assert card.publication_type == "preprint"
        assert card.peer_review_status == "not_peer_reviewed"

    def test_arxiv_sourced_published_paper_counts_as_peer_reviewed(self):
        """回归：arXiv 源 + 出版方 DOI 是已正式发表的论文，不是预印本。

        arxiv_client 对每条记录硬编码 venue="arXiv"，并从 <link title="doi">
        取回正式发表后回填的出版方 DOI。此前 venue 与 source 两处都会把这类
        记录定罪成预印本：publication_type 渲染成 [EB/OL]，peer_review_status
        又按未评审计入 global_evidence_gate 的同行评审占比。
        """
        cvpr = self._to_card(
            {"method": "x"},
            _make_paper(
                source="arxiv", venue="arXiv", arxiv_id="2201.00000",
                doi="10.1109/CVPR52688.2022.01932",
            ),
        )
        assert cvpr.publication_type == "conference_paper"
        assert cvpr.peer_review_status == "likely_peer_reviewed"

        ijcv = self._to_card(
            {"method": "x"},
            _make_paper(
                source="arxiv", venue="arXiv", arxiv_id="2303.00001",
                doi="10.1007/s11263-023-01917-4",
            ),
        )
        assert ijcv.publication_type == "journal_article"
        assert ijcv.peer_review_status == "likely_peer_reviewed"


class TestExtractPaperCard:
    def test_rule_based_extraction_without_llm(self):
        """无 LLM 时应使用规则抽取。"""
        paper = _make_paper()
        card = extract_paper_card(paper, parsed_text=None, llm=None)
        assert isinstance(card, PaperCard)
        assert card.paper_id == "test:1"
        assert card.title == "Test Paper on Vision Transformers"

    def test_full_text_extraction(self):
        """有全文时应标记 evidence_source=full_text。"""
        paper = _make_paper()
        parsed = {"full_text": "This is the full text of a paper about transformers..."}
        card = extract_paper_card(paper, parsed_text=parsed, llm=None)
        assert card.evidence_source == "full_text"

    def test_abstract_only_extraction(self):
        """仅有摘要时应标记 evidence_source=abstract。"""
        paper = _make_paper(abstract="Short abstract")
        card = extract_paper_card(paper, parsed_text={}, llm=None)
        # 没有全文时根据摘要抽取
        assert card.evidence_source in ("abstract", "metadata")

    def test_cnki_ignores_cached_full_text_and_uses_abstract_evidence(self):
        """CNKI 即使存在历史全文缓存，也必须保持摘要证据级别。"""
        paper = _make_paper(
            paper_id="cnki:abstract-only",
            source="cnki",
            abstract="This abstract reports a classroom behavior coding study.",
        )
        parsed = {
            "full_text": "A cached full text that must not be used.",
            "pages": [{"page": 1, "text": "A cached full text that must not be used."}],
        }

        card = extract_paper_card(paper, parsed_text=parsed, llm=None)

        assert card.evidence_source == "abstract"
        assert card.evidence_state.access_level.value == "abstract"
        assert card.evidence_spans
        assert any(span.source_type == "abstract" for span in card.evidence_spans)
        assert all(span.source_type != "full_text" for span in card.evidence_spans)
        assert all("cached full text" not in span.text.lower() for span in card.evidence_spans)

    def test_relevance_reason_filled(self):
        """相关性说明应有内容。"""
        paper = _make_paper()
        card = extract_paper_card(paper, parsed_text=None, llm=None, topic="vision transformer")
        assert len(card.relevance_reason) > 0

    def test_bibliographic_fields_carry_their_own_metadata_evidence(self):
        """作者、年份、来源、DOI、出版状态必须各自可溯源，而非只有内容证据。"""
        paper = _make_paper(publication_status="published")
        card = extract_paper_card(paper, parsed_text=None, llm=None, topic="课堂行为识别")

        evidence_ids = {span.evidence_id for span in card.evidence_spans}
        for field in ("authors", "year", "venue", "doi", "publication_status"):
            mapped = card.field_evidence.get(field)
            assert mapped, f"{field} 缺少元数据证据"
            assert set(mapped).issubset(evidence_ids)
        # 书目证据只用于溯源，不得作为可写入正文的内容主张。
        assert "authors" not in card.field_claims
        assert "doi" not in card.field_claims
        assert card.publication_status.value == "published"

    def test_unknown_publication_status_is_not_promoted(self):
        card = extract_paper_card(
            _make_paper(), parsed_text=None, llm=None, topic="课堂行为识别"
        )
        assert card.publication_status.value == "unknown"

    def test_abstract_card_contains_traceable_evidence(self):
        paper = _make_paper(
            abstract=(
                "We propose a transformer method for classroom behavior recognition. "
                "The method achieves 82% accuracy on the Classroom-10 dataset."
            )
        )
        card = extract_paper_card(paper, parsed_text=None, llm=None, topic="课堂行为识别")

        assert card.evidence_spans
        assert any(span.source_type == "abstract" for span in card.evidence_spans)
        assert card.field_evidence.get("method")
        evidence_ids = {span.evidence_id for span in card.evidence_spans}
        assert set(card.field_evidence["method"]).issubset(evidence_ids)

    def test_same_abstract_sentence_is_not_reused_across_semantic_fields(self):
        sentence = (
            "This paper proposes an enhanced YOLOv8 model for classroom "
            "student behavior detection."
        )
        card = extract_paper_card(
            _make_paper(abstract=sentence),
            parsed_text=None,
            llm=None,
            topic="课堂行为识别",
        )

        populated = [
            value for value in [
                card.research_problem,
                card.method,
                *card.contributions,
            ]
            if value
        ]
        assert populated == [sentence]
        assert card.method == sentence
        assert "research_problem" not in card.field_claims
        assert "contributions" not in card.field_claims

    def test_full_text_evidence_preserves_page_number(self):
        paper = _make_paper(abstract="")
        parsed = {
            "full_text": "We introduce an evidence-aware retrieval method.",
            "pages": [
                {"page": 3, "text": "We introduce an evidence-aware retrieval method for factual generation."}
            ],
        }
        card = extract_paper_card(paper, parsed_text=parsed, llm=None, topic="RAG")

        assert any(span.source_type == "full_text" for span in card.evidence_spans)
        assert any(span.page == 3 for span in card.evidence_spans)

    def test_literature_matrix_fields_are_extracted_conservatively(self):
        paper = _make_paper(
            source="arxiv",
            arxiv_id="2401.12345",
            # venue 必须与 arxiv_client 的真实输出一致（该客户端恒置 "arXiv"，
            # 见 arxiv_client.py 的 PaperMetadata 构造）。真预印本无出版方 DOI：
            # 默认夹具的 doi="10.1000/test" 是出版方 DOI，配 source=arxiv 表示
            # 正式发表后回填，那已不是预印本。此处清空以还原纯预印本形态。
            venue="arXiv",
            doi=None,
            abstract=(
                "We analyze 2018 classroom video segments using video, audio, and pose data. "
                "The study examines hand raising, student engagement, and teacher feedback."
            ),
        )
        card = extract_paper_card(paper, llm=None, topic="课堂行为分析")
        assert card.authors == ["Alice", "Bob"]
        assert card.doi is None
        assert card.publication_type == "preprint"
        assert card.peer_review_status == "not_peer_reviewed"
        assert card.evidence_level == "preprint"
        assert card.sample_size == "2018 classroom video segments"
        assert {"video", "audio", "pose_skeleton"}.issubset(card.data_modalities)
        # 开放领域类别只能由 LLM 从原文动态抽取；规则兜底不维护课堂行为词表。
        assert card.behavior_categories == []

    def test_abstract_access_rejects_author_unreported_limitations(self):
        paper = _make_paper(
            abstract=(
                "We propose a multimodal classroom behavior method. "
                "Results show that the method achieves 82% accuracy."
            )
        )
        card = extract_paper_card(paper, llm=None, topic="课堂行为分析")
        assert card.evidence_state.access_level.value == "abstract"
        assert card.method
        assert card.results
        assert card.limitations == []
        assert "limitations" in card.unsupported_fields
        assert card.field_claims["method"][0].explicitly_reported is True

    def test_metadata_only_cannot_create_content_claims(self):
        paper = _make_paper(abstract="", keywords=None, doi=None, url=None)
        card = extract_paper_card(paper, llm=None, topic="课堂行为分析")
        assert card.evidence_state.access_level.value == "metadata_only"
        assert card.research_problem == ""
        assert card.method == ""
        assert card.results is None
        assert "method" in card.unsupported_fields

    def test_full_text_requires_core_sections_otherwise_is_partial(self):
        paper = _make_paper(abstract="")
        partial = resolve_evidence_permissions(
            paper, {"full_text": "Method text", "sections": {"method": "Method text"}}
        )
        complete = resolve_evidence_permissions(
            paper,
            {
                "full_text": "Complete paper",
                "sections": {"method": "m", "experiment": "e", "references": "r"},
            },
        )
        assert partial.access_level.value == "partial_full_text"
        assert partial.can_compare_metrics is False
        assert complete.access_level.value == "full_text"
        assert complete.can_compare_metrics is True


class TestValidatePaperCard:
    def test_valid_card(self):
        card = PaperCard(
            paper_id="test:1",
            title="Test",
            research_problem="problem",
            method="method",
            relevance_reason="reason",
            evidence_source="abstract",
        )
        assert validate_paper_card(card) is True

    def test_invalid_no_paper_id(self):
        card = PaperCard(
            paper_id="",
            title="Test",
            research_problem="",
            method="",
            relevance_reason="",
            evidence_source="abstract",
        )
        assert validate_paper_card(card) is False

    def test_invalid_evidence_source(self):
        card = PaperCard(
            paper_id="test:1",
            title="Test",
            research_problem="",
            method="",
            relevance_reason="",
            evidence_source="invalid",
        )
        assert validate_paper_card(card) is False


def test_rule_card_rejects_motivation_as_method_and_affiliation_as_result():
    paper = _make_paper(
        abstract=(
            "曾建电1，北京师范大学自然科学高等研究院，广东珠海。摘要："
            "课堂行为状态分析是教育智能化研究的重要方向。"
            "然而，传统单模态方法在复杂背景下存在信息不足的问题。"
            "为此，本研究提出一种结合图像和文本的多模态数据增强方法。"
            "实验结果表明，该方法在分类准确率和F1值上优于单模态方法。"
        )
    )
    card = extract_paper_card(paper, llm=None, topic="课堂行为分析")
    assert "传统单模态" not in card.method
    assert "本研究提出" in card.method
    assert card.results and "实验结果表明" in card.results
    assert "北京师范大学" not in (card.research_problem or "")
    assert "北京师范大学" not in (card.results or "")
    assert card.limitations == []


@pytest.mark.parametrize(
    "fragment",
    [
        "© The Author(s) 2025.",
        "the models, this paper evaluates the models using Precision,",
        "A Spatio-Temporal Attention-Based Method for",
        "YOLOv8-Based Student Behavior Detection in the Classroom:",
        "Journal of Computer Science and Electrical Engineering",
        "Accuracy Precision",
        "results demonstrate superior performance in both accuracy and efficiency, validating",
        "Finally, we expect our study to contribute to classroom analytics as we investigate the",
        "In this manuscript, we propose the IEDS system for the",
        "ABSTRACT Interaction analysis supports classroom research.",
        "Introduction The classroom environment is central to student learning.",
        "Behavior, novel, AVA, behavior, SCB",
    ],
)
def test_pdf_headers_and_sentence_fragments_are_not_evidence(fragment):
    assert _is_clean_evidence_sentence(fragment) is False


def test_full_text_front_matter_cannot_replace_clean_abstract_problem():
    paper = _make_paper(
        abstract=(
            "This study examines how classroom behavior coding supports "
            "the analysis of student engagement."
        )
    )
    parsed = {
        "full_text": (
            "© The Author(s) 2025.\n"
            "Journal of Computer Science and Electrical Engineering\n"
            "the models, this paper evaluates the models using Precision,\n"
            "This study examines classroom interaction patterns in teaching."
        ),
        "pages": [{
            "page": 1,
            "text": (
                "© The Author(s) 2025.\n"
                "Journal of Computer Science and Electrical Engineering\n"
                "This study examines classroom interaction patterns in teaching."
            ),
        }],
    }

    card = extract_paper_card(paper, parsed_text=parsed, llm=None, topic="课堂行为分析")

    assert card.research_problem.startswith("This study examines how classroom")
    assert all("©" not in span.text for span in card.evidence_spans)
    assert all("Journal of Computer" not in span.text for span in card.evidence_spans)


def test_candidate_sampling_reserves_late_pages_and_auxiliary_evidence():
    from app.tools.extract_paper_card import _candidate_evidence

    paper = _make_paper(abstract="This abstract states the central research question clearly.")
    pages = []
    for page in range(1, 121):
        sentences = " ".join(
            f"Page {page} reports a complete evidence statement number {index} for this study."
            for index in range(4)
        )
        pages.append({"page": page, "text": sentences})

    candidates = _candidate_evidence(paper, {"pages": pages, "full_text": "available"})

    assert len(candidates) <= 200
    assert any(item.get("page") == 120 for item in candidates)
    assert any(item.get("source_type") == "abstract" for item in candidates)
    assert any(item.get("source_type") == "title" for item in candidates)


def test_evidence_card_database_round_trip():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.database.models import Base
    from app.database.repositories import PaperCardRepository

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    card = extract_paper_card(
        _make_paper(abstract="We propose a traceable evidence method for RAG."),
        llm=None,
        topic="RAG",
    )

    with Session(engine) as session:
        repository = PaperCardRepository(session)
        repository.save(card)
        session.commit()
        loaded = repository.get_by_paper_id(card.paper_id)

    assert loaded is not None
    assert loaded.evidence_spans
    assert loaded.field_evidence == card.field_evidence
    assert loaded.relation_type == card.relation_type
    assert loaded.authors == card.authors
    assert loaded.publication_type == card.publication_type
