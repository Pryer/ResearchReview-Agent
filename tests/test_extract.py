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
from tests.card_fixtures import source_card


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
    def test_missing_model_is_explicit_failure(self):
        from app.core.exceptions import PaperCardExtractionError
        with pytest.raises(PaperCardExtractionError):
            extract_paper_card(_make_paper(), llm=None)

    def test_abstract_extraction_is_model_driven_and_traceable(self):
        import json
        text = "We propose a traceable retrieval method for document search."
        class Model:
            def complete(self, prompt, **kwargs):
                assert text in prompt
                return json.dumps({"method": text})
        card = extract_paper_card(_make_paper(abstract=text), llm=Model())
        assert card.method == text
        assert card.evidence_source == "abstract"
        assert card.field_claims["method"][0].explicitly_reported
        assert all(s.source_status == "source_verified" for s in card.evidence_spans if s.source_type == "abstract")
        assert any(s.source_type == "metadata" for s in card.evidence_spans)
        assert card.authors == ["Alice", "Bob"]
        assert card.doi == "10.1000/test"

    def test_model_failure_does_not_create_rule_based_claims(self):
        class Model:
            def complete(self, *args, **kwargs):
                raise RuntimeError("model unavailable")
        card = extract_paper_card(_make_paper(), llm=Model())
        assert card.quality_status == "invalid"
        assert card.evidence_source == "metadata"
        assert not card.method and not card.research_problem and not card.field_claims
        assert "extraction:model_failure:RuntimeError" in card.quality_issues

    def test_unreported_fields_are_not_filled_by_rules(self):
        class Model:
            def complete(self, *args, **kwargs):
                return '{"method": ""}'
        card = extract_paper_card(_make_paper(abstract="We analyze 2018 video segments using audio and pose."), llm=Model())
        assert card.sample_size is None
        assert card.data_modalities == []
        assert card.method == ""

    def test_cnki_ignores_cached_full_text(self):
        import json
        text = "We propose a traceable retrieval method for document search."
        class Model:
            def complete(self, prompt, **kwargs):
                assert "Forbidden cached full text" not in prompt
                return json.dumps({"method": text})
        card = extract_paper_card(_make_paper(source="cnki", abstract=text),
                                  {"full_text": "Forbidden cached full text"}, llm=Model())
        assert card.evidence_source == "abstract"
        assert card.evidence_state.access_level.value == "abstract"
        assert all(span.source_type != "full_text" for span in card.evidence_spans)

    def test_no_accessible_content_is_invalid_metadata_only(self):
        card = extract_paper_card(_make_paper(abstract=""), llm=None)
        assert card.quality_status == "invalid"
        assert not card.method and not card.field_claims
        assert card.evidence_state.access_level.value == "metadata_only"
        assert "extraction:no_accessible_content" in card.quality_issues

    def test_full_text_access_still_requires_core_sections(self):
        paper = _make_paper(abstract="")
        assert resolve_evidence_permissions(paper, {"full_text": "Method text", "sections": {"method": "Method text"}}).access_level.value == "partial_full_text"
        assert resolve_evidence_permissions(paper, {"full_text": "Complete paper", "sections": {"method": "m", "experiment": "e", "references": "r"}}).access_level.value == "full_text"


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


def test_card_quality_rejects_affiliation_and_unreported_limitations():
    paper = _make_paper(
        abstract=(
            "曾建电1，北京师范大学自然科学高等研究院，广东珠海。摘要："
            "课堂行为状态分析是教育智能化研究的重要方向。"
            "然而，传统单模态方法在复杂背景下存在信息不足的问题。"
            "为此，本研究提出一种结合图像和文本的多模态数据增强方法。"
            "实验结果表明，该方法在分类准确率和F1值上优于单模态方法。"
        )
    )
    card = source_card(paper,
        method="为此，本研究提出一种结合图像和文本的多模态数据增强方法。",
        results="实验结果表明，该方法在分类准确率和F1值上优于单模态方法。")
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

    card = source_card(paper, parsed, research_problem=paper["abstract"])

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
    card = source_card(_make_paper(abstract="We propose a traceable evidence method for RAG."))

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
