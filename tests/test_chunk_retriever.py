"""第二层 RAG 的可重复离线契约；所有模型为可控替身。"""
import json

import pytest

from app.agent.execution_budget import AgentBudgetExceeded
from app.clients.retrieval_http import RetrievalProviderError
from app.core.config import Settings
from app.schemas.chunk_schema import ChunkRetrievalReport
from app.tools.chunk_pdf import chunk_pdf
from app.tools.chunk_retriever import DenseChunkIndex, retrieve_chunks


TARGET = "Results show an accuracy of 91% on the evaluated benchmark."
METHOD = "We propose a retrieval method using a dual encoder for document retrieval."


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, card_chunk_retrieval_mode="hybrid",
                    retrieval_embedding_dimension=2, retrieval_cache_dir=str(tmp_path),
                    card_chunk_top_k=1, card_extraction_max_chunks=2,
                    card_chunk_enable_rerank=False)


def chunks(*texts):
    return chunk_pdf({"pages": [{"page": i + 1, "text": text} for i, text in enumerate(texts)]},
                     paper_id="p")


class Encoder:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def embed(self, texts, *, text_type, **kwargs):
        self.calls.append((text_type, texts))
        if self.error:
            raise self.error
        return [[0.0, 1.0] if text_type == "query" or TARGET in text else [1.0, 0.0] for text in texts]


class Reranker:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def score(self, query, documents, **kwargs):
        self.calls.append(documents)
        if self.error:
            raise self.error
        return [1.0 if TARGET in text else 0.0 for text in documents]


def test_dense_only_chunk_retrieval_and_field_top_k(settings, monkeypatch):
    monkeypatch.setitem(__import__("app.tools.chunk_retriever", fromlist=["FIELD_RETRIEVAL_QUERIES"])
                        .FIELD_RETRIEVAL_QUERIES, "limitations", "局限性 caveats threats")
    source = chunks("An unrelated earlier passage.", TARGET)
    report = retrieve_chunks("p", source, settings=settings, embedding_client=Encoder())
    # limitations 查询与目标结果无词项交集，受控语义向量仍可将它召回。
    hit = report.field_hits["limitations"][0]
    assert hit.chunk_id == 1 and hit.cosine == pytest.approx(1.0)
    assert hit.bm25 is None
    assert len(report.field_hits) == 7
    assert all(len(hits) <= 1 for hits in report.field_hits.values())
    assert report.backend == "numpy_exact" and not report.degraded


def test_reranker_receives_only_local_bounded_window(settings):
    settings.card_chunk_enable_rerank = True
    model = Reranker()
    report = retrieve_chunks("p", chunks("Methods\n" + METHOD, TARGET), settings=settings,
                             embedding_client=Encoder(), reranker_client=model)
    assert len(model.calls) == 7
    assert all(len(docs) <= settings.card_chunk_rerank_window for docs in model.calls)
    assert report.field_hits["results"][0].rerank == 1.0


def test_source_isolation_before_any_model_calls(settings):
    source = chunks(TARGET)
    source[0]["paper_id"] = "other"
    encoder = Encoder()
    with pytest.raises(ValueError, match="其他论文"):
        retrieve_chunks("p", source, settings=settings, embedding_client=encoder)
    assert not encoder.calls


def test_embedding_cache_reuse_and_content_invalidation(settings):
    encoder = Encoder()
    source = chunks(METHOD, TARGET)
    retrieve_chunks("p", source, settings=settings, embedding_client=encoder)
    assert len(encoder.calls) == 2
    encoder.calls.clear()
    retrieve_chunks("p", source, settings=settings, embedding_client=encoder)
    assert not encoder.calls
    source[-1]["text"] += " More context."
    retrieve_chunks("p", source, settings=settings, embedding_client=encoder)
    assert len(encoder.calls) == 1
    assert encoder.calls[0][0] == "document" and len(encoder.calls[0][1]) == 1


def test_cache_model_changes_force_reembedding(settings):
    encoder = Encoder()
    retrieve_chunks("p", chunks(TARGET), settings=settings, embedding_client=encoder)
    encoder.calls.clear()
    settings.retrieval_embedding_model = "different-model"
    retrieve_chunks("p", chunks(TARGET), settings=settings, embedding_client=encoder)
    assert len(encoder.calls) == 2


def test_provider_failure_keeps_lexical_and_coverage_diagnostics(settings):
    source = chunks("Methods\n" + METHOD, "Results\n" + TARGET, "References\n" + TARGET)
    report = retrieve_chunks("p", source, settings=settings,
                             embedding_client=Encoder(RetrievalProviderError("private error")))
    assert report.degraded == ["dense:RetrievalProviderError"]
    assert 2 not in report.indexed_chunk_ids
    assert 2 not in report.selected_chunk_ids
    assert report.field_hits["method"]
    assert "private error" not in report.model_dump_json()


def test_reranker_failure_retains_rrf_order(settings):
    settings.card_chunk_enable_rerank = True
    report = retrieve_chunks("p", chunks(TARGET), settings=settings, embedding_client=Encoder(),
                             reranker_client=Reranker(RetrievalProviderError("unavailable")))
    assert len(report.degraded) == 7
    assert report.field_hits["results"][0].chunk_id == 0
    assert report.field_hits["results"][0].rerank is None


@pytest.mark.parametrize("stage", ["dense", "rerank"])
@pytest.mark.parametrize("error", [InterruptedError("cancel"), AgentBudgetExceeded("stop")])
def test_control_signals_propagate(settings, stage, error):
    settings.card_chunk_enable_rerank = True
    with pytest.raises(type(error)):
        retrieve_chunks("p", chunks(TARGET), settings=settings,
                        embedding_client=Encoder(error if stage == "dense" else None),
                        reranker_client=Reranker(error if stage == "rerank" else None))


def test_index_bound_and_late_section_coverage(settings):
    settings.card_chunk_index_limit = 8
    source = chunks(*(["Introduction\nEarlier context."] * 12),
                    "3.2 Experimental Results:\n" + TARGET)
    report = retrieve_chunks("p", source, settings=settings, embedding_client=Encoder())
    assert report.index_truncated and len(report.indexed_chunk_ids) == 8
    assert source[-1]["chunk_id"] in report.indexed_chunk_ids
    assert source[-1]["chunk_id"] in report.selected_chunk_ids




@pytest.mark.parametrize("vectors,query", [([[0, 0]], [1, 0]), ([[1, 0]], [float("nan"), 0]),
                                         ([[1, 0]], [1, 0, 0])])
def test_invalid_vectors_rejected(vectors, query):
    with pytest.raises(RetrievalProviderError):
        DenseChunkIndex(vectors).search(query, 1)


def test_hierarchical_extraction_uses_late_evidence_and_preserves_gates(monkeypatch, settings):
    from app.tools.extract_paper_card import extract_paper_card
    from app.tools.parse_pdf import split_paper_sections
    from app.schemas.paper_schema import PaperCard
    import app.tools.chunk_retriever as retriever
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    monkeypatch.setattr(retriever, "EmbeddingClient", lambda _: Encoder())
    pages = ["Introduction\nEarlier context."] * 10 + ["Methods\n" + METHOD, "3.2 Experimental Results:\n" + TARGET,
                                                       "References\nBibliographic entry."]
    parsed = {"full_text": "\n".join(pages), "sections": split_paper_sections("\n".join(pages)),
              "pages": [{"page": i + 1, "text": text} for i, text in enumerate(pages)]}
    class Extractor:
        def complete(self, prompt, **kwargs):
            return json.dumps({"results": [TARGET] if TARGET in prompt else [],
                               "method": [METHOD] if METHOD in prompt else []})
    card = extract_paper_card({"paper_id": "p", "title": "Study"}, parsed, Extractor())
    assert card.results == TARGET and card.method == METHOD
    assert card.quality_status == "partial"
    assert card.extraction_coverage.status == "partial"
    assert card.extraction_coverage.retrieval.mode == "hybrid"
    span = next(s for s in card.evidence_spans if s.text == TARGET)
    assert span.page == 12
    assert pages[span.page - 1][span.char_start:span.char_end] == TARGET
    assert card.field_claims["results"][0].explicitly_reported
    assert PaperCard.model_validate_json(card.model_dump_json()).extraction_coverage == card.extraction_coverage


def test_report_roundtrip(settings):
    report = retrieve_chunks("p", chunks(TARGET), settings=settings, embedding_client=Encoder())
    assert ChunkRetrievalReport.model_validate_json(report.model_dump_json()) == report
