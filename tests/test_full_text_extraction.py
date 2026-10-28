"""全文后部覆盖、精确位置、调用上限和停止信号回归；不请求外部服务。"""
import json

import pytest

from app.agent.execution_budget import AgentBudgetExceeded
from app.tools.chunk_pdf import chunk_pdf
from app.tools.extract_paper_card import extract_paper_card, batch_extract_paper_cards


METHOD = "We propose a retrieval method using a dual encoder for document retrieval."
RESULT = "Results show an accuracy of 91% on the evaluated benchmark."
LIMIT = "Our limitation is that the evaluation covers only one language."
PAPER = {"paper_id": "test:full", "title": "Retrieval Study", "source": "test", "year": 2023}


class Model:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def complete(self, prompt, **kwargs):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return json.dumps({field: [text] if text in prompt else [] for field, text in (
            ("method", METHOD), ("results", RESULT), ("limitations", LIMIT))})


def parsed(*pages):
    from app.tools.parse_pdf import split_paper_sections
    return {"full_text": "\n".join(pages), "sections": split_paper_sections("\n".join(pages)), "pages": [
        {"page": i + 1, "text": text} for i, text in enumerate(pages)]}


def test_late_results_and_limitations_survive_candidate_sampling():
    intro = "An earlier sentence discusses related retrieval work.\n" * 200
    source = parsed(intro, "Methods\n" + METHOD, "Results\n" + RESULT, "Limitations\n" + LIMIT + "\nReferences\nBibliographic entry.")
    assert source["full_text"].index(RESULT) > 8000
    model = Model()
    card = extract_paper_card(PAPER, source, model)
    assert card.method == METHOD
    assert card.results == RESULT
    assert card.limitations == [LIMIT]
    assert card.extraction_coverage.status == "complete"
    assert card.quality_status == "valid"
    for field in ("method", "results", "limitations"):
        assert card.field_claims[field][0].explicitly_reported
        for eid in card.field_evidence[field]:
            span = next(s for s in card.evidence_spans if s.evidence_id == eid)
            text = source["pages"][span.page - 1]["text"]
            assert text[span.char_start:span.char_end] == span.text
    assert card.field_claims["results"][0].source_text == RESULT
    assert card.sample_size is None


@pytest.mark.parametrize("page_mode", [True, False])
def test_chunk_offsets_overlap_and_all_text_coverage(page_mode):
    text = "Methods\n" + ("Long paragraph with Chinese 内容。\n" * 90) + "\nResults\n" + RESULT
    source = parsed(text) if page_mode else {"full_text": text}
    chunks = chunk_pdf(source, max_chars=300, overlap_chars=40)
    positions = set()
    for chunk in chunks:
        assert len(chunk["text"]) <= 300
        assert text[chunk["char_start"]:chunk["char_end"]] == chunk["text"]
        assert chunk["offset_basis"] == ("page" if page_mode else "full_text")
        positions.update(range(chunk["char_start"], chunk["char_end"]))
    assert all(i in positions for i, char in enumerate(text) if not char.isspace())
    assert any(a["char_end"] > b["char_start"] for a, b in zip(chunks, chunks[1:]))


def test_call_limit_marks_unread_pages_and_quality_partial(monkeypatch):
    from app.core.config import get_settings
    monkeypatch.setattr(get_settings(), "card_extraction_max_chunks", 1)
    model = Model()
    card = extract_paper_card(PAPER, parsed(METHOD, RESULT), model)
    assert len(model.calls) == 1
    assert card.method == METHOD
    assert card.results is None
    assert card.extraction_coverage.status == "partial"
    assert card.extraction_coverage.chunks[-1].reason == "chunk_call_limit"
    assert card.quality_status == "partial"


def test_reference_pages_excluded_but_appendix_processed():
    model = Model()
    card = extract_paper_card(PAPER, parsed(METHOD, "References\n" + RESULT,
                                          "Appendix\n" + LIMIT), model)
    assert len(model.calls) == 2
    assert card.results is None
    # 原有访问门禁仍要求完整章节标记；处理附录不自动升级访问权限。
    assert card.limitations == []
    assert LIMIT in model.calls[-1]
    assert any(r.status == "excluded" for r in card.extraction_coverage.chunks)


def test_hallucinated_or_changed_numeric_quote_is_rejected():
    class BadModel:
        def complete(self, *args, **kwargs):
            return json.dumps({"results": [RESULT.replace("91%", "99%")], "method": [METHOD]})
    card = extract_paper_card(PAPER, parsed(METHOD, RESULT), BadModel())
    assert card.results is None
    assert "results" not in card.field_claims


def test_provider_failure_preserves_prior_evidence_and_marks_rest_skipped():
    class FailingModel(Model):
        def complete(self, *args, **kwargs):
            if self.calls:
                raise TimeoutError("private detail should not enter coverage")
            return super().complete(*args, **kwargs)
    card = extract_paper_card(PAPER, parsed(METHOD, RESULT, LIMIT), FailingModel())
    assert card.method == METHOD
    assert card.extraction_coverage.status == "partial"
    assert [r.status for r in card.extraction_coverage.chunks] == ["processed", "failed", "skipped"]
    assert "private detail" not in card.model_dump_json()


@pytest.mark.parametrize("error", [InterruptedError("cancel"), AgentBudgetExceeded("budget")])
def test_stop_signals_escape_single_and_batch_extraction(error):
    with pytest.raises(type(error)):
        extract_paper_card(PAPER, parsed(METHOD), Model(error))
    with pytest.raises(type(error)):
        batch_extract_paper_cards([PAPER], {PAPER["paper_id"]: parsed(METHOD)}, Model(error))


def test_cancel_probe_stops_between_chunks():
    model = Model()
    with pytest.raises(InterruptedError):
        extract_paper_card(PAPER, parsed(METHOD, RESULT), model,
                           should_cancel=lambda: len(model.calls) >= 1)
    assert len(model.calls) == 1


def test_coverage_roundtrip_and_no_rule_fallback_for_empty_output():
    from app.schemas.paper_schema import PaperCard
    class EmptyModel:
        def complete(self, *args, **kwargs):
            return '{"method": []}'
    card = extract_paper_card(PAPER, parsed(METHOD), EmptyModel())
    assert card.method == ""
    assert card.quality_status == "invalid"
    restored = PaperCard.model_validate_json(card.model_dump_json())
    assert restored.extraction_coverage == card.extraction_coverage


def test_abstract_route_retains_existing_protocol():
    class AbstractModel:
        def complete(self, prompt, **kwargs):
            assert "research_problem" in prompt
            return json.dumps({"method": METHOD})
    card = extract_paper_card({**PAPER, "abstract": METHOD}, llm=AbstractModel())
    assert card.extraction_coverage is None
    assert card.evidence_source == "abstract"
