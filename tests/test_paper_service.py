"""PaperService 业务流程测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

import json
import pytest

from app.schemas.paper_schema import PaperMetadata, PaperSearchRequest
from app.services.paper_service import PaperService


def _paper() -> PaperMetadata:
    return PaperMetadata(
        paper_id="p1",
        title="Object Detection",
        authors=["Author A"],
        year=2024,
        source="test",
    )


def test_search_returns_results_even_if_persist_fails(monkeypatch):
    db = MagicMock()
    service = PaperService(db)
    service.paper_repo.save = MagicMock(side_effect=RuntimeError("db down"))

    monkeypatch.setattr("app.services.paper_service.search_papers", lambda **kwargs: [_paper()])

    result = service.search(PaperSearchRequest(query="目标检测"))

    assert len(result) == 1
    assert result[0].paper_id == "p1"
    db.rollback.assert_called_once()


def test_search_uses_dynamic_default_year_range(monkeypatch):
    db = MagicMock()
    service = PaperService(db)
    service._save_search_results = MagicMock()
    captured = {}

    monkeypatch.setattr("app.services.paper_service.current_year", lambda: 2026)

    def fake_search(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("app.services.paper_service.search_papers", fake_search)

    service.search(PaperSearchRequest(query="目标检测"))

    assert captured["start_year"] == 2024
    assert captured["end_year"] == 2026


def test_single_card_api_extracts_abstract_with_llm_and_preserves_source(monkeypatch):
    from app.api.routes_paper import generate_paper_card_api

    paper = _paper().model_copy(update={"abstract": "We use a dual encoder for document retrieval."})
    db = MagicMock()
    service = PaperService(db)
    service.paper_repo.get_by_id = MagicMock(return_value=paper)
    service.card_repo.save = MagicMock()
    service.download_and_parse = MagicMock(return_value={"parsed": {}})
    model = MagicMock()
    model.complete.return_value = json.dumps({"method": paper.abstract})
    monkeypatch.setattr("app.services.llm_service.LLMService", lambda: model)
    monkeypatch.setattr("app.api.routes_paper.PaperService", lambda db: service)

    response = generate_paper_card_api(paper.paper_id, db)
    card = service.card_repo.save.call_args.args[0]
    assert model.complete.call_count == 1
    assert card.method == paper.abstract
    assert card.evidence_source == "abstract"
    assert card.field_claims["method"][0].explicitly_reported
    assert any(span.text == paper.abstract for span in card.evidence_spans)
    assert response.data["method"] == paper.abstract
    assert response.data["quality_status"] == card.quality_status
    db.commit.assert_called_once()


def test_single_card_missing_paper_stays_404_without_model(monkeypatch):
    from fastapi import HTTPException
    from app.api.routes_paper import generate_paper_card_api

    service = PaperService(MagicMock())
    service.paper_repo.get_by_id = MagicMock(return_value=None)
    service.build_paper_cards = MagicMock()
    monkeypatch.setattr("app.api.routes_paper.PaperService", lambda db: service)
    with pytest.raises(HTTPException) as exc:
        generate_paper_card_api("missing", service.db)
    assert exc.value.status_code == 404
    service.build_paper_cards.assert_not_called()


@pytest.mark.parametrize("chunk_limit", [1, 8])
def test_single_full_text_card_round_trip_keeps_coverage_and_evidence(monkeypatch, chunk_limit):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.database.models import Base
    from app.schemas.paper_schema import PaperCard
    from app.core.config import get_settings
    from tests.test_full_text_extraction import METHOD, RESULT, LIMIT, Model, parsed

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    source = parsed("Methods\n" + METHOD, "Results\n" + RESULT,
                    "Limitations\n" + LIMIT + "\nReferences\nBibliographic entry.")
    monkeypatch.setattr(get_settings(), "card_extraction_max_chunks", chunk_limit)
    model = Model()
    monkeypatch.setattr("app.services.llm_service.LLMService", lambda: model)
    with Session(engine) as db:
        service = PaperService(db)
        service.paper_repo.save(_paper())
        db.commit()
        monkeypatch.setattr(service, "download_and_parse", lambda papers: {"parsed": {"p1": source}})
        card = service.build_paper_card("p1")
        restored = service.card_repo.get_by_paper_id("p1")
        serialized = PaperCard.model_validate_json(restored.model_dump_json())
        assert len(model.calls) == min(chunk_limit, 3)
        assert card.quality_status == serialized.quality_status == ("valid" if chunk_limit == 8 else "partial")
        assert serialized.extraction_coverage == card.extraction_coverage
        assert serialized.extraction_coverage.status == ("complete" if chunk_limit == 8 else "partial")
        assert serialized.field_claims == card.field_claims
        assert serialized.evidence_spans == card.evidence_spans
        for span in serialized.evidence_spans:
            if span.offset_basis == "page":
                assert source["pages"][span.page - 1]["text"][span.char_start:span.char_end] == span.text
    engine.dispose()


@pytest.mark.parametrize("failure", ["model", "vector", "budget"])
def test_single_card_failures_do_not_become_success(monkeypatch, failure):
    from app.schemas.vector_schema import VectorStoreUnavailable
    from app.agent.execution_budget import AgentBudgetExceeded
    from tests.test_full_text_extraction import METHOD, Model, parsed

    db = MagicMock()
    service = PaperService(db)
    service.paper_repo.get_by_id = MagicMock(return_value=_paper())
    service.card_repo.save = MagicMock()
    service.download_and_parse = MagicMock(return_value={"parsed": {"p1": parsed("Methods\n" + METHOD)}})
    model = Model(error=RuntimeError("provider unavailable") if failure == "model" else AgentBudgetExceeded("stop"))
    monkeypatch.setattr("app.services.llm_service.LLMService", lambda: model)
    if failure == "vector":
        def unavailable(*args, **kwargs):
            raise VectorStoreUnavailable("test unavailable")
        monkeypatch.setattr("app.tools.chunk_retriever.retrieve_chunks", unavailable)
    if failure == "model":
        card = service.build_paper_card("p1")
        assert card.quality_status == "invalid"
        assert card.extraction_coverage.status == "partial"
        assert not card.field_claims
    else:
        with pytest.raises(VectorStoreUnavailable if failure == "vector" else AgentBudgetExceeded):
            service.build_paper_card("p1")
        service.card_repo.save.assert_not_called()
        db.commit.assert_not_called()


def test_sqlite_card_coverage_migration_is_additive_and_idempotent(monkeypatch):
    from sqlalchemy import create_engine, inspect
    from app.database import db as database
    from app.core.config import get_settings

    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE paper_cards (paper_id TEXT PRIMARY KEY)")
        connection.exec_driver_sql("INSERT INTO paper_cards VALUES ('existing')")
        connection.exec_driver_sql("CREATE TABLE reviews (id INTEGER PRIMARY KEY)")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(get_settings(), "database_url", "sqlite://")
    database._migrate_sqlite_evidence_card_columns()
    database._migrate_sqlite_evidence_card_columns()
    assert "extraction_coverage" in {c["name"] for c in inspect(engine).get_columns("paper_cards")}
    with engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT paper_id, extraction_coverage FROM paper_cards").one() == ("existing", "")
    engine.dispose()
