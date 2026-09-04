"""PaperService 业务流程测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

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
