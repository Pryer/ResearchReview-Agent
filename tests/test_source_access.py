"""来源禁用必须在调度和直接客户端入口生效；所有外部调用均为 Mock。"""
from unittest.mock import Mock

import pytest

from app.core.config import Settings, get_settings
from app.tools.search_papers import search_papers


def test_default_sources_exclude_cnki_even_without_local_env(monkeypatch):
    monkeypatch.delenv("DEFAULT_SEARCH_SOURCES", raising=False)
    monkeypatch.delenv("CNKI_ENABLED", raising=False)
    settings = Settings(_env_file=None)
    assert settings.search_sources_list == ["arxiv", "semantic_scholar", "openalex", "crossref"]
    assert settings.cnki_enabled is False


def test_explicit_cnki_source_cannot_override_disabled_adapter(monkeypatch):
    client = Mock(side_effect=AssertionError("disabled client must not be called"))
    monkeypatch.setattr(get_settings(), "cnki_enabled", False)
    monkeypatch.setattr("app.tools.search_papers._search_cnki", client)
    open_client = Mock(return_value=[])
    monkeypatch.setattr("app.tools.search_papers._search_openalex", open_client)
    diagnostics = []
    assert search_papers("动作识别", 2024, 2026, 5, sources=["cnki", "openalex"],
                         diagnostics=diagnostics, enable_dual_channel=False) == []
    client.assert_not_called()
    open_client.assert_called_once()
    disabled = next(item for item in diagnostics if item.source == "cnki")
    assert disabled.outcome == "skipped"
    assert disabled.error_code == "SOURCE_DISABLED"


def test_direct_cnki_client_and_driver_cannot_bypass_disabled_setting(monkeypatch):
    from app.clients import cnki_client
    monkeypatch.setattr(cnki_client.settings, "cnki_enabled", False)
    build = Mock(side_effect=AssertionError("browser must not start"))
    events = []
    original_build = cnki_client.build_driver
    monkeypatch.setattr(cnki_client, "build_driver", build)
    monkeypatch.setattr(cnki_client, "_record_search_diagnostic",
                        lambda outcome, **kw: events.append({"outcome": outcome, **kw}))
    assert cnki_client.search_cnki("动作识别", 2024, 2026, 5) == []
    build.assert_not_called()
    assert events[0]["error_code"] == "SOURCE_DISABLED"
    with pytest.raises(PermissionError, match="CNKI_ENABLED"):
        original_build()


def test_paper_api_explicit_source_stays_disabled(monkeypatch):
    from app.api.routes_paper import search_papers_api
    from app.schemas.paper_schema import PaperSearchRequest
    monkeypatch.setattr(get_settings(), "cnki_enabled", False)
    client = Mock(side_effect=AssertionError("API must respect deployment policy"))
    monkeypatch.setattr("app.tools.search_papers._search_cnki", client)
    response = search_papers_api(PaperSearchRequest(query="动作识别", sources=["cnki"]), db=Mock())
    assert response.data == []
    client.assert_not_called()
