"""单一分层 RAG 入口、旧状态迁移及依赖失败契约；无真实模型调用。"""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.agent.orchestration import normalize_retrieval_profile
from app.services.rag_runtime import check_rag_runtime, require_rag_runtime, RagRuntimeUnavailable


def test_production_defaults_enable_hierarchical_rag():
    defaults = {name: field.default for name, field in Settings.model_fields.items()}
    assert defaults["retrieval_ranking_mode"] == "hybrid"
    assert defaults["card_chunk_retrieval_mode"] == "hybrid"
    assert defaults["card_vector_backend"] == "pgvector"
    assert defaults["card_vector_db_failure_policy"] == "fail"
    assert "enable_llm_card_extraction" not in defaults


@pytest.mark.parametrize("field,value", [
    ("retrieval_ranking_mode", "rules"), ("retrieval_ranking_mode", "hybrid_shadow"),
    ("card_chunk_retrieval_mode", "sections"), ("card_vector_db_failure_policy", "sections"),
])
def test_removed_business_modes_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize("mode", ["rules", "hybrid_shadow"])
def test_legacy_profile_keeps_evidence_constraints_and_budget_but_revokes_authorization(mode):
    state = {"retrieval_profile": {"mode": mode}, "ranked_papers": [{"paper_id": "p"}],
             "paper_cards": [{"paper_id": "p", "method": "historical"}],
             "required_reference_count": 40, "start_year": 2024, "end_year": 2026,
             "agent_execution_budget": {"llm_tokens": 1234}, "review": "Historical draft",
             "claim_plans": [{"claims": ["historical"]}], "quality_gate": {"passed": True},
             "autonomous_verified_fingerprint": "old"}
    normalize_retrieval_profile(state)
    assert state["retrieval_profile"]["mode"] == "hybrid"
    assert state["ranked_papers"] == [] and state["candidate_papers"] == [{"paper_id": "p"}]
    assert state["paper_cards"][0]["method"] == "historical" and state["review"] == "Historical draft"
    assert (state["required_reference_count"], state["start_year"], state["end_year"]) == (40, 2024, 2026)
    assert state["agent_execution_budget"] == {"llm_tokens": 1234}
    assert not any(key in state for key in ("claim_plans", "quality_gate", "autonomous_verified_fingerprint"))
    normalized = deepcopy(state)
    normalize_retrieval_profile(state)
    assert state == normalized


def test_config_only_precheck_does_not_connect_or_invoke_models(monkeypatch):
    settings = get_settings().model_copy(update={"card_vector_backend": "pgvector", "card_vector_database_url": ""})
    report = check_rag_runtime(settings, check_database=False)
    assert not report["ready"] and report["checks"]["vector_connection"] is False
    assert report["database_checked"] is False
    assert "database_url" not in str(report)


def test_missing_database_stops_before_request_planning_or_model_call(monkeypatch):
    from app.agent import graph
    settings = get_settings().model_copy(update={"card_vector_backend": "pgvector", "card_vector_database_url": ""})
    monkeypatch.setattr(graph, "get_settings", lambda: settings)
    monkeypatch.setattr(graph, "require_rag_runtime", require_rag_runtime)
    monkeypatch.setattr(graph, "_get_llm", lambda: pytest.fail("Provider 不应被调用"))
    with pytest.raises(RagRuntimeUnavailable, match="vector_connection"):
        graph.run_research_agent("生成一篇研究综述")


def test_database_precheck_requires_migrated_schema_and_dimension():
    from app.schemas.vector_schema import VectorIntegrityError
    settings = get_settings().model_copy(update={"card_vector_backend": "pgvector", "card_vector_database_url": "postgresql://private"})
    class Broken:
        def ready(self):
            raise VectorIntegrityError("private secret in driver")
    report = check_rag_runtime(settings, store=Broken())
    assert not report["ready"] and report["checks"]["vector_schema"] is False
    assert "private" not in str(report)


def test_unreranked_candidates_do_not_require_reranker_configuration(tmp_path, monkeypatch):
    from app.services.retrieval_ranking_service import rank_candidates
    from tests.retrieval_fakes import OfflineEmbedding
    settings = get_settings().model_copy(update={"retrieval_rerank_url": "", "retrieval_cache_dir": str(tmp_path)})
    class Forbidden:
        def __init__(self, *args):
            pytest.fail("未启用精排不能构造 Reranker")
    monkeypatch.setattr("app.services.retrieval_ranking_service.RerankerClient", Forbidden)
    ranked, report = rank_candidates({"topic": "retrieval", "candidate_papers": [
        {"paper_id": "p", "title": "Retrieval evidence", "abstract": "Retrieval study", "year": 2025},
    ]}, settings=settings, include_rerank=False, embedding_client=OfflineEmbedding(settings))
    assert ranked and report["rerank_calls"] is None
    assert ranked[0]["_pending_semantic_check"]


def test_cancelled_task_does_not_attempt_runtime_precheck(monkeypatch):
    from app.agent import graph
    monkeypatch.setattr(graph, "require_rag_runtime", lambda settings: pytest.fail("取消后不能检查依赖"))
    with pytest.raises(graph.AgentCancelledError):
        graph.run_research_agent("生成综述", should_cancel=lambda: True)


def test_old_pipeline_cannot_reuse_verified_draft_fingerprint(monkeypatch):
    from app.agent.graph import _draft_fingerprint
    from app.services import rag_runtime
    state = {"review": "A previously verified draft", "paper_cards": []}
    current = _draft_fingerprint(state)
    monkeypatch.setattr(rag_runtime, "RAG_PIPELINE_VERSION", "old-rules-path")
    assert _draft_fingerprint(state) != current


@pytest.mark.parametrize("entry", ["initial", "resume", "continue", "regenerate"])
def test_cancelled_entry_does_not_migrate_historical_state(entry):
    from app.agent import graph
    state = {"retrieval_profile": {"mode": "rules"}, "claim_plans": [{"historical": True}]}
    previous = deepcopy(state)
    with pytest.raises(graph.AgentCancelledError):
        if entry == "initial":
            graph.run_research_agent("生成综述", initial_state=state, should_cancel=lambda: True)
        elif entry == "continue":
            graph.continue_research_agent(state, should_cancel=lambda: True)
        elif entry == "regenerate":
            graph.regenerate_research_agent(state, should_cancel=lambda: True)
        else:
            graph._run_autonomous_pipeline(state, should_cancel=lambda: True)
    assert state == previous
