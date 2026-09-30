"""混合检索的召回互补、模型响应和语义资格边界。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.clients.embedding_client import EmbeddingClient
from app.clients.reranker_client import RerankerClient
from app.clients.retrieval_http import RetrievalProviderError
from app.schemas.retrieval_schema import RetrievalQuery
from app.services.retrieval_ranking_service import rank_candidates, screen_candidates
from app.tools.hybrid_retrieval import fuse_rankings, lexical_rank


def _settings(tmp_path):
    return SimpleNamespace(
        retrieval_provider="dashscope", dashscope_api_key="fake-key",
        retrieval_embedding_url="https://example.test/embedding",
        retrieval_rerank_url="https://example.test/rerank",
        retrieval_embedding_model="qwen3.7-text-embedding",
        retrieval_rerank_model="qwen3.7-text-rerank",
        retrieval_embedding_dimension=2, retrieval_embedding_batch_size=20,
        retrieval_rerank_batch_size=32, retrieval_request_timeout=1,
        retrieval_max_retries=0, retrieval_query_max=6,
        retrieval_bm25_top_k=4, retrieval_dense_top_k=4,
        retrieval_rrf_k=60, retrieval_cross_encoder_initial_k=4,
        retrieval_cross_encoder_max_k=4, retrieval_cache_dir=str(tmp_path),
        retrieval_cache_version="test-v1",
    )


class _FakeEmbedding:
    def __init__(self):
        self.calls = 0

    def embed(self, texts, *, text_type, should_cancel=None):
        self.calls += 1
        if text_type == "query":
            return [[1.0, 0.0] for _ in texts]
        return [[1.0, 0.0] if "pupils" in text else [0.0, 1.0] for text in texts]


class _FakeReranker:
    def __init__(self):
        self.calls = 0

    def score(self, query, documents, *, should_cancel=None):
        self.calls += 1
        return [0.9 if "pupils" in text else 0.1 for text in documents]


def test_dense_only_candidate_survives_hard_filter_and_ce_window(tmp_path):
    settings = _settings(tmp_path)
    state = {
        "topic": "classroom interaction",
        "start_year": 2024, "end_year": 2026,
        "required_concepts": [["classroom", "课堂"]],
        "candidate_papers": [
            {"paper_id": "semantic", "title": "Observing pupils and teachers",
             "abstract": "Ethnographic study of communication in lessons", "year": 2025},
            {"paper_id": "lexical", "title": "Classroom interaction index",
             "abstract": "A generic unrelated benchmark", "year": 2025},
            {"paper_id": "thesis", "title": "Classroom interaction thesis",
             "venue": "某大学", "doi": "10.1234/d.cnki.example", "year": 2025},
            {"paper_id": "old", "title": "Observing pupils", "year": 2020},
        ],
    }
    ranked, report = rank_candidates(
        state, settings=settings, include_rerank=True,
        embedding_client=_FakeEmbedding(), reranker_client=_FakeReranker(),
    )
    assert [paper["paper_id"] for paper in ranked] == ["semantic", "lexical"]
    assert ranked[0]["_retrieval_features"]["bm25"] is None
    assert ranked[0]["_retrieval_features"]["cosine"] is not None
    assert ranked[0]["_pending_semantic_check"] is True
    assert report["deterministic_exclusions"]["year"] == 1
    assert report["deterministic_exclusions"]["document_type_filter"] == 1


def test_zero_lexical_scores_do_not_create_arbitrary_matches():
    assert lexical_rank(["unrelated text", "different sample"], "课堂行为", 10) == []
    query = RetrievalQuery("课堂行为", "overall", "overall", "test")
    result = fuse_rankings(2, [query], [[]], [[(1, 0.8)]])
    assert [item["index"] for item in result] == [1]
    assert result[0]["bm25"] is None


class _FakeResponse:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload

    def post(self, *args, **kwargs):
        return _FakeResponse(self.payload)


def test_embedding_response_reorders_and_rejects_invalid(tmp_path):
    settings = _settings(tmp_path)
    payload = {"output": {"embeddings": [
        {"text_index": 1, "embedding": [0.0, 2.0]},
        {"text_index": 0, "embedding": [3.0, 0.0]},
    ]}}
    client = EmbeddingClient(settings, session=_FakeSession(payload))
    assert client.embed(["a", "b"], text_type="document") == [[1.0, 0.0], [0.0, 1.0]]
    payload["output"]["embeddings"][1]["embedding"] = [0.0, 0.0]
    with pytest.raises(RetrievalProviderError, match="范数"):
        client.embed(["a", "b"], text_type="document")


def test_reranker_response_preserves_document_identity(tmp_path):
    settings = _settings(tmp_path)
    payload = {"output": {"results": [
        {"index": 1, "relevance_score": 0.8},
        {"index": 0, "relevance_score": 0.2},
    ]}}
    client = RerankerClient(settings, session=_FakeSession(payload))
    assert client.score("q", ["a", "b"]) == [0.2, 0.8]
    payload["output"]["results"][1]["index"] = 1
    with pytest.raises(RetrievalProviderError, match="重复"):
        client.score("q", ["a", "b"])


class _FakeLLM:
    def __init__(self, include_condition):
        self.include_condition = include_condition
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        return json.dumps({"results": [{
            "paper_id": "p1", "topic_relevance": 9, "scope_alignment": 9,
            "method_alignment": 8, "decision": "include", "confidence": 0.95,
            "route_id": None, "relation_type": "direct",
            "eligible_deliverables": ["related_work"],
            "required_condition_results": ([{"condition_id": "confirmed_scope", "verdict": "satisfied"}]
                if self.include_condition else []),
        }]})


def test_unconfirmed_scope_cannot_become_evidence():
    state = {"topic": "classroom interaction", "selected_scope": {"include_terms": ["school"]}}
    paper = {"paper_id": "p1", "title": "School interaction", "abstract": "Observations", "_rank_score": 0.8}
    accepted, report = screen_candidates(state, [paper.copy()], _FakeLLM(False), target=1)
    assert accepted == []
    assert report["confirmed_count"] == 0
    accepted, _ = screen_candidates(state, [paper.copy()], _FakeLLM(True), target=1)
    assert [item["paper_id"] for item in accepted] == ["p1"]
    assert accepted[0]["_screened_content_fingerprint"]


def test_hybrid_hard_filter_does_not_mutate_confirmed_semantic_fields():
    from app.tools.rank_papers import evaluate_paper_hard_filters

    paper = {"paper_id": "p1", "title": "Classroom interaction study",
             "abstract": "Observations of teachers and students", "year": 2025,
             "_screening_decision": "include", "_topic_relation": "direct",
             "_eligible_deliverables": ["related_work"],
             "_pending_semantic_check": False}
    before = dict(paper)
    passed, _, _ = evaluate_paper_hard_filters(
        paper, topic="classroom interaction", ranking_mode="hybrid")
    assert passed
    assert paper == before


def test_detail_change_requires_fresh_semantic_screen(monkeypatch):
    from app.agent.nodes.retrieval import fetch_detail_node
    from app.services.retrieval_ranking_service import _document
    from app.tools.paper_matching import compile_scope

    original = {"paper_id": "p1", "title": "School interaction",
                "abstract": "Initial summary", "year": 2025,
                "_screening_decision": "include",
                "_semantic_conditions_confirmed": True,
                "_semantic_contract_complete": True,
                "_screening_confidence": 0.95,
                "_eligible_deliverables": ["related_work"],
                "_screened_content_fingerprint": "stale"}
    compiled = compile_scope(topic="classroom interaction")
    original["_screened_scope_fingerprint"] = compiled["fingerprint"]
    monkeypatch.setattr("app.tools.fetch_metadata.fetch_batch_details", lambda papers: [
        {**papers[0], "abstract": "Revised summary"}
    ])
    state = {"topic": "classroom interaction", "ranked_papers": [original],
             "retrieval_profile": {"mode": "hybrid"},
             "required_reference_count": 1, "max_papers": 1,
             "generation_limit": 1, "start_year": 2024, "end_year": 2026,
             "steps": []}
    fetch_detail_node(state, llm=None)
    assert state["paper_details"] == []
    assert state["retrieval_requirement_met"] is False
    state["steps"] = []
    fetch_detail_node(state, llm=_FakeLLM(True))
    assert len(state["paper_details"]) == 1
    assert state["paper_details"][0]["_screened_content_fingerprint"] == _document(
        state["paper_details"][0]
    ).content_fingerprint
    monkeypatch.setattr("app.tools.fetch_metadata.fetch_batch_details", lambda papers: [
        {**papers[0], "abstract": "Revised summary", "year": 2020}
    ])
    fetch_detail_node(state, llm=_FakeLLM(True))
    assert state["paper_details"] == []


def test_unchanged_detail_keeps_complete_semantic_admission(monkeypatch):
    from app.agent.nodes.retrieval import fetch_detail_node
    from app.agent.deliverable_router import check_generation_readiness
    from app.services.retrieval_ranking_service import _document
    from app.tools.extract_paper_card import extract_paper_card
    from app.tools.paper_matching import compile_scope

    scope = compile_scope(topic="classroom interaction")
    state = {
        "topic": "classroom interaction", "compiled_scope": scope,
        "retrieval_profile": {"mode": "hybrid"},
        "required_reference_count": 1, "max_papers": 1,
        "generation_limit": 1, "start_year": 2024, "end_year": 2026,
        "max_papers_explicit": True, "core_deliverables": ["related_work"],
        "steps": [],
    }
    llm = _FakeLLM(True)
    ranked, _ = screen_candidates(state, [{"paper_id": "p1",
        "title": "School classroom interaction",
        "abstract": ("This study examines classroom interaction using observation coding. "
                     "The analysis reports teacher student interaction patterns and student engagement."),
        "year": 2025, "_rank_score": 0.8}], llm, target=1)
    assert len(ranked) == 1
    assert ranked[0]["_screened_content_fingerprint"] == _document(ranked[0]).content_fingerprint
    state["ranked_papers"] = ranked
    monkeypatch.setattr("app.tools.fetch_metadata.fetch_batch_details", lambda papers: [dict(papers[0])])

    fetch_detail_node(state, llm=llm)

    assert llm.calls == 1
    assert len(state["paper_details"]) == 1
    detail = state["paper_details"][0]
    for key in ("_screening_decision", "_topic_relation", "_eligible_deliverables",
                "_screening_confidence", "_semantic_conditions_confirmed",
                "_semantic_contract_complete", "_screened_content_fingerprint",
                "_screened_scope_fingerprint"):
        assert detail[key] == ranked[0][key]
    assert detail["_pending_semantic_check"] is False
    card = extract_paper_card(detail, topic=state["topic"]).model_dump(mode="json")
    state["paper_cards"] = [card]
    assert card["relation_type"] == "direct"
    assert card["eligible_deliverables"] == ["related_work"]
    assert check_generation_readiness(state).usable_reference_count == 1


def test_query_expansion_keeps_admission_scope_but_user_scope_change_invalidates_it(monkeypatch):
    from app.agent.nodes.retrieval import fetch_detail_node
    from app.services.retrieval_ranking_service import admission_scope_fingerprint

    state = {
        "topic": "classroom interaction", "retrieval_profile": {"mode": "hybrid"},
        "required_reference_count": 1, "max_papers": 1, "generation_limit": 1,
        "start_year": 2024, "end_year": 2026, "steps": [],
    }
    llm = _FakeLLM(True)
    ranked, _ = screen_candidates(state, [{"paper_id": "p1",
        "title": "School classroom interaction", "abstract": "Observations",
        "year": 2025, "_rank_score": 0.8}], llm, target=1)
    original_scope = admission_scope_fingerprint(state)
    state.update(ranked_papers=ranked, paper_details=[dict(ranked[0])],
                 incremental_retrieval=True, search_branches=[{
                     "branch_type": "targeted_recovery", "queries": ["teacher dialogue"]}],
                 required_concepts=[["interaction"]], topic_anchors=[])
    fetches = []
    def fetch(papers):
        fetches.append(len(papers))
        return [dict(paper) for paper in papers]
    monkeypatch.setattr("app.tools.fetch_metadata.fetch_batch_details", fetch)

    fetch_detail_node(state, llm=llm)

    assert admission_scope_fingerprint(state) == original_scope
    assert len(state["paper_details"]) == 1
    assert state["paper_details"][0]["_topic_relation"] == "direct"
    assert llm.calls == 1
    assert fetches == []

    state["selected_scope"] = {"include_terms": ["school"]}
    fetch_detail_node(state, llm=None)
    assert admission_scope_fingerprint(state) != original_scope
    assert state["paper_details"] == []
    assert fetches == [1]


def test_screening_cache_reuses_only_unchanged_material():
    state = {"topic": "classroom interaction"}
    paper = {"paper_id": "p1", "title": "School interaction",
             "abstract": "Observations", "_rank_score": 0.8}
    llm = _FakeLLM(True)
    first, _ = screen_candidates(state, [paper.copy()], llm, target=1)
    second, diagnostics = screen_candidates(state, [paper.copy()], llm, target=1)
    assert first and second
    assert llm.calls == 1
    assert diagnostics["screening_cache_hits"] == 1
    changed = {**paper, "abstract": "New evidence about lessons"}
    screen_candidates(state, [changed], llm, target=1)
    assert llm.calls == 2


def test_embedding_and_rerank_cache_reuse_on_same_pool(tmp_path):
    settings = _settings(tmp_path)
    state = {"topic": "classroom interaction", "candidate_papers": [
        {"paper_id": "p1", "title": "Observing pupils", "year": 2025},
        {"paper_id": "p2", "title": "Classroom interaction", "year": 2025},
    ]}
    first_embed, first_rerank = _FakeEmbedding(), _FakeReranker()
    rank_candidates(state, settings=settings, include_rerank=True,
                    embedding_client=first_embed, reranker_client=first_rerank)
    second_embed, second_rerank = _FakeEmbedding(), _FakeReranker()
    _, report = rank_candidates(state, settings=settings, include_rerank=True,
                                embedding_client=second_embed, reranker_client=second_rerank)
    assert first_embed.calls > 0 and first_rerank.calls > 0
    assert second_embed.calls == 0 and second_rerank.calls == 0
    assert report["embedded_new_count"] == report["reranked_new_count"] == 0


def test_embedding_usage_shares_execution_budget(tmp_path):
    from app.agent.execution_budget import AgentBudgetExceeded, budget_scope
    from app.core.config import get_settings

    settings = _settings(tmp_path)
    payload = {"output": {"embeddings": [{"text_index": 0, "embedding": [1.0, 0.0]}]},
               "usage": {"total_tokens": 7}}
    state = {}
    with budget_scope(state):
        EmbeddingClient(settings, session=_FakeSession(payload)).embed(["sample"], text_type="query")
        ledger = state["agent_execution_budget"]
        assert ledger["retrieval_embedding_requests"] == 1
        assert ledger["retrieval_model_tokens"] == 7
        ledger["retrieval_model_requests"] = get_settings().retrieval_model_request_limit
        with pytest.raises(AgentBudgetExceeded, match="retrieval model request"):
            EmbeddingClient(settings, session=_FakeSession(payload)).embed(["sample"], text_type="query")
