"""评测输入完整性与跨语言排名位置。"""

import json

import pytest

from scripts.evaluate_hybrid_retrieval import _metrics, evaluate


def test_language_metric_keeps_global_rank_positions():
    result = _metrics(["english", "chinese"], {"chinese": 3}, 2)
    assert result["recall"] == 1.0
    assert result["ndcg"] < 1.0


def test_qrels_must_cover_entire_candidate_pool(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    qrels = tmp_path / "qrels.jsonl"
    profile = tmp_path / "profile.json"
    snapshot.write_text(json.dumps({"tasks": [{"task_id": "t", "topic": "example",
        "candidate_papers": [{"paper_id": "a", "title": "A"},
                             {"paper_id": "b", "title": "B"}]}]}), encoding="utf-8")
    qrels.write_text(json.dumps({"task_id": "t", "paper_id": "a", "relevance": 1}), encoding="utf-8")
    profile.write_text(json.dumps({"k": 1, "split": "dev"}), encoding="utf-8")
    with pytest.raises(ValueError, match="每一篇"):
        evaluate(snapshot, qrels, profile)


def test_single_channel_ablation_keeps_hits_outside_fusion_window(monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.services import retrieval_ranking_service as ranking

    settings = get_settings().model_copy(update={
        "retrieval_cross_encoder_initial_k": 1, "retrieval_cross_encoder_max_k": 1,
        "language_branch_min_zh": 0, "language_branch_min_en": 0,
        "retrieval_cache_dir": str(tmp_path),
    })
    state = {"topic": "retrieval", "candidate_papers": [
        {"paper_id": "a", "doi": "10.1/a", "title": "Retrieval A", "year": 2024},
        {"paper_id": "b", "doi": "10.1/b", "title": "Retrieval B", "year": 2024},
    ]}
    monkeypatch.setattr(ranking, "lexical_rank", lambda *a, **k: [(0, 1.0)])
    monkeypatch.setattr(ranking, "dense_rank", lambda *a, **k: [(1, 0.9)])
    monkeypatch.setattr(ranking, "_embed_cached", lambda client, cache, settings, texts, *a: ([[1.0]] * len(texts), len(texts)))
    papers, report = ranking.rank_candidates(state, settings=settings, include_channel_rankings=True)
    assert len(papers) == 1
    assert report["channel_rankings"]["bm25"] == ["a"]
    assert report["channel_rankings"]["dense"] == ["b"]
    assert report["channel_rankings"]["dense"][0] not in [p["paper_id"] for p in papers]


def test_paid_semantic_evaluation_stops_on_first_failed_batch():
    from app.tools.paper_rerank import llm_rerank_papers

    class FailedProvider:
        calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("offline provider failure")

    llm = FailedProvider()
    papers = [{"paper_id": str(i), "title": "Retrieval", "abstract": "A study"} for i in range(20)]
    with pytest.raises(RuntimeError, match="offline provider failure"):
        llm_rerank_papers(papers, "retrieval", llm=llm, batch_size=1, strict_provider_errors=True)
    assert llm.calls == 1
