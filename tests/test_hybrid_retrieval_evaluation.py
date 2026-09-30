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
