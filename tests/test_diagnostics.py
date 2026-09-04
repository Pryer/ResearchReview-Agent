"""诊断仪表盘与评测数据包导出测试。"""

from __future__ import annotations

from app.agent.diagnostics import collect_diagnostics, export_evaluation_bundle


def _state() -> dict:
    return {"user_query": "课堂行为分析综述", "review": "## 研究现状\n\n正文 [p1]。", "claim_plans": []}


def test_retrieval_diagnostics_aggregate_outcomes_and_language_gap():
    result = collect_diagnostics({
        "source_diagnostics": [
            {"source": "openalex", "status": "empty", "outcome": "success_empty"},
            {"source": "arxiv", "status": "failed", "outcome": "rate_limited", "error_code": "HTTP_429", "message": "限流"},
            {"source": "cnki", "status": "skipped", "outcome": "query_not_adapted"},
        ],
    })["retrieval"]
    assert result["outcome_counts"] == {
        "success_empty": 1, "rate_limited": 1, "query_not_adapted": 1,
    }
    assert result["language_gap_sources"] == ["cnki"]
    assert result["recent_errors"][0]["error_code"] == "HTTP_429"


def test_export_bundle_rotates_old_directories(tmp_path):
    """每次导出后只保留最新 20 个 eval_bundle_* 目录，不无限增长。"""
    bundles_root = tmp_path / "eval_bundles"
    bundles_root.mkdir()
    for index in range(25):
        legacy = bundles_root / f"eval_bundle_20260101_{index:06d}"
        legacy.mkdir()
        (legacy / "metadata.json").write_text("{}", encoding="utf-8")

    target = bundles_root / "eval_bundle_20260821_120000"
    result = export_evaluation_bundle(_state(), output_dir=str(target))

    assert result == str(target.resolve())
    remaining = sorted(p.name for p in bundles_root.iterdir())
    # 最老的 6 个被清理（25 个遗留 + 新 1 个 - 保留上限 20）
    assert len(remaining) == 20
    assert target.name in remaining
    assert "eval_bundle_20260101_000000" not in remaining
    assert (target / "writer_output.md").read_text(encoding="utf-8").startswith("## 研究现状")


def test_export_bundle_without_eval_bundles_parent_does_not_rotate(tmp_path):
    """默认/自定义目录名不匹配 eval_bundles 约定时不得误删任何目录。"""
    custom_root = tmp_path / "custom_output"
    sibling = custom_root / "important_data"
    sibling.mkdir(parents=True)

    export_evaluation_bundle(_state(), output_dir=str(custom_root / "bundle"))

    assert sibling.exists()
