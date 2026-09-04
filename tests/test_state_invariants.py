from app.agent.state_invariants import validate_research_state_invariants


def test_time_window_mismatch_is_blocking():
    result = validate_research_state_invariants({
        "research_request": {"start_year": 2022, "end_year": 2024},
        "start_year": 2024,
        "end_year": 2026,
    })
    assert result["valid"] is False
    assert result["blocking_issues"][0]["code"] == "state_time_window_mismatch"


def test_stale_gap_snapshot_is_blocking():
    result = validate_research_state_invariants({
        "evidence_snapshot_version": 3,
        "evidence_snapshot_fingerprint": "new",
        "evidence_gap_report": {
            "evidence_snapshot_version": 2,
            "evidence_snapshot_fingerprint": "old",
        },
    })
    codes = {item["code"] for item in result["blocking_issues"]}
    assert "stale_evidence_snapshot" in codes


def test_needs_recovery_cannot_be_ready_without_explanation():
    result = validate_research_state_invariants({
        "evidence_snapshot_version": 1,
        "evidence_gap_report": {
            "needs_recovery": True,
            "evidence_snapshot_version": 1,
        },
        "generation_readiness": {"ready": True, "blocking_issues": []},
    })
    codes = {item["code"] for item in result["blocking_issues"]}
    assert "recovery_readiness_conflict" in codes


def test_empty_source_is_not_healthy_and_mismatch_is_reported():
    result = validate_research_state_invariants({
        "source_diagnostics": [{"source": "openalex", "status": "empty"}],
        "evidence_gap_report": {"source_health": "healthy"},
    })
    assert result["source_health"] == "empty"
    assert result["warnings"][0]["code"] == "source_health_snapshot_mismatch"


def test_consistent_state_has_no_violation():
    result = validate_research_state_invariants({
        "research_request": {"start_year": 2024, "end_year": 2026},
        "start_year": 2024,
        "end_year": 2026,
        "source_diagnostics": [{"source": "openalex", "status": "success"}],
        "evidence_gap_report": {"source_health": "healthy"},
    })
    assert result["valid"] is True
    assert result["blocking_issues"] == []
