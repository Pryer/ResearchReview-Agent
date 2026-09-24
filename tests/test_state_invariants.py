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


def test_stale_global_evidence_gate_is_blocking():
    result = validate_research_state_invariants({
        "evidence_snapshot_version": 3,
        "evidence_snapshot_fingerprint": "new",
        "global_evidence_gate": {
            "status": "EVALUATED",
            "evidence_snapshot_version": 2,
            "evidence_snapshot_fingerprint": "old",
        },
    })
    codes = {item["code"] for item in result["blocking_issues"]}
    assert "stale_global_evidence_gate" in codes


def test_current_global_evidence_gate_is_not_blocking():
    result = validate_research_state_invariants({
        "evidence_snapshot_version": 3,
        "evidence_snapshot_fingerprint": "same",
        "global_evidence_gate": {
            "status": "EVALUATED",
            "evidence_snapshot_version": 3,
            "evidence_snapshot_fingerprint": "same",
        },
    })
    assert result["valid"] is True


def test_unversioned_legacy_gate_is_not_blocking():
    # 历史会话的门禁没有快照版本，不能因为缺字段就判成陈旧并丢弃。
    result = validate_research_state_invariants({
        "evidence_snapshot_version": 3,
        "global_evidence_gate": {"status": "EVALUATED", "passed": True},
    })
    assert result["valid"] is True


def test_failed_gate_is_not_checked_for_staleness():
    result = validate_research_state_invariants({
        "evidence_snapshot_version": 3,
        "global_evidence_gate": {"status": "FAILED", "error": "boom"},
    })
    codes = {item["code"] for item in result["blocking_issues"]}
    assert "stale_global_evidence_gate" not in codes


def test_gate_counting_split_away_route_is_reported():
    # 复现真实 blocked 会话：恢复轮把 R1 拆成 R1_S1/S2/S3，门禁却仍在统计 R1。
    result = validate_research_state_invariants({
        "validated_routes": [
            {"route_id": "R1_S1"}, {"route_id": "R1_S2"}, {"route_id": "R3"},
        ],
        "global_evidence_gate": {
            "status": "EVALUATED",
            "metrics": {"route_stats": [
                {"route_id": "R1", "paper_count": 45},
                {"route_id": "R3", "paper_count": 17},
            ]},
        },
    })
    warnings = {item["code"]: item for item in result["warnings"]}
    assert "gate_route_universe_mismatch" in warnings
    assert warnings["gate_route_universe_mismatch"]["details"]["gate_only_routes"] == ["R1"]


def test_gate_matching_current_routes_has_no_warning():
    result = validate_research_state_invariants({
        "validated_routes": [{"route_id": "R1"}, {"route_id": "R3"}],
        "global_evidence_gate": {
            "status": "EVALUATED",
            "metrics": {"route_stats": [
                {"route_id": "R1", "paper_count": 45},
                {"route_id": "R3", "paper_count": 17},
            ]},
        },
    })
    codes = {item["code"] for item in result["warnings"]}
    assert "gate_route_universe_mismatch" not in codes
