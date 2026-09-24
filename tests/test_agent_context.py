"""主 Agent 五字段上下文、压缩与版本边界。"""

from __future__ import annotations

from app.agent.context_builder import build_context_snapshot, build_main_agent_context
from app.agent.context_compaction import compact_main_context


def _state() -> dict:
    return {
        "state_schema_version": "2",
        "user_query": "近三年研究现状，引用不少于2篇",
        "research_request": {"original_query": "近三年研究现状，引用不少于2篇"},
        "topic": "示例主题",
        "canonical_topic": "示例主题",
        "core_deliverables": ["research_status"],
        "start_year": 2024,
        "end_year": 2026,
        "year_range_explicit": True,
        "required_reference_count": 2,
        "max_papers_explicit": True,
        "language": "zh",
        "paper_cards": [
            {
                "paper_id": "p1",
                "evidence_source": "abstract",
                "evidence_state": {"access_level": "abstract"},
            },
            {
                "paper_id": "p2",
                "evidence_source": "full_text",
                "evidence_state": {"access_level": "full_text"},
            },
        ],
        "claim_plans": [{
            "route_id": "r1",
            "claims": [{
                "claim_id": "c1",
                "claim_text": "两项研究分别报告了相关发现。",
                "evidence_ids": ["p1:e1", "p2:e1"],
            }],
        }],
        "steps": [{"step": "claim_plan", "status": "success"}],
        "quality_gate": {
            "passed": False,
            "blocking_issues": [{"code": "minimum_references_not_met", "message": "引用不足"}],
        },
    }


def test_main_context_has_exactly_five_top_level_fields_and_preserves_hard_constraints():
    context = build_main_agent_context(_state())

    assert set(context.model_dump()) == {
        "goal", "state", "key_evidence", "decisions", "open_questions",
    }
    constraints = {item["name"]: item for item in context.goal.explicit_constraints}
    assert constraints["time_range"]["value"] == {"start_year": 2024, "end_year": 2026}
    assert constraints["required_reference_count"]["value"] == 2
    assert context.key_evidence[0].paper_ids == ["p1", "p2"]
    assert context.key_evidence[0].source_refs == [
        "state://paper-card/p1", "state://paper-card/p2",
    ]


def test_compaction_never_drops_goal_or_blocking_question():
    context = build_main_agent_context(_state())
    context.open_questions.extend([
        context.open_questions[0].model_copy(update={
            "question_id": f"optional-{index}",
            "blocking": False,
        })
        for index in range(20)
    ])

    compacted = compact_main_context(context, max_chars=2000)

    assert compacted.goal.explicit_constraints == context.goal.explicit_constraints
    assert any(item.blocking for item in compacted.open_questions)


def test_context_snapshot_rejects_unknown_evidence_paper():
    state = _state()
    state["claim_plans"][0]["claims"][0]["evidence_ids"] = ["missing:e1"]

    snapshot = build_context_snapshot(state)

    assert "unknown_evidence_paper:c1" in snapshot.validation_errors


def test_context_reads_step_name_and_separates_execution_from_research_status():
    state = _state()
    state.update({
        "steps": [{"step_name": "fetch_detail", "status": "success"}],
        "agent_execution_status": "running",
        "result_status": "partial",
    })

    context = build_main_agent_context(state)

    assert context.state.current_stage == "fetch_detail"
    assert context.state.completed_actions == ["fetch_detail"]
    assert context.state.execution_status == "running"
    assert context.state.research_status == "partial"


def test_blocking_questions_are_prioritized_before_limit(monkeypatch):
    from app.core.config import get_settings

    state = _state()
    state["evidence_gap_report"] = {
        "needs_recovery": False,
        "gaps": [
            {"route_id": f"gap-{index}", "reason": f"建议问题 {index}"}
            for index in range(20)
        ],
    }
    state["quality_gate"] = {
        "blocking_issues": [{"code": "hard-stop", "message": "必须阻断"}],
    }
    monkeypatch.setattr(get_settings(), "main_context_max_open_questions", 4)

    context = build_main_agent_context(state)

    assert context.open_questions[0].question_id == "hard-stop"
    assert context.open_questions[0].blocking is True


def test_context_fingerprint_tracks_direct_evidence_and_constraint_changes():
    from app.agent.context_builder import context_source_fingerprint

    state = _state()
    first = context_source_fingerprint(state)
    state["paper_cards"][0]["evidence_state"]["access_level"] = "full_text"
    second = context_source_fingerprint(state)
    state["required_reference_count"] = 41
    third = context_source_fingerprint(state)

    assert first != second
    assert second != third


def test_context_reports_budget_exceeded_without_dropping_hard_constraint(monkeypatch):
    from app.core.config import get_settings

    state = _state()
    state["research_semantic_frame"] = {
        "evidence_requirements": [{"label": "不可删除的限制" * 800, "source": "user_explicit"}],
    }
    monkeypatch.setattr(get_settings(), "main_context_max_chars", 2000)

    snapshot = build_context_snapshot(state)

    assert "context_budget_exceeded" in snapshot.validation_errors
    assert snapshot.context.goal.explicit_constraints
