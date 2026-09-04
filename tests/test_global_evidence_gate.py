"""全局证据门（Global Evidence Gate）测试。

全部确定性测试：不访问真实 LLM、不访问外部平台。
"""

from __future__ import annotations

import pytest

from app.agent.decorators import get_node_metadata
from app.agent.global_evidence_gate import evaluate_global_sufficiency
from app.agent.graph import _build_output, derive_result_status, run_research_agent
from app.agent.nodes import global_evidence_gate_node
from app.core.config import Settings


# ---------- helpers ----------

def _paper(paper_id: str, year: int = 2025) -> dict:
    return {"paper_id": paper_id, "title": f"Paper {paper_id}", "year": year}


def _card(paper_id: str, peer_review_status: str = "peer_reviewed") -> dict:
    return {
        "paper_id": paper_id,
        "year": 2025,
        "peer_review_status": peer_review_status,
    }


def _route(
    route_id: str,
    paper_ids: list | None = None,
    status: str = "KEEP",
    sufficiency: float | dict | None = 0.8,
) -> dict:
    paper_ids = list(paper_ids or [])
    return {
        "route_id": route_id,
        "name": route_id,
        "status": status,
        "paper_ids": paper_ids,
        "core_paper_ids": paper_ids,
        "supporting_paper_ids": [],
        "evidence_sufficiency": (
            {"score": sufficiency} if isinstance(sufficiency, (int, float)) else sufficiency
        ),
    }


def _make_state(
    *,
    n_details: int = 0,
    detail_years: list[int] | None = None,
    required: int = 0,
    max_papers_explicit: bool = False,
    start_year: int | None = None,
    end_year: int | None = None,
    year_range_explicit: bool = False,
    cards: list[dict] | None = None,
    routes: list[dict] | None = None,
    user_query: str = "",
    recovery_status: str | None = None,
) -> dict:
    details = [
        {
            "paper_id": f"p{i}",
            "title": f"Paper {i}",
            "year": detail_years[i] if detail_years else 2025,
        }
        for i in range(n_details)
    ]
    return {
        "user_query": user_query,
        "required_reference_count": required,
        "max_papers": required,
        "max_papers_explicit": max_papers_explicit,
        "start_year": start_year,
        "end_year": end_year,
        "year_range_explicit": year_range_explicit,
        "strict_year_range": False,
        "paper_details": details,
        "paper_cards": cards or [],
        "validated_routes": routes or [],
        "route_validation_report": {},
        "evidence_recovery_status": recovery_status,
        "steps": [],
        "errors": [],
    }


# ---------- 维度评估 ----------

def test_pass_all_dimensions():
    state = _make_state(
        n_details=40,
        required=40,
        max_papers_explicit=True,
        start_year=2024,
        end_year=2026,
        year_range_explicit=True,
        cards=[_card(f"p{i}") for i in range(34)]
        + [_card(f"p{i}", peer_review_status="not_peer_reviewed") for i in range(34, 40)],
        routes=[
            _route("r1", [f"p{i}" for i in range(15)]),
            _route("r2", [f"p{i}" for i in range(15, 29)]),
            _route("r3", [f"p{i}" for i in range(29, 40)]),
        ],
        user_query="引用不少于40篇近三年同行评审论文，并生成研究现状",
    )

    result = evaluate_global_sufficiency(state)

    assert result["passed"] is True
    assert result["deficits"] == []
    assert result["recommended_actions"] == ["CONTINUE"]
    assert result["explicit_constraint_unmet"] is False
    assert result["metrics"]["peer_review_ratio"] == pytest.approx(0.85)


def test_citation_deficit_explicit_is_blocking():
    state = _make_state(
        n_details=25,
        required=40,
        max_papers_explicit=True,
        user_query="引用不少于40篇论文",
    )

    result = evaluate_global_sufficiency(state)

    assert result["passed"] is False
    assert result["explicit_constraint_unmet"] is True
    deficit = result["deficits"][0]
    assert deficit["type"] == "citation_count"
    assert deficit["severity"] == "blocking"
    assert deficit["required"] == 40
    assert deficit["available"] == 25
    assert deficit["missing"] == 15
    assert result["evidence_debt"] == {"citation_count": 15}
    assert "TARGETED_GLOBAL_SEARCH" in result["recommended_actions"]


def test_citation_deficit_implicit_measures_only():
    state = _make_state(
        n_details=25,
        required=40,
        max_papers_explicit=False,
        user_query="生成目标检测综述",
    )

    result = evaluate_global_sufficiency(state)

    deficit = result["deficits"][0]
    assert deficit["type"] == "citation_count"
    assert deficit["severity"] == "non_blocking"
    assert result["passed"] is True
    assert result["explicit_constraint_unmet"] is False
    assert result["recommended_actions"] == ["CONTINUE"]


def test_recency_deficit_explicit_blocks_and_implicit_measures_only():
    years = [2024] * 10 + [2020] * 30
    explicit_state = _make_state(
        n_details=40,
        detail_years=years,
        required=40,
        start_year=2024,
        end_year=2026,
        year_range_explicit=True,
        user_query="引用不少于40篇近三年论文",
    )
    result = evaluate_global_sufficiency(explicit_state)
    deficit = result["deficits"][0]
    assert deficit["type"] == "recency"
    assert deficit["severity"] == "blocking"
    assert deficit["available"] == 10
    assert deficit["missing"] == 30
    assert result["evidence_debt"] == {"recency": 30}
    assert result["passed"] is False
    assert result["explicit_constraint_unmet"] is True

    implicit_state = _make_state(
        n_details=40,
        detail_years=years,
        required=40,
        start_year=2024,
        end_year=2026,
        year_range_explicit=False,
    )
    result = evaluate_global_sufficiency(implicit_state)
    assert result["deficits"][0]["severity"] == "non_blocking"
    assert result["passed"] is True
    assert result["explicit_constraint_unmet"] is False


def test_peer_review_explicit_enforcement():
    cards = [_card(f"p{i}") for i in range(20)] + [
        _card(f"p{i}", peer_review_status="not_peer_reviewed") for i in range(20, 40)
    ]
    state = _make_state(
        n_details=40,
        required=40,
        cards=cards,
        user_query="引用不少于40篇同行评审的期刊论文",
    )

    result = evaluate_global_sufficiency(state)

    deficit = result["deficits"][0]
    assert deficit["type"] == "quality"
    assert deficit["severity"] == "blocking"
    assert deficit["required"] == 32  # ceil(0.8 * 40)
    assert deficit["available"] == 20
    assert deficit["missing"] == 12
    assert result["evidence_debt"] == {"quality": 12}
    assert result["passed"] is False
    assert result["explicit_constraint_unmet"] is True


def test_peer_review_implicit_measure_only():
    cards = [_card(f"p{i}") for i in range(20)] + [
        _card(f"p{i}", peer_review_status="not_peer_reviewed") for i in range(20, 40)
    ]
    state = _make_state(
        n_details=40,
        required=40,
        cards=cards,
        user_query="综述近三年目标检测论文",
    )

    result = evaluate_global_sufficiency(state)

    assert result["deficits"][0]["severity"] == "non_blocking"
    assert result["passed"] is True
    assert result["explicit_constraint_unmet"] is False


def test_route_imbalance_detected_only_on_keep_routes():
    routes = [
        _route("r1", [f"a{i}" for i in range(35)]),
        _route("r2", [f"b{i}" for i in range(10)]),
        _route("r3", [f"c{i}" for i in range(2)]),
    ]
    state = _make_state(n_details=47, routes=routes)

    result = evaluate_global_sufficiency(state)

    deficit = result["deficits"][0]
    assert deficit["type"] == "route_coverage"
    assert deficit["severity"] == "blocking"
    assert deficit["required"] == 3
    assert deficit["available"] == 2
    assert deficit["missing"] == 1
    assert result["evidence_debt"] == {"route_coverage": 1}
    assert result["passed"] is False
    # 路线均衡是综述质量信号，不是用户显式约束 → 不影响结果状态
    assert result["explicit_constraint_unmet"] is False
    assert "REBALANCE_ROUTE" in result["recommended_actions"]


def test_weak_routes_do_not_double_count():
    routes = [
        _route("r1", [f"a{i}" for i in range(35)]),
        _route("r2", [f"b{i}" for i in range(10)]),
        _route("r3", [f"c{i}" for i in range(2)]),
        _route("r_weak", [f"w{i}" for i in range(1)], status="WEAK"),
    ]
    state = _make_state(n_details=48, routes=routes)

    result = evaluate_global_sufficiency(state)

    # WEAK 路线已由 Route Validator 判定证据不充分，不重复计入 deficit
    assert result["evidence_debt"] == {"route_coverage": 1}
    assert result["metrics"]["weak_route_count"] == 1


def test_balanced_routes_pass():
    routes = [
        _route("r1", [f"a{i}" for i in range(15)]),
        _route("r2", [f"b{i}" for i in range(14)]),
        _route("r3", [f"c{i}" for i in range(11)]),
    ]
    state = _make_state(n_details=40, routes=routes)

    result = evaluate_global_sufficiency(state)

    assert all(d["type"] != "route_coverage" for d in result["deficits"])
    assert result["metrics"]["route_balance_ratio"] == pytest.approx(11 / (40 / 3))


def test_debt_and_action_recommendation_rules():
    exhausted = _make_state(
        n_details=25,
        required=40,
        max_papers_explicit=True,
        user_query="引用不少于40篇论文",
        recovery_status="EXHAUSTED",
    )
    result = evaluate_global_sufficiency(exhausted)
    actions = result["recommended_actions"]
    assert "TARGETED_GLOBAL_SEARCH" in actions
    assert "ASK_USER" in actions
    assert actions.index("TARGETED_GLOBAL_SEARCH") < actions.index("ASK_USER")

    passed_state = _make_state(n_details=40, required=40)
    assert evaluate_global_sufficiency(passed_state)["recommended_actions"] == ["CONTINUE"]


def test_claim_support_uses_claim_plan_statistics():
    """有 claim_plans 时，主张强度必须取自主张分级计数而非路线证据体量。"""
    routes = [
        _route("r1", [f"a{i}" for i in range(10)], sufficiency=1.0),
        _route("r2", [f"b{i}" for i in range(10)], sufficiency=1.0),
    ]
    state = _make_state(n_details=20, routes=routes)
    # 实测形态：绝大多数主张只有单篇证据，而路线体量代理恒为 1.0。
    state["claim_plans"] = [
        {"total_claims": 200, "strong_plus_claims": 1, "single_evidence_claims": 197},
        {"total_claims": 33, "strong_plus_claims": 0, "single_evidence_claims": 30},
    ]

    result = evaluate_global_sufficiency(state)
    metrics = result["metrics"]

    assert metrics["claim_support_proxy"] == pytest.approx(1 / 233)
    assert metrics["claim_total_count"] == 233
    assert metrics["claim_strong_plus_count"] == 1
    assert metrics["claim_single_evidence_count"] == 227
    assert metrics["claim_support_source"] == "claim_plans"
    # 主张维度仍不在本门禁阻断，由 claim_evidence_gate 判定。
    assert all(d["type"] != "claim_support" for d in result["deficits"])
    assert any("claim_evidence_gate" in note for note in result["notes"])


def test_claim_support_falls_back_before_claim_plans_exist():
    """claim_plans 未生成时回退到路线体量，并标明来源以免误读。"""
    routes = [
        _route("r1", [f"a{i}" for i in range(10)], sufficiency={"score": 0.9}),
        _route("r2", [f"b{i}" for i in range(10)], sufficiency=0.5),
    ]
    state = _make_state(n_details=20, routes=routes)

    result = evaluate_global_sufficiency(state)
    metrics = result["metrics"]

    assert metrics["claim_support_proxy"] == pytest.approx(0.7)
    assert metrics["claim_support_source"] == "route_evidence_volume_fallback"
    assert metrics["claim_total_count"] == 0
    assert any("回退" in note for note in result["notes"])


def test_route_volume_proxy_is_saturated_and_not_used_as_claim_strength():
    """回归护栏：路线体量分数会被削平到 1.0，不得再作为主张强度上报。"""
    from app.agent.global_evidence_gate import _route_evidence_volume_proxy

    # 实测数据：7 条路线核心证据 9~29 篇，除以固定阈值 3 后全部溢出。
    routes = [
        _route(f"r{i}", [f"p{i}_{j}" for j in range(23)], sufficiency=1.0)
        for i in range(7)
    ]
    state = _make_state(n_details=60, routes=routes)
    assert _route_evidence_volume_proxy(state) == pytest.approx(1.0)

    # 同一状态下若已有主张统计，上报值必须反映主张强度而非这个饱和值。
    state["claim_plans"] = [
        {"total_claims": 233, "strong_plus_claims": 1, "single_evidence_claims": 227},
    ]
    metrics = evaluate_global_sufficiency(state)["metrics"]
    assert metrics["claim_support_proxy"] < 0.01


# ---------- 节点包装 ----------

def test_node_wrapper_success_and_step_record():
    state = _make_state(n_details=40, required=40, user_query="综述论文")

    global_evidence_gate_node(state)

    assert state["global_evidence_gate"]["passed"] is True
    step = state["steps"][-1]
    assert step["step_name"] == "global_evidence_gate"
    assert step["status"] == "success"


def test_node_wrapper_degraded_when_deficit():
    state = _make_state(
        n_details=25,
        required=40,
        max_papers_explicit=True,
        user_query="引用不少于40篇论文",
    )

    global_evidence_gate_node(state)

    assert state["global_evidence_gate"]["passed"] is False
    assert state["steps"][-1]["status"] == "degraded"


def test_node_wrapper_failed_path(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.agent.global_evidence_gate.evaluate_global_sufficiency", boom
    )
    state = _make_state(n_details=25, required=40, user_query="引用不少于40篇论文")

    global_evidence_gate_node(state)

    assert state["global_evidence_gate"]["status"] == "FAILED"
    assert any("global_evidence_gate" in err for err in state["errors"])
    assert state["steps"][-1]["status"] == "failed"
    # FAILED 状态不参与 derive_result_status 判定
    assert derive_result_status(state) == "success"


def test_node_registered():
    metadata = get_node_metadata("global_evidence_gate")
    assert metadata is not None
    assert "global_evidence_gate" in metadata.provided_fields


# ---------- 结果状态与输出 ----------

def test_derive_result_status_partial_on_explicit_unmet():
    state = {
        "global_evidence_gate": {
            "status": "EVALUATED",
            "passed": False,
            "explicit_constraint_unmet": True,
        }
    }
    assert derive_result_status(state) == "partial"

    state["global_evidence_gate"]["explicit_constraint_unmet"] = False
    assert derive_result_status(state) == "success"

    # FAILED（节点异常）不改变结果状态
    state["global_evidence_gate"] = {"status": "FAILED", "deficits": []}
    assert derive_result_status(state) == "success"


def test_build_output_includes_global_gate():
    state = {"global_evidence_gate": {"passed": True, "evidence_debt": {}}}

    output = _build_output(state)

    assert output["global_evidence_gate"] == {"passed": True, "evidence_debt": {}}
    assert output["research_state"]["global_evidence_gate"] == {
        "passed": True,
        "evidence_debt": {},
    }


def test_build_output_never_leaks_quarantined_draft():
    """未获授权的失败草稿不得通过 answer/body/兼容字段二次泄漏。"""
    import json

    state = {
        "answer": "## 正式正文已被质量门禁阻止\n\n阻断原因见下。",
        "review": "## 研究现状\n\nQUARANTINE_SENTINEL 未经验证的正文[p1]。",
        "related_work": "## 相关工作\n\nQUARANTINE_SENTINEL",
        "introduction": "## 引言\n\nQUARANTINE_SENTINEL",
        "body": "## 研究现状\n\nQUARANTINE_SENTINEL",
        "quarantined_draft": "## 研究现状\n\nQUARANTINE_SENTINEL",
        "quality_gate": {
            "passed": False,
            "draft_available": True,
            "draft_released": False,
            "draft_disposition": "quarantined",
            "partial_success": False,
            "phase": "post_generation",
            "blocking_issues": [{"code": "claim_evidence_quality_not_met", "message": "支持率不足"}],
        },
    }

    output = _build_output(state)

    assert output["status"] == "blocked"
    assert output["body"] == ""
    assert "QUARANTINE_SENTINEL" not in output["answer"]
    assert output["related_work"] is None
    assert output["introduction"] is None
    assert output["draft_disposition"] == "quarantined"
    # research_state 供内部恢复使用，不属于对外展示字段；这里只校验公开层。
    public = {key: value for key, value in output.items() if key != "research_state"}
    assert "QUARANTINE_SENTINEL" not in json.dumps(public, ensure_ascii=False, default=str)


def test_build_output_keeps_body_when_global_gate_only_reports_partial():
    """全局证据门导致的 partial 不应清空已通过写作后门禁的正文。"""
    state = {
        "answer": "## 研究现状\n\n可交付正文[p1]。",
        "review": "## 研究现状\n\n可交付正文[p1]。",
        "body": "## 研究现状\n\n可交付正文[p1]。",
        "quality_gate": {"passed": True, "draft_released": True, "draft_disposition": "approved"},
        "global_evidence_gate": {
            "status": "EVALUATED",
            "passed": False,
            "explicit_constraint_unmet": True,
        },
    }

    output = _build_output(state)

    assert output["status"] == "partial"
    assert "可交付正文" in output["body"]
    assert "可交付正文" in output["answer"]


def test_global_gate_control_limits_are_configurable_and_clamped():
    settings = Settings(
        global_gate_min_recency_ratio=1.5,
        global_gate_route_balance_min_ratio=-0.2,
        global_gate_peer_review_ratio=2.0,
    )

    assert settings.global_gate_min_recency_ratio == 1.0
    assert settings.global_gate_route_balance_min_ratio == 0.0
    assert settings.global_gate_peer_review_ratio == 1.0


# ---------- 图集成 smoke ----------

def _install_graph_fakes(monkeypatch, *, n_papers: int) -> None:
    """把 run_research_agent 的节点全部替换为填充 state 的假节点，
    只保留真实的 global_evidence_gate_node 和 final_answer_node。"""

    def fake_plan(current, llm=None, current_year=None):
        current.update({
            "user_query": "引用不少于40篇近三年目标检测论文，并生成研究现状",
            "intent": "generate_review",
            "confidence": 0.9,
            "topic": "目标检测",
            "canonical_topic": "目标检测",
            "keywords": ["object detection"],
            "required_concepts": [],
            "excluded_title_terms": [],
            "start_year": 2024,
            "end_year": 2026,
            "year_range_explicit": True,
            "strict_year_range": False,
            "max_papers": 40,
            "required_reference_count": 40,
            "max_papers_explicit": True,
            "retrieval_target": 120,
            "generation_limit": 80,
            "requested_sections": ["research_status"],
            "core_deliverables": ["research_status"],
            "language": "zh",
            "citation_style": "gbt7714",
        })
        return current

    def fake_provisional_routes(current, llm=None):
        current["provisional_framework"] = {
            "provisional_routes": [{"route_id": "r0", "name": "R0"}]
        }
        return current

    def fake_search_rank(current, **kwargs):
        current["candidate_papers"] = [_paper(f"p{i}") for i in range(n_papers)]
        current["ranked_papers"] = list(current["candidate_papers"])
        current["searched_keywords"] = ["object detection"]
        current["focus_coverage"] = {"missing_focuses": []}
        return current

    def fake_expand_year(current, should_cancel=None):
        return current

    def fake_fetch(current, should_cancel=None):
        current["paper_details"] = [_paper(f"p{i}") for i in range(n_papers)]
        return current

    def fake_download(current, should_cancel=None):
        return current

    def fake_extract(current, llm=None, should_cancel=None):
        current["paper_cards"] = [_card(f"p{i}") for i in range(n_papers)]
        return current

    def fake_validate(current, llm=None):
        third = max(n_papers // 3, 1)
        current["validated_routes"] = [
            _route("r1", [f"p{i}" for i in range(third)]),
            _route("r2", [f"p{i}" for i in range(third, 2 * third)]),
            _route("r3", [f"p{i}" for i in range(2 * third, n_papers)]),
        ]
        current["route_decisions"] = []
        current["route_validation_report"] = {
            "coverage": {"evidence_understood_rate": 1.0},
            "assignment_map": {},
        }
        return current

    def fake_recovery(current, should_cancel=None):
        return current

    def fake_claim_plan(current, llm=None):
        current["claim_plans"] = [{"route_id": "r1", "route_name": "R1", "total_claims": 3}]
        return current

    def fake_claim_gate(current):
        current["claim_evidence_gate"] = {"passed": True}
        return current

    def fake_generate(current, llm=None, should_cancel=None):
        current["review"] = "## 研究现状\n\n示例正文。"
        current["body"] = current["review"]
        current["writing_plans"] = [{"sections": []}]
        return current

    def fake_verify(current, llm=None):
        return current

    def fake_citation(current, llm=None):
        current["references"] = ["[1] Paper 0"]
        current["citation_map"] = {"p0": 1}
        current["unique_cited_paper_count"] = 1
        return current

    monkeypatch.setattr("app.agent.graph._get_llm", lambda: None)
    monkeypatch.setattr("app.agent.graph.plan_node", fake_plan)
    monkeypatch.setattr("app.agent.nodes.provisional_route_node", fake_provisional_routes)
    monkeypatch.setattr("app.agent.graph._search_rank_with_refinement", fake_search_rank)
    monkeypatch.setattr("app.agent.graph.expand_search_year_node", fake_expand_year)
    monkeypatch.setattr("app.agent.graph.fetch_detail_node", fake_fetch)
    monkeypatch.setattr("app.agent.graph.download_pdf_node", fake_download)
    monkeypatch.setattr("app.agent.graph.should_parse_pdf", lambda current: False)
    monkeypatch.setattr("app.agent.graph.extract_card_node", fake_extract)
    monkeypatch.setattr("app.agent.graph.validate_routes_node", fake_validate)
    monkeypatch.setattr("app.agent.graph._run_route_evidence_recovery", fake_recovery)
    monkeypatch.setattr("app.agent.graph.claim_plan_node", fake_claim_plan)
    monkeypatch.setattr("app.agent.graph.claim_evidence_gate_node", fake_claim_gate)
    monkeypatch.setattr("app.agent.graph.generate_deliverables_node", fake_generate)
    monkeypatch.setattr("app.agent.graph._claim_alignment_check", lambda current: None)
    monkeypatch.setattr("app.agent.graph.verify_claims_node", fake_verify)
    monkeypatch.setattr("app.agent.graph.citation_check_node", fake_citation)
    # 隔离收尾阶段的完整性校验与评估导出，让 smoke 聚焦门禁集成
    monkeypatch.setattr(
        "app.tools.validate_deliverable.validate_final_review_integrity",
        lambda text, state: {"valid": True, "errors": []},
    )
    monkeypatch.setattr(
        "app.agent.claim_plan.validate_claim_citation_consistency",
        lambda *args, **kwargs: {
            "inconsistent_sentences": 0,
            "consistent_sentences": 0,
        },
    )
    monkeypatch.setattr(
        "app.agent.diagnostics.export_evaluation_bundle",
        lambda state, output_dir: "",
    )


def test_graph_smoke_gate_passes_with_sufficient_pool(monkeypatch):
    _install_graph_fakes(monkeypatch, n_papers=40)

    result = run_research_agent(
        "引用不少于40篇近三年目标检测论文，并生成研究现状",
        current_year=2026,
    )

    step_names = [step["step_name"] for step in result["steps"]]
    assert "global_evidence_gate" in step_names
    assert result["global_evidence_gate"]["passed"] is True
    assert result["status"] == "success"


def test_graph_smoke_gate_flags_deficit_and_partial_status(monkeypatch):
    _install_graph_fakes(monkeypatch, n_papers=25)

    result = run_research_agent(
        "引用不少于40篇近三年目标检测论文，并生成研究现状",
        current_year=2026,
    )

    gate = result["global_evidence_gate"]
    assert gate["passed"] is False
    assert gate["evidence_debt"] == {"citation_count": 15}
    assert gate["explicit_constraint_unmet"] is True
    assert result["status"] == "partial"
    assert "全局证据门提示" in result["answer"]


def test_gate_runs_after_claim_plan_and_reports_real_claim_source(monkeypatch):
    """全局门禁必须排在 claim_plan 之后，否则主张强度永远落到回退值。

    回归 2026-08-29 实测缺陷：门禁在 claim_plan 之前执行，
    ``claim_support_proxy`` 恒为路线体量的饱和值 1.0，而同一次运行里
    claim_plan 统计的是绝大多数主张仅有单篇证据。判据取门禁上报的
    ``claim_support_source``：它只有在 claim_plans 已存在时才会是
    ``claim_plans``，因此直接反映了两个节点的先后次序。
    """
    _install_graph_fakes(monkeypatch, n_papers=40)

    result = run_research_agent(
        "引用不少于40篇近三年目标检测论文，并生成研究现状",
        current_year=2026,
    )

    gate = result["global_evidence_gate"]
    metrics = gate["metrics"]
    assert metrics["claim_support_source"] == "claim_plans"
    assert metrics["claim_total_count"] == 3
    assert not any("回退" in note for note in gate["notes"])
