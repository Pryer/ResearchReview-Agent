"""标准化诊断仪表盘。

架构冻结后，每次运行输出统一指标，用于跨领域/跨版本对比。
不引入新逻辑，只聚合 state 中已有数据。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

# eval_bundles 保留数量上限：长期运行的服务每次研究都会导出一个
# 带时间戳的 bundle，不轮转会无限增长。
_EVAL_BUNDLE_KEEP = 20


def _rotate_evaluation_bundles(bundle_dir: Path, keep: int = _EVAL_BUNDLE_KEEP) -> int:
    """按名称（时间戳序）只保留父目录 ``eval_bundles`` 下最新 keep 个 bundle。

    仅处理以 ``eval_bundle_`` 为前缀的兄弟目录；目录名不符合约定或
    keep 非正时跳过。返回删除的 bundle 数。
    """
    parent = bundle_dir.parent
    if keep <= 0 or parent.name != "eval_bundles":
        return 0
    try:
        siblings = sorted(
            (
                p for p in parent.iterdir()
                if p.is_dir() and p.name.startswith("eval_bundle_")
            ),
            key=lambda p: p.name,
        )
    except OSError:
        return 0
    stale = siblings[:-keep] if len(siblings) > keep else []
    removed = 0
    for directory in stale:
        try:
            import shutil
            shutil.rmtree(directory, ignore_errors=True)
            removed += 1
        except OSError:
            continue
    return removed


def export_evaluation_bundle(state: dict[str, Any], output_dir: str = "eval_bundle") -> str:
    """导出 Alignment 评测所需的完整数据包。

    生成目录：eval_bundle/
      ├── user_query.txt
      ├── claim_plan.json
      ├── writer_output.md
      ├── claim_alignment.json
      ├── final_routes.json
      ├── citation_map.json
      └── metadata.json
    """
    import json as _json
    from pathlib import Path as _Path

    bundle_dir = _Path(output_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / "user_query.txt").write_text(
        str(state.get("user_query") or ""), encoding="utf-8"
    )

    claim_plans = state.get("claim_plans") or []
    sanitized = []
    for plan in claim_plans:
        sanitized.append({
            "route_id": plan.get("route_id", ""),
            "route_name": plan.get("route_name", ""),
            "total_claims": plan.get("total_claims", 0),
            "claims": [
                {k: v for k, v in c.items() if k in (
                    "claim_id", "claim_text", "claim_type", "evidence_ids",
                    "evidence_count", "support_level", "allowed_language",
                )}
                for c in plan.get("claims", [])
            ],
        })
    (bundle_dir / "claim_plan.json").write_text(
        _json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    review = str(state.get("review") or state.get("answer") or "")
    (bundle_dir / "writer_output.md").write_text(review, encoding="utf-8")

    alignment = state.get("claim_alignment") or {}
    (bundle_dir / "claim_alignment.json").write_text(
        _json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    routes = state.get("validated_routes") or []
    (bundle_dir / "final_routes.json").write_text(
        _json.dumps([
            {"route_id": r.get("route_id",""), "name": r.get("name",""),
             "research_question": r.get("research_question",""),
             # 阶段字段用于复盘「上游产物是否被当作下游证据」：
             # route_role 由 LLM 声明（实测常缺失），pipeline_stage 由语义帧推导。
             "route_role": r.get("route_role") or "",
             "pipeline_stage": r.get("pipeline_stage") or "",
             "core_paper_ids": (r.get("core_paper_ids") or [])[:10]}
            for r in routes
        ], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    citation_map = state.get("citation_map") or {}
    (bundle_dir / "citation_map.json").write_text(
        _json.dumps(citation_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    gate = state.get("global_evidence_gate") or {}
    (bundle_dir / "global_evidence_gate.json").write_text(
        _json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "topic": str(state.get("canonical_topic") or state.get("topic") or ""),
        "start_year": state.get("start_year"),
        "end_year": state.get("end_year"),
        "required_references": state.get("required_reference_count"),
        "total_planned_claims": sum(p.get("total_claims", 0) for p in claim_plans),
        "total_factual_sentences": alignment.get("total_factual_sentences", 0),
        "alignment_rate": alignment.get("support_rate", 0),
    }
    (bundle_dir / "metadata.json").write_text(
        _json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _rotate_evaluation_bundles(bundle_dir)

    return str(bundle_dir.resolve())


def collect_diagnostics(state: dict[str, Any]) -> dict[str, Any]:
    """从 state 聚合各层指标，返回结构化诊断 dict。"""
    return {
        "retrieval": _retrieval_metrics(state),
        "route": _route_metrics(state),
        "claims": _claim_metrics(state),
        "writing": _writing_metrics(state),
        "citation": _citation_metrics(state),
        "gate": _gate_metrics(state),
    }


def _gate_metrics(state: dict[str, Any]) -> dict[str, Any]:
    """全局证据门指标；只聚合 state 中已有数据。"""
    gate = state.get("global_evidence_gate") or {}
    metrics = gate.get("metrics") or {}
    return {
        "status": gate.get("status", "NOT_REQUIRED"),
        "passed": gate.get("passed"),
        "explicit_constraint_unmet": gate.get("explicit_constraint_unmet", False),
        "blocking_deficits": [
            d.get("type") for d in gate.get("deficits") or []
            if d.get("severity") == "blocking"
        ],
        "evidence_debt": gate.get("evidence_debt", {}),
        "recency_ratio": metrics.get("recency_ratio"),
        "route_balance_ratio": metrics.get("route_balance_ratio"),
        "peer_review_ratio": metrics.get("peer_review_ratio"),
    }


def format_diagnostics(state: dict[str, Any]) -> str:
    """渲染为可读的多行文本。"""
    d = collect_diagnostics(state)
    lines = ["=" * 50, "DIAGNOSTICS REPORT", "=" * 50, ""]

    for section, metrics in d.items():
        lines.append(f"--- {section.upper()} ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.1%}" if value <= 1 else f"  {key}: {value:.1f}")
            elif isinstance(value, list):
                lines.append(f"  {key}: [{', '.join(str(v) for v in value[:8])}]")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# 各层指标提取
# ============================================================

def _retrieval_metrics(state: dict[str, Any]) -> dict[str, Any]:
    papers = state.get("paper_details") or []
    cards = state.get("paper_cards") or []
    candidates = state.get("candidate_papers") or []
    search_report = state.get("search_report") or {}
    raw_diagnostics = state.get("source_diagnostics") or search_report.get("source_diagnostics") or []
    diagnostics = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        for item in raw_diagnostics
    ]
    outcome_counts = dict(Counter(
        str(item.get("outcome") or item.get("status") or "unknown")
        for item in diagnostics
    ))
    affected_sources = sorted({
        str(item.get("source") or "") for item in diagnostics
        if item.get("outcome") not in {"success_with_results", "success_empty"}
        and item.get("source")
    })
    language_gap_sources = sorted({
        str(item.get("source") or "") for item in diagnostics
        if item.get("outcome") == "query_not_adapted"
    })

    usable = sum(
        1 for card in cards
        if str(getattr(
            (card.get("evidence_state") or {}).get("access_level"),
            "value",
            (card.get("evidence_state") or {}).get("access_level") or "",
        )) in {"abstract", "partial_full_text", "full_text"}
    )

    return {
        "retrieved": len(papers) or search_report.get("candidate_count", 0),
        "candidates": len(candidates) or len(papers),
        "screened": search_report.get("screened_count", len(cards)),
        "usable_cards": usable,
        "sources": search_report.get("sources", []),
        "outcome_counts": outcome_counts,
        "affected_sources": affected_sources,
        "language_gap_sources": language_gap_sources,
        "recent_errors": [
            {"source": item.get("source"), "outcome": item.get("outcome"),
             "error_code": item.get("error_code"), "message": item.get("message")}
            for item in diagnostics[-8:]
            if item.get("error_code") or item.get("message")
        ],
        "keywords_count": len(state.get("keywords") or state.get("searched_keywords") or []),
    }


def _route_metrics(state: dict[str, Any]) -> dict[str, Any]:
    decisions = state.get("route_decisions") or []
    # 路线验证结果由 validate_routes 节点统一写入 route_validation_report
    # （state["coverage"] 是同一次写入的冗余副本）。此前引用的
    # validate_routes_result 键从未有任何写入方，属于幽灵引用。
    report = state.get("route_validation_report") or {}
    coverage = report.get("coverage") or state.get("coverage") or {}

    action_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for d in decisions:
        action = str(d.get("action") or "")
        action_counts[action] = action_counts.get(action, 0) + 1
        status = str(d.get("status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "provisional_routes": len((state.get("provisional_framework") or {}).get("provisional_routes") or []),
        "final_routes": len(state.get("validated_routes") or []),
        "keep": action_counts.get("KEEP", 0),
        "weak": status_counts.get("WEAK", 0),
        "merge": action_counts.get("MERGE", 0) + action_counts.get("MERGED_INTO", 0),
        "split": action_counts.get("SPLIT", 0) + action_counts.get("SPLIT_INTO", 0),
        "add_new": action_counts.get("ADD_NEW_ROUTE", 0),
        "drop": action_counts.get("DROP", 0),
        "provisional_route_survival_rate": coverage.get("provisional_route_survival_rate", 0),
        "provisional_route_keep_rate": coverage.get("provisional_route_keep_rate", 0),
        "route_validator_recheck": coverage.get("route_validator_recheck", False),
        "single_route_pct": coverage.get("single_route_confident", 0) / max(1, coverage.get("total_papers", 1)),
        "cross_route_pct": coverage.get("cross_route_confident", 0) / max(1, coverage.get("total_papers", 1)),
        "ambiguous_pct": coverage.get("ambiguous_uncertain", 0) / max(1, coverage.get("total_papers", 1)),
        "unassigned_pct": coverage.get("unassigned", 0) / max(1, coverage.get("total_papers", 1)),
        "coverage_rate": coverage.get("evidence_understood_rate", coverage.get("coverage_rate", 0)),
    }


def _claim_metrics(state: dict[str, Any]) -> dict[str, Any]:
    plans = state.get("claim_plans") or []

    total_claims = sum(p.get("total_claims", 0) for p in plans)
    single = sum(p.get("single_evidence_claims", 0) for p in plans)
    strong_plus = sum(p.get("strong_plus_claims", 0) for p in plans)
    moderate = total_claims - single - strong_plus

    bg_plans = [p for p in plans if str(p.get("route_id", "")).startswith("background_")]
    route_plans = [p for p in plans if not str(p.get("route_id", "")).startswith("background_")]

    return {
        "planned_claims": total_claims,
        "route_claims": sum(p.get("total_claims", 0) for p in route_plans),
        "background_claims": sum(p.get("total_claims", 0) for p in bg_plans),
        "single_evidence": single,
        "moderate_evidence": moderate,
        "strong_plus": strong_plus,
        "filtered": sum(
            p.get("total_claims", 0)
            for p in plans
            if p.get("total_claims", 0) == 0
        ),
    }


def _writing_metrics(state: dict[str, Any]) -> dict[str, Any]:
    alignment = state.get("claim_alignment") or {}
    review_text = str(state.get("review") or "")

    import re
    factual_sentences = len(re.findall(r"\[\d+\]", review_text))

    return {
        "factual_sentences": alignment.get("total_factual_sentences", factual_sentences),
        "authorized": alignment.get("supported_sentences", 0),
        "unauthorized": alignment.get("unsupported_sentences", 0),
        "alignment_rate": alignment.get("support_rate", 0),
        "overclaimed": len(alignment.get("overclaimed_samples") or []),
        "cited_refs": len(set(re.findall(r"\[(\d+)\]", review_text))),
        "total_chars": len(review_text),
    }


def _citation_metrics(state: dict[str, Any]) -> dict[str, Any]:
    validation = state.get("citation_validation") or {}
    refs = state.get("references") or []
    ccc = state.get("claim_citation_consistency") or {}

    return {
        "total_references": len(refs),
        "structurally_valid": bool(validation.get("structurally_valid")),
        "metadata_quality_valid": bool(validation.get("metadata_quality_valid")),
        "missing_citations": len(validation.get("missing_citations") or []),
        "unused_references": len(validation.get("unused_references") or []),
        "incomplete_metadata": len(validation.get("incomplete_metadata") or []),
        "duplicate_dois": len(validation.get("duplicate_dois") or []),
        "claim_citation_consistency": ccc.get("consistency_rate", 0),
        "inconsistent_sentences": ccc.get("inconsistent_sentences", 0),
        "unmapped_papers": ccc.get("unmapped_papers", []),
    }
