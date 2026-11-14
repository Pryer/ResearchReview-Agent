"""同研究任务的角色工作摘要与依赖增量；历史建议不构成证据授权。"""
import copy
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:20]


def task_identity(state):
    request = state.get("research_request") or {}
    return digest([state.get("session_id"), request.get("original_query") or
                   state.get("quality_original_query") or state.get("user_query")])


def dependencies(state, role):
    from app.agent.action_contracts import CONTRACTS
    from app.agent.action_registry import ACTION_REGISTRY
    names = set().union(*(CONTRACTS[name].inputs for name, spec in ACTION_REGISTRY.items()
                         if spec.role and spec.role.value == role and name in CONTRACTS))
    # WHY: 跟踪研究材料的内容，排除报告UUID、历史、当前委派和预算等执行噪声。
    relevant = names & {"paper_details", "paper_cards", "claim_plans", "review", "writing_plans",
        "parsed_papers", "searched_keywords", "searched_query_windows", "candidate_papers",
        "source_diagnostics", "target_length", "start_year", "end_year", "required_reference_count",
        "canonical_topic", "research_semantic_frame", "citation_allocation_plan", "check_report"}
    return {key: digest(state[key]) for key in sorted(relevant) if key in state}


def prepare_work_context(state, role):
    session = (state.get("subagent_sessions") or {}).get(role) or {}
    old = session.get("work_summary") or {}
    identity = task_identity(state)
    if old.get("task_identity") != identity:
        old = {}
    now = dependencies(state, role)
    previous = old.get("dependencies") or {}
    changed = [key for key in now if key in previous and now[key] != previous[key]]
    result = {"task_identity": identity, "prior_result": copy.deepcopy(old.get("last_result") or {}),
        "input_delta": {"changed_fields": changed, "new_fields": [key for key in now if key not in previous],
                        "removed_fields": [key for key in previous if key not in now]},
        "dependencies": now, "history_is_evidence": False}
    if role == "search":
        result["searched_queries"] = list(state.get("searched_keywords") or [])[-16:]
        result["searched_windows"] = copy.deepcopy((state.get("searched_query_windows") or [])[-8:])
        result["candidate_count"] = len(state.get("candidate_papers") or [])
    elif role == "analysis":
        result["coverage"] = [{"paper_id": c.get("paper_id"), "status": c.get("quality_status"),
            "coverage_status": (c.get("extraction_coverage") or {}).get("status"),
            "unsupported_fields": (c.get("unsupported_fields") or [])[:8]} for c in (state.get("paper_cards") or [])[:12]]
    elif role == "writing":
        result["selected_candidate_id"] = state.get("selected_candidate_id")
        result["candidate_refs"] = [f"state://writer-candidate/{r['candidate_id']}" for r in (state.get("writer_candidates") or []) if r.get("task_identity") == identity][-3:]
        result["latest_issues"] = copy.deepcopy((state.get("check_report") or {}).get("issues") or state.get("writer_attempt_issues") or [])[:8]
        result["target_section_ids"] = list(state.get("target_section_ids") or [])
    elif role == "check":
        from app.agent.checking import check_is_current
        report = state.get("check_report") or {}
        result["previous_check"] = {"report_id": report.get("report_id"), "current": bool(report) and check_is_current(state),
                                    "issues": copy.deepcopy(report.get("issues") or [])[:8]}
        result["independence"] = "只根据当前稿件、授权证据和规则裁决；历史解释和写作者自述不是核验结论"
    return result


def complete_work_context(state, role, result):
    session = state["subagent_sessions"][role]
    session["work_summary"] = {"task_identity": task_identity(state), "dependencies": dependencies(state, role),
        "last_result": {key: copy.deepcopy(result.get(key)) for key in
            ("task_id", "operation", "status", "outcome", "issues", "check_report_ref", "evidence_refs", "findings", "task_executed", "model_report")}}


def begin_delegation(state, role):
    session = state["subagent_sessions"][role]
    trajectory = session.get("main_agent_trajectory")
    if not trajectory:
        return
    from app.services.research_execution_service import artifact_store
    from app.agent.react_context import archive_payload
    refs = []
    if artifact_store() is not None and state.get("session_id"):
        refs = archive_payload(state, "context_epoch", {"messages": trajectory.get("messages", []),
            "memory": trajectory.get("memory", []), "previous_archive_refs": trajectory.get("archive_refs", [])},
            scope_id=trajectory.get("scope_id"))
    else:
        # WHY: 离线/无数据库调用保留历史本体；生产运行使用已有Artifact归档与预算控制。
        session.setdefault("delegation_history", []).append(copy.deepcopy(trajectory))
    old_identity = (session.get('work_summary') or {}).get('task_identity')
    if old_identity and old_identity != task_identity(state):
        from uuid import uuid4
        # WHY: 新研究任务保留历史审计，但当前可回查链不能连到旧研究上下文。
        trajectory = {**trajectory, 'scope_id': uuid4().hex}
        refs = []
    session["main_agent_trajectory"] = {**trajectory, "messages": [], "memory": [], "goals": [],
        "pending": None, "archive_refs": refs, "epoch": trajectory.get("epoch", 0) + 1}
