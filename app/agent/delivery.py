"""Controller-owned graded delivery; immutable publication survives later failures."""
from __future__ import annotations
import copy
import hashlib
import re
import time
from app.agent.runtime_config import delivery_mode

def draft_first(state):
    return delivery_mode(state) == "draft_first"

def draft_materials(state):
    from app.services.evidence_support import independent_spans
    from app.agent.evidence_roles import citation_eligible_paper_ids
    from app.agent.deliverable_router import unconfirmed_reference_ids
    cards=state.get("paper_cards") or []
    eligible=citation_eligible_paper_ids(state.get("research_semantic_frame") or {}, cards)
    rejected=unconfirmed_reference_ids(state)
    details={str(p.get("paper_id") or ""):p for p in state.get("paper_details") or []}
    request=state.get("research_request") or {}
    explicit=bool(state.get("year_range_explicit") or state.get("strict_year_range") or request.get("year_range_explicit"))
    from app.utils.search_constraints import exact_search_title
    title=exact_search_title(state)
    def scope_ok(card):
        paper=details.get(str(card.get("paper_id") or ""),card)
        if title and " ".join(str(paper.get("title") or "").split()).casefold() != " ".join(title.split()).casefold():return False
        if explicit:
            year=paper.get("year")
            if isinstance(year,bool) or not isinstance(year,int):return False
            if state.get("start_year") is not None and year<int(state["start_year"]):return False
            if state.get("end_year") is not None and year>int(state["end_year"]):return False
        return True
    return {str(c["paper_id"]): independent_spans(c) for c in cards
            if c.get("paper_id") in eligible and c.get("paper_id") not in rejected
            and c.get("quality_status") != "invalid" and scope_ok(c) and independent_spans(c)}

def supplement_draft_plans(state):
    if not draft_first(state): return
    materials=draft_materials(state)
    existing=state.get("claim_plans") or []
    from app.agent.claim_plan import claim_authorized_paper_ids
    known=claim_authorized_paper_ids(existing)
    claims=[]
    for pid,spans in materials.items():
        if pid in known: continue
        span=spans[0]
        claims.append({"claim_id":"draft_"+hashlib.sha256(span["evidence_id"].encode()).hexdigest()[:16],
            "claim_text":span["text"],"claim_type":"background_fact", "evidence_ids":[span["evidence_id"]],
            "evidence_count":1,"support_level":"single", "verification_status":"unverified",
            "authorization_kind":"draft_only", "allowed_language":"原文报告／研究涉及",
            "allowed_verbs":["报告","提出","讨论"], "evidence_access_limit":
                "abstract_only" if span.get("source_type")=="abstract" else "source_bound"})
    if claims:
        state["claim_plans"]=[*existing,{"route_id":"draft_source_bound","route_name":"已有证据概述",
            "claims":claims,"total_claims":len(claims),"total_evidence_papers":len(claims),
            "strong_plus_claims":0,"single_evidence_claims":len(claims)}]

def draft_preflight(state):
    from app.services.writer_support import substantive
    from app.core.citation_syntax import extract_citation_ids
    text=str(state.get("review") or "").strip()
    errors=[]
    if not substantive(text) or len(re.sub(r"\s", "", text))<100:
        errors.append("no_substantive_body")
    if re.search(r"正式正文已被|正文生成已阻止|localhost|127\.0\.0\.1|(?:state|artifact)://|<svg|[A-Za-z]:\\",text,re.I):
        errors.append("internal_or_placeholder_content")
    source=str(state.get("citation_source_text") or text)
    # WHY: 数字编号必须绑定同版渲染前正文，不能按当前卡片顺序猜测。
    if state.get("citation_source_text") and state.get("citation_rendered_text") != text:
        errors.append("citation_source_version_mismatch")
    ids=set(extract_citation_ids(source))
    materials=draft_materials(state)
    if not ids or ids-set(materials): errors.append("citation_source_unresolved")
    validation=state.get("citation_validation") or {}
    if not validation or set(validation.get("cited_ids") or []) != ids or any(validation.get(key) for key in
            ("missing_citations", "unverified_metadata", "retracted_references", "duplicate_dois")):
        errors.append("citation_validation_failed")
    report=state.get("check_report") or {}
    from app.agent.checking import check_is_current
    current_check=check_is_current(state)
    if report and not current_check: errors.append("stale_check_report")
    issues=(report.get("issues") or []) if current_check else []
    if any(i.get("blocks_draft",True) for i in issues): errors.append("draft_blocking_issue")
    actual=len(ids & set(materials))
    return {"ready":not errors,"errors":errors,"cited_paper_ids":sorted(ids & set(materials)),
        "metadata_warnings":validation.get("incomplete_metadata") or validation.get("unknown_publication_status") or [],
        "actual_reference_count":actual,"requested_reference_count":int(state.get("required_reference_count") or 0),
        "verification_status":"complete" if current_check and report.get("verdict")=="passed" else "partial" if report else "not_started"}

def publication_identity(state):
    from app.agent.delegation_memory import task_identity
    return task_identity(state)

def published_delivery(state):
    # WHY: 同会话可以有不同研究任务；旧任务草稿不能冒充当前任务产物。
    reviews=state.get("publication_reviews") or {}
    matches=[v for v in state.get("published_versions") or [] if v.get("task_identity")==publication_identity(state)
             and not (reviews.get(str(v.get("version"))) or {}).get("withdrawn")]
    return copy.deepcopy(matches[-1]) if matches else None

def publication_fingerprint(state, preflight=None):
    """绑定公开内容及核验语义，供准入与发布共用；随机报告ID不产生新版本。"""
    import json
    from app.agent.checking import check_is_current, check_input_fingerprint
    preflight = preflight or draft_preflight(state)
    final = bool((state.get("quality_gate") or {}).get("passed") is True and check_is_current(state)
        and (state.get("check_report") or {}).get("verdict") == "passed"
        and (not preflight["requested_reference_count"] or
             preflight["actual_reference_count"] >= preflight["requested_reference_count"]))
    report = state.get("check_report") or {}
    payload = {"grade": "final" if final else "draft", "input": check_input_fingerprint(state),
        "references": state.get("references") or [], "citation_map": state.get("citation_map") or {},
        "verification": state.get("claim_verification") or {},
        "verdict": report.get("verdict"), "execution_status": report.get("execution_status"),
        "issues": [{k: v for k, v in issue.items() if k != "issue_id"}
                   for issue in report.get("issues") or []], "preflight": preflight}
    return payload["grade"], hashlib.sha256(json.dumps(payload, ensure_ascii=False,
        sort_keys=True, default=str).encode()).hexdigest()


def has_unpublished_draft(state):
    if not draft_first(state):
        return False
    preflight = draft_preflight(state)
    if not preflight["ready"]:
        return False
    previous = published_delivery(state)
    return not previous or previous.get("fingerprint") != publication_fingerprint(state, preflight)[1]


def publish_draft(state):
    from app.agent.execution_budget import check_execution
    check_execution()
    if not draft_first(state): raise ValueError("draft publication is not authorized by task policy")
    preflight=draft_preflight(state)
    if not preflight["ready"]: raise ValueError("draft preflight failed: "+",".join(preflight["errors"]))
    from app.agent.checking import check_input_fingerprint
    text=str(state["review"])
    grade,digest=publication_fingerprint(state, preflight)
    final=grade == "final"
    versions=state.setdefault("published_versions",[])
    previous=published_delivery(state)
    if previous and previous["fingerprint"]==digest: return previous
    version=len(versions)+1
    notes=[]
    if not final:
        notes.append("草稿")
        notes.append(f"已引用{preflight['actual_reference_count']}篇；最终目标{preflight['requested_reference_count']}篇" if preflight['requested_reference_count'] else f"已引用{preflight['actual_reference_count']}篇")
        if preflight["verification_status"]!="complete": notes.append("语义核验尚未全部通过，待完善项见质量报告")
    public_text=("> "+"；".join(notes)+"。\n\n" if notes else "")+text
    if not final:
        pending=[]
        for claim in (state.get("claim_verification") or {}).get("claims") or []:
            if claim.get("support_status") in {"unsupported","partially_supported"} or claim.get("verification_status") in {"not_completed","unverified"}:
                sentence=re.sub(r"\[[^\]]*\]", "",str(claim.get("sentence") or "")).strip()
                if sentence and sentence not in pending: pending.append(sentence)
        if pending:
            public_text+="\n\n## 待完善内容\n\n以下论述仍需收窄或补充直接证据，暂不作为已确认结论：\n\n"+"\n".join("- “"+x[:240]+"”" for x in pending[:12])
        if preflight.get("metadata_warnings"):
            public_text+="\n\n部分书目信息或出版状态尚待补全。"
    refs=copy.deepcopy(state.get("references") or [])
    if refs and not re.search(r"^#{1,6}\s*参考文献",public_text,re.M):
        public_text+="\n\n## 参考文献\n\n"+"\n".join(str(x) for x in refs)
    record={"version":version,"fingerprint":digest,"task_identity":publication_identity(state),
        "body":public_text,"answer":public_text,"references":refs,"grade":grade,
        "verification_status":preflight["verification_status"],"requirements_satisfied":final,
        "quality_report":copy.deepcopy(state.get("check_report") or {}),
        "quality_gate":copy.deepcopy(state.get("quality_gate") or {}),
        "citation_map":copy.deepcopy(state.get("citation_map") or {}),
        "citation_registry":copy.deepcopy(state.get("citation_registry") or {}),
        "public_related_work":bool(state.get("related_work")), "public_introduction":bool(state.get("introduction")),
        "preflight":preflight,
        "writing_version":int(state.get("writing_version") or 0),"published_at":time.time(),
        "source_fingerprint":check_input_fingerprint(state),
        "source_artifact_manifest":copy.deepcopy({k:v for k,v in (state.get("artifact_manifest") or {}).items()
            if k in {"paper_details","paper_cards","claim_plans","parsed_papers","check_report"}})}
    versions.append(record)
    state.update(delivery_status=grade,published_draft_version=version,
        published_artifact_ref=f"state://published-draft/{version}",verification_status=record["verification_status"],
        requirements_satisfied=final)
    metrics=state.setdefault("delivery_metrics",{})
    metrics.setdefault("first_draft_at",record["published_at"])
    started=(state.get("agent_execution_budget") or {}).get("started_at")
    if started: metrics.setdefault("first_draft_seconds",round(record["published_at"]-float(started),3))
    return copy.deepcopy(record)

def delivery_output(state, output):
    status=str(state.get("result_status") or output.get("status") or "completed")
    record=published_delivery(state)
    execution=("cancelled" if status=="cancelled" else "failed" if status=="failed" else
               "stopped" if status in {"blocked","waiting_user"} else "running" if status in {"running","queued","cancel_requested"} else "completed")
    errors=state.get("errors") or []
    reason=next((e.get("code") for e in reversed(errors) if isinstance(e,dict)),"")
    output.update(delivery_metrics=copy.deepcopy(state.get("delivery_metrics") or {}),execution_status=execution,stop_reason=reason,delivery_status=record["grade"] if record else "none",
        verification_status=record["verification_status"] if record else "not_started",
        requirements_satisfied=record["requirements_satisfied"] if record else False,
        published_draft_version=record["version"] if record else None,
        published_artifact_ref=state.get("published_artifact_ref") if record else None)
    if record:
        output.update(answer=record["answer"],body=record["body"],references=record["references"],
            draft_available=True,draft_released=True,draft_disposition="approved" if record["grade"]=="final" else "released_best_effort")
        output["citation_map"]=copy.deepcopy(record.get("citation_map") or {})
        output["citation_registry"]=copy.deepcopy(record.get("citation_registry") or {})
        output["published_quality_report"]=copy.deepcopy(record.get("quality_report") or {})
        output["related_work"]=record["body"] if record.get("public_related_work") else None
        output["introduction"]=record["body"] if record.get("public_introduction") else None
        output["related_work_data"]=None
        output["introduction_data"]=None
        output["quality_gate"]={**(record.get("quality_gate") or {}),"passed":record["grade"]=="final","draft_released":True,
            "published_version":record["version"],"published_verification_status":record["verification_status"]}
        # Keep cancellation/failure visible; a stopped optimization with a draft is partial delivery.
        if status not in {"cancelled","waiting_user"}: output["status"]="completed" if record["grade"]=="final" else "partial"
    return output


def review_published_delivery(state):
    """A later Check can withdraw the same bad version, never a different earlier candidate."""
    from app.agent.checking import check_is_current, check_input_fingerprint
    report=state.get("check_report") or {}
    if not check_is_current(state):return
    if not any(i.get("blocks_draft",True) for i in report.get("issues") or []):return
    fingerprint=check_input_fingerprint(state)
    for version in state.get("published_versions") or []:
        if version.get("source_fingerprint")==fingerprint:
            state.setdefault("publication_reviews",{})[str(version["version"])]= {
                "withdrawn":True,"report_id":report.get("report_id"),"reason":"独立核验发现必须修正的来源或事实错误"}
