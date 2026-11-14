"""写作候选审计与授权投影；不授予发布权限。"""
from __future__ import annotations

import hashlib
import json
import re
from uuid import uuid4

from app.services.evidence_support import independent_spans


def substantive(text):
    body = re.sub(r"^#{1,6}[^\n]*", "", str(text or ""), flags=re.M)
    body = re.sub(r"\[[^\]]*\]", "", body)
    body = re.sub(r"各文献的具体问题设定与方法细节以其原文报告为准，本节不作外推。|现有证据的适用范围以各文献原文报告为准。", "", body)
    body = re.sub(r"《[^》]+》[^。\n]*直接讨论了[^。\n]*。|本节纳入的文献包括：[^\n]*", "", body)
    return len(re.sub(r"\s", "", body)) >= 40


def record_candidate(state, plan, stage, raw, normalized, validation, *, issues=()):
    from app.agent.task_context import current_task_brief
    from app.services.research_execution_service import artifact_store
    from app.agent.execution_budget import check_execution
    check_execution()
    brief = current_task_brief("writing") or {}
    candidate_id = uuid4().hex
    from app.agent.delegation_memory import task_identity
    payload = {"task_identity": task_identity(state), "candidate_id": candidate_id, "parent_candidate_id": state.get("selected_candidate_id"),
               "delegation_id": brief.get("delegation_id"), "stage": stage,
               "writing_version": state.get("writing_version", 0),
               "input_fingerprint": hashlib.sha256(json.dumps({
                   "plan": plan.model_dump(mode="json"), "cards": state.get("paper_cards"),
                   "claims": state.get("claim_plans"), "query": state.get("user_query"),
               }, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest(),
               "normalization_actions": ["removed_executor_report"] if strip_writer_report(raw)!=str(raw or "") and str(raw or "")!=str(normalized or "") else [],
               "citation_protocol": "paper_id-v1", "raw_text": str(raw or ""),
               "normalized_text": str(normalized or ""), "validation": validation,
               "issues": list(issues), "substantive": substantive(normalized),
               "chars": len(str(normalized or ""))}
    store = artifact_store()
    record = dict(payload)
    if store is not None and state.get("session_id"):
        # WHY: 原始候选先于验证结果提交审计资料；它不能更新公开稿或通过状态。
        refs = []
        value = json.dumps(payload, ensure_ascii=False)
        for i in range(0, len(value), 16000):
            refs.append(store.persist_payload(session_id=str(state["session_id"]),
                artifact_type="writer_candidate", payload={"value": value[i:i+16000]},
                identity=f"{candidate_id}:{i}", provenance={"candidate_id": candidate_id,
                    "delegation_id": brief.get("delegation_id"), "offset": i}))
        store.repo.db.flush()
        record.pop("raw_text")
        record.pop("normalized_text")
        record["artifact_refs"] = refs
    state.setdefault("writer_candidates", []).append(record)
    state.setdefault("writer_diagnostics", []).append({
        "candidate_id": candidate_id, "stage": stage, "chars": payload["chars"],
        "valid": validation.get("valid", False), "errors": validation.get("errors") or [],
        "issues": list(issues), "candidate_ref": f"state://writer-candidate/{candidate_id}"})
    return candidate_id


def read_candidate(state, candidate_id, *, role="writing"):
    record = next((r for r in state.get("writer_candidates") or [] if r["candidate_id"] == candidate_id), None)
    if not record:
        raise ValueError("writer candidate not found")
    from app.agent.delegation_memory import task_identity
    if role != "service" and record.get("task_identity") != task_identity(state):
        raise PermissionError("writer candidate belongs to another research task")
    if role not in {"main", "writing", "check", "analysis", "service"}:
        raise PermissionError("role cannot read writer candidate")
    if not record.get("artifact_refs"):
        return dict(record)
    from app.services.research_execution_service import artifact_store
    store = artifact_store()
    if store is None:
        raise ValueError("writer candidate requires artifact store")
    return json.loads("".join(store.resolve(str(state["session_id"]), ref,
        role=role, allowed_types={"writer_candidate"})["payload"]["value"] for ref in record["artifact_refs"]))


def citation_aliases(cards):
    groups = {}
    for card in cards:
        pid = str(card.get('paper_id') or '')
        if not pid:
            continue
        values = {pid}
        if re.fullmatch(r's2:[0-9a-fA-F]{40}', pid):
            for n in range(8, 41):
                values.update({pid[3:3+n], pid[:3+n]})
        doi = str(card.get('doi') or '').removeprefix('https://doi.org/').strip()
        if doi:
            values.add('doi:' + doi)
            m = re.fullmatch(r'10\.48550/arxiv\.(\d{4}\.\d{4,5})', doi, re.I)
            if m:
                values.add('arxiv:' + m.group(1))
        for alias in values:
            groups.setdefault(alias.casefold(), set()).add(pid)
    # WHY: 只在当前输入中唯一匹配才展开；短前缀冲突/未知标识留给门禁，不能猜测。
    aliases = {k: next(iter(v)) for k, v in groups.items() if len(v) == 1}
    for card in cards:
        pid = str(card.get('paper_id') or '')
        if pid:
            aliases[pid.casefold()] = pid
    return aliases


def strip_writer_report(text):
    """仅去除正文后明确的执行报告；不把真实数字引用重新解释成论文身份。"""
    value=str(text or "")
    pattern=re.compile(r"(?m)^(?:#{1,6}\s*)?(?:\*\*)?(?:生成诊断|执行诊断|生成自检|写作自检|结果状态|建议动作)(?:[:：][^\n]*)?(?:\*\*)?\s*$")
    for match in pattern.finditer(value):
        body=value[:match.start()].rstrip()
        suffix=value[match.start():]
        # WHY: 仅处理正文后的控制面报告；后面还有学术章节时不擅自裁剪内容。
        later_headings=re.findall(r"(?m)^#{1,6}\s+([^\n]+)",suffix.split("\n",1)[-1])
        if len(re.sub(r"\s", "",body))<100 or any(not re.match(r"生成诊断|执行诊断|生成自检|写作自检|结果状态|建议动作",h) for h in later_headings):continue
        if re.search(r"target_char_range|WritingPlan|required_reference_count|\bMain\b|已生成候选|字数约|结果状态[：:]\s*(?:partial|blocked|completed|failed|needs_help)",suffix):
            return body
    return value


def normalize_writer_citations(text, cards, *, trusted_numbers=None):
    from app.deliverables.renderers.base_renderer import _normalize_evidence_citations, _strip_evidence_meta_language
    normalized = _strip_evidence_meta_language(_normalize_evidence_citations(strip_writer_report(text).strip(), cards))
    from app.core.citation_syntax import split_citation_group
    aliases = citation_aliases(cards)
    for number, pid in (trusted_numbers or {}).items():
        if pid not in {c.get('paper_id') for c in cards}:
            raise ValueError('citation alias target not in current input')
        aliases[str(number)] = pid
        label = 'p' + str(number)
        canonical = {str(c.get('paper_id')).casefold() for c in cards}
        if label not in canonical or aliases.get(label) == pid:
            aliases[label] = pid
    def expand(match):
        tokens = split_citation_group(match.group(1) or match.group(2))
        keys = [t.rstrip('.…').casefold() for t in tokens]
        return '[' + '; '.join(aliases[k] for k in keys) + ']' if tokens and all(k in aliases for k in keys) else match.group(0)
    normalized = re.sub(r'\[([^\]\n]+)\]|〔([^〕\n]+)〕', expand, normalized)
    # WHY: 没有可信编号映射时保留原稿并报告，不猜测论文顺序或静默删引用。
    issues = ([{"code": "citation_protocol_invalid", "reason": "内部候选含未映射数字引用；请使用[paper_id]"}]
              if re.search(r"\[\s*\d+(?:\s*[,，\-–]\s*\d+)*\s*\]", normalized) else [])
    return normalized, issues


def authorized_writer_claims(state, paper_id):
    card = next((c for c in state.get("paper_cards") or [] if c.get("paper_id") == paper_id), {})
    from app.agent.claim_plan import _paper_id_from_evidence
    spans = {s["evidence_id"]: s for s in independent_spans(card)
             if _paper_id_from_evidence(str(s["evidence_id"])) == paper_id}
    claims = []
    for plan in state.get("claim_plans") or []:
        for claim in plan.get("claims") or []:
            from app.agent.delivery import draft_first
            if claim.get("authorization_kind") == "draft_only" and not draft_first(state):
                continue
            ids = [str(e) for e in claim.get("evidence_ids") or [] if str(e) in spans]
            if not ids or claim.get("support_level") in {"insufficient", "unsupported", "none"}:
                continue
            text = str(claim.get("claim_text") or claim.get("claim") or "").strip()
            if not text:
                continue
            claims.append({"claim_id": claim.get("claim_id"), "claim": text,
                "field": "authorized_claim", "explicitly_reported": True,
                "evidence_id": ids[0], "evidence_ids": ids,
                "source_text": spans[ids[0]]["text"],
                "source_records": [spans[e] for e in ids],
                "authorization_kind": claim.get("authorization_kind") or ("coverage" if plan.get("route_id") == "reference_coverage" else "claim_plan"),
                "verification_status": claim.get("verification_status", "planned"),
                "draft_boundary": "按原文范围陈述；语义核验未完成，不外推数字、共识或因果" if claim.get("authorization_kind")=="draft_only" else "",
                "allowed_language": claim.get("allowed_language"),
                "allowed_verbs": claim.get("allowed_verbs"),
                "evidence_access_limit": claim.get("evidence_access_limit") or
                    ("abstract_only" if all(spans[e].get("source_type") == "abstract" for e in ids) else "source_bound")})
    return claims


def writer_issues(state):
    issues = list(state.get("writer_attempt_issues") or [])
    for report in state.get("deliverable_validation") or []:
        for error in report.get("errors") or []:
            issues.append({"code": "writer_structure_invalid", "reason": str(error)})
    if not substantive(state.get("review")):
        issues.append({"code": "writer_no_substantive_candidate", "reason": "本次写作没有生成可核验的实质正文"})
    result = []
    for issue in issues:
        if issue not in result:
            result.append(issue)
    return result[:12]
