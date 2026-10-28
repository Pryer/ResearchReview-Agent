"""A/B/C/D 的最小反例与正向约束，所有模型调用均为替身。"""
import json
from copy import deepcopy
import pytest

from app.schemas.paper_schema import PaperCard
from app.tools.extract_paper_card import _attach_evidence, _quality_gate, extract_paper_card
from app.tools.verify_claims import verify_review_claims
from app.utils.deduplicate import deduplicate_papers


METHOD = "We use a dual encoder method for document retrieval."


class Echo:
    def __init__(self):
        self.calls = []
    def complete(self,prompt,**kwargs):
        self.calls.append(prompt)
        return '{"results":[{"claim_id":"c001u01","label":"entailed","confidence":0.99}]}'


def test_a_legacy_fields_are_not_independent_evidence():
    model = Echo()
    card = {"paper_id":"p1","title":"Study","method":METHOD,"results":"The model achieves 89% accuracy.",
            "evidence_source":"full_text","evidence_state":{"access_level":"full_text"}}
    report = verify_review_claims("The model achieves 89% accuracy [p1].",[card],llm=model)
    assert not report["valid"]
    assert report["claims"][0]["support_status"] == "unsupported"
    assert not model.calls


def test_a_real_abstract_span_remains_verifiable():
    card = extract_paper_card({"paper_id":"p1","title":"Study","abstract":METHOD},llm=None).model_dump(mode="json")
    model = Echo()
    report = verify_review_claims(METHOD+" [p1]",[card],llm=model)
    assert report["claims"][0]["support_status"] == "supported"
    assert len(model.calls) == 1


def test_b_conflicting_dois_do_not_merge_via_title():
    papers = [{"paper_id":"a","title":"The same research title","doi":"10.1/a","authors":["Alice"],"year":2023},
              {"paper_id":"b","title":"The same research title","doi":"10.1/b","authors":["Bob"],"year":2024}]
    assert len(deduplicate_papers(deepcopy(papers))) == 2


def test_b_shared_doi_and_cross_key_bridge_still_merge():
    papers = [{"paper_id":"a","title":"First title","doi":"10.1/a"},
              {"paper_id":"b","title":"Second title","arxiv_id":"2401.00001"},
              {"paper_id":"c","title":"Third title","doi":"10.1/a","arxiv_id":"2401.00001"}]
    assert len(deduplicate_papers(papers)) == 1


def test_b_correction_relation_is_not_identity_evidence():
    papers = [{"paper_id":"a","title":"The same research title","doi":"10.1/a"},
              {"paper_id":"b","title":"The same research title","doi":"10.1/b","relation":{"is-correction-of":"10.1/a"}}]
    assert len(deduplicate_papers(papers)) == 2


def test_c_negated_method_never_receives_explicit_support():
    paper = {"paper_id":"p1","title":"Study","abstract":METHOD}
    wrong = METHOD.replace("We use","We do not use")
    card = _attach_evidence(PaperCard(paper_id="p1",title="Study",method=wrong,evidence_source="abstract"),paper,None)
    assert not any(c.explicitly_reported for c in card.field_claims.get("method",[]))
    gated = _quality_gate(card)
    assert not gated.method and "method" in gated.unsupported_fields
    restored = PaperCard.model_validate_json(gated.model_dump_json())
    assert not restored.field_claims.get("method")


def test_c_exact_method_still_receives_support():
    paper = {"paper_id":"p1","title":"Study","abstract":METHOD}
    card = _attach_evidence(PaperCard(paper_id="p1",title="Study",method=METHOD,evidence_source="abstract"),paper,None)
    assert card.field_claims["method"][0].explicitly_reported


def test_d_late_qualifier_survives_semantic_context():
    qualifier = "However, this result is not statistically significant and applies only to a pilot sample."
    original = "Results show 89% accuracy in the experiment. "+"Additional context. "*30+qualifier
    card = {"paper_id":"p1","title":"Study","evidence_source":"full_text","evidence_state":{"access_level":"full_text"},
            "evidence_spans":[{"evidence_id":"p1:e1","text":original,"source_type":"full_text","page":1}]}
    model = Echo()
    report = verify_review_claims("Results show 89% accuracy in the experiment [p1].",[card],llm=model)
    assert qualifier in model.calls[0]
    assert len(report["claims"][0]["evidence_snippets"][0]["text"]) <= 360


def test_d_over_budget_context_does_not_claim_verified_support(monkeypatch):
    from app.core.config import get_settings
    monkeypatch.setattr(get_settings(),"claim_verification_max_context_chars",1000,raising=False)
    card = {"paper_id":"p1","title":"Study","evidence_source":"full_text","evidence_state":{"access_level":"full_text"},
            "evidence_spans":[{"evidence_id":"p1:e1","text":"Results show 89% accuracy. "+"Context. "*200,"source_type":"full_text","page":1}]}
    model = Echo()
    report = verify_review_claims("Results show 89% accuracy [p1].",[card],llm=model)
    assert not model.calls
    assert "insufficient_context" in report["claims"][0]["issues"]
    assert report["claims"][0]["support_status"] == "unsupported"


@pytest.mark.parametrize("span", [
    {"evidence_id":"e","text":METHOD,"source_type":"full_text"},
    {"evidence_id":"e","text":METHOD,"source_type":"abstract","char_start":-1,"char_end":len(METHOD)-1},
    {"evidence_id":"e","text":METHOD,"source_type":"abstract","char_start":0,"char_end":1},
    {"evidence_id":"e","text":METHOD,"source_type":"abstract","char_start":0,"char_end":len(METHOD),"source_status":"source_unverified"},
])
def test_a_nonempty_span_without_valid_source_is_not_authorized(span):
    model = Echo()
    report = verify_review_claims(METHOD+" [p1]",[{"paper_id":"p1","evidence_spans":[span]}],llm=model)
    assert not report["valid"] and not model.calls


def test_a_tampered_source_hash_is_not_authorized():
    card = extract_paper_card({"paper_id":"p1","title":"Study","abstract":METHOD},llm=None).model_dump(mode="json")
    for span in card["evidence_spans"]:
        if span["source_type"] == "abstract":
            span["text"] = span["text"].replace("We use","We ban")
    model = Echo()
    report = verify_review_claims("We ban a dual encoder method for document retrieval [p1].",[card],llm=model)
    assert not report["valid"] and not model.calls


def test_b_conflict_cannot_be_hidden_by_identifier_bridge():
    papers = [{"paper_id":"a","title":"Same research title","doi":"10.1/a"},
              {"paper_id":"b","title":"Same research title","doi":"10.1/b"},
              {"paper_id":"c","title":"Same research title","arxiv_id":"2401.00001"},
              {"paper_id":"d","title":"Same research title","doi":"10.1/b","arxiv_id":"2401.00001"}]
    result = deduplicate_papers(papers)
    assert {p.get("doi") for p in result} == {"10.1/a","10.1/b"}
    assert len(result) == 2


def test_b_shared_doi_is_stronger_than_conflicting_weak_metadata():
    result = deduplicate_papers([
        {"paper_id":"a","title":"One title","doi":"10.1/a","arxiv_id":"2401.00001"},
        {"paper_id":"b","title":"Another title","doi":"10.1/a","arxiv_id":"2401.00002"}])
    assert len(result) == 1


def test_c_serialized_legacy_true_flag_is_reauthorized_before_writing():
    from app.agent.claim_plan import _is_author_stated
    from app.tools.write_deliverable import _writer_inputs
    from app.schemas.deliverable_schema import WritingPlan, WritingSection
    card = extract_paper_card({"paper_id":"p1","title":"Study","abstract":METHOD},llm=None).model_dump(mode="json")
    claim = card["field_claims"]["method"][0]
    claim["claim"] = METHOD.replace("We use","We do not use")
    claim["explicitly_reported"] = True
    restored = json.loads(json.dumps(card))
    assert not _is_author_stated(claim["evidence_id"],{"p1":restored})
    # 下游 Writer 必须重新判断，不能相信历史布尔标记。
    from app.services.evidence_support import authorized_field_claim
    assert not authorized_field_claim(restored,restored["field_claims"]["method"][0])
    plan = WritingPlan(deliverable_type="research_background",purpose="test",organizing_strategy="test",
                       sections=[WritingSection(id="s1",title="Background",purpose="test",supporting_paper_ids=["p1"])])
    safe_cards,_ = _writer_inputs(plan,{"paper_cards":[restored]})
    assert not safe_cards[0]["claims"]


def test_d_context_after_display_boundary_invalidates_cache():
    prefix = "Results show 89% accuracy. "+"Context. "*60
    card = {"paper_id":"p1","evidence_source":"full_text","evidence_state":{"access_level":"full_text"},
            "evidence_spans":[{"evidence_id":"e","text":prefix+"Pilot study only.","source_type":"full_text","page":1}]}
    cache = {}
    first = verify_review_claims("Results show 89% accuracy [p1].",[card],llm=Echo(),entailment_cache=cache)
    card["evidence_spans"][0]["text"] = prefix+"No significant difference."
    model = Echo()
    second = verify_review_claims("Results show 89% accuracy [p1].",[card],llm=model,entailment_cache=cache)
    assert first["claims"][0]["evidence_snippets"][0]["text"] == second["claims"][0]["evidence_snippets"][0]["text"]
    assert len(model.calls) == 1 and second["entailment_cache_stats"]["reused"] == 0


def test_old_verifier_version_cannot_reuse_local_report_or_cache(monkeypatch):
    import app.tools.verify_claims as verifier
    card = extract_paper_card({"paper_id":"p1","title":"Study","abstract":METHOD},llm=None).model_dump(mode="json")
    cache = {}
    with monkeypatch.context() as old:
        old.setattr(verifier,"_ENTAILMENT_VERIFIER_VERSION","atomic-entailment-v3")
        report = verify_review_claims(METHOD+" [p1]",[card],llm=Echo(),entailment_cache=cache)
    model = Echo()
    new = verify_review_claims(METHOD+" [p1]",[card],llm=model,entailment_cache=cache,
                              target_claim_ids=["c001"],verification_scope={"mode":"local","previous_report":report})
    assert new["verification_scope"]["mode"] == "full"
    assert new["entailment_cache_stats"] == {"reused":0,"computed":1}
    assert len(model.calls) == 1


def test_old_verified_draft_is_quarantined_even_with_best_effort():
    from app.agent.nodes.synthesis import _apply_final_quality_gate
    state = {"review":"A synthetic draft.","best_effort_generation":True,
             "claim_verification":{"valid":True,"claims":[{"claim_id":"c1","support_status":"supported"}],
                                   "verifier_version":"atomic-entailment-v3"},
             "generation_quality":{"passed":True,"support_rate":1.0},"citation_validation":{"valid":True}}
    _apply_final_quality_gate(state)
    assert not state["quality_gate"]["passed"]
    assert not state["quality_gate"]["draft_released"]
    assert state["quality_gate"]["draft_disposition"] == "quarantined"


@pytest.mark.parametrize("issue", ["source_unverified","insufficient_context"])
def test_source_and_context_failures_are_hard_even_with_high_overall_support(issue):
    from app.tools.verify_claims import _ENTAILMENT_VERIFIER_VERSION
    from app.agent.nodes.synthesis import _apply_final_quality_gate
    state = {"review":"A synthetic draft.","best_effort_generation":True,
             "claim_verification":{"valid":False,"claims":[{"issues":[issue]}],
                                   "verifier_version":_ENTAILMENT_VERIFIER_VERSION},
             "generation_quality":{"passed":True,"support_rate":0.99,"unsupported_claims":1},
             "citation_validation":{"valid":True}}
    _apply_final_quality_gate(state)
    assert not state["quality_gate"]["passed"]
    assert not state["quality_gate"]["draft_released"]


def test_exact_text_with_invalid_anchor_is_not_explicit(monkeypatch):
    monkeypatch.setattr("app.tools.extract_paper_card._candidate_evidence",lambda *args:[{
        "text":METHOD,"source_type":"abstract","section":"abstract","char_start":0,"char_end":1}])
    paper = {"paper_id":"p1","title":"Study","abstract":METHOD}
    card = _attach_evidence(PaperCard(paper_id="p1",title="Study",method=METHOD,evidence_source="abstract"),paper,None)
    assert card.evidence_spans[0].source_status == "source_unverified"
    assert not card.field_claims["method"][0].explicitly_reported
    assert not _quality_gate(card).method


@pytest.mark.parametrize("defect", ["missing_source", "negated", "unreported"])
def test_legacy_claim_is_rejected_by_quality_planning_and_theme_synthesis(defect):
    from app.agent.claim_plan import _extract_route_claims
    from app.agent.writing_plan import _all_claim_ids
    from app.tools.synthesize_themes import synthesize_themes

    card = extract_paper_card({"paper_id": "p1", "title": "Study", "abstract": METHOD}, llm=None).model_dump(mode="json")
    assert _extract_route_claims(["p1"], {"p1": card})
    if defect == "missing_source":
        card["evidence_spans"] = []
    elif defect == "negated":
        card["field_claims"]["method"][0]["claim"] = METHOD.replace("We use", "We do not use")
    else:
        card["field_claims"]["method"][0]["explicitly_reported"] = False
    restored = json.loads(json.dumps(card))
    gated = _quality_gate(PaperCard.model_validate(restored))
    assert not gated.method
    assert not _extract_route_claims(["p1"], {"p1": restored})
    assert not _all_claim_ids([restored], ["p1"])
    assert not synthesize_themes([restored], None)[0]["reported_methods"]


def test_pre_generation_context_over_budget_is_failed_without_model_call(monkeypatch):
    from app.agent.claim_plan import enforce_claim_evidence_gate
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "claim_verification_max_context_chars", 1000)
    model = Echo()
    text = METHOD + " Context." * 150
    cards = [{"paper_id": "p1", "evidence_spans": [{
        "evidence_id": "p1:e1", "text": text, "source_type": "abstract",
        "char_start": 0, "char_end": len(text),
    }]}]
    plans, report = enforce_claim_evidence_gate([{"route_id": "r1", "claims": [{
        "claim_id": "c1", "claim_text": METHOD, "claim_type": "finding",
        "evidence_ids": ["p1:e1"], "support_level": "single",
    }]}], cards, llm=model)
    assert not plans[0]["claims"] and not report["passed"]
    assert report["entailment_failed_claims"] == 1
    assert not model.calls


def test_context_budget_change_invalidates_local_verification(monkeypatch):
    from app.core.config import get_settings
    prefix = METHOD + " Additional context." * 60
    card = {"paper_id": "p1", "evidence_source": "abstract", "evidence_spans": [{
        "evidence_id": "p1:e1", "text": prefix, "source_type": "abstract", "page": 1,
    }]}
    cache = {}
    report = verify_review_claims(METHOD + " [p1]", [card], llm=Echo(), entailment_cache=cache)
    monkeypatch.setattr(get_settings(), "claim_verification_max_context_chars", 1000)
    model = Echo()
    updated = verify_review_claims(METHOD + " [p1]", [card], llm=model,
                                  entailment_cache=cache, target_claim_ids=["c001"],
                                  verification_scope={"mode": "local", "previous_report": report})
    assert updated["verification_scope"]["mode"] == "full"
    assert "insufficient_context" in updated["claims"][0]["issues"]
    assert not updated["valid"] and not model.calls


def test_semantic_payload_keeps_all_selected_evidence_within_budget():
    from app.schemas.verification_schema import ClaimEvidenceResult
    from app.tools.verify_claims import _llm_entailment_results
    model = Echo()
    qualifier = "However, no statistically significant improvement was observed."
    claim = ClaimEvidenceResult(claim_id="c1", sentence=METHOD, citations=["p1"], support_status="unsupported",
                                evidence_snippets=[{"text": f"Evidence {i}."} for i in range(3)]
                                + [{"text": qualifier}])
    _llm_entailment_results([claim], model)
    assert qualifier in model.calls[0]
