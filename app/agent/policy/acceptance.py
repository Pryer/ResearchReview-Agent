"""阶段验收读取当前产物；工具成功不能代替业务满足。"""
from dataclasses import dataclass
from app.agent.action_registry import admitted_ranked_papers


@dataclass(frozen=True)
class Acceptance:
    satisfied: bool
    missing: tuple[str, ...]


def search_acceptance(grant, state, results):
    completed = {r['operation'] for r in results if r.get('status') == 'completed'}
    search = 'targeted_search' if grant.profile == 'gap_recovery' else 'search_and_rank'
    if search not in completed:
        return Acceptance(False, ('search_not_completed',))
    ranked = (admitted_ranked_papers(state) if (state.get('retrieval_profile') or {}).get('mode') == 'hybrid'
              else state.get('ranked_papers') or [])
    if not ranked:
        return Acceptance(False, ('no_eligible_candidates',))
    if state.get('intent') == 'search_papers':
        return Acceptance(True, ())
    target = max(1, int(state.get('generation_limit') or state.get('max_papers') or len(ranked)))
    ids = {p.get('paper_id') for p in ranked[:target]}
    ready = set()
    for paper in state.get('paper_details') or []:
        year = paper.get('year')
        if (not paper.get('title') or not paper.get('authors') or type(year) is not int
                or paper.get('_metadata_mismatch')):
            continue
        if (state.get('start_year') and year < state['start_year']) or (state.get('end_year') and year > state['end_year']):
            continue
        ready.add(paper.get('paper_id'))
    if 'fetch_metadata' not in completed or not ids or None in ids or not ids <= ready:
        return Acceptance(False, ('candidate_details_incomplete',))
    # WHY: 阶段只证明指定候选已补全；最终引用数量与正文支持仍由后续门禁验收。
    return Acceptance(True, ())


def accept_delegation(grant, state, results):
    if grant.profile in {'initial_research', 'gap_recovery'}:
        return search_acceptance(grant, state, results)
    completed = {r['operation'] for r in results if r.get('status') == 'completed'}
    if grant.profile == 'evidence_build':
        if grant.scope.get('paper_ids'):
            return Acceptance(False, ('global_claim_planning_required_after_scoped_extraction',))
        from app.agent.claim_plan import claim_has_authorized_support, claim_source_strength_violations
        cards = {str(card.get('paper_id')): card for card in state.get('paper_cards') or []}
        claims = [claim for plan in state.get('claim_plans') or [] for claim in plan.get('claims') or []]
        if 'plan_claims' not in completed or not claims:
            return Acceptance(False, ('authorized_claim_plan_missing',))
        if (any(not claim_has_authorized_support(c, cards) for c in claims)
                or claim_source_strength_violations(state.get('claim_plans') or [], list(cards.values()))):
            return Acceptance(False, ('claim_evidence_authorization_incomplete',))
        if any((state.get(key) or {}).get('passed') is False for key in ('claim_evidence_gate', 'global_evidence_gate')):
            return Acceptance(False, ('evidence_quality_gate_not_met',))
        return Acceptance(True, ())
    if not results or not completed:
        return Acceptance(False, ('delegated_action_not_completed',))
    if grant.profile in {'draft_generation', 'section_revision'}:
        ready = bool(str(state.get('review') or '').strip() and state.get('pending_result_validation'))
        return Acceptance(ready, () if ready else ('candidate_not_ready_for_check',))
    if grant.profile == 'independent_check':
        from app.agent.checking import check_is_current
        ready = check_is_current(state) and (state.get('check_report') or {}).get('execution_status') == 'completed'
        return Acceptance(ready, () if ready else ('check_incomplete_or_stale',))
    if grant.profile == 'gap_diagnosis':
        ready = bool(state.get('evidence_gap_report'))
        return Acceptance(ready, () if ready else ('gap_diagnosis_missing',))
    if grant.profile == 'scope_revision':
        ready = bool(state.get('provisional_framework') and not state.get('scope_revision_failed')
                     and 'scope_revision_count' in results[-1].get('changed_state_fields', []))
        return Acceptance(ready, () if ready else ('scope_revision_not_completed',))
    if grant.profile in {'single_action', 'legacy_single_v1'}:
        ready = results[-1].get('outcome') not in {'blocked', 'unknown'}
        return Acceptance(ready, () if ready else ('single_action_did_not_satisfy_goal',))
    raise ValueError('unknown delegation acceptance profile')
