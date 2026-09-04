# Quality Gates

Quality gates are behavioral checkpoints, not logging-only diagnostics. Their result
must affect execution, draft status, output wording, or recovery options.

## Gates

| Gate | Implementation | Behavior |
|---|---|---|
| Unsupported task | app/agent/unsupported_task_guard.py | Blocks before retrieval and returns a supported-scope message |
| Route validity | app/agent/route_validator.py | Keeps, weakens, revises, splits, or drops routes; weak routes may trigger recovery |
| Claim-evidence | app/agent/claim_plan.py and app/agent/nodes/verification.py | Binds claims to evidence and lowers language strength or blocks generation readiness |
| Global evidence | app/agent/global_evidence_gate.py | Deterministically measures citation, recency, route balance, quality, and claim support |
| Deliverable validation | app/tools/validate_deliverable.py | Rejects invalid structure, unauthorized citations, leakage, and evidence-limited sections |
| Post-generation quality | app/agent/nodes/synthesis.py | Quarantines an unfit draft, marks partial success, blocks formal delivery, or records an explicit warning |
| Claim verification | app/tools/verify_claims.py | Marks unsupported claims and supplies revision/removal decisions |

## Per-Route Evidence Targets

`route_min_core_evidence` answers only "is this route sufficiently supported"; it is
not a recovery goal. `app/agent/route_targets.py` derives a separate per-route target
from the deliverable type and the user's requested reference count:

- `research_status`: the requested reference count is shared across active routes and
  clamped to `[route_recovery_target_min, route_recovery_target_max]`.
- `related_work`: a route that carries competing-work comparison gets
  `route_recovery_competing_work_bonus` extra papers, because a missing competing work
  makes the comparison meaningless while a missing prior work does not.
- `narrative_review`: year-span diversity is tracked as a separate deficit. Reaching
  the paper count does not satisfy it, since same-year evidence cannot support a
  research trajectory.
- Any other deliverable (including `research_background`, which organizes by
  argumentative role rather than routes) gets no extra target and keeps the previous
  behavior.

The recovery loop uses these targets to allocate query budget per route, to keep
targeted recall inside `top_k`, and to decide convergence per route: evidence that
lands on an already-satisfied route does not count as progress for a route still short
of its target. When recovery is exhausted and a route is still short, the route is
merged for writing as before and the remaining deficit is reported as a
`route_evidence_target_not_met` warning together with any `section_floor_deficits`. It
is deliberately non-blocking: relevance filters are never relaxed to reach a count.

## Blocking Semantics

User-explicit requirements such as a minimum unique reference count or explicit date
window are blocking when unmet. Implicit defaults may be non-blocking, but the
result must remain visibly degraded or best-effort.

The global evidence gate is intentionally measurement/recommendation-only in the
current version. It does not itself execute recovery; final quality, claim, and
deliverable checks enforce delivery behavior. This distinction is covered by
tests/test_global_evidence_gate.py and tests/test_generation_quality_gate.py.

No gate may fabricate evidence, silently lower an explicit constraint, or convert an
unsupported claim into confident prose. Gate outputs are persisted in state and
included in diagnostics/evaluation bundles for auditability.

## State Invariants and Recovery Observability

At the write boundary, `app/agent/state_invariants.py` checks that the requested
and top-level year windows agree, evidence-gap diagnostics refer to the current
snapshot version/fingerprint, unresolved recovery needs are not silently paired
with `generation_readiness.ready=true`, and source-health summaries agree with
this round's source diagnostics. Blocking violations quarantine the draft before
writing; source-health disagreements remain explicit warnings.

Incremental regeneration also persists `recovery_statistics`, including reused
and recomputed claim counts and observed LLM calls. These counters are
observability data only: they do not weaken claim, citation, or final-integrity
gates.

## Screening Confirmation Boundary

`rule_screened_reserve` papers are rule-ranked overflow candidates that have not been
semantically confirmed. They may remain in retrieval diagnostics, but they are not
eligible writing evidence and do not count toward a user-explicit reference minimum.
The marker is read from `paper_details`, because private retrieval keys are not carried
into `PaperCard`.

The same exclusion set is enforced at all three boundaries: generation readiness,
WritingPlan construction, and final citation validation/counting. If a best-effort
draft still contains such an ID, citation validation reports it as missing and the
post-generation gate reports the real confirmed-reference shortfall instead of
marking the requirement as met.

## Test Evidence

Focused behavior is covered by tests/test_unsupported_task_guard.py,
tests/test_route_validator.py, tests/test_route_targets.py,
tests/test_route_recovery_gold.py, tests/test_evidence_recovery.py,
tests/test_global_evidence_gate.py, tests/test_generation_quality_gate.py,
tests/test_verify_claims.py, and tests/test_deliverables.py.
