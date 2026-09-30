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

Reference-count recovery has one configurable delivery fallback. When the only
pre-generation issue is a reference-count shortfall, at least the configured ratio
of the requested references must be evidence-backed and claim-authorized before
Writer can build a fresh writing plan. When the *only* final hard issue is
`minimum_cited_references_not_met` and the final-valid/reference target ratio reaches
`reference_coverage_best_effort_ratio` (default `0.85`), it may release the draft as
`partial` / `released_best_effort`. The original requested count and blocking issue
remain in the quality gate and the response includes the exact numerator,
denominator, ratio, and threshold. Release does not wait for recovery budget exhaustion.
Claim support, citation authorization, metadata,
language, structure, and final-integrity failures are never softened by this ratio.
Explicit "generate usable draft" authorization also cannot release factual prose
with remaining unsupported/unverified claims, failed citation authorization or
metadata verification, or failed structure/integrity checks. Such drafts remain
quarantined until repaired; `partial_success` alone is not proof of safe delivery.
Set `enable_reference_coverage_best_effort_release=false` to require the exact count
for every release.

When the bounded recovery controller exhausts all actions and usable evidence still
exists, `enable_recovery_exhausted_best_effort_generation=true` permits one final
generation pass over that evidence. This pass keeps the original constraints and all
failed gate details, sets the result to `partial` / `released_best_effort`, and adds a
visible limitation banner. For an explicit reference target it also requires at least
the configured coverage ratio; below that ratio it does not start an automatic final
generation or release its draft. It runs at most once. Runtime errors, stale state, missing
required user material, authentication requirements, and a complete lack of usable
evidence remain blocked because no trustworthy draft can be produced from them.
An explicitly requested draft that has a current verification fingerprint and a
releasable gate finishes deterministically after mandatory recovery actions have
completed, without another model decision. Both writing entrypoints share the
same persisted rejected-input guard. Rejected candidates restore the previous
text, citation counts, verification reports, and chapter checkpoints; rollback
does not run the mutating verification/repair chain again. Changed evidence
invalidates the restored verification fingerprint and still requires validation.
The public request field `best_effort_on_failure=false` overrides the enabled
default for callers that require strict blocking. A manual “生成可用草稿” action
on an existing blocked session is still an explicit per-session authorization.

Focus coverage also records requirement provenance. Only a requirement grounded
to user-explicit source entities can block formal writing. An LLM-inferred focus is
reported as `advisory_missing` and remains visible in diagnostics and draft
limitations. Requirement aliases may help evidence matching, but cannot by
themselves prove that the user explicitly requested the underlying focus.

The global evidence gate is intentionally measurement/recommendation-only in the
current version. It does not itself execute recovery; final quality, claim, and
deliverable checks enforce delivery behavior. This distinction is covered by
tests/test_global_evidence_gate.py and tests/test_generation_quality_gate.py.

No gate may fabricate evidence, silently lower an explicit constraint, or convert an
unsupported claim into confident prose. A control exception raised inside a gate,
entailment batch, or clustering handler — token-budget exhaustion, cancellation, lease
staleness, or a ledger conflict — propagates to the execution boundary instead of being
caught and degraded into a soft `passed=False`; only ordinary provider failures are
recorded and recovered. Gate outputs are persisted in state and
included in diagnostics/evaluation bundles for auditability.

## State Invariants and Recovery Observability

At the write boundary, `app/agent/state_invariants.py` checks that the requested
and top-level year windows agree, evidence-gap diagnostics refer to the current
snapshot version/fingerprint, unresolved recovery needs are not silently paired
with `generation_readiness.ready=true`, and source-health summaries agree with
this round's source diagnostics. Blocking violations quarantine the draft before
writing; source-health disagreements remain explicit warnings.

Regeneration clears writing-derived readiness and section diagnostics before this
boundary is evaluated, then recomputes them from the current evidence snapshot.
This prevents an old `ready` value or an old failed section from blocking an
otherwise valid retry. Internal state conflicts remain blocking even when the user
has enabled best-effort generation; an empty result is reported as a
pre-generation block. The best-effort strategy marker is excluded from recovery
failure classification, so a later conservative rewrite can target the actual
section or claim failure.

Incremental regeneration also persists `recovery_statistics`, including reused
and recomputed claim counts and observed LLM calls. These counters are
observability data only: they do not weaken claim, citation, or final-integrity
gates.

Citation-gap repair uses an authorization-safe acceptance rule. It records the
claim-citation mismatch count before and after repair and rejects a candidate that
adds unauthorized mismatches, even when its raw cited-paper count is higher. The
section writer maps evidence IDs found in either `evidence_spans` or structured
`claims` to the owning paper ID before citation validation; unknown IDs remain
invalid.

Generation readiness records seven distinct reference counts: raw candidates,
confirmed in-scope papers, evidence-backed papers, claim-authorized papers,
writing-plan coverage, actual citations, and final valid citations. A paper-card
count is never used as a substitute for claim authorization. Whenever an explicit
reference target is set and the claim-authorized union falls below it,
`minimum_planned_references_not_met` blocks the writer and triggers claim-plan
rebuilding — including when `claim_plans` is empty (expired authorization cleared or
planning failed), so an empty plan can never inherit `ready=true` and bypass the
authorization check. Coverage claims rebuilt to fill the shortfall pass through the
same entailment gate as route claims; coverage-added claims that fail entailment are
dropped rather than authorized. The coverage builder computes the shortfall only from
the current eligible set, so stale out-of-scope or unconfirmed claims cannot consume
the target.

All route and writing recovery actions consume `recovery_total_action_budget`.
The deterministic controller classifies internal-state, structure, reference,
claim/citation, section, metadata, and user-input failures; it then chooses state
recomputation, structure or claim rebuilding, citation reallocation, local section
rewriting, or in-scope targeted retrieval. An action that makes no component of the
quality vector improve is not repeated for the same evidence/scope fingerprint.
User input is requested only when access/material is missing or continuing requires
changing a user constraint such as the existing-evidence-only policy.

Candidate adoption compares every quality-vector component independently. A lower
reference shortfall cannot compensate for more unsupported claims, citation
mismatches, missing sections, metadata failures, or new hard issue codes. Failed
candidates leave the prior text, plans, mappings, and verification report intact.
Budget exhaustion returns a concrete remaining-issue report and preserves any real
section checkpoints or quarantined draft; it does not show another generic retry
menu.

Claim support and verification completion are separate gate dimensions. Each claim
records `support_status`, `verification_status`, and `verification_method`; reports
also expose `unverified`, `structural_failures`, and `verification_coverage`. A claim
that meets the verification preconditions (citations and evidence snippets) but has no
valid provider verdict stays conservatively unsupported and is blocked as
`claim_verification_incomplete` rather than reported as proven evidence insufficiency.
A claim missing a structural precondition (no citation, an invalid citation, or no
evidence snippet) is settled as `verified` / `deterministic` / `unsupported` and counted
under `structural_failures`; it is excluded from the re-verification queue so it can
never drive a zero-submit reverify loop, while still blocking the gate as unsupported.
Recovery keeps the current draft and retries only the genuinely pending semantic claims
once for the same input fingerprint; it does not search for papers or regenerate the
document. The progress vector keeps unverified claims as an independent component, so
relabelling an unsupported claim as unverified cannot count as improvement.

Semantic cache keys cover claim text, cited paper identities, evidence text and
location, evidence access level, verifier version, and threshold policy. Local
revalidation reuses unchanged sentences by exact content including citations, so a
sentence insertion or deletion cannot attach an old verdict to a different sentence
number. Legacy reports without the verification contract are fully revalidated.

Section candidates are promotable checkpoints only after their local writer checks,
configured route density floor, and cross-sentence duplicate checks pass. The route
body default is controlled by `route_section_min_plain_chars`; it is a compatibility
default rather than evidence that every domain or deliverable is sufficiently
covered. Metadata-only title entries, token-only metric fragments, and generic
citation placeholders do not count as body evidence.

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
tests/test_verify_claims.py, tests/test_deliverables.py,
tests/test_generation_recovery.py, tests/test_hybrid_retrieval.py, and
tests/test_hybrid_retrieval_integration.py.

# 混合检索的资格边界

BM25、向量余弦相似度、RRF 和专用 reranker 只确定优先检查的候选。语义筛选未确认的论文保留为待检查，不进入详情证据池；论文相关性通过后仍需完成元数据、证据跨度、主张授权、引用一致性与最终发布门禁。`required_reference_count` 始终代表最终实际使用的唯一有效参考文献数量，不由检索 Top-K 或候选池规模替代。远程模型失败会记录降级，不能让未确认论文取得正式资格。

恢复进展只根据当前证据对应的重新验证结果结算。清理旧 gate、主张或引用计划只是失效操作；阻塞代码消失本身不证明质量改善。写作前可用当前卡片和主张重新计算 readiness；写作后要求当前正文的验证指纹。无进展终止且旧 gate 已失效时，系统重新计算当前写作前缺口，并向用户区分证据卡、可用证据和正文有效引用数量。

语义验证前提由同一判定维护：有引用、有证据片段，且不存在引用无效、访问等级不足或结果证据等级不足等结构性硬失败。结构失败按确定性不受支持结算，仍阻断门禁，但不发送蕴含请求、不进入重验队列；旧报告里误标为 `not_completed` 的结构失败同样排除。模型返回缺失或非法蕴含标签时保持语义待验，记录安全的缺项/响应错误类型，只缓存合法判定；同批合法的否定结论继续保留。

重验直接入口同时过滤主张 ID 和句子索引。已有报告且过滤后没有目标时，保留证据、原始篇数要求和隔离稿，以 `blocked` 返回，不再启动验证动作；缺少旧报告或验证状态字段时沿用完整校验兼容路径，不能将未知状态视为已经核验。

恢复历史保存验证前指纹。正文和证据不变时，新判定改变当前验证版本且质量向量改善，也可以记为 `improved`；旧指纹、报告重放、清空门禁或候选回滚不算进展。旧历史没有验证前指纹时保留原有保守判定。无进展重写历史和候选拒绝记录共同限制后续重写，动作可用性、Controller 直接执行和外层恢复共用该限制，改变决策措辞或重启不能重获额度。
