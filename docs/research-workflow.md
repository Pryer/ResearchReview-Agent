# Research Workflow

The workflow is implemented by app/agent/graph.py::run_research_agent and the
stage nodes under app/agent/nodes/.

## Clarification and working query

For a scope clarification, the conversation service sends the original user request,
the clarification question, prior clarification answers, and the latest answer to
`research_query_rewrite`. Explicit count, year, and section revisions are applied in
user-turn order; the latest explicit revision is authoritative.
The result is a versioned working query with a fingerprint and user source spans.
Time, reference-count, and requested-section constraints are checked against user
input before the rewrite enters semantic parsing and search planning. A failed or
invalid model rewrite falls back to a conservative merge of user text when that
merge still preserves the effective constraints. Conflicting count or year revisions
that cannot be safely merged stop planning with a technical error. Candidate
scope descriptions, exclusions, and seed queries are never treated as user rules.
The semantic frame is grounded against the original user text plus clarification,
and its cache key is the working query passed to planning.

## Citation and release boundary

Writing starts with stable paper identifiers. `citation_source_text` preserves that
version while `citation_rendered_text` records the numbered display version. A
repeated citation check or claim verification uses the stable version only when
the displayed body matches its stored snapshot. Legacy numbered bodies without a
trusted source version fail identity validation and stay quarantined; the current
paper-card order is never used to guess their meaning. Recovery snapshots carry
the body, both citation versions, mapping, references, and verification reports.
Blocked, failed, waiting, and cancelled results have no public body or released
draft flag, even if an older gate snapshot retained a release flag. A verification-only
recovery must run `validate_result`; it passes target claims and the prior report to
local verification when available, and otherwise runs full verification.

The graph supplies node adapters to the single autonomous scheduler. The main Agent receives a five-field context
view (`goal`, `state`, `key_evidence`, `decisions`, and `open_questions`) and proposes
one registered action per round. Search, evidence extraction, and writing execute
through versioned Search/Analysis/Writing Agent task contracts on isolated input
projections. Each operation has explicit read/write allowlists and typed patch checks.
The underlying nodes and all quality gates remain the same. A stale task
fingerprint or state version prevents an old specialist result from being committed
after scope or evidence changes.

New and resumed sessions use autonomous orchestration. Persisted `legacy` or absent
mode markers from older sessions are normalized before execution. A main-Agent clarification
pauses as `needs_clarification`, persists the question, and resumes the same evidence
pool with updated user constraints. A hard execution stop — token-budget exhaustion,
lease staleness, or a ledger conflict — is never reframed as a scope clarification: it
settles as `blocked`, or `failed` for a stale lease, with one consistent reason across
the job, session, and result, and only asks the user a question when an answer can
actually change the outcome. Deterministic verification controls publication;
failed/cancelled/waiting runs do not publish a body. See
[context architecture](agent-context-architecture.md) for artifact storage, budget
accounting and compatibility boundaries; see the
[validation report](validation/2026-09-22-runtime-cache.md) for the successful real
commit/recovery smoke and the budget-blocked 40-reference scenario.

Quality recovery and best-effort requests can preselect a short list of registered
actions. The same main loop executes them before requesting a fresh model decision;
unfinished actions remain in the checkpoint. A full rebuild discards the old draft
and its citation authorization before writing against changed evidence. A local
rewrite retains the old draft only while its evidence and claim authorization match.

Conversation-service execution persists a lease and cumulative budget. Successful
actions atomically commit their result, artifacts and checkpoint through database CAS;
cancelled or stale results cannot overwrite newer state. Real provider attempts retain
their charges independently of task success. Display history is archived before
truncation, and summaries advance by event cursor.

An explicit `resume_from_checkpoint=true` request resumes an interrupted session from
its last committed research state, preserving hard constraints and cumulative usage.
Uncertain external requests are not automatically replayed; crash takeover waits for
lease expiry. This is separate from answering clarification or revising a completed
review. See [runtime boundaries](agent-runtime-and-cache.md) for the API and migration.

## Stages

1. **Capability guard**: unsupported_task_guard rejects tasks outside the four
   supported deliverables before retrieval.
2. **Scope understanding**: intent, slots, topic disambiguation, time range, source
   language, reference target, and deliverable requirements are normalized.
3. **Planning**: plan_node creates concepts, exclusions, and search plans.
   Research-status tasks may receive provisional routes before search.
4. **Retrieval**: language-compatible sources are queried through bounded dual-channel
   search. Results are deduplicated, filtered, ranked, and optionally refined. A
   bounded rule-ranked tail may be screened when exclusions shrink the main window;
   tail papers that were not semantically confirmed remain diagnostic-only.
5. **Metadata verification**: details are enriched by the matching source; missing
   values stay missing. Search-only requests can return at this point.
6. **Evidence extraction**: optional PDF download/parse produces page-aware text;
   extract_card_node creates PaperCards and Evidence Cards.
7. **Route and evidence checks**: routes are validated, bounded evidence recovery may
   run, and evidence-backed clusters are produced as a fallback.
8. **Claim planning and gates**: claims are bound to evidence before writing;
   claim-evidence and global evidence gates record blocking or degradable deficits.
9. **Synthesis and writing**: generate_deliverables_node uses a WritingPlan and the
   single deliverable renderer pipeline. WritingPlan construction re-applies the
   semantic-confirmation boundary so an explicit best-effort override cannot
   reintroduce diagnostic-only reserve papers.
10. **Output validation**: claim alignment, sentence-level claim verification,
    citation validation, citation-gap repair, and the final quality gate run before
    final_answer_node. Final citation sources and the unique-reference count exclude
    unconfirmed reserve papers; any occurrence in draft text is reported as a missing
    citation.

## Recovery and regeneration

Regeneration starts by restoring explicit user constraints from `research_request`
and migrating old session counters, permissions, writing versions, and private
section checkpoints. It clears only the current round's invalid writing products.
Routes and claim plans are reused only when their evidence, scope, screening state,
and policy fingerprints still match. A previous
`user_accepted_best_effort_generation` marker describes a user strategy and is not
treated as a new writing failure.

The write boundary is checked again after derived readiness is recomputed. A stale
snapshot, year-window mismatch, or recovery/readiness conflict remains a
pre-generation block. The final gate preserves that phase and does not describe an
empty result as a post-generation draft failure. Citation-gap repair records the
pre-repair quality vector and rolls back a candidate if any hard component becomes
worse, even if its raw citation count increases.

Section writers may emit an evidence span identifier such as `paper_id:e001`.
Before validation, identifiers from both `evidence_spans` and structured `claims`
are normalized to the owning paper ID. Unknown identifiers remain errors. This
keeps evidence provenance while preventing a valid span citation from being
mistaken for an unauthorized paper citation.

Before writing, the system separately counts raw candidates, confirmed in-scope
papers, evidence-backed papers, claim-authorized papers, and planned papers. It
rebuilds claim coverage when usable evidence exists but is not authorized, and
starts targeted retrieval only when the current pool cannot support the hard target
and the user permits evidence expansion. The target, time window, scope, and
required focuses remain copied from `research_request`; no lower count is inferred.
Each focus-coverage diagnostic includes its grounded source, explicit/inferred
status, matched count, required count, and whether it participates in the hard
gate. Model-inferred suggestions remain advisory. A model-generated evidence alias
cannot promote a focus to a user-explicit constraint.

The same deterministic controller handles pre- and post-generation failures. Route
recovery and writing recovery share `recovery_total_action_budget`, action history,
input fingerprints, cancellation boundaries, and no-progress detection. Internal
state is recomputed before search; structure failures rebuild routes; citation
allocation and claim failures rebuild only the necessary products; metadata gaps
refresh existing papers before searching. If that shared budget is exhausted and
usable evidence remains, the conversation service performs one final best-effort
generation pass. The final gate still reports every unmet constraint and publishes
the result only as an explicitly limited `partial` draft. Technical failures, stale
state, missing user material, and absent usable evidence do not enter this fallback.
API callers can set `best_effort_on_failure=false` to retain strict blocking even
when the deployment default is enabled. A blocked session can later submit the
explicit “生成可用草稿” action; the service recognizes it before new-topic parsing
and reuses the saved evidence and recovery budget. Metadata-only records are not
enough to release a draft. If the user specified existing evidence only, no
retrieval action is executed.

Each section checkpoint stores text, writing version, local validation state, and an
input fingerprint that covers its plan, allocation, evidence content/access, screening,
actual authorized card projection, deterministic draft, claim constraints, topic/scope/
focus, citation policy and cross-route synthesis obligation. The authorized survey list
is included because every section receives it; unrelated global papers do not invalidate
all sections merely by changing the global snapshot. During local recovery, matching validated sections bypass the LLM
section writer and only failed sections are generated. The merged document always
runs the complete claim, citation, metadata, structure, and final-integrity chain
again. Checkpoint text and quarantined drafts remain private until the final gate
permits release.

The claim verifier uses one contract before and after sentence repair. Deterministic
checks first enforce citation identity, numeric values, access level, and evidence
binding; eligible atomic claims are then batch-checked for semantic entailment. A claim
that meets the verification preconditions (it has citations and evidence snippets) but
receives no valid provider verdict is recorded as `not_completed` semantic verification,
not silently turned into an evidence finding. A claim that fails a structural
precondition (no citation, an invalid citation, or no evidence snippet) is instead
settled deterministically as `verified` / `unsupported`, because re-submitting it would
always yield zero entailment batches; it stays blocking but is never queued for semantic
re-verification. The recovery controller preserves the written document and retries only
the genuinely pending semantic claims, while successful cached decisions remain reusable.
If the same input makes no progress, recovery stops with the exact incomplete items
instead of switching to retrieval or full-document generation.

Fallback writing only emits a citation when a card contains an attributable problem,
method, result, or other authorized claim. It does not append generic source sentences
or title-only entries to reach a reference target. Token-only field fragments are
rejected both at section-candidate validation and at final integrity validation.
Conservative regeneration still uses the configured section writer; conservatism is
enforced by its evidence and claim inputs rather than by disabling the writer.

An explicit reference target remains the success target throughout recovery. If all
other final checks pass but the valid reference count is still short, the
configurable best-effort coverage policy may
release the verified text with `partial` status. The default threshold is 85%; this
does not rewrite `required_reference_count` or report the task as fully successful.
Writer may start at that threshold only when the same proportion of requested
papers is evidence-backed and claim-authorized. Recovery budget exhaustion is not
required for this count-only partial release.
An automatic final best-effort pass after recovery exhaustion also observes the
same minimum ratio when the user specified a reference target.
Any claim, citation-consistency, metadata, language, section, or integrity failure
continues to quarantine the draft regardless of reference coverage.

Background jobs expose real recovery actions and reference-coverage counters. While
an action is running, the frontend shows that revalidation is pending; after the
gate runs it shows whether revalidation passed. If the shared budget is exhausted,
the frontend shows the final best-effort generation step. An eligible old blocked
result exposes a “生成可用草稿” button, and a released partial draft can be downloaded
with the same limitation banner contained in the API body. Counts that have not yet
been calculated display as “尚未统计” rather than zero.

## Branch Guarantees

- Cancellation, token-budget exhaustion, lease staleness, and ledger conflicts are
  checked at node boundaries and inside every gate, entailment, and clustering handler;
  they propagate to the execution boundary and must never be swallowed or degraded into
  a soft gate failure.
- PDF disabled is a valid path; metadata/abstract evidence remains usable.
- External source failure should allow other sources to continue and must not be
  misreported as a verified empty result. A CNKI chromedriver/browser version mismatch
  is classified distinctly from access-restricted, network, and success-but-zero-results
  outcomes, and short-circuits further driver launches in the same process instead of
  repeatedly restarting a known-incompatible driver.
- Explicit user constraints are never silently expanded or weakened.
- Incremental revision reuses the session and paper set, then reruns affected
  validation, writing, and citation stages.
- Reverification submits only claims whose citation, evidence and access prerequisites
  hold. Structural failures stay unsupported; invalid model labels stay pending.
  Both claim IDs and sentence indices are filtered before execution, and a known
  report with no eligible target stops without running another validation action.
- Fresh verification can improve recovery without changing the draft or evidence.
  The current validation fingerprint must match the draft and differ from the
  pre-recovery version. Replayed reports, cleared gates and rejected candidates do
  not count as progress. Persisted rejection history also constrains section rewrites
  through action availability, direct Controller calls and the recovery controller.

For a worked, mock-friendly narrative see docs/react_workflow_example.md. Canonical
machine-readable inputs are under examples/.
# 混合检索候选筛选

新任务可通过 `RETRIEVAL_RANKING_MODE=hybrid` 选择 BM25、dense embedding、RRF 与专用文本 reranker。输入是去重候选的标题、摘要、关键词；检索 query 由语义主题、已确认范围和检索路线派生。语言及定向恢复路线获得有限检查机会，排名不会代替语义准入。缺少摘要的论文仍可凭标题参与排序，完全没有文本的记录保留在候选池并报告数量。

hybrid 把主题别名、分支来源和语义范围的词面未命中标为待核验。已知年份超界、学位论文及明确的标题排除条件仍直接排除。进入详情和证据流程前，LLM 必须返回 `include`、`direct/near`、可用交付物，并逐项确认用户已确认的必要条件。LLM 缺项、失败或详情改变后未重新确认的记录不得计入正式候选。最终引用和主张支持继续由原有门禁判定。

`rules` 是默认模式；`hybrid_shadow` 只记录候选数量、融合与模型调用诊断，不改变规则结果。检索 profile 随会话保存；旧会话缺 profile 时使用 `rules`。向量和精排分数是可重建缓存，筛选结论按文本、范围、协议、prompt 和模型指纹失效。

确定性硬过滤只判断年份、文献类型和显式范围规则，不重写已有的逐篇语义结论。融合候选在排序入口标记 pending；详情复核只复用内容、用户语义范围和完整合同均有效的授权。新增检索分支或关键词不改变用户确认的准入范围指纹；真实范围或逐篇协议变化仍需重新筛选。无 LLM 时 pending 留在候选层，详情动作不可消费它。检查点若停在 pending 窗口且没有新查询，下一次检索动作直接完成语义准入。

质量恢复中新论文获准入后，主控优先推进详情、证据卡、路线和主张；证据变更会失效旧授权与写作门禁。终态可编辑状态保留本会话 retrieval profile，后续恢复沿用同一检索模式。
