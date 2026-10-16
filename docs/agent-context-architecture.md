# Agent Context Architecture

ResearchReview-Agent separates authoritative research state from the context shown
to the main Agent. The authoritative `ResearchAgentState` retains retrieval data,
paper cards, evidence spans, validation reports, drafts, checkpoints, and recovery
history. The model-facing `MainAgentContext` is rebuilt from that state and contains
exactly five top-level fields:

```text
goal
state
key_evidence
decisions
open_questions
```

The authoritative state also stores a clarification rewrite with source anchors and
the source query used for semantic parsing. Citation identity is tied to the
pre-render body and a matching numbered render snapshot. These fields remain in
the private research state and do not add fields to `MainAgentContext`.

`goal` preserves the user request, deliverables, scope, and explicit constraints.
`state` contains separate execution/research status, the current stage, allowed and
pending actions, a bounded recent trajectory, failure reasons, artifact counts, gate
summary, and remaining action/token/retrieval/recovery budgets. `key_evidence` contains bounded findings with paper and
artifact references. `decisions` records still-valid scope, route, and recovery
choices. `open_questions` contains unresolved clarification, evidence, and quality
problems.

## Stable and dynamic context

Role rules form a stable system prefix. Native calls carry the tool catalog only
through `tools`; the JSON fallback retains the textual catalog. Dynamic goals, evidence,
and task state are serialized separately. `LLMService.for_agent(role)` injects the
stable prefix for specialist calls; `MainAgentPolicy` owns the main prefix and
sorted action catalog. Both retain provider fallback, timeout, and metrics behavior. Raw document text is treated as research material and cannot redefine the
role or tool boundary.

The context builder is deterministic. It may shorten repeated or low-priority
material to fit configured limits, but it does not remove the goal, explicit user
constraints, or blocking questions. It does not use an LLM to invent summaries.
Every key-evidence entry must resolve to a paper card in the authoritative state.
The total character budget reserves room for the stable system/tool prefix and the
structured decision output. If the irreducible context still exceeds the dynamic
budget, the request fails with `context_budget_exceeded` rather than dropping a hard
constraint.

## Main-Agent decision loop

`MainAgentPolicy` receives the latest five-field status bar and the persisted
append-only trajectory per round and returns
exactly one `AgentDecision`. The stable action registry is the source for tool
descriptions, parameter schemas, roles, and controller authorization. Providers with
native tool calling use a required single tool call; deployments that disable native
tools, or explicitly reject the native protocol, use strict JSON validated against
the same contract. Tool schemas stay stable across rounds; the current allowed
actions and correction reasons live inside `state`.

`MainAgentLoop` executes `build -> decide -> validate -> execute -> commit observation -> rebuild`.
The model may request completion, but deterministic quality gates decide whether the
current draft can be released. Invalid actions receive one bounded correction chance;
repeated no-progress actions, round limits, action limits, cancellation, and token
budgets end in an explicit terminal state.

There is a single production scheduler. The initial, incremental, regeneration,
checkpoint-resume and verification-only entry points all reach the same five-field
loop; verification-only sessions expose only validation and terminal control
actions. Historical sessions whose persisted mode is absent or `legacy` are
normalized to this scheduler at the execution boundary: the original research goal,
explicit constraints, papers, evidence cards, recovery history and consumed budget
are preserved, and only results derived from the retired scheduler or a stale
evidence version are recomputed. `agent_orchestration_mode` survives as a persisted
audit field, not as a routing switch, and migration is idempotent.
When a quality-recovery decision or best-effort request has already selected work,
the same main loop executes those registered actions before asking the model for a
new choice. The pending action list is checkpointed; each action still uses the
normal permission, budget, cancellation and version checks. A missing prerequisite
blocks explicitly instead of silently skipping the selected remedy.
Request parsing runs in the search/planning adapter before the five-field decision
loop; it is not a main-Agent decision call.

`request_clarification` returns public `needs_clarification`, stores its question and
private `waiting_user` state, and resumes with the original evidence and the latest
user constraints. A hard execution stop (budget exhaustion, lease staleness, or a
ledger conflict) is distinct from a clarification: it settles as `blocked`, or `failed`
for a stale lease, with one consistent reason across job, session, and result, and is
never reported as a missing-checkpoint question. Cancellation, failure and waiting
states do not publish a draft.
Generation and citation-repair candidates run the normal verification chain; a
worse candidate restores the previous generation products and is verified again.

The budget ledger is shared across conversation parsing, graph entry points,
specialist workers and automatic recovery. Each real LLM attempt reserves prompt
bytes plus its output token cap before execution. When the remaining budget cannot
cover a reservation, the attempt is rejected before any provider call: the request is
not counted as sent, no charge is incurred, and a structured diagnostic records the
operation, used, reserved, requested, and remaining amounts. Provider usage settles an
accepted reservation; missing usage or failed attempts retain the conservative
reservation and set `usage_estimated`. Retries and fallback requests are charged
separately.
Retrieval is counted per search-node round, and recovery has its own sublimit.
Nested controller actions charge the parent action once. The deadline spans the
current user turn; cumulative usage survives pause/resume. In-flight external calls
remain subject to their client timeouts; late results cannot commit after expiry.

## ReAct epochs and context capacity

Within an epoch, the original goal and subsequent goal/constraint revisions, status
bars, assistant decisions, and observations are appended without rewriting prior
messages. Native assistant calls retain their provider IDs, raw argument JSON,
content, and DeepSeek continuation fields; `tool` observations match the IDs.
JSON mode uses assistant JSON decisions followed by explicitly marked user
observations. Transport/provider changes can invalidate cache prefixes; DeepSeek
private continuation fields are omitted for other providers.

L1 projects immutable bounded observations (status, outcome, findings, errors,
evidence and output references). Full task results remain in authoritative state.
Large results are split into immutable artifacts behind one service-only archive
index, so the index list itself cannot fill the model context. Oversized reference
lists retain their full values in that archive and expose a count plus the index;
no truncated JSON or fabricated evidence references are used. Specialists continue
to load only fields and artifact types allowed by their existing task contracts.

Before each model request, the capacity guard counts the serialized system, native
tools and complete trajectory with a conservative UTF-8 byte upper bound plus
message framing overhead; this is an estimate, not provider usage. The configured
`AGENT_CONTEXT_WINDOW_TOKENS` (default 65536) must be no larger than the smallest
usable window among the primary and backup models. Output reserve equals
`LLM_CONTROL_PLANE_MAX_TOKENS`. The status bar exposes the configured limit and
previous-request estimates/usage. Actual prompt/completion/cache tokens come from
provider usage, including OpenAI `prompt_tokens_details.cached_tokens`.

At `AGENT_CONTEXT_COMPACT_RATIO` (default 0.8), L2 archives the entire epoch and
starts a new one containing all goal revisions, a deterministic checkpoint with
current evidence/decisions/questions, the compact observation/decision ledger,
archive references and the latest complete interactions (default 2). No LLM is used
to invent a summary. The previous archive is linked from the new archive. A short
cooldown avoids repeated threshold compaction until more than the retained turn
count has accumulated; a hard window overflow bypasses that cooldown.

L3 blocks before model invocation if the retained set still exceeds the hard window,
configuration is invalid, an archive fails, or a pending call is unresolved. L2
requires the session Artifact Store; standalone in-memory graph calls explicitly
block at this boundary instead of silently discarding history. The existing
character limits still govern each rebuildable five-field status view separately.

The active trajectory is stored as service-only JSON fragments in checkpoints and
restored through the existing hydration path. Controller commits pair the task
result and its observation in one transaction. A process restart after that commit
recovers a complete interaction. Existing leases, CAS, cancellation, task
idempotency and quality gates remain authoritative. This design enables shared
prefix reuse; it does not implement an inference engine KV cache or guarantee a
provider cache hit or a measured latency/cost improvement.

## Version and invalidation boundary

Each context snapshot records a source fingerprint and the state/evidence/writing
versions used to build it. Scope, hard constraints, evidence snapshots, route
decisions, recovery decisions, or gate results change that fingerprint. A task whose
input fingerprint or state version no longer matches is returned as `stale`. The
controller checks before execution and again immediately before commit.

Context snapshots are persisted as immutable, session-scoped research artifacts. Artifact reads
must match both `session_id` and `artifact_id`; knowing an artifact ID from another
session is insufficient. Corrupt JSON is reported rather than converted to an empty
payload. Registered types include raw results, metadata, document fragments, cards,
claim/writing plans, drafts and task results. The session repository stores these
heavy editable-state fields as an artifact manifest and hydrates them on reads;
the public result snapshot is retained for API compatibility. Production task
reads check role, type and version as well as session ownership. Missing, forbidden,
corrupt and mismatched-version references have distinct errors.

Artifacts carry schema versions, payload hashes and source references. Drafts and
parsed-document JSON are split into bounded fragments with character offsets;
page information is preserved when supplied by the parser, never inferred.
Individual payloads/provenance are limited to 256 KiB; session JSON to 8 MiB.
Embedded PDF data is rejected. PDF references must resolve inside configured PDF
or parsed-document storage, and their content hash is checked again on read.

Research requests, clarification answers, result events and conversation entries are
archived before the display history is truncated. A durable cursor rebuilds summaries
from new events only. Old sessions import the still-available history once using a
database migration marker; earlier truncated data remains explicitly unknown. Resolved
questions and superseded decisions stay resolved after resume. The compact context
stores references rather than copying full documents.

## Specialist Agents

The deterministic controller delegates bounded tasks through `AgentTask` and
`AgentTaskResult` contracts:

- Search Agent: search, ranking, metadata retrieval, and targeted retrieval.
- Analysis Agent: evidence cards, route validation, synthesis, and claim planning.
- Writing Agent: deliverable generation and authorized section rewriting.

Each task declares its objective, constraints, input artifacts, expected outputs,
budget, and source-state fingerprint. Specialist results contain findings, evidence
references, unresolved questions, changed fields, and input/output fingerprints.
`action_contracts.py` defines explicit input and output allowlists for every executable
operation. Specialists receive a deep copy of the input projection; newly added state
fields are not automatically visible. Patch keys, removals and types are checked by the
specialist and again by the merger. The controller merges only after identity and
version checks. Specialists cannot change the user query, explicit time/reference constraints,
selected scope, or deliverables. The Writing Agent also cannot mutate the evidence
pool or claim authorization.

The controller updates task-graph status and rebuilds the main context after every
specialist result. Existing retrieval loops, evidence recovery, writing renderers,
claim verification, and final quality gates remain authoritative. A successful
specialist call does not imply that the overall research result passed its gates.

During quality recovery, the main loop deterministically advances newly admitted
hybrid candidates through detail, card, route, and claim actions before requesting
another model decision. This uses the existing action contracts and keeps the
model-facing context at the same five fields. Evidence changes invalidate dependent
plans and gates after the specialist result commits; removed artifacts are also
removed from the manifest before the next checkpoint can hydrate them.

## Configuration

The compact view is controlled by:

```text
MAIN_CONTEXT_MAX_CHARS
MAIN_CONTEXT_MAX_EVIDENCE
MAIN_CONTEXT_MAX_DECISIONS
MAIN_CONTEXT_MAX_OPEN_QUESTIONS
MAIN_CONTEXT_SYSTEM_RESERVE_CHARS
MAIN_CONTEXT_OUTPUT_RESERVE_CHARS
AGENT_EXECUTION_ACTION_BUDGET
AGENT_MAIN_MAX_ROUNDS
AGENT_MAIN_NO_PROGRESS_LIMIT
AGENT_MAIN_TOKEN_BUDGET
AGENT_RETRIEVAL_BUDGET
AGENT_EXECUTION_DEADLINE_SECONDS
LLM_NATIVE_TOOLS_ENABLED
```

Context limits control the model-facing view; execution budgets can stop work.
Neither lowers reference targets, deletes evidence, or changes validation counts.
Database-backed conversation execution uses a session lease and database CAS. Each
successful action commits its task result, artifact references, checkpoint, budget and
event in one transaction. Attempts reserve and settle independently so failed, cancelled
or stale research results cannot erase incurred usage. Public session saves also check
the execution lease/version. The graph-only Python entry remains an in-memory mode.
See [runtime, recovery and cache boundaries](agent-runtime-and-cache.md) for migration,
explicit checkpoint resume, action dependencies and validation evidence.

恢复细节继续外置在现有历史与候选拒绝记录中，不增加 Main Agent 的五字段。
Controller 在创建重写任务和预留预算前执行与动作注册表、外层恢复一致的资格检查；
task 幂等键包含 objective，不能替代这一限制。自主重写候选被回滚时保存操作和
恢复输入指纹，随现有状态持久化。恢复历史新增可选的验证前指纹，允许同输入的
新语义判定结算真实进展；旧历史默认空值，保持兼容和保守判断。
