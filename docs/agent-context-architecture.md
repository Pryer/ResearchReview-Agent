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
stable prefix while retaining the existing provider fallback, timeout, and metrics
behavior. Raw document text is treated as research material and cannot redefine the
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

`MainAgentPolicy` receives one freshly built `MainAgentContext` per round and returns
exactly one `AgentDecision`. The stable action registry is the source for tool
descriptions, parameter schemas, roles, and controller authorization. Providers with
native tool calling use a required single tool call; deployments that disable native
tools, or explicitly reject the native protocol, use strict JSON validated against
the same contract. Tool schemas stay stable across rounds; the current allowed
actions and correction reasons live inside `state`.

`MainAgentLoop` executes `build -> decide -> validate -> execute -> commit -> rebuild`.
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
user constraints. Cancellation, failure and waiting states do not publish a draft.
Generation and citation-repair candidates run the normal verification chain; a
worse candidate restores the previous generation products and is verified again.

The budget ledger is shared across conversation parsing, graph entry points,
specialist workers and automatic recovery. Each real LLM attempt reserves prompt
bytes plus its output token cap before execution. Provider usage settles that
reservation; missing usage or failed attempts retain the conservative reservation
and set `usage_estimated`. Retries and fallback requests are charged separately.
Retrieval is counted per search-node round, and recovery has its own sublimit.
Nested controller actions charge the parent action once. The deadline spans the
current user turn; cumulative usage survives pause/resume. In-flight external calls
remain subject to their client timeouts; late results cannot commit after expiry.

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
