# Agent Context Architecture

## Stable and dynamic context

TaskMainLoop is the production entry. Model requests combine a stable role prefix,
semantic trajectory and current Status Bar. Its five fields are goal, state,
key_evidence, decisions and open_questions. context_builder.py derives a bounded
view from authoritative state.

## Main-Agent decision loop

TaskMainPolicy exposes delegate_agent and registered controls through native tools
or strict JSON. Main requests a role/Profile, goal and scope; AuthorizationPolicy
issues the Grant. Mandatory recovery uses exact server-issued single-action grants.
Business progress and explored queries bound repetition.

## Specialist planning and action context

| Consumer | Input | Responsibility |
| --- | --- | --- |
| Main | Five-field Status Bar and bounded memory | Tasks and controls |
| Local strategy | PlanningContext, constraints, summaries and recent results | Actions within a live Grant |
| Action handler | Contract inputs, authorized artifacts and task brief | Research work and typed patches |

Search and Analysis support deterministic/ReAct; evidence_build also supports
Plan-and-Act. Plans are validated and persisted. Each step refreshes admission,
resource manifest and business version. Writing/Check use direct projection.

Task briefs distinguish delegation, invocation, attempt and input version.
Workers inherit the bounded identity. Trusted results extend resources within an
attempt; committed lineage is saved with the action. A summary does not grant
access to raw documents.

## ReAct epochs and context capacity

ReActContext pairs actual tool calls and observations. Role model/context settings
remain frozen for a task and active delegation. Each actual RPC checks its input
and output reservation.

The Artifact Store archives observations. Working context is bounded by item,
character and window limits. Compaction retains goals, unresolved work and failures.
ContextBudgetExceeded stops execution without weakening evidence rules.
Terminal observations use archive=False after cancellation or loss of ownership.

## Version and invalidation boundary

Snapshot revision, business version, source fingerprint and execution profile have
different lifetimes. Billing and heartbeats do not advance business versions.
Every action checks current inputs and live ownership at commit. Evidence changes
invalidate derived plans and reports.

Check reads candidate/evidence and returns a bound report. Controller settles
rollback or recovery. Published body, references and quality metadata retain their
own version independently of the latest candidate and job execution status.

## Research memory and historical state

Memory search and recall preserve provenance and bounded identifiers. Historical
recommendations do not grant actions or establish facts. Known v1 activity migrates
through submission proofs; unknown records stop for diagnosis. Legacy fields
remain readable for migration and reports, with no old scheduler selectable.

The retired context-selection and one-action delegation protocol is a historical
test reference. Current decisions use Profiles.
See [runtime and cache](agent-runtime-and-cache.md) and
[retirement verification](validation/2026-11-18-v2-runtime-retirement.md).
