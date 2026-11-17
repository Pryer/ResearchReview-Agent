# Architecture

Production uses one architecture: task-level Main, bounded specialist delegations,
and a deterministic commit kernel. Search and Analysis default to deterministic
local flows. Both can use ReAct, and Analysis evidence building can also use Plan-and-Act.

## Layers

- app/api and app/services: requests, conversations, jobs, durable execution and resources.
- app/agent/task_main.py: task-level Main policy and controls.
- app/agent/runtime: DelegationRunner, strategies, ActionExecutor and shared progress.
- app/agent/policy and registry: capabilities, scope, admission and acceptance.
- app/agent/controller.py: task identity, typed input/output, evidence invalidation and atomic commit.
- app/clients and tools: source adapters and research operations.
- app/schemas and database: grants, invocations, attempts, leases, budgets and checkpoints.
- app/deliverables and prompt: writing specifications, renderers and role prompts.

## Ownership Rules

Main requests a capability, goal and scope. The server derives the role from the
profile registry and issues the Grant.
Effective actions intersect registration, role capabilities, Grant, latest admission
and remaining budget. Every action independently checks its inputs and version.
ActionExecutor invokes Controller; ToolExecutor checks external capabilities and
persists receipts. Billing uses the existing ledger.

Search and Analysis can run multiple authorized actions. Writing and Check use
direct parameter projection. Writing commits a candidate; independent Check binds
its report to that candidate and evidence. Main can request targeted section
revisions, check the new candidate, or delegate claim-plan rebuilding to Analysis
when the revision limit is reached. Draft and formal publication apply their
respective source, citation and quality gates.

## Specialist context and communication

Main sees the five-field Status Bar. Local planning receives bounded summaries in
PlanningContext. Each action receives only contract-projected business fields and
authorized artifact references. Planning summaries never grant raw artifacts.
Delegations return structured results to Main; suggestions do not authorize work.
Plans, action results and observation cursors are durable.
See [context architecture](agent-context-architecture.md).

## Durable execution and recovery

Jobs and direct Python graph entry points use the same durable kernel. Services
own an outer scope; standalone calls create one and bind an Artifact Store.
Nested graph calls reuse their outer scope. Snapshot revisions are independent
from committed business versions.

Known historical delegations migrate after goal/source checks and durable
completion-proof validation. Unknown formats stop explicitly. Old frozen
architecture settings upgrade to v2 after full settings validation; constraints,
delivery mode, models and cumulative spend remain authoritative.

Calls with durable starts but missing receipts become uncertain on takeover.
Cancellation, ownership loss and CAS conflicts prevent late business commits.
Binary rollback requires stopping writes and restoring a compatible backup.
Changing local strategy is a separate operation.

## Runtime Components and retired code

The former Main loop, policy, subagent loop and delegation-context builder have
been removed from app/. Historical protocol references live only under
tests/reference_v1 with an explicit manifest. Production and scripts cannot import
them; a static regression checks that boundary.

API and Streamlit use run_api.py and run_chat_frontend.py. Configuration lives in
app/core/config.py and .env.example. Restart services after upgrading code.
The latest `rragent` regression passed 2,769 tests, including native pgvector
integration. Live runs verified exact-title retrieval through the HTTP Job API
and produced a partial draft with five references. Formal review quality,
independent human review and strategy comparisons remain open.
See the [live validation record](validation/2026-11-18-v2-live-closeout.md) and
[runtime retirement record](validation/2026-11-18-v2-runtime-retirement.md).
