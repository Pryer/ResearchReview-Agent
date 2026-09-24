# Architecture

ResearchReview-Agent uses a layered Python application layout. The production
implementation is under app/, not src/.

## Layers

- app/api/: FastAPI HTTP routes. Routes validate requests and delegate to services.
- app/services/: application services for jobs, conversations, papers, reviews,
  citations, libraries, and LLM calls.
- app/agent/: stateful workflow orchestration, planning, routing, nodes, and bounded
  retrieval/evidence-recovery loops. `main_loop.py` / `main_policy.py` select actions;
  `controller.py`, `action_contracts.py`, `task_context.py` and `result_merger.py`
  enforce specialist input/output and commit boundaries. `context_builder.py` and
  `subagents/` supply the five-field view and specialist adapters.
- app/clients/: thin adapters for arXiv, Semantic Scholar, OpenAlex, Crossref,
  and CNKI. They map remote responses to PaperMetadata.
- app/tools/: single-purpose search, ranking, deduplication, PDF, card,
  writing-dispatch, and citation operations.
- app/schemas/: Pydantic contracts shared across layers.
- app/core/: configuration, logging, security, rate limiting, circuit breakers,
  metrics, citation syntax, and text-quality infrastructure.
- app/database/: SQLAlchemy models, SQLite initialization, and repositories.
  `runtime_repository.py` owns execution leases, task/attempt records, budget
  persistence and atomic CAS checkpoints.
- app/deliverables/: deliverable specifications and renderers.
- app/prompt/: lazy-loaded prompts and writing prompts.

## Ownership Rules

`main_loop.py` selects one registered action from the five-field context while
graph.py supplies the node adapters and final quality boundary. New and resumed
sessions use this scheduler. Historical sessions whose mode is absent or `legacy`
are normalized at the execution boundary; their evidence, explicit constraints,
recovery history and consumed budget remain authoritative.
The artifact repository externalizes heavy editable-state fields and validates
session, role, type and version on reads. A shared execution ledger covers token
reservations, search rounds, recovery actions and cancellation/deadline boundaries.
Retrieval refinement belongs in
retrieval_loop.py; evidence recovery belongs in recovery_loop.py. External requests
do not belong in nodes, and writing uses the single
write_deliverable.py -> deliverables/renderers path.

Cross-layer data uses Pydantic models where a schema exists. Metadata is authoritative
from retrieval/verification layers; LLM output cannot overwrite provenance fields.

The full `ResearchAgentState` is the authoritative workflow record. The main Agent
receives a rebuildable `MainAgentContext` containing only `goal`, `state`,
`key_evidence`, `decisions`, and `open_questions`. Specialist tasks use versioned
operation-level allowlists and typed patches; they cannot change user goals, hard
constraints or the shared budget. See [context architecture](agent-context-architecture.md).

## Durable execution and recovery

Conversation service entry points bind the artifact store, shared budget and durable
execution lease. Action success commits state, artifacts, task result, checkpoint,
budget and event atomically. Attempts reserve and settle separately, preserving usage
when a task fails, is cancelled or becomes stale. Public session saves check the
lease/version too. Direct graph calls remain an in-memory execution mode.

History entries are archived before display truncation; summaries consume only new
events after a durable cursor. Session reads restore newer committed research
checkpoints. `AgentRequest.resume_from_checkpoint=true` explicitly resumes interrupted
sessions with the original constraints and accumulated budget; it does not replay an
uncertain external request or take over a live lease. Pending clarification and
completed-session revision retain their separate entry points.

Fixed prompt rules precede dynamic material. Native requests send tool definitions
only through `tools`; JSON fallback retains the textual catalog. Section reuse hashes
the actual authorized writer inputs and deterministic draft, then reruns final
validation. Observed cache usage is distinct from conservative budget estimates.
See [runtime and cache boundaries](agent-runtime-and-cache.md).

## Runtime Components

- API entry: run_api.py -> app.main:app
- Streamlit entry: run_chat_frontend.py -> app/frontend/chat_app.py
- Database default: data/research_review.db
- Configuration: app/core/config.py, .env, and .env.example
- Additive execution-table migration: scripts/migrate_agent_runtime.py; also run by
  normal startup table creation. Restart services after deployment.
- Validation evidence: [2026-09-22 report](validation/2026-09-22-runtime-cache.md).
  The real 3-paper commit/recovery smoke passed; the 40-reference deliverable remained
  budget-blocked, so it is not recorded as a completed end-to-end review.

The root ARCHITECTURE.md remains a detailed operational reference. This document is
the stable architecture contract; update both when a structural change affects
developers or operators.
