# Book-guided Agent Harness: small, reversible improvement

Basis: [深入理解 AI Agent](https://github.com/bojieli/ai-agent-book), chapters 1–10. This branch deliberately implements a **small subset** of book-inspired changes rather than replacing the existing autonomous loop.

## The architecture

1. **Context:** Keep the five-field working view, authoritative state, stable role prompts and scoped tool schemas. Add the last four completed specialist task observations (status/outcome, bounded findings, evidence references). This makes Tool Result -> Observe -> next decision explicit without exposing raw documents or increasing top-level context fields.
2. **ReAct:** Keep `MainAgentLoop` as the only decision loop; `MainAgentPolicy` selects one allowed action. Preserve deterministic recovery handoffs, bounded rounds and no-progress detectors.
3. **Tools / execution:** `AgentController` remains responsible for action allowlists, role-scoped inputs, state fingerprint/CAS, task budget, cancellation, idempotency and persistent commit.
4. **Optional Sidecar:** `ActionSidecar` performs **independent, read-only review of just the proposed action name and its structured arguments**. It never receives the main reasoning, papers, user transcript or research state. It is an additional check, **not** an authorization replacement. Enabled only by an explicit flag and reviewed-action list. Its `deny` response causes a rejection observed on the next round; `escalate` or reviewer errors return `waiting_user`. No automatic retry on a denied action.
5. **Verification:** The existing deterministic claim/evidence and quality gates remain the final authority. The writing model cannot approve its own result.

## Configuration

```dotenv
# Defaults: disabled. Start in a disposable evaluation environment.
AGENT_SIDECAR_ENABLED=false
AGENT_SIDECAR_REVIEW_ACTIONS=generate_deliverables,rewrite_sections
```

Setting the flag true requires a functional LLM service because the Sidecar performs a separate model request. Run-time prompt/inputs are bounded and independent of main-agent reasoning. By design, enabling it for a writing action reviews the *proposed action*, **not** its scientific accuracy. It is optional: the existing claim/evidence verification handles output quality.

**Important:** The current code calls Sidecar after the main model returns an action. It is synchronous for that action and does not yet overlap with streaming model generation. Do not describe it as fully parallel Sidecar execution.

## Book-to-implementation map

| Book topic | What this branch changes | What stays unchanged |
| --- | --- | --- |
| Ch1–2 ReAct / context | Preserve recent typed tool-result observations in existing status bar | Five fields, compression, role prompts |
| Ch3 Agentic RAG | No retrieval re-architecture | Existing evidence retrieval and source attribution |
| Ch4 Sidecar | Optional classifier + fail-closed escalation | Controller remains final authority |
| Ch5–6 reliable runtime | No rewrite | Persistent checkpoints, cancellation and consistency |
| Ch7 evaluation | Add deterministic unit regressions | Existing full suite; live E2E eval still required |
| Ch8–9 training / self-evolution | Intentionally out of scope | No unsafe runtime prompt/skill auto-updates |
| Ch10 collaboration | No extra agents | Existing specialist role separation |

## Acceptance plan

Run locally on this branch:
```bash
pytest -q tests/test_action_sidecar.py tests/test_agent_context.py tests/test_main_agent_loop.py tests/test_main_agent_policy.py
pytest -q
```

Then conduct a matched **baseline vs opt-in Sidecar** comparison on representative research tasks. Measure: completion quality, repeated-action rate, false denials, escalation count, token usage, and latency. Include benign research and intentionally malicious tool-argument cases (boundary set), plus normal tasks unchanged (retention set). Do not claim a performance improvement until measured.

## Follow-on only after measurement

- Selectively surface evidence details via authorized artifact references if a real task requires them (progressive disclosure).
- Add an actual streaming-overlapped Sidecar if latency warrants it.
- Consider prompt-cache layout changes only after measuring provider cache hits.

This preserves the core principle: **Model proposes, Harness authorizes, Environment executes, verification decides whether to finish.**


## Proposed next migration: adaptive long-context compaction (not implemented yet)

This section is the **target architecture**, not a statement that the running agent already has an append-only API trajectory. Currently `main_policy.py` sends a freshly serialized five-field snapshot per decision and `context_compaction.py` enforces a deterministic **character** cap. The new design must first implement protocol-correct multi-turn messages and token accounting. Do not present these as completed features until the migration and tests pass.

### Context layout

- Stable system prompt and stable-order tool definitions, with Controller still authoritative over per-turn allowed actions.
- Original user objective and constraints (immutable except explicit user amendments).
- Append-only, protocol-correct assistant tool calls and matching tool results. Store long tool results in artifacts; expose bounded, source-linked observations.
- A framework-generated dynamic status bar at the end with stage, remaining budget, incomplete TODOs, blocking issues and most recent failure.
- Long-lived research state, full event log, source materials and authoritative evidence remain outside the model context.

### Three compaction tiers

1. **L1 / per-tool output bound:** on oversized observations, preserve the full payload in an authorized artifact and expose a bounded, immutable summary plus source/fragment references. Do not mutate earlier visible tool messages in place.
2. **L2 / batch trajectory compaction:** when (estimated input tokens + reserved output tokens) approaches a *configurable* threshold (start experiments at 80% of the model's actual usable context window), archive a complete original trajectory, then create a new context epoch: stable prefix + original task constraints + verified structured summary + source refs + recent complete tool-call/result pairs + latest status bar. This checkpoint is deliberately a cache break; do it infrequently, not every round.
3. **L3 / fail-safe:** if post-compaction still exceeds the reserved token budget, do not drop immutable constraints or evidence identities. Report an explicit blocked state or request narrower scope / larger window; guard against repeated unsuccessful compaction attempts.

### Protected information

- Exact user requirements, priorities and any amendments, with provenance.
- Most recent verified progress and outstanding tasks; hard quality-gate failures.
- Failed or rejected operations and the reasons, especially repeated non-progress patterns.
- Evidence and claim IDs, authorization boundaries, artifact refs and uncertainty.
- Last few complete tool interactions where the provider API protocol requires coherent assistant/tool pairs.

### Cache and correctness constraints

- Compaction of old messages intentionally invalidates the cache from the first changed token. It trades some cache reuse for bounded context and reliable completion.
- Prompt Cache matches **token prefixes**, not JSON keys, so field count is not the issue; frequently rewriting fields near the front is.
- Track real provider token usage (prompt, cached tokens, completion), not just existing `main_context_max_chars`. Keep a conservative estimator with a safety margin for unknown providers.
- Do not place a changing timestamp, counters or status in the system prompt or sort tool definitions differently per turn. Evaluate whether dynamically scoped tool schemas are worth their cost; never weaken Controller authorization to win a cache hit.
- Keep history and authoritative state immutable/auditable in durable storage; compaction changes only the model-visible working view and explicit checkpoint pointer, never past facts or versions.
- Do not blindly repass provider-specific hidden reasoning after compaction; respect the model/provider's documented thinking and tool-message protocol.

### Acceptance tests

- Below threshold -> no compaction; append-only prefix equality across ordinary turns.
- Oversized tool result -> saved artifact and bounded, source-linked observation; no content loss in durable storage.
- Above threshold -> one checkpoint and a new epoch; no repeated compaction loop.
- User constraints, unresolved blockers, citations and latest successful/failed observations remain available.
- All tool-call IDs and result messages remain correctly paired; stale or unauthorized tool calls are still rejected.
- Compare completion rate, citation precision, number of redundant actions, input tokens, cached tokens, provider cost, latency and compaction failure cases against baseline.

**Rollout sequence:** (1) stable message protocol and trajectory persistence, (2) token-budget accounting and status bar, (3) L1 bounding, (4) L2/L3 batch compaction with a feature flag, (5) provider-level caching and end-to-end evaluation. Preserve the existing five-field fallback until the new path is proven safe.
