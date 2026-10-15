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
