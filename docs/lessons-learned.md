# Lessons Learned

This document records recurring failure modes found during review and production
diagnosis. It is a rationale record, not a substitute for the rules in AGENTS.md or
for executable tests.

## Research Correctness

- A user reference target is not a per-paper citation threshold. Keep
  required_reference_count separate from citation_count.
- Metadata authority must remain in the retrieval layer; an LLM card extractor must
  not overwrite title, authors, year, DOI, or source.
- Repeated citations must be counted by unique paper, and reference numbering must
  follow the configured citation syntax rather than card order.
- A route that is structurally valid but evidence-poor is a recovery candidate, not
  automatically a dropped route.

## State and Workflow

- Every declared node output must be written on every applicable path; contract
  violations must be visible in state and diagnostics.
- Cancellation must pass through detail, download, parse, extraction, and generation
  loops; catching it as a generic exception makes cancellation ineffective.
- Repair flows need snapshots and rollback. A failed citation repair must not leave
  the body and reference map inconsistent.
- Progress counters and reset logic must be shared by run, continue, and regenerate
  so their lifecycles do not drift.

## External Sources

- A client returning [] can mean either a true empty result or a source failure.
  Preserve source diagnostics at the scheduling layer before refining keywords or
  expanding years.
- CNKI automation must stop for CAPTCHA, login, or institution walls. Ordinary text
  alerts may be dismissed; authentication barriers may not be bypassed.
- Rate limits, timeouts, and circuit breakers are part of correctness because a
  partial source outage must not be mistaken for evidence absence.

## Writing and Output

- Fallback writers must receive the same authorized paper/claim/theme subset as the
  LLM writer; otherwise unrelated themes or plan-external citations leak into text.
- Abstract-only evidence must not support precise experiment, ablation, or full
  method claims.
- Internal IDs, debug traces, local URLs, and quality-gate implementation language
  must stay out of the user-facing final answer.

Each new root cause should add a focused regression test and, when the fix required
multiple iterations, trigger the mandatory convergence review in AGENTS.md.

