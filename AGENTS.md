# AGENTS.md

Talk to me ELI5 by default and go deeper if I ask.

## Project

ResearchReview-Agent is an evidence-grounded academic literature review agent.
The implementation lives under `app/`; tests live under `tests/`; design rationale
and operational knowledge live under `docs/`; standard scenarios live under
`examples/`.

Core workflow:

```text
User Request
→ Scope Understanding
→ Retrieval
→ Metadata Verification
→ Evidence Extraction
→ Claim-Evidence Alignment
→ Quality Gates
→ Synthesis
→ Candidate Writing
→ Independent Check (revise affected sections and recheck when needed)
→ Final Acceptance
```

## Where to start testing

Run the command for the behavior you changed from the repository root in the
`rragent` environment.

| Changed behavior | Focused regression |
| --- | --- |
| Mapping a PDF quote back to its original text and offsets | `python -m pytest -q tests/test_source_alignment.py` |
| Publishing a draft with a reference shortfall | `python -m pytest -q tests/test_draft_delivery_policy.py` |
| Passing revised sections back to Check before completion | `python -m pytest -q tests/test_v2_frozen_business.py::test_section_revision_hands_off_to_check_again_before_finish` |

## Non-Negotiable Rules

1. Never fabricate papers, metadata, DOI, results, datasets, or citations.
2. Important factual claims must be traceable to evidence.
3. Citations must actually support their claims.
4. Evidence insufficiency must never be silently converted into confident prose.
5. Quality gates must change system behavior, or explicitly mark the result as
   degraded/best-effort; a warning alone is not a successful validation.
6. Retrieval, evidence processing, synthesis, and writing must remain logically
   separable.
7. Do not weaken evidence validation for speed, simplicity, or token reduction.
8. Do not mix research methods, research objects, variables, and application goals
   within the same taxonomy level.
9. Unknown metadata remains `None` or the schema-defined empty value; never use
   zero or an LLM guess as a substitute.
10. A requested reference count means the number of unique references used by the
    final review (`required_reference_count`), not a minimum citation count for
    each paper.
11. Do not bypass login, CAPTCHA, institution access, paid access, or rate limits.

## Codex Change Rules

Before modifying code:

- Read the relevant implementation, callers, downstream consumers, schemas, and tests.
- Trace the component's role in the research workflow.
- Preserve existing behavior unless the requested change explicitly changes it.
- Prefer minimal targeted changes over large rewrites.
- Keep `clients`, `tools`, `services`, `agent`, and `schemas` logically separate.
- Keep prompts in `app/prompt/`; use the existing deliverable renderer pipeline for
  writing changes.
- Do not remove provenance, validation, quality gates, uncertainty handling, or
  cancellation boundaries.
- Do not silently swallow errors that affect research correctness.
- Do not revert or overwrite user changes outside the current task.
- Do not add AI co-author attribution to Git commits or pull requests unless the
  user explicitly requests it.

## Change Log (Required)

After every completed code change, append one entry to docs/change-log.md before
declaring the task complete. Use the Singapore timezone and an ISO-like timestamp
such as 2026-08-30 18:47:19 +08:00.

Each entry must include:

- timestamp;
- modified files;
- root cause or reason for the change;
- behavior change;
- tests and validation results;
- known limitations or unverified areas.

The detailed `docs/change-log.md` is local-only and ignored by Git because it may
contain development context and personal schedules. If absent in a fresh clone,
create it before appending. Keep public, reader-facing changes in `CHANGELOG.md`.
Do not force-add local-only documents.

Append to the existing file; do not rewrite or delete earlier entries. Do not record
secrets, API keys, private user data, full prompts, or large debug traces. This is an
Agent completion requirement; it is not a claim that Git automatically logs every
working-tree edit.

## Convergence Review (Required)

When the root cause is found only after multiple adjustments to the same problem,
the Agent must autonomously perform one final minimal-behavior review before
declaring completion.

The review must confirm:

- the root-cause fix covers the target behavior;
- the change matches the established pattern for similar functionality;
- temporary branches, fallbacks, duplicate logic, and unnecessary refresh/reload
  steps introduced during exploration have been removed;
- only the Agent's own temporary changes are cleaned up;
- non-obvious retained constraints have a short Chinese `WHY` comment explaining
  field semantics, state lifetime, or timing boundaries;
- regression tests cover the root-cause behavior, not only the visible symptom.

## Validation

After meaningful changes verify, as applicable:

- retrieval still works and source failures remain diagnosable;
- metadata and evidence provenance are preserved;
- claim-evidence alignment and citation validation still run;
- quality gates still execute and affect the result status, draft quarantine, or
  explicit best-effort downgrade;
- unsupported claims are removed, weakened, or clearly reported;
- final output contains no debug/UI/tool artifacts, internal IDs, hidden prompts,
  temporary paths, or local service URLs.

Start with the affected regression. For the full suite, run `python -m pytest -q`.
[Offline CI](.github/workflows/ci.yml) runs it on Windows and Ubuntu with
`RRA_OFFLINE_TESTS=1` and `CNKI_ENABLED=false`. Mock external APIs, LLM calls,
and Selenium in CI. Report offline test results, live provider runs, and
independent human review separately. Identify any relevant checks left unrun.

For a claimed improvement in quality, latency, or cost, measure the baseline
and the change on the same inputs with the same budget and metric. For PDF
layout changes, compare reading order, table values against their row and column
headers, and whether each excerpt can be located again in the source. Record
both gains and regressions. Where results are equivalent, prefer the simpler
implementation.

## Critical Failure Policy

For example, a user requests 10 unique references within a specific year range.
The review cites 7 eligible papers. Keep the years and the target of 10 throughout.

1. Continue recovery while an allowed action and sufficient budget remain.
2. If recovery is exhausted and no specific user input can resolve the gap,
   strict delivery ends as `blocked`. Report the shortfall of 3.
3. In a mode that permits drafts, publish only when the draft checks pass.
   Label the published version as a draft, show the shortfall, and keep
   `requirements_satisfied=false`. An unpublished candidate stays private.

## User-facing Reviews and Exports

Never expose:

- localhost or internal service URLs;
- SVG/UI placeholders;
- tool traces or debug logs;
- internal paper IDs or database identifiers;
- hidden prompts;
- temporary paths;
- unsupported factual claims presented as established findings.

Save requested execution traces and diagnostic artifacts in a separate local
report. Redact credentials and private user data before sharing diagnostics.

## Definition of Done

A change is complete only when:

- requested behavior works;
- the research workflow remains intact;
- evidence provenance is preserved;
- quality gates still execute and their status is honored;
- relevant tests pass;
- the convergence review was completed when triggered;
- no internal artifacts leak into output;
- documentation is updated when behavior, configuration, API, or workflow changes.

## Document Responsibilities

```text
AGENTS.md   → what an Agent must and must not do
README.md   → what the project is and how to install/use it
docs/       → why the system is designed this way and how mechanisms work
docs/change-log.md
            → append-only record of completed Agent changes
tests/      → executable behavioral constraints and regression proof
examples/   → canonical inputs, expected states, and standard scenarios
app/        → production implementation
```
