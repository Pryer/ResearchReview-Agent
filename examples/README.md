# Standard Examples

These examples are canonical inputs for documentation, manual inspection, and
mock-based tests. They do not contain API keys and do not require live data sources.

- narrative_review_request.json: supported review request with an explicit reference target.
- related_work_request.json: related-work request with required our_work context.
- unsupported_request.json: request that must be stopped by the capability guard.
- generation_recovery_scenarios.json: expected recovery decisions for sufficient,
  insufficient, existing-evidence-only, and cross-domain reference targets.
- checkpoint_resume_request.json: explicit recovery of an interrupted session,
  including committed-checkpoint and lease preconditions and retained budgets.

Submit only the `request` object to `POST /api/reviews/jobs`; `expected` describes
validation expectations, not API input. A resume request needs the real prior
session ID and cannot create its prerequisite checkpoint. Clarification uses
`clarification_answer`; completed-result revision uses `/api/reviews/jobs/revise`.
See the [session API guide](../docs/multi_turn_research.md).

Expected behavior is part of each file. Actual paper results are deliberately absent:
tests must inject deterministic mock clients and evidence.
