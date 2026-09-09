# Standard Examples

These examples are canonical inputs for documentation, manual inspection, and
mock-based tests. They do not contain API keys and do not require live data sources.

- narrative_review_request.json: supported review request with an explicit reference target.
- related_work_request.json: related-work request with required our_work context.
- unsupported_request.json: request that must be stopped by the capability guard.
- generation_recovery_scenarios.json: expected recovery decisions for sufficient,
  insufficient, existing-evidence-only, and cross-domain reference targets.

Expected behavior is part of each file. Actual paper results are deliberately absent:
tests must inject deterministic mock clients and evidence.
