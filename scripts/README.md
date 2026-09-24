# Utility scripts

Only reusable operations and manual validation entry points belong in this
directory. The E2E runner writes under `data/`; the older submit/monitor utilities
write `result_*` files to the current directory. These outputs are not tracked.

## Service operations

- `update_session_budget.py`: explicitly update an idle session's persistent token
  limit with `--session-id`, `--expected-limit` and `--token-limit`. Preserves usage,
  rejects active jobs/leases/unsettled requests, and records an audit event. It does
  not start or resume research; see the [budget contract](../docs/agent-runtime-and-cache.md).

- `check_llm_api.py`: validate the primary or backup OpenAI-compatible LLM endpoint.
- `submit_research.py`: submit a research job from the command line.
- `monitor_job.py`: monitor a submitted background job.
- `show_metrics.py`: inspect in-process metrics during local debugging.
- `migrate_db_v1.3.0.py`: migrate databases created before the v1.3 schema change.
- `migrate_agent_runtime.py`: idempotently add the six runtime, task, attempt,
  checkpoint, event and memory-cursor tables to the configured `DATABASE_URL`.
  Run `python scripts/migrate_agent_runtime.py` from the repository root.
  Normal API startup also creates missing tables; this does not replay jobs.

`submit_research.py` contains an editable example payload, not a general CLI for
checkpoint recovery. Both monitors stop on `partial`, `blocked` and
`needs_clarification` as well as completed/failed/cancelled jobs. Clarification and
checkpoint recovery require a new request; see the
[session API guide](../docs/multi_turn_research.md) and
[checkpoint example](../examples/checkpoint_resume_request.json).
`monitor_job.py` is a local utility without an API-key header; for authenticated
deployments use the authenticated API client or the main frontend.

`show_metrics.py` runs in a separate process when launched from the shell, so it
does not read a running API server's metrics or the persistent budget ledger.
Inspect `get_metrics_collector().get_report()` in the process that made the calls.
Missing provider cache telemetry produces a `None` hit rate, not zero.

## Manual validation

- `run_agent_tests.py`: run the real-provider Agent scenario set.
- `run_classroom_behavior_e2e.py`: run the 40-reference classroom-behavior scenario.
  It uses the persisted conversation service and automatic recovery controller by
  default, includes the education-technology focus in the request, and writes each
  run to a new `data/e2e_runs/<timestamp>/` directory. Use `--graph-only` to isolate
  the raw Agent graph or `--query` / `--output-dir` for another explicit scenario.
  Here automatic recovery means evidence/quality repair within a run. The script
  submits one turn; it does not answer clarification or resume after a crash.
  Graph-only runs do not validate database CAS, persistent budgets or restoration.
- `cnki_selenium_smoke.py` and `test_cnki_headless.py`: validate CNKI browser access.
- `inspect_eval_bundle.py`: inspect an exported evaluation bundle.

## Dataset utilities

- `export_claim_verification_data.py`: export claim-verification candidates.
- `build_claim_verifier_dataset.py`: build a paper-level split training dataset.

The real-provider scripts may consume API quota and can require interactive CNKI
access. They do not bypass login, CAPTCHA, paywalls, or rate limits.
The [2026-09-22 acceptance record](../docs/validation/2026-09-22-runtime-cache.md)
distinguishes the passing small commit/reopen smoke test from the still-unaccepted
40-reference full deliverable.
