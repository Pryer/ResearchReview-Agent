# Utility scripts

Only reusable operations and manual validation entry points belong in this
directory. Runtime output is written under `data/` or `logs/` and is not tracked.

## Service operations

- `check_llm_api.py`: validate the primary or backup OpenAI-compatible LLM endpoint.
- `submit_research.py`: submit a research job from the command line.
- `monitor_job.py`: monitor a submitted background job.
- `show_metrics.py`: inspect in-process metrics during local debugging.
- `migrate_db_v1.3.0.py`: migrate databases created before the v1.3 schema change.

## Manual validation

- `run_agent_tests.py`: run the real-provider Agent scenario set.
- `run_classroom_behavior_e2e.py`: run the 40-reference classroom-behavior scenario.
- `cnki_selenium_smoke.py` and `test_cnki_headless.py`: validate CNKI browser access.
- `inspect_eval_bundle.py`: inspect an exported evaluation bundle.

## Dataset utilities

- `export_claim_verification_data.py`: export claim-verification candidates.
- `build_claim_verifier_dataset.py`: build a paper-level split training dataset.

The real-provider scripts may consume API quota and can require interactive CNKI
access. They do not bypass login, CAPTCHA, paywalls, or rate limits.
