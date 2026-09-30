# ResearchReview-Agent

> An evidence-grounded literature review agent for multidisciplinary research topics.

[中文 README](README.md) | **English README**

ResearchReview-Agent interprets research constraints and topic ambiguity from natural-language requests, retrieves openly available papers, builds structured paper cards, synthesizes research lines from the available evidence, and produces literature reviews with verifiable citations.

## Core capabilities

- **Research request understanding:** Extracts topics, years, reference counts, scope, and deliverable types with LLM-assisted semantic parsing and deterministic validation.
- **Multi-turn research sessions:** Persists clarifications, paper collections, generation versions, and revision history; papers can be excluded by stable ID, title, or natural-language instruction.
- **Five-field main agent:** Each round selects one registered action from the goal, state, key evidence, decisions, and open questions.
- **Controlled specialist tasks:** The controller projects action inputs and validates authorization, types, identities, versions, budgets, and evidence before committing results.
- **Background task control:** Long-running jobs expose progress and support cooperative cancellation at node boundaries.
- **Durable recovery:** Session leases and database CAS protect commits; checkpoints preserve evidence, explicit constraints, and cumulative budgets.
- **Caching and usage accounting:** Reuses eligible work while tracking cache hits, misses, retries, failover requests, and model usage.
- **Incremental regeneration:** Rebuilds clustering, writing, and citation verification after paper-set edits without repeating retrieval and metadata completion.
- **Multi-source retrieval:** arXiv, Semantic Scholar, OpenAlex, Crossref, and CNKI through Selenium.
- **Hybrid retrieval:** Optional BM25, dense embeddings, reciprocal-rank fusion, and a dedicated text reranker.
- **Evidence validation:** PaperCard extraction, evidence roles, claim-evidence alignment, citation validation, and quality gates.
- **Four deliverables:** Research background, state of the art, related work, and a narrative review draft.

## Architecture

The system has one execution scheduler. In each round, the main agent selects one registered action; deterministic code enforces action authorization, budgets, cancellation, evidence versions, and quality gates. Historical sessions are normalized at the recovery boundary without discarding their goals, constraints, papers, evidence cards, recovery history, or budget usage.

```text
User request -> scope understanding -> five-field context -> action selection
                                                        |
                                         controller validation and delegation
                                                        |
                                         search / analysis / writing
                                                        |
                                version checks -> transaction commit -> next round

Deliverable request -> deterministic quality gates -> complete / degraded / blocked
```

The model context is intentionally limited to `goal`, `state`, `key_evidence`, `decisions`, and `open_questions`. Full papers, evidence, and history remain in authoritative state and database records. See [ARCHITECTURE.md](ARCHITECTURE.md) and [runtime, recovery, and cache notes](docs/agent-runtime-and-cache.md).

## Quick start

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate rragent
```

For an existing environment:

```bash
conda activate rragent
python -m pip install -r requirements.txt
python -m pip install pytest==7.4.0
```

### 2. Configure environment variables

```bash
copy .env.example .env
# Edit .env and provide LLM_API_KEY and other local settings.
```

You can validate an OpenAI-compatible provider before starting the full service:

```bash
python scripts/check_llm_api.py --target primary
```

After upgrading an existing deployment, restart the service so it loads the new code. Startup creates the execution and history tables. You can also run the idempotent migration:

```bash
python scripts/migrate_agent_runtime.py
```

### 3. Start the API

```bash
conda activate rragent
python run_api.py
```

Open <http://127.0.0.1:8000/docs> for Swagger UI.

### 4. Start the frontend (optional)

```bash
conda activate rragent
python run_chat_frontend.py
```

Open <http://127.0.0.1:8501> for the Streamlit frontend.

### pip installation

```bash
pip install -r requirements.txt
```

## API examples

### Submit and monitor a background job

```bash
curl -X POST http://localhost:8000/api/reviews/jobs \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-001","user_query":"Review papers from the last three years and generate a literature review"}'
```

Poll `GET /api/reviews/jobs/{job_id}` for status or call `POST /api/reviews/jobs/{job_id}/cancel` to request cooperative cancellation. External API and LLM calls still obey their client timeouts, and already incurred model usage is settled.

### Revise a paper set

```bash
curl -X POST http://localhost:8000/api/reviews/jobs/revise \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-001","excluded_paper_ids":["doi:example"],"instruction":"This paper is not directly relevant"}'
```

The revision endpoint also accepts natural-language instructions such as “remove papers 2 and 5 and regenerate”.

### Resume from a checkpoint

```json
{
  "session_id": "demo-001",
  "user_query": "Continue the previous research",
  "resume_from_checkpoint": true
}
```

Resume preserves the original topic, year range, reference requirement, and cumulative budget. It does not automatically increase limits or replay unresolved requests. Active jobs must finish first, and an expired crash lease must be recoverable before takeover.

### Natural-language agent request

```bash
curl -X POST http://localhost:8000/api/reviews/jobs \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-001","user_query":"Review classroom behavior analysis papers from the last three years, cite at least 40 papers, and generate the research background and state of the art"}'
```

When the result is `status=needs_clarification`, submit a free-text answer using the same `session_id` and `clarification_answer`.

### Structured paper search

```bash
curl -X POST http://localhost:8000/api/papers/search \
  -H "Content-Type: application/json" \
  -d '{"query":"vision transformer","start_year":2021,"end_year":2025,"max_results":20}'
```

## Tests

```bash
conda activate rragent
pytest
```

The repository uses mocked external LLM and retrieval boundaries for deterministic tests. Real CNKI, external LLM, and live end-to-end runs require local credentials and are not part of the default test suite.

## Project structure

```text
ResearchReview-Agent/
├── app/
│   ├── agent/        # intent, planning, routing, nodes, graph, recovery, and state
│   ├── api/          # FastAPI routes
│   ├── clients/      # arXiv, Semantic Scholar, OpenAlex, Crossref, CNKI, and retrieval clients
│   ├── core/         # configuration, logging, exceptions, and metrics
│   ├── database/     # SQLAlchemy runtime repository, leases, CAS, jobs, and snapshots
│   ├── frontend/     # Streamlit frontend
│   ├── schemas/      # Pydantic models
│   ├── services/     # sessions, jobs, artifacts, retrieval ranking, and LLM services
│   ├── tools/        # retrieval, ranking, PDF, clustering, writing, and citation tools
│   ├── deliverables/ # deliverable specifications and renderers
│   ├── prompt/       # prompts and lazy-loaded prompt catalog
│   └── main.py       # FastAPI application and startup recovery
├── tests/            # pytest tests
├── data/             # runtime PDFs, parsed data, reviews, caches, and evaluation bundles
├── environment.yml   # conda environment
├── run_api.py        # API entry point
├── run_chat_frontend.py # Streamlit entry point
└── requirements.txt  # base dependencies
```

## Configuration

Edit `.env` using `.env.example` as the template. Important settings include:

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | Primary LLM provider | `deepseek` |
| `LLM_API_KEY` | Primary API key | required |
| `LLM_BASE_URL` | Provider base URL | provider-specific |
| `LLM_MODEL` | Model name | `deepseek-v4-flash` |
| `LLM_THINKING_ENABLED` | Enable provider reasoning mode | `false` |
| `LLM_REASONING_MODE` | `static` respects the global switch; `auto` selects per operation | `static` |
| `LLM_NATIVE_TOOLS_ENABLED` | Use native tool calls for the main agent | `true` |
| `AGENT_MAIN_MAX_ROUNDS` | Maximum main-agent decision rounds | `24` |
| `AGENT_EXECUTION_ACTION_BUDGET` | Cumulative specialist-action reservation | `64` |
| `AGENT_MAIN_TOKEN_BUDGET` | Shared cumulative token budget | `10000000` |
| `AGENT_RETRIEVAL_BUDGET` | Maximum retrieval rounds | `64` |
| `RETRIEVAL_RANKING_MODE` | `rules`, `hybrid_shadow`, or `hybrid` | `rules` |
| `DASHSCOPE_API_KEY` | Separate key for hybrid retrieval | empty |
| `RETRIEVAL_EMBEDDING_MODEL` / `RETRIEVAL_RERANK_MODEL` | Dense and reranking models | Qwen models |
| `AGENT_EXECUTION_DEADLINE_SECONDS` | Execution deadline and lease duration | `1800` |
| `DATABASE_URL` | Database URL | SQLite |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional Semantic Scholar key | empty |
| `APP_API_KEY` | Optional API protection key | empty |
| `CORS_ALLOWED_ORIGINS` | Allowed frontend origins | local Streamlit origin |
| `RECOVERY_TOTAL_ACTION_BUDGET` | Shared recovery-action budget | `6` |
| `REFERENCE_COVERAGE_BEST_EFFORT_RATIO` | Minimum coverage for an explicitly marked partial draft | `0.85` |

To use hybrid retrieval, install its optional dependencies and configure the embedding and reranker endpoints locally:

```bash
python -m pip install -r requirements-retrieval.txt
```

The real `DASHSCOPE_API_KEY` and endpoint URLs belong only in local `.env`. Environment variables override `.env`; clear stale shell values before starting a service if necessary. Hybrid retrieval builds BM25 and dense views from titles, abstracts, and keywords, fuses candidates with reciprocal rank fusion, and applies a dedicated reranker. Ranking does not replace semantic admission or evidence validation. Cached vectors and reranker scores are stored under `data/retrieval_cache/` and can be rebuilt. `hybrid_shadow` calls remote models for diagnostics while keeping the research output on the rules path.

Quality-gate failures trigger bounded recovery: the agent may recompute state, restore citation allocation, or rewrite failed sections within budget. It does not silently expand explicit topic, time-range, or reference requirements. When evidence is insufficient, the result is blocked or explicitly marked as partial/best-effort rather than presented as a complete review.

## Documentation and examples

- [AGENTS.md](AGENTS.md): coding-agent rules, validation requirements, and completion criteria.
- [ARCHITECTURE.md](ARCHITECTURE.md): control loop, modules, and runtime boundaries.
- [docs/architecture.md](docs/architecture.md): system layers and component boundaries.
- [docs/agent-context-architecture.md](docs/agent-context-architecture.md): five-field context and specialist-agent boundaries.
- [docs/agent-runtime-and-cache.md](docs/agent-runtime-and-cache.md): action contracts, commits, budgets, recovery, and caching.
- [docs/research-workflow.md](docs/research-workflow.md): workflow stages and branch guarantees.
- [docs/evidence-model.md](docs/evidence-model.md): relationships between metadata, evidence, claims, and citations.
- [docs/quality-gates.md](docs/quality-gates.md): capability, route, claim, deliverable, and generation gates.
- [docs/change-log.md](docs/change-log.md): append-only record of completed changes.
- [examples/](examples/): standard requests that do not require live external services.

## Compliance

ResearchReview-Agent follows these principles:

1. Retrieve public paper metadata only.
2. Download open-access PDFs only; do not bypass paywalls or access controls.
3. Never fabricate papers, authors, years, citation counts, results, datasets, or references.
4. Review citations must come from papers actually retrieved and verified by the system.
5. Evidence sources must be labeled, such as `abstract` or `full_text`.
6. When evidence is insufficient, retrieve more within configured bounds, downgrade explicitly, or report the unmet requirement.
