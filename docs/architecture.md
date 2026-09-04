# Architecture

ResearchReview-Agent uses a layered Python application layout. The production
implementation is under app/, not src/.

## Layers

- app/api/: FastAPI HTTP routes. Routes validate requests and delegate to services.
- app/services/: application services for jobs, conversations, papers, reviews,
  citations, libraries, and LLM calls.
- app/agent/: stateful workflow orchestration, planning, routing, nodes, and bounded
  retrieval/evidence-recovery loops.
- app/clients/: thin adapters for arXiv, Semantic Scholar, OpenAlex, Crossref,
  and CNKI. They map remote responses to PaperMetadata.
- app/tools/: single-purpose search, ranking, deduplication, PDF, card,
  writing-dispatch, and citation operations.
- app/schemas/: Pydantic contracts shared across layers.
- app/core/: configuration, logging, security, rate limiting, circuit breakers,
  metrics, citation syntax, and text-quality infrastructure.
- app/database/: SQLAlchemy models, SQLite initialization, and repositories.
- app/deliverables/: deliverable specifications and renderers.
- app/prompt/: lazy-loaded prompts and writing prompts.

## Ownership Rules

graph.py owns node order and branch selection. Retrieval refinement belongs in
retrieval_loop.py; evidence recovery belongs in recovery_loop.py. External requests
do not belong in nodes, and writing uses the single
write_deliverable.py -> deliverables/renderers path.

Cross-layer data uses Pydantic models where a schema exists. Metadata is authoritative
from retrieval/verification layers; LLM output cannot overwrite provenance fields.

## Runtime Components

- API entry: run_api.py -> app.main:app
- Streamlit entry: run_chat_frontend.py -> app/frontend/chat_app.py
- Database default: data/research_review.db
- Configuration: app/core/config.py, .env, and .env.example

The root ARCHITECTURE.md remains a detailed operational reference. This document is
the stable architecture contract; update both when a structural change affects
developers or operators.

