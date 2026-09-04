# Research Workflow

The workflow is implemented by app/agent/graph.py::run_research_agent and the
stage nodes under app/agent/nodes/.

## Stages

1. **Capability guard**: unsupported_task_guard rejects tasks outside the four
   supported deliverables before retrieval.
2. **Scope understanding**: intent, slots, topic disambiguation, time range, source
   language, reference target, and deliverable requirements are normalized.
3. **Planning**: plan_node creates concepts, exclusions, and search plans.
   Research-status tasks may receive provisional routes before search.
4. **Retrieval**: language-compatible sources are queried through bounded dual-channel
   search. Results are deduplicated, filtered, ranked, and optionally refined. A
   bounded rule-ranked tail may be screened when exclusions shrink the main window;
   tail papers that were not semantically confirmed remain diagnostic-only.
5. **Metadata verification**: details are enriched by the matching source; missing
   values stay missing. Search-only requests can return at this point.
6. **Evidence extraction**: optional PDF download/parse produces page-aware text;
   extract_card_node creates PaperCards and Evidence Cards.
7. **Route and evidence checks**: routes are validated, bounded evidence recovery may
   run, and evidence-backed clusters are produced as a fallback.
8. **Claim planning and gates**: claims are bound to evidence before writing;
   claim-evidence and global evidence gates record blocking or degradable deficits.
9. **Synthesis and writing**: generate_deliverables_node uses a WritingPlan and the
   single deliverable renderer pipeline. WritingPlan construction re-applies the
   semantic-confirmation boundary so an explicit best-effort override cannot
   reintroduce diagnostic-only reserve papers.
10. **Output validation**: claim alignment, sentence-level claim verification,
    citation validation, citation-gap repair, and the final quality gate run before
    final_answer_node. Final citation sources and the unique-reference count exclude
    unconfirmed reserve papers; any occurrence in draft text is reported as a missing
    citation.

## Branch Guarantees

- Cancellation is checked at node boundaries and must not be swallowed.
- PDF disabled is a valid path; metadata/abstract evidence remains usable.
- External source failure should allow other sources to continue and must not be
  misreported as a verified empty result.
- Explicit user constraints are never silently expanded or weakened.
- Incremental revision reuses the session and paper set, then reruns affected
  validation, writing, and citation stages.

For a worked, mock-friendly narrative see docs/react_workflow_example.md. Canonical
machine-readable inputs are under examples/.
