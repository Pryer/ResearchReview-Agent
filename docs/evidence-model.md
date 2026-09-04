# Evidence Model

The evidence model separates what a paper is, what the review claims, and how a
claim is cited.

## Objects

- **PaperMetadata**: retrieved bibliographic identity and source-specific metadata.
  Unknown year, DOI, publication status, keywords, or citation count remain unknown.
  Bibliographic fields are authoritative from retrieval; an extraction model must
  never overwrite title, authors, year, venue, DOI, URL, or publication status.
- **PaperCard**: structured fields such as problem, method, data, metrics, results,
  and limitations. Each field must retain its evidence origin.
- **EvidenceSpan**: source text plus source type, section/page, provider and position
  where available. Abstract and metadata evidence are weaker than verified full text.
  Bibliographic facts (authors, year, venue, DOI, publication status) carry their own
  metadata spans so they stay auditable, and they never become content claims.
- **Claim**: a sentence-level container for one or more atomic factual propositions.
- **AtomicClaim**: the smallest verifiable proposition inside a sentence. It is bound
  to a single cited paper's evidence. Numbers, strong language, access level and
  entailment are evaluated per atomic claim; evidence from different papers is never
  merged into one pool. When a clause carries several citations with no reliable way
  to attribute facts, every cited paper must independently support the whole clause.
- **Citation**: a reference to a paper in the final selected set; it is not evidence
  by itself until the cited paper supports the claim. A rule-ranked reserve paper that
  has not received semantic confirmation remains retrieval diagnostics, not citable
  evidence, and cannot satisfy an explicit minimum-reference requirement.

The implementation contracts are in app/schemas/paper_schema.py,
app/schemas/verification_schema.py, and app/schemas/global_evidence_schema.py.

## Evidence Flow

retrieved metadata/PDF
-> PaperCard + EvidenceSpan
-> Claim-Evidence Plan
-> generated sentence with citation
-> verify_claims / citation validation

Claim verification returns supported, partially_supported, unsupported, or
not_applicable, together with evidence IDs, snippets, access level, issues, and a
suggested revision. Each sentence result also carries its atomic claims; a sentence
is supported only when every atomic claim is supported by its own cited paper.

## Draft Release Boundary

A generated draft is a candidate, not a deliverable. When the post-generation quality
gate fails, the draft is quarantined: it stays available for internal repair and
diagnostics but must not appear in the public answer, body, related_work, or
introduction. Only an explicit user decision to accept a best-effort result releases
the draft, and it is then labelled as a degraded, non-submittable version. A warning
banner alone is not a release.

## Strength Rules

- Metadata can establish identity and bibliographic facts, not detailed methods,
  quantitative results, or author-intended limitations.
- Abstract evidence can support claims explicitly stated in the abstract; it cannot
  be silently promoted to full-text evidence.
- Numeric values and strong result language require matching evidence.
- Cross-language claims use configured concept aliases and still require semantic
  support.
- A citation count is a property of a paper/source, not the user's requested number
  of references. Source counts are stored separately in citation_count_by_source
  and must not be added together.

Evidence provenance must survive extraction, clustering, writing, repair, and
serialization. When evidence is insufficient, weaken/remove the claim, retrieve
more evidence within bounds, or report the limitation.
