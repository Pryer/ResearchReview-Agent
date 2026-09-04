"""交付物渲染器包。"""

from __future__ import annotations

from app.deliverables.renderers.base_renderer import (
    BaseRenderer,
    _allocated_paper_ids,
    _allocation_by_section,
    _citation_ids,
    _citation_recovery_paragraph,
    _citation_target_met,
    _clean_section_response,
    _english_residue_repair_prompt,
    _ensure_citation_coverage,
    _evidence_limited_section,
    _heading,
    _is_safe_partial_section,
    _merge_failed_or_missing_sections,
    _normalize_evidence_citations,
    _only_english_residue,
    _record_writer_diagnostic,
    _render_claim_constraints,
    _section_candidate_score,
    _section_heading,
    _section_merge_score,
    _section_rewrite_prompt,
    _split_planned_sections,
    _strip_agent_process_clauses,
    _strip_stray_numeric_citations,
    _style_guidance,
    _survey_papers,
    _validate_rewritten_section,
    _write_sections_in_chinese,
)
from app.deliverables.renderers.background_renderer import BackgroundRenderer
from app.deliverables.renderers.narrative_review_renderer import NarrativeReviewRenderer
from app.deliverables.renderers.related_work_renderer import RelatedWorkRenderer
from app.deliverables.renderers.status_renderer import StatusRenderer
from app.schemas.deliverable_schema import CoreDeliverableType

_RENDERERS = {
    CoreDeliverableType.RESEARCH_BACKGROUND: BackgroundRenderer(),
    CoreDeliverableType.RESEARCH_STATUS: StatusRenderer(),
    CoreDeliverableType.RELATED_WORK: RelatedWorkRenderer(),
    CoreDeliverableType.NARRATIVE_REVIEW: NarrativeReviewRenderer(),
}

def get_renderer(deliverable_type: CoreDeliverableType | str) -> BaseRenderer:
    dtype = CoreDeliverableType(deliverable_type)
    renderer = _RENDERERS.get(dtype)
    if not renderer:
        raise ValueError(f"No renderer registered for deliverable type: {deliverable_type}")
    return renderer

__all__ = [
    "get_renderer",
    "BaseRenderer",
    "BackgroundRenderer",
    "StatusRenderer",
    "RelatedWorkRenderer",
    "NarrativeReviewRenderer",
    "_allocated_paper_ids",
    "_allocation_by_section",
    "_citation_ids",
    "_citation_recovery_paragraph",
    "_citation_target_met",
    "_clean_section_response",
    "_english_residue_repair_prompt",
    "_ensure_citation_coverage",
    "_evidence_limited_section",
    "_heading",
    "_is_safe_partial_section",
    "_merge_failed_or_missing_sections",
    "_normalize_evidence_citations",
    "_only_english_residue",
    "_record_writer_diagnostic",
    "_render_claim_constraints",
    "_section_candidate_score",
    "_section_heading",
    "_section_merge_score",
    "_section_rewrite_prompt",
    "_split_planned_sections",
    "_strip_agent_process_clauses",
    "_strip_stray_numeric_citations",
    "_style_guidance",
    "_survey_papers",
    "_validate_rewritten_section",
    "_write_sections_in_chinese",
]
