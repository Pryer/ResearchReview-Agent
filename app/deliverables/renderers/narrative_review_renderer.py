"""叙述性综述初稿渲染器 (NARRATIVE_REVIEW)。"""

from __future__ import annotations

from typing import Any
from app.schemas.deliverable_schema import CoreDeliverableType, WritingPlan
from app.deliverables.renderers.base_renderer import BaseRenderer

class NarrativeReviewRenderer(BaseRenderer):
    def __init__(self):
        super().__init__(CoreDeliverableType.NARRATIVE_REVIEW)
