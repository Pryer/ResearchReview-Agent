"""论文相关工作渲染器 (RELATED_WORK)。"""

from __future__ import annotations

from typing import Any
from app.schemas.deliverable_schema import CoreDeliverableType, WritingPlan
from app.deliverables.renderers.base_renderer import BaseRenderer

class RelatedWorkRenderer(BaseRenderer):
    def __init__(self):
        super().__init__(CoreDeliverableType.RELATED_WORK)
