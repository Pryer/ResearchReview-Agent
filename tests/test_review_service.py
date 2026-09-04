"""ReviewService 辅助逻辑测试。"""

from __future__ import annotations

from datetime import datetime

from app.database.models import ReviewModel
from app.services.review_service import ReviewService


def test_review_row_to_dict_serializes_json_fields():
    row = ReviewModel(
        id=1,
        topic="目标检测",
        review_text="# 目标检测",
        sections_json='[{"title": "研究背景", "content": "正文"}]',
        references_json='["ref1"]',
        paper_ids_json='["p1"]',
        citation_validation_json='{"valid": true}',
        citation_style="gbt7714",
        language="zh",
        created_at=datetime(2026, 7, 9, 12, 0, 0),
    )

    result = ReviewService._review_row_to_dict(row)

    assert result["id"] == 1
    assert result["sections"][0]["title"] == "研究背景"
    assert result["references"] == ["ref1"]
    assert result["paper_ids"] == ["p1"]
    assert result["citation_validation"] == {"valid": True}
    assert result["created_at"] == "2026-07-09T12:00:00"
