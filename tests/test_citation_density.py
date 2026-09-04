# -*- coding: utf-8 -*-
"""引用密度与堆砌检测测试。"""

from __future__ import annotations

import pytest

from app.core.citation_density import detect_citation_dumps, has_citation_dumps


def test_detect_no_dumps():
    text = "在少样本动作识别中，Huang等[1]提出复合原型，Lee等[2, 3]探讨了时序对齐。"
    assert not has_citation_dumps(text, max_per_group=3)
    assert len(detect_citation_dumps(text, max_per_group=3)) == 0


def test_detect_single_bracket_dump():
    text = "该领域得到了广泛关注[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]。"
    dumps = detect_citation_dumps(text, max_per_group=5)
    assert len(dumps) == 1
    assert dumps[0]["type"] == "single_bracket_dump"
    assert dumps[0]["citation_count"] == 10
    assert has_citation_dumps(text, max_per_group=5)


def test_detect_consecutive_brackets_dump():
    text = "多项代表性工作展开了探索[1][2][3][4][5][6][7]。"
    dumps = detect_citation_dumps(text, max_per_group=5)
    assert len(dumps) == 1
    assert dumps[0]["type"] == "consecutive_brackets_dump"
    assert dumps[0]["citation_count"] == 7
    assert has_citation_dumps(text, max_per_group=5)


def test_default_threshold_catches_four_citations():
    """默认阈值3应该检测到4篇引用堆砌。"""
    text = "少样本动作识别领域近年取得积极进展[1, 2, 3, 4]。"
    # 默认阈值为3，4篇应被检测为堆砌
    assert has_citation_dumps(text)
    assert len(detect_citation_dumps(text)) == 1
    # 3篇不应被检测为堆砌
    text_ok = "相关工作从不同角度展开[1, 2, 3]。"
    assert not has_citation_dumps(text_ok)
