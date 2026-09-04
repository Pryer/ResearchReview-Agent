# -*- coding: utf-8 -*-
"""测试增强版筛选协议兜底功能。"""

import pytest
from app.agent.planner import (
    _enhanced_fallback_screening_protocol,
    build_screening_protocol,
)
from app.schemas.screening_schema import ScreeningProtocol


def test_enhanced_fallback_with_branches_and_frame():
    """测试从 search_branches 和 semantic_frame 自动构建 soft_criteria 与 routes。"""
    topic = "少样本动作识别"
    search_branches = [
        {
            "branch_type": "technical_method",
            "required_concepts": [
                ["few-shot", "少样本", "小样本"],
                ["action recognition", "动作识别", "video classification"],
            ],
            "rationale": "方法与领域核心概念",
        },
        {
            "branch_type": "domain_foundation",
            "required_concepts": [["temporal alignment", "时序对齐"]],
            "rationale": "时序建模",
        },
    ]
    semantic_frame = {
        "canonical_topic": "少样本动作识别",
        "method_roles": {
            "metric_learning": "primary_method",
            "temporal_alignment": "auxiliary_method",
        },
    }

    protocol = _enhanced_fallback_screening_protocol(
        topic=topic,
        search_branches=search_branches,
        semantic_frame=semantic_frame,
    )

    assert isinstance(protocol, ScreeningProtocol)
    assert protocol.generated_by == "enhanced_fallback"
    assert len(protocol.soft_include_criteria) == 3
    # 验证提取了中英对照
    criterion_0 = protocol.soft_include_criteria[0]
    assert "few-shot" in criterion_0.terms_en
    assert "少样本" in criterion_0.terms_zh

    # 验证提取了 routes
    assert len(protocol.routes) == 2
    route_ids = [r.route_id for r in protocol.routes]
    assert "metric_learning" in route_ids
    assert "temporal_alignment" in route_ids


def test_build_screening_protocol_single_turn_uses_enhanced_fallback():
    """测试单轮无 LLM 场景下 build_screening_protocol 返回 enhanced_fallback。"""
    topic = "少样本动作识别"
    search_branches = [
        {
            "branch_type": "technical_method",
            "required_concepts": [["few-shot", "少样本"]],
            "rationale": "技术方法",
        },
    ]

    result = build_screening_protocol(
        original_query="帮我调研少样本动作识别",
        user_query="帮我调研少样本动作识别",
        topic=topic,
        conversation_history=[],
        selected_scope=None,
        semantic_frame=None,
        search_branches=search_branches,
        llm=None,
    )

    assert result["generated_by"] == "enhanced_fallback"
    assert len(result["soft_include_criteria"]) == 1
    assert result["soft_include_criteria"][0]["terms"] == ["few-shot", "少样本"]


def test_enhanced_fallback_empty_branches():
    """测试空分支时优雅降级为 minimal_fallback。"""
    protocol = _enhanced_fallback_screening_protocol(topic="通用主题")
    assert protocol.generated_by == "minimal_fallback"
    assert len(protocol.soft_include_criteria) == 0
    assert len(protocol.routes) == 0
