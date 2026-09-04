"""pytest 共享 fixtures。"""

from __future__ import annotations

import os
import sys

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest


@pytest.fixture
def sample_paper():
    """返回一个示例论文字典。"""
    return {
        "paper_id": "test:1",
        "title": "Vision Transformer for Image Classification",
        "authors": ["Alice", "Bob"],
        "year": 2023,
        "venue": "CVPR",
        "abstract": "This paper proposes a vision transformer.",
        "doi": "10.1000/test",
        "arxiv_id": None,
        "url": "https://example.com",
        "pdf_url": None,
        "citation_count": 50,
        "source": "test",
    }


@pytest.fixture
def sample_card():
    """返回一个示例 PaperCard 字典。"""
    return {
        "paper_id": "test:1",
        "title": "Vision Transformer for Image Classification",
        "year": 2023,
        "venue": "CVPR",
        "research_problem": "Improve image classification",
        "method": "Vision Transformer with self-attention",
        "dataset": "ImageNet",
        "metrics": ["Top-1 Accuracy"],
        "results": "85% accuracy",
        "contributions": ["New attention mechanism"],
        "limitations": ["Computationally expensive"],
        "relevance_reason": "Highly relevant to vision transformers",
        "evidence_source": "abstract",
    }
