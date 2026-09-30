"""pytest 共享 fixtures。"""

from __future__ import annotations

import os
import sys

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 测试默认钉在 rules 检索模式：绝大多数单元/集成测试验证的是既有规则链路
# （rule_filter/llm_rerank 诊断等），不应因为本机 .env 启用 hybrid 而改变
# 行为。hybrid 行为由 test_hybrid_retrieval*.py 通过 state 的
# retrieval_profile 或 monkeypatch settings 显式开启。必须在导入 app
# 之前设置，pydantic-settings 在实例化时读取环境变量。
os.environ.setdefault("RETRIEVAL_RANKING_MODE", "rules")

# 把测试期间的文件日志隔离到 logs/test/：很多用例直接调用 run_research_agent
# 等函数，会真实打印"Agent started/step"等运行痕迹；若写入生产 logs/app.log，
# 排查真实任务时会混入"幽灵执行"。必须在 app.core.logger 被首次导入前设置。
os.environ.setdefault("APP_LOG_DIR", os.path.join("logs", "test"))

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
