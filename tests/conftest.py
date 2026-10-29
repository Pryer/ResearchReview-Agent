"""pytest 共享 fixtures。"""

from __future__ import annotations

import os
import sys

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 测试运行同一条 hybrid 业务路径，模型使用确定性替身；数据库集成单独显式启用。
os.environ.setdefault("RETRIEVAL_RANKING_MODE", "hybrid")
os.environ["CARD_CHUNK_RETRIEVAL_MODE"] = "hybrid"
os.environ["CARD_VECTOR_DB_FAILURE_POLICY"] = "fail"
os.environ["CARD_VECTOR_BACKEND"] = "numpy"

# 把测试期间的文件日志隔离到 logs/test/：很多用例直接调用 run_research_agent
# 等函数，会真实打印"Agent started/step"等运行痕迹；若写入生产 logs/app.log，
# 排查真实任务时会混入"幽灵执行"。必须在 app.core.logger 被首次导入前设置。
os.environ.setdefault("APP_LOG_DIR", os.path.join("logs", "test"))

import pytest


@pytest.fixture(autouse=True)
def offline_rag_models(monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.services import retrieval_ranking_service
    from app.tools import chunk_retriever
    from app.agent import graph
    from tests.retrieval_fakes import OfflineEmbedding, OfflineReranker

    settings = get_settings()
    monkeypatch.setattr(settings, "retrieval_cache_dir", str(tmp_path / "retrieval"))
    monkeypatch.setattr(settings, "dashscope_api_key", "offline-test")
    monkeypatch.setattr(settings, "retrieval_embedding_url", "https://example.invalid/embedding")
    monkeypatch.setattr(settings, "retrieval_rerank_url", "https://example.invalid/rerank")
    for module in (retrieval_ranking_service, chunk_retriever):
        monkeypatch.setattr(module, "EmbeddingClient", OfflineEmbedding)
        monkeypatch.setattr(module, "RerankerClient", OfflineReranker)
    # WHY: 本 fixture 只隔离环境检查；启动依赖检查本身由专用用例调用真实实现。
    monkeypatch.setattr(graph, "require_rag_runtime", lambda settings: {"ready": True})


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


@pytest.fixture
def retry_clock(monkeypatch):
    """Advance both retry sleeps and cooldown clocks without real network waits."""
    import time
    from types import SimpleNamespace
    clock = SimpleNamespace(now=time.monotonic(), waits=[])
    def sleep(seconds):
        clock.waits.append(seconds)
        clock.now += seconds + 0.000001
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(time, "sleep", sleep)
    return clock
