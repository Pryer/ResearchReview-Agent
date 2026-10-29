"""分层 RAG 启动检查：不调用模型，不迁移数据库，不输出连接信息。"""
from __future__ import annotations

from app.core.exceptions import AgentBaseError

RAG_PIPELINE_VERSION = "hierarchical-rag-v2"


class RagRuntimeUnavailable(AgentBaseError):
    status_code = 503


def check_rag_runtime(settings, *, check_database=True, store=None) -> dict:
    database_checked = False
    checks = {
        "paper_hybrid": settings.retrieval_ranking_mode == "hybrid",
        "pdf_pipeline": settings.enable_pdf_pipeline,
        "chunk_hybrid": settings.card_chunk_retrieval_mode == "hybrid",
        "llm": bool(settings.llm_api_key and settings.llm_base_url and settings.llm_model),
        "embedding": bool(settings.dashscope_api_key and settings.retrieval_embedding_url),
        "reranker": bool(settings.retrieval_rerank_url),
    }
    if settings.card_vector_backend == "pgvector":
        checks["vector_connection"] = bool(settings.card_vector_database_url)
        if check_database and checks["vector_connection"]:
            database_checked = True
            from app.storage.pgvector_store import PgVectorStore
            from app.schemas.vector_schema import VectorStoreUnavailable, VectorIntegrityError
            try:
                checks["vector_schema"] = bool((store or PgVectorStore(settings)).ready())
            except (VectorStoreUnavailable, VectorIntegrityError):
                checks["vector_schema"] = False
    return {"ready": all(checks.values()), "checks": checks,
            "database_checked": database_checked, "backend": settings.card_vector_backend}


def require_rag_runtime(settings) -> dict:
    report = check_rag_runtime(settings)
    if not report["ready"]:
        failed = ", ".join(key for key, ok in report["checks"].items() if not ok)
        raise RagRuntimeUnavailable(f"分层 RAG 依赖未就绪：{failed}；请运行启动检查并完成配置或迁移")
    return report
