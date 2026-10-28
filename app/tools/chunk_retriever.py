"""单篇论文内的字段级 BM25/dense/RRF/重排与有界章节补充。"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from app.agent.execution_budget import check_execution
from app.clients.embedding_client import EmbeddingClient
from app.clients.reranker_client import RerankerClient
from app.clients.retrieval_http import RetrievalProviderError
from app.prompt.paper_card import FIELD_RETRIEVAL_QUERIES
from app.schemas.chunk_schema import ChunkHit, ChunkRetrievalReport
from app.schemas.retrieval_schema import RetrievalQuery, fingerprint
from app.services.retrieval_ranking_service import VectorCache
from app.tools.hybrid_retrieval import lexical_rank, fuse_rankings
from app.tools.chunk_pdf import section_family
from app.schemas.vector_schema import ChunkVectorIndex, IndexCoverage, VectorIntegrityError, VectorStoreUnavailable


def _check(should_cancel):
    check_execution()
    if should_cancel and should_cancel():
        raise InterruptedError("全文检索已取消")


def coverage_order(chunks):
    """章节优先轮转；未知章节按页分组，防止前部页面占满额度。"""
    groups = OrderedDict()
    for chunk in chunks:
        if chunk["excluded"]:
            continue
        family = section_family(chunk["section"])
        key = (family, chunk["page"] if family == "full_text" else None)
        groups.setdefault(key, []).append(chunk["chunk_id"])
    priority = {name: i for i, name in enumerate((
        "method", "results", "limitations", "discussion", "abstract", "conclusion", "introduction"))}
    keys = sorted(groups, key=lambda key: priority.get(key[0], 99))
    # WHY: 长章节先补尾块，再补内部块；索引预算不足时也保留后半篇候选。
    for key in keys:
        ids = groups[key]
        if len(ids) > 2:
            groups[key] = [ids[0], ids[-1], *ids[1:-1]]
    return [groups[key][i] for i in range(max(map(len, groups.values()), default=0))
            for key in keys if i < len(groups[key])]


class DenseChunkIndex:
    """归一化精确向量索引；仅在本次单篇论文范围内持有。"""
    def __init__(self, vectors):
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise RetrievalProviderError("无效 Chunk 向量矩阵")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise RetrievalProviderError("Chunk 向量范数无效")
        self.matrix = matrix / norms

    def search(self, query, limit):
        vector = np.asarray(query, dtype=np.float32)
        if vector.shape != (self.matrix.shape[1],) or not np.isfinite(vector).all():
            raise RetrievalProviderError("Chunk 查询向量无效")
        norm = np.linalg.norm(vector)
        if norm <= 0:
            raise RetrievalProviderError("Chunk 查询范数无效")
        scores = self.matrix @ (vector / norm)
        return sorted(enumerate(map(float, scores)), key=lambda pair: (-pair[1], pair[0]))[:limit]


def _embed(texts, text_type, client, settings, should_cancel):
    cache = VectorCache(settings.retrieval_cache_dir, settings.retrieval_embedding_dimension)
    keys = [fingerprint(("chunk.v1", settings.retrieval_provider, settings.retrieval_embedding_url,
                         settings.retrieval_embedding_model, settings.retrieval_embedding_dimension,
                         settings.retrieval_cache_version, text_type, text)) for text in texts]
    vectors = [cache.get(key) for key in keys]
    missing = [i for i, value in enumerate(vectors) if value is None]
    if missing:
        encoded = client.embed([texts[i] for i in missing], text_type=text_type, should_cancel=should_cancel)
        if len(encoded) != len(missing):
            raise RetrievalProviderError("Chunk 向量数量不匹配")
        for i, vector in zip(missing, encoded):
            if len(vector) != settings.retrieval_embedding_dimension:
                raise RetrievalProviderError("Chunk 向量维度不匹配")
            cache.put(keys[i], vector)
            vectors[i] = vector
    return vectors


def select_extraction_chunks(chunks, report, limit):
    """一半额度用于章节覆盖，其余按字段检索名次轮转；不重复抽取同一块。"""
    coverage = coverage_order(chunks)
    selected = coverage[:max(1, limit // 2)]
    hits = list(report.field_hits.values())
    for rank in range(max(map(len, hits), default=0)):
        for field_hits in hits:
            if rank < len(field_hits) and field_hits[rank].chunk_id not in selected and len(selected) < limit:
                selected.append(field_hits[rank].chunk_id)
    selected.extend(chunk_id for chunk_id in coverage if chunk_id not in selected)
    report.selected_chunk_ids = selected[:limit]
    return report


def retrieve_chunks(paper_id, chunks, *, settings, topic="", should_cancel=None,
                    embedding_client=None, reranker_client=None, parsed_text=None, vector_store=None):
    """只接受同篇论文的原始 Chunk；失败可诊断降级，停止信号不捕获。"""
    _check(should_cancel)
    if any(chunk.get("paper_id") != paper_id for chunk in chunks):
        raise ValueError("Chunk 索引禁止混入其他论文")
    report = ChunkRetrievalReport(mode=settings.card_chunk_retrieval_mode)
    if report.mode == "sections":
        report.backend = "sections"
        return select_extraction_chunks(chunks, report, settings.card_extraction_max_chunks)
    order = coverage_order(chunks)
    persistent = settings.card_vector_backend == "pgvector"
    limit = settings.card_vector_index_max_chunks if persistent else settings.card_chunk_index_limit
    ids = order[:limit]
    report.indexed_chunk_ids = ids
    report.index_truncated = len(ids) < len(order)
    if not ids:
        return report
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    texts = [by_id[i]["text"] for i in ids]
    query_texts = [f"{text}\n研究主题：{topic[:500]}" for text in FIELD_RETRIEVAL_QUERIES.values()]
    dense = [[] for _ in query_texts]
    client = embedding_client or EmbeddingClient(settings)
    try:
        if persistent:
            if parsed_text is None:
                raise VectorIntegrityError("持久化检索必须提供当前原文以验证位置")
            from app.storage.pgvector_store import PgVectorStore
            from app.services.chunk_indexer import ensure_chunk_index, PersistentChunkIndex
            store = vector_store or PgVectorStore(settings)
            spec, snapshot, revision, report.index_coverage = ensure_chunk_index(
                paper_id, parsed_text, chunks, [by_id[i] for i in ids], store=store, settings=settings,
                embed=lambda batch: _embed(batch, "document", client, settings, should_cancel),
                check=lambda: _check(should_cancel))
            report.backend = "pgvector_exact"
            index: ChunkVectorIndex = PersistentChunkIndex(store,spec,snapshot,revision,ids)
        else:
            vectors = _embed(texts, "document", client, settings, should_cancel)
            index = DenseChunkIndex(vectors)
        query_vectors = _embed(query_texts, "query", client, settings, should_cancel)
        for position, query in enumerate(query_vectors):
            _check(should_cancel)
            dense[position] = index.search(query, settings.card_chunk_rerank_window)
    except InterruptedError:
        # WHY: InterruptedError 继承 OSError，必须先透传，不能误降级后继续调用重排。
        raise
    except VectorStoreUnavailable:
        if settings.card_vector_db_failure_policy == "fail":
            raise
        report.backend = "sections"
        report.degraded.append("vector_store:unavailable")
        report.index_coverage = IndexCoverage(status="failed", total_chunks=len(order))
        return select_extraction_chunks(chunks, report, settings.card_extraction_max_chunks)
    except (RetrievalProviderError, OSError) as exc:
        report.degraded.append(f"dense:{type(exc).__name__}")
        if persistent and report.index_coverage is None:
            report.index_coverage = IndexCoverage(status="failed", total_chunks=len(order))
    reranker = reranker_client or RerankerClient(settings)
    for position, (field, text) in enumerate(zip(FIELD_RETRIEVAL_QUERIES, query_texts)):
        _check(should_cancel)
        query = RetrievalQuery(text=text, role="field", family=field, source="paper_chunk")
        lex = lexical_rank(texts, text, settings.card_chunk_rerank_window)
        fused = fuse_rankings(len(texts), [query], [lex], [dense[position]], rrf_k=settings.retrieval_rrf_k)
        window = fused[:settings.card_chunk_rerank_window]
        scores = {}
        if window and settings.card_chunk_enable_rerank:
            try:
                values = reranker.score(text, [texts[item["index"]] for item in window], should_cancel=should_cancel)
                if len(values) != len(window) or not np.isfinite(values).all():
                    raise RetrievalProviderError("Chunk 重排响应无效")
                scores = {item["index"]: float(score) for item, score in zip(window, values)}
                window.sort(key=lambda item: (-scores[item["index"]], -item["rrf"], item["index"]))
            except RetrievalProviderError as exc:
                report.degraded.append(f"rerank:{field}:{type(exc).__name__}")
        report.field_hits[field] = [ChunkHit(
            chunk_id=ids[item["index"]], bm25=item["bm25"], cosine=item["cosine"],
            rrf=item["rrf"], rerank=scores.get(item["index"]),
        ) for item in window[:settings.card_chunk_top_k]]
    _check(should_cancel)
    return select_extraction_chunks(chunks, report, settings.card_extraction_max_chunks)
