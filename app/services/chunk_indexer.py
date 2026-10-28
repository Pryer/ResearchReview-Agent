"""将当前可用原文发布为有版本、可复用的向量快照。"""
from __future__ import annotations

from app.agent.execution_budget import check_execution, is_control_exception
from app.core.logger import get_logger
from app.schemas.retrieval_schema import fingerprint
from app.schemas.vector_schema import IndexCoverage, IndexSpec, VectorIntegrityError
from app.services.document_ingestion import (
    source_fingerprint, validate_chunk_sources, processing_profile, embedding_profile,
)

logger = get_logger(__name__)


class PersistentChunkIndex:
    def __init__(self, store, spec, snapshot, revision, chunk_ids):
        self.store, self.spec, self.snapshot, self.revision = store, spec, snapshot, revision
        self.positions = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}

    def search(self, query, limit):
        hits = self.store.search(self.spec, self.snapshot, self.revision, query, limit)
        if any(chunk_id not in self.positions for chunk_id, _ in hits):
            raise VectorIntegrityError("向量库返回当前快照之外的块")
        return [(self.positions[chunk_id], score) for chunk_id, score in hits]


def ensure_chunk_index(paper_id, parsed, all_chunks, selected_chunks, *, store, settings, embed, check):
    if not str(paper_id).strip():
        raise VectorIntegrityError("缺少论文身份，不能建立持久化快照")
    validate_chunk_sources(paper_id, all_chunks, parsed)
    source_hash = source_fingerprint(parsed)
    spec = IndexSpec(
        tenant=settings.card_vector_tenant, paper_id=paper_id, source_hash=source_hash,
        processing_profile=processing_profile(parsed, selected_chunks, settings),
        embedding_profile=embedding_profile(settings), dimension=settings.retrieval_embedding_dimension,
        expected_count=len(selected_chunks), total_count=sum(not chunk["excluded"] for chunk in all_chunks),
    )
    check()
    revision = store.register_source(spec.tenant, paper_id, source_hash,
                                     expected_revision=parsed.get("source_revision"))
    snapshot, ticket = store.claim(spec, revision)
    if ticket is not None:
        try:
            size = settings.retrieval_embedding_batch_size
            for start in range(0, len(selected_chunks), size):
                check()
                store.renew(spec, ticket)
                batch = selected_chunks[start:start + size]
                vectors = embed([chunk["text"] for chunk in batch])
                check()
                store.write(spec, ticket, batch, vectors, start_order=start)
            check()
            store.activate(spec, ticket)
        except Exception as exc:
            try:
                # WHY: 已停止的执行不能为了写诊断继续发请求；未完成 building 等租约过期再回收。
                check_execution()
                store.fail(spec, ticket, cancelled=isinstance(exc, InterruptedError) or is_control_exception(exc))
            except Exception as cleanup:
                logger.warning("Vector build cleanup deferred: %s", type(cleanup).__name__)
            raise
    rows = store.load(spec, snapshot, revision)
    for order, (chunk, row) in enumerate(zip(selected_chunks, rows)):
        if (row["chunk_id"] != chunk["chunk_id"] or row["chunk_order"] != order
                or row["text"] != chunk["text"] or row["content_hash"] != fingerprint(chunk["text"])
                or row["page"] != chunk["page"] or row["section"] != chunk["section"]
                or row["offset_basis"] != chunk["offset_basis"] or row["start_offset"] != chunk["char_start"]
                or row["end_offset"] != chunk["char_end"]):
            raise VectorIntegrityError("持久化块与当前原文/位置不一致")
    coverage = IndexCoverage(
        snapshot_id=snapshot, source_revision=revision,
        status="partial" if spec.expected_count < spec.total_count else "complete",
        total_chunks=spec.total_count, indexed_chunks=len(rows), reused=ticket is None,
        index_truncated=spec.expected_count < spec.total_count,
    )
    logger.info("Vector index %s: indexed=%s total=%s coverage=%s",
                "reused" if ticket is None else "published", len(rows), spec.total_count, coverage.status)
    return spec, snapshot, revision, coverage
