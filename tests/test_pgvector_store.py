"""真实 PostgreSQL 协议集成；无测试 DSN 时明确跳过，不使用业务数据库。"""
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from uuid import uuid4

import numpy as np
import pytest

from app.core.config import Settings
from app.schemas.vector_schema import IndexSpec, VectorIntegrityError, VectorStoreUnavailable
from app.storage.pgvector_store import PgVectorStore
from app.tools.chunk_retriever import DenseChunkIndex, retrieve_chunks

DSN = os.environ.get("RRA_VECTOR_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="requires dedicated RRA_VECTOR_TEST_DATABASE_URL")


@pytest.fixture
def settings(tmp_path):
    pytest.importorskip("psycopg")
    return Settings(_env_file=None, card_vector_database_url=DSN,
                    card_vector_tenant="test_" + uuid4().hex, card_vector_pool_size=1,
                    card_vector_prepared_statements=os.environ.get("RRA_VECTOR_TEST_NATIVE") == "1",
                    retrieval_embedding_dimension=1024, retrieval_cache_dir=str(tmp_path),
                    card_chunk_retrieval_mode="hybrid", card_vector_backend="pgvector",
                    card_chunk_enable_rerank=False)


@pytest.fixture
def store(settings):
    backend = PgVectorStore(settings)
    backend.migrate()
    yield backend
    with backend._db() as cur:
        cur.execute("DELETE FROM rra_vector_sources WHERE tenant=%s", (settings.card_vector_tenant,))


def spec(settings, count=2, total=None):
    return IndexSpec(settings.card_vector_tenant,"p","source-v1","processing-v1","embedding-v1",1024,count,total or count)


def test_runtime_ready_checks_migration_and_dimension(store, settings):
    assert store.ready() is True
    incompatible = settings.model_copy(update={"retrieval_embedding_dimension": 2})
    with pytest.raises(VectorIntegrityError):
        PgVectorStore(incompatible).ready()


def items(count):
    return [dict(paper_id="p", chunk_id=i, section="full_text", page=i + 1,
                 char_start=0,char_end=len(f"Original source sentence {i}."),offset_basis="page",
                 excluded=False,text=f"Original source sentence {i}.") for i in range(count)]


def vectors(count):
    values = np.random.default_rng(42).normal(size=(count,1024)).astype(np.float32)
    return values / np.linalg.norm(values,axis=1,keepdims=True)


def build(store, descriptor):
    rev = store.register_source(descriptor.tenant,descriptor.paper_id,descriptor.source_hash)
    snapshot,ticket = store.claim(descriptor,rev)
    if ticket:
        store.write(descriptor,ticket,items(descriptor.expected_count),vectors(descriptor.expected_count),start_order=0)
        store.activate(descriptor,ticket)
    return snapshot,rev,ticket


def test_migrations_are_idempotent_and_health_reports_actual_versions(store):
    store.migrate()
    assert store.health()["vector"]
    assert "PostgreSQL" in store.health()["server"]


def test_index_reuse_and_delete_cascade(store,settings):
    descriptor = spec(settings)
    snapshot,rev,_ = build(store,descriptor)
    assert store.claim(descriptor,rev) == (snapshot,None)
    assert len(store.load(descriptor,snapshot,rev)) == 2
    store.delete(descriptor.tenant,descriptor.paper_id)
    with pytest.raises(VectorIntegrityError):
        store.load(descriptor,snapshot,rev)
    with store._db() as cur:
        cur.execute("SELECT count(*) AS n FROM rra_vector_chunks WHERE snapshot_id=%s", (snapshot,))
        assert cur.fetchone()["n"] == 0


@pytest.mark.parametrize("ties", [False, True])
def test_exact_cosine_top_k_matches_numpy_for_200_chunks(store,settings,ties):
    descriptor = spec(settings,200)
    rev = store.register_source(descriptor.tenant,"p",descriptor.source_hash)
    snapshot,ticket = store.claim(descriptor,rev)
    matrix = vectors(200)
    if ties:
        matrix[1] = matrix[0]
    store.write(descriptor,ticket,items(200),matrix,start_order=0)
    store.activate(descriptor,ticket)
    query = matrix[0]
    expected = DenseChunkIndex(matrix).search(query,20)
    actual = store.search(descriptor,snapshot,rev,query,20)
    assert [i for i,_ in actual] == [i for i,_ in expected]
    assert np.allclose([s for _,s in actual],[s for _,s in expected],atol=2e-6)


def test_source_update_requires_cas_and_fences_old_builder(store,settings):
    old = spec(settings)
    rev = store.register_source(old.tenant,"p",old.source_hash)
    _,ticket = store.claim(old,rev)
    with pytest.raises(VectorIntegrityError):
        store.register_source(old.tenant,"p","new-source")
    new_rev = store.register_source(old.tenant,"p","new-source",expected_revision=rev)
    assert new_rev == rev + 1
    with pytest.raises(VectorIntegrityError):
        store.write(old,ticket,items(2),vectors(2),start_order=0)
    with pytest.raises(VectorIntegrityError):
        store.activate(old,ticket)
    with pytest.raises(VectorIntegrityError):
        store.register_source(old.tenant,"p",old.source_hash,expected_revision=rev)


def test_lease_takeover_rejects_old_writer(store,settings):
    descriptor = spec(settings)
    rev = store.register_source(descriptor.tenant,"p",descriptor.source_hash)
    snapshot,first = store.claim(descriptor,rev)
    with pytest.raises(VectorStoreUnavailable):
        store.claim(descriptor,rev)
    with store._db() as cur:
        cur.execute("UPDATE rra_vector_snapshots SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE id=%s", (snapshot,))
    _,second = store.claim(descriptor,rev)
    assert second.fencing_token > first.fencing_token
    with pytest.raises(VectorIntegrityError):
        store.renew(descriptor,first)
    with pytest.raises(VectorIntegrityError):
        store.write(descriptor,first,items(2),vectors(2),start_order=0)
    store.write(descriptor,second,items(2),vectors(2),start_order=0)
    store.activate(descriptor,second)


def test_incomplete_build_does_not_publish_or_destroy_old_active(store,settings):
    old = spec(settings)
    old_id,rev,_ = build(store,old)
    new = replace(old,processing_profile="new")
    _,ticket = store.claim(new,rev)
    store.write(new,ticket,items(1),vectors(1),start_order=0)
    with pytest.raises(VectorIntegrityError):
        store.activate(new,ticket)
    store.fail(new,ticket)
    assert len(store.load(old,old_id,rev)) == 2


def test_cross_tenant_paper_and_profile_queries_rejected(store,settings):
    descriptor = spec(settings)
    snapshot,rev,_ = build(store,descriptor)
    for other in (replace(descriptor,tenant="other"),replace(descriptor,paper_id="other"),
                  replace(descriptor,embedding_profile="different"),replace(descriptor,dimension=3)):
        with pytest.raises(VectorIntegrityError):
            store.search(other,snapshot,rev,vectors(1)[0] if other.dimension==1024 else [1,0,0],5)


def test_expired_snapshot_queries_rejected_after_source_update(store,settings):
    descriptor = spec(settings)
    snapshot,rev,_ = build(store,descriptor)
    store.register_source(descriptor.tenant,"p","new",expected_revision=rev)
    with pytest.raises(VectorIntegrityError):
        store.search(descriptor,snapshot,rev,vectors(1)[0],5)


def test_wrong_dimension_refuses_migration_without_changing_schema(store,settings):
    settings.retrieval_embedding_dimension = 3
    with pytest.raises(VectorIntegrityError):
        store.migrate()
    settings.retrieval_embedding_dimension = 1024
    assert store.health()["vector"]


class Encoder:
    def __init__(self, fail_documents=False):
        self.calls = []
        self.fail_documents = fail_documents

    def embed(self,texts,*,text_type,**kwargs):
        self.calls.append(text_type)
        if text_type == "document" and self.fail_documents:
            raise AssertionError("active snapshot must not re-embed documents")
        return [([0.,1.] if text_type=="query" or "Results" in text else [1.,0.]) + [0.]*1022 for text in texts]


def retrieval_input():
    from app.tools.chunk_pdf import chunk_pdf
    pages = [{"page":1,"text":"Methods\nWe propose a retrieval method using a dual encoder."},
             {"page":2,"text":"Results\nResults show an accuracy of 91% on the test data."}]
    parsed = {"pages":pages,"full_text":"\n".join(p["text"] for p in pages)}
    return parsed,chunk_pdf(parsed,paper_id="p")


def test_persistent_reuse_bypasses_expired_document_cache_and_matches_numpy(store,settings):
    parsed,chunks = retrieval_input()
    first = retrieve_chunks("p",chunks,settings=settings,parsed_text=parsed,embedding_client=Encoder(),vector_store=store)
    assert not first.index_coverage.reused
    # 模拟向量计算缓存已清理：持久化快照仍必须支持文档索引复用。
    from pathlib import Path
    for path in Path(settings.retrieval_cache_dir).glob("*.npy"):
        path.unlink()
    encoder = Encoder(fail_documents=True)
    second = retrieve_chunks("p",chunks,settings=settings,parsed_text=parsed,embedding_client=encoder,vector_store=store)
    assert second.index_coverage.reused and "document" not in encoder.calls
    settings.card_vector_backend = "numpy"
    baseline = retrieve_chunks("p",chunks,settings=settings,embedding_client=Encoder())
    assert {k:[h.chunk_id for h in v] for k,v in second.field_hits.items()} == {
        k:[h.chunk_id for h in v] for k,v in baseline.field_hits.items()}
    assert second.selected_chunk_ids == baseline.selected_chunk_ids


def test_source_position_corruption_rejected_before_model_extraction(store,settings):
    parsed,chunks = retrieval_input()
    report = retrieve_chunks("p",chunks,settings=settings,parsed_text=parsed,embedding_client=Encoder(),vector_store=store)
    with store._db() as cur:
        cur.execute("UPDATE rra_vector_chunks SET text='Tampered source text.' WHERE snapshot_id=%s",(report.index_coverage.snapshot_id,))
    with pytest.raises(VectorIntegrityError):
        retrieve_chunks("p",chunks,settings=settings,parsed_text=parsed,embedding_client=Encoder(),vector_store=store)


def test_actual_database_to_paper_card_pipeline_with_mock_models(store,settings,monkeypatch):
    import json
    from app.tools.extract_paper_card import extract_paper_card
    from app.tools.parse_pdf import split_paper_sections
    monkeypatch.setattr("app.core.config.get_settings",lambda:settings)
    monkeypatch.setattr("app.tools.chunk_retriever.EmbeddingClient",lambda _:Encoder())
    parsed,chunks = retrieval_input()
    parsed["full_text"] += "\nReferences\nBibliographic entry."
    parsed["sections"] = split_paper_sections(parsed["full_text"])
    method = "We propose a retrieval method using a dual encoder."
    result = "Results show an accuracy of 91% on the test data."
    class Extractor:
        def complete(self,prompt,**kwargs):
            return json.dumps({"method":[method] if method in prompt else [],
                               "results":[result] if result in prompt else []})
    card = extract_paper_card({"paper_id":"p","title":"Study"},parsed,Extractor())
    assert card.method == method and card.results == result
    assert card.extraction_coverage.retrieval.backend == "pgvector_exact"
    assert card.extraction_coverage.retrieval.index_coverage.status == "complete"
    assert card.field_claims["results"][0].explicitly_reported
    for span in card.evidence_spans:
        if span.source_type == "full_text":
            page = parsed["pages"][span.page-1]["text"]
            assert page[span.char_start:span.char_end] == span.text


def test_index_truncation_is_separate_from_successful_snapshot_plan(store,settings):
    from app.tools.chunk_pdf import chunk_pdf
    settings.card_vector_index_max_chunks = 8
    parsed = {"pages":[{"page":i+1,"text":f"Original source paragraph on page {i}."} for i in range(12)]}
    report = retrieve_chunks("p",chunk_pdf(parsed,paper_id="p"),settings=settings,parsed_text=parsed,
                             embedding_client=Encoder(),vector_store=store)
    assert report.index_coverage.status == "partial"
    assert report.index_coverage.total_chunks == 12
    assert report.index_coverage.indexed_chunks == 8
    assert report.index_coverage.index_truncated
    assert report.index_coverage.snapshot_id


def test_failed_embedding_build_is_retryable_and_never_active(store,settings):
    from app.clients.retrieval_http import RetrievalProviderError
    parsed,chunks = retrieval_input()
    class FailedEncoder:
        def embed(self,*args,**kwargs):
            raise RetrievalProviderError("provider unavailable")
    report = retrieve_chunks("p",chunks,settings=settings,parsed_text=parsed,embedding_client=FailedEncoder(),vector_store=store)
    assert report.index_coverage.status == "failed"
    with store._db() as cur:
        cur.execute("SELECT status FROM rra_vector_snapshots WHERE tenant=%s",(settings.card_vector_tenant,))
        assert [row["status"] for row in cur.fetchall()] == ["failed"]
    retried = retrieve_chunks("p",chunks,settings=settings,parsed_text=parsed,embedding_client=Encoder(),vector_store=store)
    assert retried.index_coverage.status == "complete"


@pytest.mark.skipif(os.environ.get("RRA_VECTOR_TEST_NATIVE") != "1", reason="requires native PostgreSQL multi-connection semantics")
def test_native_concurrent_claim_has_only_one_owner(store,settings):
    settings.card_vector_pool_size = 2
    descriptor = spec(settings)
    rev = store.register_source(descriptor.tenant,"p",descriptor.source_hash)
    barrier = Barrier(2)
    def claim():
        barrier.wait(timeout=10)
        try:
            return store.claim(descriptor,rev)[1]
        except VectorStoreUnavailable:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _:claim(),range(2)))
    assert sum(ticket is not None for ticket in results) == 1
