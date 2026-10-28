"""可选后端的默认行为、故障及控制边界，无数据库和模型请求。"""
import pytest

from app.agent.execution_budget import AgentBudgetExceeded
from app.core.config import Settings
from app.schemas.vector_schema import VectorStoreUnavailable, VectorIntegrityError
from app.tools.chunk_pdf import chunk_pdf
from app.tools.chunk_retriever import retrieve_chunks
from app.tools.extract_paper_card import extract_paper_card, batch_extract_paper_cards
from app.storage.pgvector_store import normalized_vectors


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None,card_vector_backend="pgvector",card_chunk_retrieval_mode="hybrid",
                    retrieval_cache_dir=str(tmp_path))


def source():
    parsed = {"full_text":"We propose a dual encoder method for document retrieval."}
    return parsed,chunk_pdf(parsed,paper_id="p")


class BrokenStore:
    def __init__(self,error):
        self.error = error
    def register_source(self,*args,**kwargs):
        raise self.error


class NoModel:
    def embed(self,*args,**kwargs):
        raise AssertionError("must not invoke embedding")


def test_sections_mode_does_not_require_pg_database(settings):
    settings.card_chunk_retrieval_mode = "sections"
    parsed,chunks = source()
    report = retrieve_chunks("p",chunks,settings=settings,parsed_text=parsed,embedding_client=NoModel())
    assert report.backend == "sections" and not report.degraded


def test_database_unavailable_explicit_sections_fallback(settings):
    parsed,chunks = source()
    report = retrieve_chunks("p",chunks,settings=settings,parsed_text=parsed,embedding_client=NoModel(),
                             vector_store=BrokenStore(VectorStoreUnavailable("private password")))
    assert report.backend == "sections"
    assert report.index_coverage.status == "failed"
    assert report.degraded == ["vector_store:unavailable"]
    assert "private password" not in report.model_dump_json()


@pytest.mark.parametrize("error", [VectorIntegrityError("stale"),AgentBudgetExceeded("budget"),InterruptedError("cancel")])
def test_integrity_and_stop_signals_never_downgrade(settings,error):
    parsed,chunks = source()
    with pytest.raises(type(error)):
        retrieve_chunks("p",chunks,settings=settings,parsed_text=parsed,embedding_client=NoModel(),
                        vector_store=BrokenStore(error))


def test_explicit_fail_policy_propagates(settings):
    settings.card_vector_db_failure_policy = "fail"
    parsed,chunks = source()
    with pytest.raises(VectorStoreUnavailable):
        retrieve_chunks("p",chunks,settings=settings,parsed_text=parsed,embedding_client=NoModel(),
                        vector_store=BrokenStore(VectorStoreUnavailable("down")))


def test_extraction_does_not_replace_integrity_error_with_rule_based_claims(monkeypatch,settings):
    monkeypatch.setattr("app.core.config.get_settings",lambda:settings)
    monkeypatch.setattr("app.storage.pgvector_store.PgVectorStore",lambda _:BrokenStore(VectorIntegrityError("stale")))
    parsed,_ = source()
    paper = {"paper_id":"p","title":"Study"}
    with pytest.raises(VectorIntegrityError):
        extract_paper_card(paper,parsed,llm=object())
    with pytest.raises(VectorIntegrityError):
        batch_extract_paper_cards([paper],{"p":parsed},llm=object())


@pytest.mark.parametrize("vectors", [[[float("nan"),0]],[[0,0]],[[1,0,0]]])
def test_storage_vector_validation(vectors):
    with pytest.raises(VectorIntegrityError):
        normalized_vectors(vectors,2)
