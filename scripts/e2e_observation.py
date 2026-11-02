"""仅供独立 E2E 进程使用的运行观察；不改变业务返回值和异常。"""
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
import json
from hashlib import sha256
from threading import Lock
from unittest.mock import patch


def paper_identity(paper):
    return {**{key: paper.get(key) for key in ("paper_id", "doi", "arxiv_id", "title", "year", "source")},
            "content_sha256": sha256(json.dumps({k: paper.get(k) for k in ("title", "abstract", "keywords")},
                                                ensure_ascii=False, sort_keys=True).encode()).hexdigest()}


class RunObserver:
    def __init__(self, path):
        self.path = path
        self.events = []
        self.lock = Lock()
        self.context = ContextVar("e2e_observation", default={"stage": "other"})

    def emit(self, kind, **values):
        with self.lock:
            event = {"sequence": len(self.events) + 1, "kind": kind, **self.context.get(), **values}
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.events.append(event)

    def model_wrapper(self, method, kind):
        @wraps(method)
        def wrapped(client, *args, **kwargs):
            before = client.calls
            try:
                result = method(client, *args, **kwargs)
            except BaseException as exc:
                self.emit("model", model_kind=kind, outcome="failed", error_type=type(exc).__name__,
                          http_responses=client.calls - before)
                raise
            self.emit("model", model_kind=kind, outcome="succeeded", http_responses=client.calls - before)
            return result
        return wrapped

    @contextmanager
    def installed(self):
        from app.clients.embedding_client import EmbeddingClient
        from app.clients.reranker_client import RerankerClient
        from app.services import retrieval_ranking_service as ranking
        from app.storage.pgvector_store import PgVectorStore
        from app.tools import chunk_retriever
        from app.services.llm_service import LLMService

        rank_method, screen_method = ranking.rank_candidates, ranking.screen_candidates
        chunk_method, search_method = chunk_retriever.retrieve_chunks, PgVectorStore.search
        usage_method = LLMService._record_usage

        def rank(state, **kwargs):
            token = self.context.set({"stage": "paper"})
            candidates = [paper_identity(p) for p in state.get("candidate_papers") or []]
            try:
                papers, report = rank_method(state, **kwargs)
                self.emit("paper_ranking", candidates=candidates,
                          ranked=[paper_identity(p) for p in papers], report=report)
                return papers, report
            except BaseException as exc:
                self.emit("paper_ranking", candidates=candidates, ranked=None, error_type=type(exc).__name__)
                raise
            finally:
                self.context.reset(token)

        def screen(state, candidates, llm, **kwargs):
            papers, report = screen_method(state, candidates, llm, **kwargs)
            self.emit("paper_admission", candidates=[paper_identity(p) for p in candidates],
                      admitted=[paper_identity(p) for p in papers], report=report)
            return papers, report

        def chunks(paper_id, *args, **kwargs):
            token = self.context.set({"stage": "chunk", "paper_id": paper_id})
            try:
                report = chunk_method(paper_id, *args, **kwargs)
                self.emit("chunk_retrieval", report=report.model_dump(mode="json"))
                return report
            except BaseException as exc:
                self.emit("chunk_retrieval", error_type=type(exc).__name__)
                raise
            finally:
                self.context.reset(token)

        def search(store, spec, snapshot, revision, *args, **kwargs):
            try:
                hits = search_method(store, spec, snapshot, revision, *args, **kwargs)
            except BaseException as exc:
                self.emit("vector_query", outcome="failed", paper_id=spec.paper_id,
                          snapshot_id=snapshot, error_type=type(exc).__name__)
                raise
            self.emit("vector_query", outcome="succeeded", paper_id=spec.paper_id,
                      snapshot_id=snapshot, returned_count=len(hits))
            return hits

        def usage(client, raw, **kwargs):
            tokens = usage_method(client, raw, **kwargs)
            self.emit("llm_usage", provider=kwargs.get("provider"), model=kwargs.get("model"),
                      operation=kwargs.get("operation"), usage_reported=raw is not None, usage=tokens)
            return tokens

        with ExitStack() as stack:
            for obj, name, replacement in (
                (ranking, "rank_candidates", rank), (ranking, "screen_candidates", screen),
                (chunk_retriever, "retrieve_chunks", chunks), (PgVectorStore, "search", search),
                (EmbeddingClient, "embed", self.model_wrapper(EmbeddingClient.embed, "embedding")),
                (RerankerClient, "score", self.model_wrapper(RerankerClient.score, "reranker")),
                (LLMService, "_record_usage", usage),
            ):
                stack.enter_context(patch.object(obj, name, replacement))
            yield self
