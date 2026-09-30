"""DashScope 原生文本重排客户端。"""

from __future__ import annotations

from app.clients.retrieval_http import RetrievalProviderError, finite_number, post_json


class RerankerClient:
    def __init__(self, settings, *, session=None):
        self.settings = settings
        self.session = session
        self.calls = 0
        self.usage = []

    def score(self, query, documents, *, should_cancel=None):
        if not documents:
            return []
        settings = self.settings
        scores = []
        for start in range(0, len(documents), settings.retrieval_rerank_batch_size):
            batch = list(documents[start:start + settings.retrieval_rerank_batch_size])
            payload = post_json(
                settings.retrieval_rerank_url, settings.dashscope_api_key,
                {"model": settings.retrieval_rerank_model,
                 "input": {"query": query, "documents": batch},
                 "parameters": {"top_n": len(batch)}},
                timeout=settings.retrieval_request_timeout,
                retries=settings.retrieval_max_retries,
                session=self.session, should_cancel=should_cancel,
                kind="rerank",
            )
            self.calls += 1
            self.usage.append(payload.get("usage"))
            from app.agent.execution_budget import record_retrieval_usage

            record_retrieval_usage(payload.get("usage"))
            records = (payload.get("output") or {}).get("results")
            if not isinstance(records, list) or len(records) != len(batch):
                raise RetrievalProviderError("rerank 返回数量与请求不一致")
            ordered = [None] * len(batch)
            for record in records:
                index = record.get("index") if isinstance(record, dict) else None
                if type(index) is not int or not 0 <= index < len(batch) or ordered[index] is not None:
                    raise RetrievalProviderError("rerank index 重复或越界")
                ordered[index] = finite_number(record.get("relevance_score"))
            if any(score is None for score in ordered):
                raise RetrievalProviderError("rerank 返回索引缺失")
            scores.extend(ordered)
        return scores
