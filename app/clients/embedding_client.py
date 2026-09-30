"""DashScope 原生 dense embedding 客户端。"""

from __future__ import annotations

import math

from app.clients.retrieval_http import RetrievalProviderError, finite_number, post_json


class EmbeddingClient:
    def __init__(self, settings, *, session=None):
        self.settings = settings
        self.session = session
        self.calls = 0
        self.usage = []

    def embed(self, texts, *, text_type, should_cancel=None):
        if text_type not in {"query", "document"}:
            raise ValueError("text_type 必须是 query 或 document")
        if not texts:
            return []
        if any(not str(text).strip() for text in texts):
            raise ValueError("空文本不能请求 embedding")
        settings = self.settings
        vectors = []
        for start in range(0, len(texts), settings.retrieval_embedding_batch_size):
            batch = list(texts[start:start + settings.retrieval_embedding_batch_size])
            payload = post_json(
                settings.retrieval_embedding_url, settings.dashscope_api_key,
                {"model": settings.retrieval_embedding_model,
                 "input": {"texts": batch},
                 "parameters": {"dimension": settings.retrieval_embedding_dimension,
                                "text_type": text_type, "output_type": "dense"}},
                timeout=settings.retrieval_request_timeout,
                retries=settings.retrieval_max_retries,
                session=self.session, should_cancel=should_cancel,
                kind="embedding",
            )
            self.calls += 1
            self.usage.append(payload.get("usage"))
            from app.agent.execution_budget import record_retrieval_usage

            record_retrieval_usage(payload.get("usage"))
            records = (payload.get("output") or {}).get("embeddings")
            if not isinstance(records, list) or len(records) != len(batch):
                raise RetrievalProviderError("embedding 返回数量与请求不一致")
            ordered = [None] * len(batch)
            for record in records:
                index = record.get("text_index") if isinstance(record, dict) else None
                raw = record.get("embedding") if isinstance(record, dict) else None
                if type(index) is not int or not 0 <= index < len(batch) or ordered[index] is not None:
                    raise RetrievalProviderError("embedding text_index 重复或越界")
                if not isinstance(raw, list) or len(raw) != settings.retrieval_embedding_dimension:
                    raise RetrievalProviderError("embedding 维度错误")
                vector = [finite_number(value) for value in raw]
                norm = math.sqrt(sum(value * value for value in vector))
                if not math.isfinite(norm) or norm == 0:
                    raise RetrievalProviderError("embedding 向量范数无效")
                ordered[index] = [value / norm for value in vector]
            if any(vector is None for vector in ordered):
                raise RetrievalProviderError("embedding 返回索引缺失")
            vectors.extend(ordered)
        return vectors
