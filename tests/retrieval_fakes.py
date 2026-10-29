"""离线模型替身；业务检索仍运行 BM25/dense/RRF 和精排，不请求远程端点。"""
import hashlib
import re


def admitted_paper(state, **paper):
    from app.services.retrieval_ranking_service import _document, admission_scope_fingerprint
    return {**paper, "_screening_decision": "include", "_topic_relation": "direct",
            "_eligible_deliverables": ["narrative_review", "research_background", "research_status", "related_work"],
            "_semantic_conditions_confirmed": True, "_semantic_contract_complete": True,
            "_screening_confidence": 0.99, "_pending_semantic_check": False,
            "_screened_content_fingerprint": _document(paper).content_fingerprint,
            "_screened_scope_fingerprint": admission_scope_fingerprint(state)}


class OfflineEmbedding:
    def __init__(self, settings):
        self.dimension = settings.retrieval_embedding_dimension
        self.calls = 0
        self.usage = []

    def embed(self, texts, *, text_type, should_cancel=None):
        self.calls += 1
        result = []
        for text in texts:
            vector = [0.0] * self.dimension
            vector[0] = 1.0
            for word in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower()):
                position = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % self.dimension
                vector[position] += 1
            result.append(vector)
        return result


class OfflineReranker:
    def __init__(self, settings):
        self.calls = 0
        self.usage = []

    def score(self, query, documents, *, should_cancel=None):
        self.calls += 1
        terms = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", query.lower()))
        return [len(terms & set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower()))) / max(1, len(terms))
                for text in documents]
