"""小候选池的 BM25、dense cosine 与 RRF 排序。"""

from __future__ import annotations

import re
from typing import Sequence

from app.schemas.retrieval_schema import RetrievalQuery


def tokenize(text: str) -> list[str]:
    try:
        import jieba
    except ImportError as exc:
        raise RuntimeError("hybrid 检索需要 jieba；请安装 requirements-retrieval.txt") from exc
    tokens: list[str] = []
    for segment in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*", text.casefold()):
        if re.search(r"[\u4e00-\u9fff]", segment):
            tokens.extend(term for term in jieba.lcut(segment) if term.strip())
        else:
            tokens.append(segment)
            if "-" in segment or "_" in segment:
                tokens.extend(part for part in re.split(r"[-_]", segment) if part)
    return tokens


def lexical_rank(texts: Sequence[str], query: str, limit: int) -> list[tuple[int, float]]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError("hybrid 检索需要 rank-bm25；请安装 requirements-retrieval.txt") from exc
    corpus = [tokenize(text) for text in texts]
    terms = tokenize(query)
    if not terms or not any(corpus):
        return []
    # WHY: 单文档/高频术语可能导致 BM25 原始分数为零或负数；只有真实词项交集才是词法命中。
    scores = BM25Okapi(corpus).get_scores(terms)
    query_terms = set(terms)
    matches = [
        (index, float(scores[index])) for index, words in enumerate(corpus)
        if query_terms.intersection(words)
    ]
    matches.sort(key=lambda item: (-item[1], item[0]))
    return matches[:limit]


def dense_rank(vectors: Sequence[Sequence[float]], query_vector: Sequence[float], limit: int) -> list[tuple[int, float]]:
    scores = [sum(left * right for left, right in zip(vector, query_vector)) for vector in vectors]
    return sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))[:limit]


def fuse_rankings(
    count: int,
    queries: Sequence[RetrievalQuery],
    lexical: Sequence[list[tuple[int, float]]],
    dense: Sequence[list[tuple[int, float]]],
    *,
    rrf_k: int = 60,
) -> list[dict]:
    families = list(dict.fromkeys(query.family for query in queries))
    if not families:
        return []
    combined: dict[int, dict] = {}
    for family in families:
        members = [index for index, query in enumerate(queries) if query.family == family]
        for channel, rankings in (("bm25", lexical), ("dense", dense)):
            best: dict[int, tuple[int, float, str]] = {}
            for query_index in members:
                for rank, (doc_index, raw_score) in enumerate(rankings[query_index], 1):
                    if not 0 <= doc_index < count:
                        raise ValueError("检索排名包含越界论文索引")
                    current = best.get(doc_index)
                    if current is None or rank < current[0]:
                        best[doc_index] = (rank, raw_score, queries[query_index].query_fingerprint)
            for doc_index, (rank, raw_score, query_fp) in best.items():
                record = combined.setdefault(doc_index, {
                    "index": doc_index, "rrf": 0.0, "bm25": None, "cosine": None,
                    "bm25_rank": None, "dense_rank": None,
                    "matched_queries": set(),
                })
                record["rrf"] += 1 / (len(families) * (rrf_k + rank))
                key = "bm25" if channel == "bm25" else "cosine"
                if record[key] is None or raw_score > record[key]:
                    record[key] = raw_score
                rank_key = "bm25_rank" if channel == "bm25" else "dense_rank"
                if record[rank_key] is None or rank < record[rank_key]:
                    record[rank_key] = rank
                record["matched_queries"].add(query_fp)
    result = list(combined.values())
    for record in result:
        record["matched_queries"] = sorted(record["matched_queries"])
    result.sort(key=lambda item: (-item["rrf"], item["index"]))
    return result
