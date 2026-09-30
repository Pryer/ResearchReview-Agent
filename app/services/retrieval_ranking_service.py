"""候选论文的混合排序和可重建向量缓存。"""

from __future__ import annotations

import os
from pathlib import Path
import json
import tempfile
import time
from typing import Any

import numpy as np

from app.clients.embedding_client import EmbeddingClient
from app.clients.reranker_client import RerankerClient
from app.clients.retrieval_http import RetrievalProviderError
from app.schemas.retrieval_schema import RetrievalDocument, RetrievalQuery, fingerprint
from app.tools.hybrid_retrieval import dense_rank, fuse_rankings, lexical_rank
from app.tools.language_router import detect_paper_language
from app.utils.deduplicate import deduplicate_papers


class VectorCache:
    """仅存可重建计算结果；缓存损坏时重新编码。"""

    def __init__(self, directory: str, dimension: int):
        self.directory = Path(directory)
        self.dimension = dimension

    def get(self, key: str):
        path = self.directory / f"{key}.npy"
        try:
            if time.time() - path.stat().st_mtime > 7 * 86400:
                return None
            with path.open("rb") as handle:
                value = np.load(handle, allow_pickle=False)
            if value.shape != (self.dimension,) or value.dtype != np.float32 or not np.isfinite(value).all():
                return None
            norm = float(np.linalg.norm(value))
            if not 0.99 <= norm <= 1.01:
                return None
            return value.astype(float).tolist()
        except (OSError, ValueError, EOFError):
            return None

    def put(self, key: str, vector):
        value = np.asarray(vector, dtype=np.float32)
        if value.shape != (self.dimension,) or not np.isfinite(value).all():
            raise ValueError("无效 embedding 不得写入缓存")
        self.directory.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as handle:
                temp_name = handle.name
                np.save(handle, value, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.directory / f"{key}.npy")
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)


class ScoreCache:
    def __init__(self, directory: str):
        self.directory = Path(directory) / "rerank"

    def get(self, key: str):
        path = self.directory / f"{key}.json"
        try:
            if time.time() - path.stat().st_mtime > 7 * 86400:
                return None
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, bool) and isinstance(value, (int, float)) and np.isfinite(value):
                return float(value)
        except (OSError, ValueError, TypeError):
            return None
        return None

    def put(self, key: str, value: float):
        if not np.isfinite(value):
            raise ValueError("无效 rerank 分数不得写入缓存")
        self.directory.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", mode="w",
                                             encoding="utf-8", delete=False) as handle:
                temp_name = handle.name
                json.dump(value, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.directory / f"{key}.json")
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)


def _document(paper: dict[str, Any]) -> RetrievalDocument:
    from app.agent.nodes.base import _paper_identity_key

    raw_keywords = paper.get("keywords") or []
    if isinstance(raw_keywords, str):
        raw_keywords = [raw_keywords]
    return RetrievalDocument(
        identity=_paper_identity_key(paper),
        title=str(paper.get("title") or ""),
        abstract=str(paper.get("abstract") or ""),
        keywords=tuple(str(value) for value in raw_keywords if str(value).strip()),
        language=str(paper.get("_language_branch") or detect_paper_language(paper)),
    )


def admission_scope_fingerprint(state: dict[str, Any]) -> str:
    """语义准入边界；检索查询和路线调整不改变已确认的逐篇条件。"""
    return fingerprint((
        "semantic_admission.v1",
        state.get("topic"), state.get("selected_scope") or {},
        state.get("research_semantic_frame") or {},
        state.get("screening_protocol") or {},
        required_semantic_conditions(state),
        state.get("start_year"), state.get("end_year"),
        state.get("excluded_title_terms") or [],
    ))


def has_semantic_contract(paper: dict[str, Any]) -> bool:
    """已获得完整逐篇准入结论；内容变化后仍须重新确认。"""
    try:
        confidence = float(paper.get("_screening_confidence") or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        paper.get("_screening_decision") == "include"
        and paper.get("_topic_relation") in {"direct", "near"}
        and paper.get("_eligible_deliverables")
        and paper.get("_semantic_conditions_confirmed") is True
        and paper.get("_semantic_contract_complete") is True
        and confidence >= 0.80
    )


def is_semantically_admitted(
    paper: dict[str, Any], *, scope_fingerprint: str,
    legacy_scope_fingerprint: str | None = None,
) -> bool:
    """完整合同与有效指纹同时成立时才能成为正式证据。"""
    valid_scopes = {scope_fingerprint}
    if legacy_scope_fingerprint:
        valid_scopes.add(legacy_scope_fingerprint)
    return bool(
        has_semantic_contract(paper)
        and paper.get("_screened_content_fingerprint") == _document(paper).content_fingerprint
        and paper.get("_screened_scope_fingerprint") in valid_scopes
    )


def build_queries(state: dict[str, Any], max_queries: int) -> list[RetrievalQuery]:
    """使用研究主题、确认范围与路线生成少量多语言 query family。"""
    frame = state.get("research_semantic_frame") or {}
    topic = str(frame.get("canonical_topic") or state.get("canonical_topic") or state.get("topic") or "").strip()
    queries: list[RetrievalQuery] = []
    seen: set[str] = set()

    def add(text: str, role: str, family: str, source: str):
        value = " ".join(str(text or "").split())
        if value and value.casefold() not in seen and len(queries) < max_queries:
            seen.add(value.casefold())
            queries.append(RetrievalQuery(value, role, family, source))

    scope = state.get("selected_scope") or {}
    context = " ".join(str(value) for value in (scope.get("include_terms") or [])[:3])
    add(" ".join(part for part in (topic, context) if part), "overall", "overall", "semantic_frame")
    for index, branch in enumerate(state.get("search_branches") or []):
        if not isinstance(branch, dict):
            continue
        for query in (branch.get("queries") or [])[:2]:
            add(" ".join(part for part in (topic, str(query)) if part),
                "route", f"route:{index}", "search_branch")
    for keyword in state.get("core_keywords") or state.get("keywords") or []:
        add(str(keyword), "lexical_variant", "overall", "search_plan")
    return queries


def _cache_key(settings, text: str, text_type: str) -> str:
    return fingerprint((settings.retrieval_provider, settings.retrieval_embedding_url,
        settings.retrieval_embedding_model, settings.retrieval_cache_version,
        settings.retrieval_embedding_dimension, text_type, text))


def _embed_cached(client, cache, settings, texts, text_type, should_cancel):
    result = [None] * len(texts)
    missing = []
    for index, text in enumerate(texts):
        key = _cache_key(settings, text, text_type)
        value = cache.get(key)
        if value is None:
            missing.append((index, key, text))
        else:
            result[index] = value
    if missing:
        encoded = client.embed([item[2] for item in missing], text_type=text_type,
                               should_cancel=should_cancel)
        for (index, key, _), vector in zip(missing, encoded):
            cache.put(key, vector)
            result[index] = vector
    return result, len(missing)


def _validate_profile(settings):
    if settings.retrieval_provider != "dashscope":
        raise ValueError("hybrid 目前只支持 dashscope")
    if not settings.dashscope_api_key or not settings.retrieval_embedding_url or not settings.retrieval_rerank_url:
        raise ValueError("hybrid 缺少 DashScope 密钥或模型端点")
    if settings.retrieval_cross_encoder_initial_k > settings.retrieval_cross_encoder_max_k:
        raise ValueError("hybrid 初始精排窗口超过最大窗口")


def _usage_total(usages):
    values = []
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        value = usage.get("total_tokens")
        if isinstance(value, (int, float)):
            values.append(int(value))
    return sum(values) if values else None


def _rerank_cached(client, settings, query: str, documents, should_cancel):
    cache = ScoreCache(settings.retrieval_cache_dir)
    scores = [None] * len(documents)
    missing = []
    for index, text in enumerate(documents):
        key = fingerprint((settings.retrieval_provider, settings.retrieval_rerank_url,
            settings.retrieval_rerank_model, settings.retrieval_cache_version,
            query, text))
        value = cache.get(key)
        if value is None:
            missing.append((index, key, text))
        else:
            scores[index] = value
    if missing:
        fresh = client.score(query, [item[2] for item in missing], should_cancel=should_cancel)
        for (index, key, _), score in zip(missing, fresh):
            cache.put(key, score)
            scores[index] = score
    return scores, len(missing)


def rank_candidates(state, *, settings, include_rerank=False, should_cancel=None,
                    embedding_client=None, reranker_client=None):
    """返回全部融合候选的有界检查窗口及不含原文的诊断。"""
    _validate_profile(settings)
    from app.tools.paper_matching import compile_scope
    from app.tools.rank_papers import evaluate_paper_hard_filters

    papers = deduplicate_papers([dict(item) for item in state.get("candidate_papers") or []])
    compiled_scope = state.get("compiled_scope") or compile_scope(
        selected_scope=state.get("selected_scope") or {},
        semantic_frame=state.get("research_semantic_frame") or {},
        screening_protocol=state.get("screening_protocol") or {},
        required_concepts=state.get("required_concepts") or [],
        topic_anchors=state.get("topic_anchors") or [],
        search_branches=state.get("search_branches") or [],
        excluded_title_terms=state.get("excluded_title_terms") or [],
        topic=state.get("topic") or "",
    )
    retained = []
    exclusions: dict[str, int] = {}
    start_year, end_year = state.get("start_year"), state.get("end_year")
    for paper in papers:
        if should_cancel and should_cancel():
            raise InterruptedError("混合检索已取消")
        year = paper.get("year")
        try:
            numeric_year = int(year) if year is not None else None
        except (TypeError, ValueError):
            numeric_year = None
        if numeric_year is not None and numeric_year <= 0:
            numeric_year = None
        if numeric_year is not None and ((start_year and numeric_year < int(start_year)) or (end_year and numeric_year > int(end_year))):
            exclusions["year"] = exclusions.get("year", 0) + 1
            continue
        passed, stage, _ = evaluate_paper_hard_filters(
            paper, topic=state.get("topic") or "",
            keywords=state.get("keywords") or [],
            required_concepts=state.get("required_concepts") or [],
            excluded_title_terms=state.get("excluded_title_terms") or [],
            scope=state.get("selected_scope") or {},
            search_branches=state.get("search_branches") or [],
            research_mode=str((state.get("research_semantic_frame") or {}).get("research_mode") or ""),
            screening_protocol=state.get("screening_protocol") or {},
            language_branch=str(paper.get("_language_branch") or detect_paper_language(paper)),
            compiled_scope=compiled_scope, ranking_mode="hybrid",
        )
        if not passed:
            exclusions[stage] = exclusions.get(stage, 0) + 1
            continue
        # WHY: 融合排序只有候选资格；旧结论即使来自同一篇论文，也要由本轮
        # 语义筛选按当前内容与范围重新确认，不能从候选字段直接进入证据池。
        paper["_pending_semantic_check"] = True
        paper["_screening_decision"] = "pending_semantic_check"
        paper["_topic_relation"] = "indirect"
        paper["_eligible_deliverables"] = []
        retained.append(paper)
    retained.sort(key=lambda paper: _document(paper).identity)
    documents = [_document(paper) for paper in retained]
    searchable = [(index, doc) for index, doc in enumerate(documents) if doc.text]
    queries = build_queries(state, settings.retrieval_query_max)
    if not queries or not searchable:
        return [], {"candidate_count": len(papers), "deterministic_exclusions": exclusions,
                    "stop_reason": "no_query_or_text"}
    texts = [doc.text for _, doc in searchable]
    lex = [lexical_rank(texts, query.text, settings.retrieval_bm25_top_k) for query in queries]
    dense = [[] for _ in queries]
    degraded = []
    embed = embedding_client or EmbeddingClient(settings)
    cache = VectorCache(settings.retrieval_cache_dir, settings.retrieval_embedding_dimension)
    embedded_new = 0
    try:
        vectors, new_docs = _embed_cached(embed, cache, settings, texts, "document", should_cancel)
        query_vectors, new_queries = _embed_cached(embed, cache, settings,
            [query.text for query in queries], "query", should_cancel)
        embedded_new = new_docs + new_queries
        dense = [dense_rank(vectors, vector, settings.retrieval_dense_top_k)
                 for vector in query_vectors]
    except InterruptedError:
        raise
    except (RetrievalProviderError, OSError, ValueError) as exc:
        degraded.append(f"dense:{type(exc).__name__}")
    fused = fuse_rankings(len(texts), queries, lex, dense, rrf_k=settings.retrieval_rrf_k)
    ranked = []
    for position, item in enumerate(fused, 1):
        source_index = searchable[item["index"]][0]
        paper = retained[source_index]
        paper["_retrieval_features"] = {
            "bm25": item["bm25"], "cosine": item["cosine"],
            "bm25_rank": item["bm25_rank"], "dense_rank": item["dense_rank"],
            "rrf": item["rrf"], "ce": None, "fusion_rank": position,
            "matched_query_fingerprints": item["matched_queries"],
            "content_fingerprint": documents[source_index].content_fingerprint,
        }
        paper["_rank_score"] = 1 / position
        paper["_retrieval_stage"] = "fusion"
        ranked.append(paper)
    fused_count = len(ranked)
    max_window = min(fused_count, settings.retrieval_cross_encoder_max_k)
    protected = []
    for language, minimum in (("zh", getattr(settings, "language_branch_min_zh", 0)),
                              ("en", getattr(settings, "language_branch_min_en", 0))):
        if minimum > 0:
            protected.extend([
                paper for paper in ranked if _document(paper).language == language
            ][:minimum])
    for branch in state.get("search_branches") or []:
        if not isinstance(branch, dict) or branch.get("constraint_level") != "targeted_recovery":
            continue
        branch_type = str(branch.get("branch_type") or "")
        protected.extend([
            paper for paper in ranked
            if branch_type in (paper.get("_search_branches") or [])
        ][:1])
    selected_ids = set()
    window_papers = []
    for paper in [*protected, *ranked]:
        identity = _document(paper).identity
        if identity in selected_ids:
            continue
        selected_ids.add(identity)
        window_papers.append(paper)
        if len(window_papers) >= max_window:
            break
    ranked = window_papers
    rerank = reranker_client or RerankerClient(settings)
    reranked_new = 0
    if include_rerank and max_window:
        window = min(max_window, settings.retrieval_cross_encoder_initial_k)
        try:
            scores, reranked_new = _rerank_cached(rerank, settings, queries[0].text,
                [_document(paper).text for paper in ranked[:window]], should_cancel)
            for paper, score in zip(ranked[:window], scores):
                paper["_retrieval_features"]["ce"] = score
                paper["_retrieval_stage"] = "cross_encoder"
            ranked[:window] = sorted(ranked[:window], key=lambda paper: (
                -paper["_retrieval_features"]["ce"],
                paper["_retrieval_features"]["fusion_rank"],
            ))
        except InterruptedError:
            raise
        except (RetrievalProviderError, OSError, ValueError) as exc:
            degraded.append(f"rerank:{type(exc).__name__}")
    # WHY: 未进入当前窗口的候选仍留在原始 candidate_papers；不得记为已排除。
    window_size = len(ranked)
    for position, paper in enumerate(ranked, 1):
        paper["_rank_score"] = 1 / position
    report = {
        "candidate_count": len(papers), "deterministic_exclusions": exclusions,
        "searchable_count": len(searchable), "fused_count": fused_count,
        "window_count": window_size, "pending_count": window_size,
        "missing_text_count": len(documents) - len(searchable),
        "protected_count": min(len(protected), window_size),
        "embedded_new_count": embedded_new, "embedding_calls": getattr(embed, "calls", None),
        "embedding_total_tokens": _usage_total(getattr(embed, "usage", [])),
        "rerank_calls": getattr(rerank, "calls", None),
        "reranked_new_count": reranked_new,
        "rerank_total_tokens": _usage_total(getattr(rerank, "usage", [])),
        "degraded": degraded,
        "query_fingerprints": [query.query_fingerprint for query in queries],
        "scope_fingerprint": compiled_scope.get("fingerprint"),
    }
    return ranked[:window_size], report


def required_semantic_conditions(state: dict[str, Any]) -> list[dict[str, str]]:
    """仅把用户确认的逐篇条件交给 LLM 作必须满足的判断。"""
    conditions: list[dict[str, str]] = []
    scope = state.get("selected_scope") or {}
    if any(scope.get(key) for key in ("include_terms", "seed_queries", "branches")):
        conditions.append({
            "condition_id": "confirmed_scope",
            "description": str({key: scope.get(key) for key in
                ("include_terms", "seed_queries", "branches") if scope.get(key)}),
        })
    for index, criterion in enumerate((state.get("screening_protocol") or {}).get("hard_include_criteria") or []):
        if not isinstance(criterion, dict) or not criterion.get("applies_to_each_paper", True):
            continue
        if str(criterion.get("source") or "") not in {"user_explicit", "confirmed_scope"}:
            continue
        conditions.append({
            "condition_id": str(criterion.get("criterion_id") or f"protocol_{index}"),
            "description": str(criterion.get("label") or criterion.get("terms")
                or criterion.get("terms_zh") or criterion.get("terms_en") or ""),
        })
    return conditions


def screen_candidates(state, candidates, llm, *, target: int, should_cancel=None):
    """LLM 准入结论是 hybrid 论文进入详情/证据池的唯一出口。"""
    from app.tools.paper_rerank import llm_rerank_papers

    if llm is None:
        return [], {"mode": "pending_no_llm", "confirmed_count": 0,
                    "pending_count": len(candidates)}
    conditions = required_semantic_conditions(state)
    diagnostics: dict[str, Any] = {}
    from app.core.config import get_settings

    settings = get_settings()
    scope_fingerprint = admission_scope_fingerprint(state)
    namespace = fingerprint(("paper_screening.v3", scope_fingerprint,
        settings.llm_provider, settings.llm_model))
    screening_cache = state.setdefault("retrieval_screening_cache", {})
    screened = llm_rerank_papers(
        candidates,
        topic=str(state.get("topic") or ""),
        scope=state.get("selected_scope") or {},
        llm=llm,
        top_k=len(candidates),
        research_mode=str((state.get("research_semantic_frame") or {}).get("research_mode") or ""),
        screening_protocol=state.get("screening_protocol") or {},
        rerank_diagnostics=diagnostics,
        minimum_required=int(state.get("required_reference_count") or 0),
        required_conditions=conditions,
        screening_cache=screening_cache,
        cache_namespace=namespace,
        should_cancel=should_cancel,
    )
    accepted = []
    for paper in screened:
        if has_semantic_contract(paper):
            paper["_pending_semantic_check"] = False
            paper["_screened_content_fingerprint"] = _document(paper).content_fingerprint
            paper["_screened_scope_fingerprint"] = scope_fingerprint
            accepted.append(paper)
    diagnostics["confirmed_count"] = len(accepted)
    diagnostics["pending_or_excluded_count"] = len(candidates) - len(accepted)
    return accepted[:target], diagnostics
