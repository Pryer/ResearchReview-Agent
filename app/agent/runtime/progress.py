"""业务进展摘要与运行内缓存，不依赖调度策略。"""
import hashlib
import copy
import json
from typing import Any
from contextvars import ContextVar
from contextlib import contextmanager
from app.agent.action_registry import admitted_ranked_papers

_PROGRESS_CACHE = ContextVar("progress_digest_cache", default=None)


@contextmanager
def progress_cache_scope():
    if _PROGRESS_CACHE.get() is not None:
        yield
        return
    token = _PROGRESS_CACHE.set({})
    try:
        yield
    finally:
        _PROGRESS_CACHE.reset(token)


def _progress_fingerprint(state: dict[str, Any]) -> str:
    from app.agent.evidence_invalidation import _detail_signature
    def stable_items(name: str) -> list[str]:
        value = state.get(name) or []
        cache = _PROGRESS_CACHE.get()
        previous = cache.get(name, []) if cache is not None else []
        current, hashes = [], []
        for index, item in enumerate(value):
            # WHY: 运行内缓存用独立快照比较，原地修改嵌套Span也会失效；不能仅信任版本号或对象身份。
            if index < len(previous) and previous[index][0] == item:
                saved, digest = previous[index]
            else:
                digest = hashlib.sha256(json.dumps(item, ensure_ascii=False, sort_keys=True,
                    default=str).encode("utf-8")).hexdigest()
                saved = copy.deepcopy(item) if cache is not None else None
            hashes.append(digest)
            if len(value) <= 512:
                current.append((saved, digest))
        if cache is not None:
            cache[name] = current
        # WHY: 内容摘要排序保留集合语义，避免再次JSON转义整段证据原文；缓存不进入持久化状态。
        return sorted(hashes)

    payload = {
        "scope": state.get("selected_scope") or {},
        "candidates": sorted({str(p.get("doi") or p.get("arxiv_id") or p.get("paper_id") or p.get("title")) for p in state.get("candidate_papers") or []}),
        # WHY: 排序抖动不算进展；逐篇语义合同确认后才算新增可处理材料。
        "admitted_ranked": sorted({json.dumps({
            "id": paper.get("paper_id") or paper.get("id") or paper.get("title"),
            "content": paper.get("_screened_content_fingerprint"),
            "scope": paper.get("_screened_scope_fingerprint"),
            "relation": paper.get("_topic_relation"),
            "eligible": sorted(paper.get("_eligible_deliverables") or []),
        }, ensure_ascii=False, sort_keys=True) for paper in admitted_ranked_papers(state)})
        if (state.get("retrieval_profile") or {}).get("mode") == "hybrid" else [],
        "details": sorted(_detail_signature(p) for p in state.get("paper_details") or []),
        "cards": stable_items("paper_cards"),
        "routes": stable_items("validated_routes"),
        "taxonomy": state.get("dynamic_taxonomy") or {},
        "claims": stable_items("claim_plans"),
        "gaps": state.get("evidence_gap_report") or {},
        "review": state.get("review") or "",
        "published_version": state.get("published_draft_version"),
        "quality": state.get("quality_gate") or {},
        # WHY: 完成独立复验是研究进展；同一报告重放不变，仍会被无进展/循环限制。
        "pending_validation": bool(state.get("pending_result_validation")),
        "verification": {key: (state.get("claim_verification") or {}).get(key)
                         for key in ("valid", "supported", "unsupported", "unverified", "support_rate")},
        "valid_citations": state.get("unique_valid_cited_paper_count") or 0,
    }
    cache = _PROGRESS_CACHE.get()
    cached = cache.get("final_payload") if cache is not None else None
    if cached is not None and cached[0] == payload:
        return cached[1]
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:20]
    if cache is not None:
        # WHY: 只复用实际内容相等的进展载荷；独立快照能发现原地质量/范围修改，不持久化最新指纹。
        cache["final_payload"] = (copy.deepcopy(payload), digest)
    return digest
