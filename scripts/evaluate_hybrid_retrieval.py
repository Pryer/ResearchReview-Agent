"""在完整标注的冻结候选池上对照规则、单路与混合检索。

输入 snapshot.json: {"tasks": [{"task_id", "topic", "candidate_papers", ...}]}
输入 qrels.jsonl: 每行 {"task_id", "paper_id", "relevance": 0..3, "language"}
输入 profile.json: {"k": 正整数, "split": "dev" 或 "test"}
输出只含指标/哈希/配置，不复制私人论文文本。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.services.retrieval_ranking_service import rank_candidates
from app.tools.rank_papers import deduplicate_and_rank


def _metrics(ranking, labels, k, relevance_threshold=1):
    """relevance_threshold=1 时 1 分（仅背景）也算相关（旧口径，向后兼容）；
    threshold=2 时只有可作为证据的 2/3 分算相关（plan 正式门槛口径）。"""
    def relevant_value(value):
        return value >= relevance_threshold

    ranked = ranking[:k]
    relevant = sum(relevant_value(value) for value in labels.values())
    hits = sum(relevant_value(labels.get(paper_id, 0)) for paper_id in ranked)
    gain = sum((2 ** max(0, labels.get(paper_id, 0) - (relevance_threshold - 1)) - 1)
               / math.log2(index + 2) for index, paper_id in enumerate(ranked))
    ideal_grades = sorted(
        (max(0, value - (relevance_threshold - 1)) for value in labels.values()),
        reverse=True)[:k]
    ideal = sum((2 ** value - 1) / math.log2(index + 2)
                for index, value in enumerate(ideal_grades))
    return {"recall": hits / relevant if relevant else None,
            "precision": hits / k,
            "ndcg": gain / ideal if ideal else None,
            "relevant": relevant, "retrieved": len(ranked),
            "relevance_threshold": relevance_threshold}


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def evaluate(snapshot_path, qrels_path, profile_path, allow_subset_qrels=False,
             relevance_threshold=1):
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    k = int(profile["k"])
    if k <= 0 or k > 240 or profile.get("split") not in {"dev", "test"}:
        raise ValueError("profile 需要 split=dev/test 与 1..240 的 k")
    # 同一份排名可在多个截断点上计算指标（如 Recall@120 与 Precision/nDCG@72），
    # 不产生任何额外模型调用。
    metric_ks = sorted({int(value) for value in profile.get("metric_ks", [k])})
    if any(value <= 0 or value > 240 for value in metric_ks):
        raise ValueError("metric_ks 必须是 1..240 的正整数")
    qrels = defaultdict(dict)
    languages = defaultdict(dict)
    for line in Path(qrels_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        task_id, paper_id = str(item["task_id"]), str(item["paper_id"])
        relevance = int(item["relevance"])
        if relevance not in range(4) or paper_id in qrels[task_id]:
            raise ValueError("qrels 存在非法相关性等级或重复论文")
        qrels[task_id][paper_id] = relevance
        languages[task_id][paper_id] = str(item.get("language") or "unknown")
    settings = get_settings()
    settings = settings.model_copy(update={
        "retrieval_bm25_top_k": max(k, settings.retrieval_bm25_top_k),
        "retrieval_dense_top_k": max(k, settings.retrieval_dense_top_k),
        "retrieval_cross_encoder_initial_k": max(k, settings.retrieval_cross_encoder_initial_k),
        "retrieval_cross_encoder_max_k": max(k, settings.retrieval_cross_encoder_max_k),
    })
    results = []
    durations = []
    for task in snapshot.get("tasks") or []:
        task_id = str(task["task_id"])
        papers = list(task.get("candidate_papers") or [])
        ids = [str(paper.get("paper_id") or "") for paper in papers]
        if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("snapshot 每篇论文必须有唯一 paper_id")
        qrel_ids = set(qrels[task_id])
        if not allow_subset_qrels and set(ids) != qrel_ids:
            raise ValueError("qrels 必须覆盖候选池中的每一篇论文")
        if allow_subset_qrels and not qrel_ids <= set(ids):
            raise ValueError("子集 qrels 中的 paper_id 必须都在候选池中")
        state = dict(task)
        state["candidate_papers"] = papers
        t0 = time.monotonic()
        rules = deduplicate_and_rank(
            [dict(paper) for paper in papers], str(task.get("topic") or ""),
            top_k=len(papers), keywords=task.get("keywords") or [],
            required_concepts=task.get("required_concepts") or [],
            excluded_title_terms=task.get("excluded_title_terms") or [],
            scope=task.get("selected_scope") or {},
            search_branches=task.get("search_branches") or [],
            screening_protocol=task.get("screening_protocol") or {},
        )
        fusion, fusion_report = rank_candidates(state, settings=settings, include_rerank=False)
        ce, ce_report = rank_candidates(state, settings=settings, include_rerank=True)
        durations.append(time.monotonic() - t0)
        by_id = lambda collection: [str(paper.get("paper_id")) for paper in collection]
        rankings = {
            "rules": by_id(rules),
            "bm25": by_id(sorted((p for p in fusion if p["_retrieval_features"]["bm25"] is not None),
                                   key=lambda p: (p["_retrieval_features"]["bm25_rank"],
                                                  -p["_retrieval_features"]["bm25"]))),
            "dense": by_id(sorted((p for p in fusion if p["_retrieval_features"]["cosine"] is not None),
                                   key=lambda p: (p["_retrieval_features"]["dense_rank"],
                                                  -p["_retrieval_features"]["cosine"]))),
            "rrf": by_id(fusion), "rrf_ce": by_id(ce),
        }
        task_result = {"task_id": task_id, "candidate_count": len(papers),
                       "rankings": {}, "degraded": sorted(set(fusion_report["degraded"] + ce_report["degraded"]))}
        for mode, ranking in rankings.items():
            metrics_by_k = {str(cut): _metrics(ranking, qrels[task_id], cut,
                                               relevance_threshold)
                            for cut in metric_ks}
            by_language = {}
            for language in sorted(set(languages[task_id].values())):
                subset = {paper_id: value for paper_id, value in qrels[task_id].items()
                          if languages[task_id][paper_id] == language}
                by_language[language] = {
                    str(cut): _metrics(ranking, subset, cut, relevance_threshold)
                    for cut in metric_ks
                }
            # 向后兼容：overall 仍给出主 K（profile.k）指标，另给 overall_by_k。
            task_result["rankings"][mode] = {
                "overall": metrics_by_k[str(k)],
                "overall_by_k": metrics_by_k,
                "by_language": by_language,
            }
        results.append(task_result)
    return {
        "input_sha256": {"snapshot": _hash(snapshot_path), "qrels": _hash(qrels_path),
                         "profile": _hash(profile_path)},
        "profile": profile, "embedding_model": settings.retrieval_embedding_model,
        "rerank_model": settings.retrieval_rerank_model,
        "model_revision": None, "cache_version": settings.retrieval_cache_version,
        "task_count": len(results), "task_duration_seconds": durations,
        "tasks": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-subset-qrels", action="store_true",
                        help="允许 qrels 只覆盖合格论文子集（如剔除学位论文后）")
    parser.add_argument("--relevance-threshold", type=int, default=1, choices=(1, 2),
                        help="相关阈值：1=含背景级（旧口径），2=仅可引用证据（正式门槛）")
    args = parser.parse_args()
    report = evaluate(args.snapshot, args.qrels, args.profile,
                      allow_subset_qrels=args.allow_subset_qrels,
                      relevance_threshold=args.relevance_threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"tasks": report["task_count"], "output_written": True,
           "degraded_tasks": sum(bool(task["degraded"]) for task in report["tasks"])})


if __name__ == "__main__":
    main()
