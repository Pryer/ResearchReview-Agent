"""测量 rules 主题锚点误杀的相关论文被混合检索找回的比例（plan §11.1）。

对评测子集中每篇标注为相关（relevance≥2）但被 rules 主题锚点排除的论文，
检查混合检索（RRF + 专用 rerank）是否将其带回 Top-K，并分中英文报告。
embedding/rerank 走本地缓存指纹，同一快照重复运行不重复计费。

用法：
  python scripts/evaluate_miskill_recovery.py \
    --snapshot data/retrieval_eval/real_merged_snapshot.json \
    --qrels data/retrieval_eval/real_merged_qrels.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.services.retrieval_ranking_service import rank_candidates
from app.tools.language_router import detect_paper_language
from app.tools.paper_matching import compile_scope
from app.tools.rank_papers import evaluate_paper_hard_filters


def evaluate(snapshot_path: str, qrels_path: str, k: int = 120) -> dict:
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    qrels: dict[tuple[str, str], int] = {}
    for line in Path(qrels_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            qrels[(item["task_id"], item["paper_id"])] = int(item["relevance"])

    settings = get_settings().model_copy(update={
        "retrieval_bm25_top_k": k, "retrieval_dense_top_k": k,
        "retrieval_cross_encoder_initial_k": k, "retrieval_cross_encoder_max_k": k,
    })

    summary = {}
    for task in snapshot.get("tasks") or []:
        tid = task["task_id"]
        compiled = compile_scope(
            selected_scope=task.get("selected_scope") or {},
            semantic_frame=task.get("research_semantic_frame") or {},
            screening_protocol=task.get("screening_protocol") or {},
            required_concepts=task.get("required_concepts") or [],
            topic_anchors=task.get("topic_anchors") or [],
            search_branches=task.get("search_branches") or [],
            excluded_title_terms=task.get("excluded_title_terms") or [],
            topic=task.get("topic") or "",
        )
        killed_relevant = []
        for paper in task["candidate_papers"]:
            if qrels.get((tid, paper["paper_id"]), 0) < 2:
                continue
            language = detect_paper_language(paper)
            passed, stage, _ = evaluate_paper_hard_filters(
                paper, topic=task.get("topic") or "",
                keywords=task.get("keywords") or [],
                required_concepts=task.get("required_concepts") or [],
                excluded_title_terms=task.get("excluded_title_terms") or [],
                scope=task.get("selected_scope") or {},
                search_branches=task.get("search_branches") or [],
                screening_protocol=task.get("screening_protocol") or {},
                language_branch=language, compiled_scope=compiled,
                ranking_mode="rules",
            )
            if not passed and stage == "topic_anchor_filter":
                killed_relevant.append((paper["paper_id"], language))
        state = dict(task)
        ranked, _ = rank_candidates(state, settings=settings, include_rerank=True)
        top_k = {paper["paper_id"] for paper in ranked[:k]}
        recovered = [item for item in killed_relevant if item[0] in top_k]
        zh_killed = [item for item in killed_relevant if item[1] == "zh"]
        zh_recovered = [item for item in recovered if item[1] == "zh"]
        summary[tid] = {
            "anchor_killed_relevant": len(killed_relevant),
            "recovered_in_top_k": len(recovered),
            "recovery_rate": round(len(recovered) / len(killed_relevant), 4)
            if killed_relevant else None,
            "zh_killed": len(zh_killed), "zh_recovered": len(zh_recovered),
            "zh_recovery_rate": round(len(zh_recovered) / len(zh_killed), 4)
            if zh_killed else None,
        }
    return {"k": k, "tasks": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--k", type=int, default=120)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate(args.snapshot, args.qrels, args.k)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
