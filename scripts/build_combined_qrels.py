"""合并标签全集：人工标签优先，未人工标注项保留银标（480 篇 @120 口径）。

并输出人工封闭集中 rrf+ce 未召回的中文相关论文及其在各模式下的排名位置。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.services.retrieval_ranking_service import rank_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--silver-qrels", required=True)
    parser.add_argument("--human-qrels", required=True)
    parser.add_argument("--combined-qrels-out", required=True)
    args = parser.parse_args()

    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    silver = {}
    for line in Path(args.silver_qrels).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            silver[(item["task_id"], item["paper_id"])] = item
    human = {}
    for line in Path(args.human_qrels).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            human[(item["task_id"], item["paper_id"])] = item

    combined_lines = []
    human_count = 0
    for key, item in silver.items():
        chosen = human.get(key, item)
        if key in human:
            human_count += 1
        combined_lines.append(json.dumps(chosen, ensure_ascii=False))
    Path(args.combined_qrels_out).write_text("\n".join(combined_lines), encoding="utf-8")

    # 定位课堂封闭池中 rrf+ce @72 未召回的中文相关论文
    settings = get_settings().model_copy(update={
        "retrieval_bm25_top_k": 120, "retrieval_dense_top_k": 120,
        "retrieval_cross_encoder_initial_k": 120, "retrieval_cross_encoder_max_k": 120,
    })
    missing = []
    for task in snapshot["tasks"]:
        tid = task["task_id"]
        state = dict(task)
        ranked, _ = rank_candidates(state, settings=settings, include_rerank=True)
        positions = {str(p["paper_id"]): index + 1 for index, p in enumerate(ranked)}
        for paper in task["candidate_papers"]:
            key = (tid, str(paper["paper_id"]))
            label = human.get(key)
            if not label or label["language"] != "zh" or label["relevance"] < 2:
                continue
            position = positions.get(str(paper["paper_id"]))
            if position is None or position > 72:
                missing.append({
                    "task_id": tid, "paper_id": paper["paper_id"],
                    "human_relevance": label["relevance"],
                    "rrf_ce_position_in_closed_pool": position,
                    "pool_size": len(ranked),
                    "title": str(paper.get("title") or "")[:100],
                    "has_abstract": bool(str(paper.get("abstract") or "").strip()),
                })
    print(json.dumps({"combined_qrels": len(combined_lines),
                      "human_labels_used": human_count,
                      "human_zh_relevant_missing_top72": missing}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
