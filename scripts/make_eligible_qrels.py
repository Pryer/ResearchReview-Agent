"""把确定性元数据边界（学位论文等）从 Recall 分母中剔除，生成 eligible qrels。

主题相关但文献形态不合格的论文不是任何检索模式的合法召回目标
（plan §3.1/§3.2：语义检索不改变学位论文排除政策）。被剔除项单独
输出计数，保证口径可审计，不删除原始 qrels。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools.paper_matching import compile_scope
from app.tools.rank_papers import evaluate_paper_hard_filters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    papers: dict[tuple[str, str], dict] = {}
    compiled_by_task = {}
    for task in snapshot["tasks"]:
        compiled_by_task[task["task_id"]] = compile_scope(
            selected_scope=task.get("selected_scope") or {},
            semantic_frame=task.get("research_semantic_frame") or {},
            screening_protocol=task.get("screening_protocol") or {},
            required_concepts=task.get("required_concepts") or [],
            topic_anchors=task.get("topic_anchors") or [],
            search_branches=task.get("search_branches") or [],
            excluded_title_terms=task.get("excluded_title_terms") or [],
            topic=task.get("topic") or "",
        )
        for paper in task["candidate_papers"]:
            papers[(task["task_id"], str(paper["paper_id"]))] = (
                task, paper)

    kept, removed = [], []
    for line in Path(args.qrels).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = (item["task_id"], item["paper_id"])
        task, paper = papers[key]
        compiled = compiled_by_task[item["task_id"]]
        passed, stage, _ = evaluate_paper_hard_filters(
            paper, topic=task.get("topic") or "",
            keywords=task.get("keywords") or [],
            required_concepts=task.get("required_concepts") or [],
            excluded_title_terms=task.get("excluded_title_terms") or [],
            scope=task.get("selected_scope") or {},
            search_branches=task.get("search_branches") or [],
            screening_protocol=task.get("screening_protocol") or {},
            language_branch=item.get("language") or "zh",
            compiled_scope=compiled, ranking_mode="hybrid",
        )
        if passed:
            kept.append(item)
        else:
            removed.append({**item, "excluded_stage": stage})

    Path(args.output).write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in kept),
        encoding="utf-8")
    print(json.dumps({"kept": len(kept), "deterministic_excluded": len(removed),
                      "by_stage": _count(removed, "excluded_stage"),
                      "excluded_relevant": sum(1 for x in removed if x["relevance"] >= 2)},
                     ensure_ascii=False))


def _count(items, key):
    counts = {}
    for item in items:
        counts[item[key]] = counts.get(item[key], 0) + 1
    return counts


if __name__ == "__main__":
    main()
