"""把人工复核队列导出为 Excel 可直接打开的 CSV（utf-8-sig）。

你只需要填写 human_relevance 列：3=核心就是该主题，2=明确相关可引用，
1=仅背景/邻接，0=不相关；拿不准留空（不计入测试集）。
不要修改 paper_id / task_id 列。

用法：
  python scripts/export_human_review_sheet.py \
    --snapshot data/retrieval_eval/real_merged_snapshot_enriched.json \
    --qrels data/retrieval_eval/real_merged_qrels.jsonl \
    --output data/retrieval_eval/human_review_sheet.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOPIC_LABELS = {
    "real_classroom": "课堂行为分析",
    "real_rag_hallucination": "RAG幻觉",
    "real_battery_recycling": "锂电池回收",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--queue", nargs="*", default=[],
        help="可选：旧 human_review JSON 文件，只导出其中 paper_id 的复核队列",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[],
        help="可选：已标注 qrels/jsonl，其中 paper_id 从本表排除（用于第二批）",
    )
    args = parser.parse_args()

    queue_ids: set[tuple[str, str]] = set()
    for queue_file in args.queue:
        for item in json.loads(Path(queue_file).read_text(encoding="utf-8")):
            queue_ids.add((item["task_id"], item["paper_id"]))
    excluded_ids: set[tuple[str, str]] = set()
    for exclude_file in args.exclude:
        for line in Path(exclude_file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                excluded_ids.add((item["task_id"], item["paper_id"]))

    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    papers: dict[tuple[str, str], dict] = {}
    for task in snapshot["tasks"]:
        for paper in task["candidate_papers"]:
            key = (task["task_id"], str(paper["paper_id"]))
            if key in excluded_ids:
                continue
            if not queue_ids or key in queue_ids:
                papers[key] = paper
    silver: dict[tuple[str, str], int] = {}
    for line in Path(args.qrels).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            key = (item["task_id"], item["paper_id"])
            if key in papers:
                silver[key] = int(item["relevance"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "task_id", "topic", "paper_id", "silver_relevance",
            "title", "abstract", "year", "venue", "human_relevance", "notes",
        ])
        # 优先排列：中文（@120 中文门槛关键）→ 银标相关分 → 边界分 → 无摘要。
        def order(key):
            from app.tools.language_router import detect_paper_language

            task_id, paper_id = key
            paper = papers[key]
            zh_first = 0 if detect_paper_language(paper) == "zh" else 1
            borderline = 0 if silver[key] in (1, 2) else 1
            no_abstract = 0 if str(paper.get("abstract") or "").strip() else 1
            return (task_id, zh_first, -silver[key], borderline, -no_abstract, paper_id)

        for task_id, paper_id in sorted(papers, key=order):
            paper = papers[(task_id, paper_id)]
            writer.writerow([
                task_id, TOPIC_LABELS.get(task_id, task_id), paper_id,
                silver[(task_id, paper_id)],
                str(paper.get("title") or "").replace("\n", " "),
                str(paper.get("abstract") or "").replace("\n", " ")[:1500],
                paper.get("year") or "", paper.get("venue") or "",
                "", "",
            ])
    print({"output": str(output), "rows": len(papers)})


if __name__ == "__main__":
    main()
