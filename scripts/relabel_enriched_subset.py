"""对补全摘要后信息发生变化的论文重新做独立银标，其余标签保持不变。

只重标 provenance 中 title_only=true 且当前快照已有摘要的论文；这样评测集
的论文身份集合不变（可与上一轮直接对比），仅让此前"仅标题猜测"的标签在
看到摘要后更新。会话真实 LLM 标签一律不动。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_real_eval_subset import label_papers

TASK_IDS = ("real_classroom", "real_rag_hallucination", "real_battery_recycling")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--snapshot-out", required=True)
    parser.add_argument("--qrels-out", required=True)
    parser.add_argument("--provenance-out", required=True)
    args = parser.parse_args()

    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    provenance = {}
    for line in Path(args.provenance).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            provenance[item["paper_id"]] = item
    old_labels = {}
    for line in Path(args.qrels).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            old_labels[(item["task_id"], item["paper_id"])] = int(item["relevance"])

    from app.services.llm_service import LLMService
    llm = LLMService()
    changes = []
    for task in snapshot["tasks"]:
        to_relabel = []
        for paper in task["candidate_papers"]:
            pid = str(paper["paper_id"])
            record = provenance.get(pid) or {}
            if (record.get("title_only")
                    and record.get("label_source") == "silver_llm"
                    and str(paper.get("abstract") or "").strip()):
                to_relabel.append(paper)
        if not to_relabel:
            continue
        fresh_labels, fresh_sources = label_papers(to_relabel, task, llm)
        for paper in to_relabel:
            pid = str(paper["paper_id"])
            new_value = fresh_labels[pid]
            old_value = old_labels[(task["task_id"], pid)]
            if new_value != old_value:
                changes.append({"task_id": task["task_id"], "paper_id": pid,
                                "old": old_value, "new": new_value})
            provenance[pid] = {
                "paper_id": pid, "label_source": "silver_llm_relabeled_after_enrichment",
                "relevance": new_value, "title_only": False,
                "language": provenance[pid].get("language"),
            }

    qrels_lines = []
    for task in snapshot["tasks"]:
        for paper in task["candidate_papers"]:
            pid = str(paper["paper_id"])
            qrels_lines.append(json.dumps({
                "task_id": task["task_id"], "paper_id": pid,
                "relevance": provenance[pid]["relevance"],
                "language": provenance[pid].get("language"),
            }, ensure_ascii=False))
    Path(args.snapshot_out).write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.qrels_out).write_text("\n".join(qrels_lines), encoding="utf-8")
    Path(args.provenance_out).write_text(
        "\n".join(json.dumps(provenance[pid], ensure_ascii=False)
                  for pid in provenance), encoding="utf-8")
    print(json.dumps({"relabeled_changes": len(changes), "changes": changes[:20]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
