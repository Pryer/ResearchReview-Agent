"""用人工标签构造封闭测试快照（只用人工标注过的论文）。

每主题候选池收窄为 human qrels 覆盖的论文，排名在同一封闭池上重算，
保证 Recall 分母确定、五模式公平可比。K 必须不大于最小主题池。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--human-qrels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    labeled: dict[str, set[str]] = defaultdict(set)
    for line in Path(args.human_qrels).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            labeled[item["task_id"]].add(item["paper_id"])

    tasks = []
    for task in snapshot["tasks"]:
        keep = labeled.get(task["task_id"], set())
        task_out = {key: value for key, value in task.items() if key != "candidate_papers"}
        task_out["candidate_papers"] = [
            paper for paper in task["candidate_papers"]
            if str(paper.get("paper_id")) in keep
        ]
        tasks.append(task_out)

    out = {**{k: v for k, v in snapshot.items() if k != "tasks"},
           "dataset_kind": "human_labeled_closed_test_set", "tasks": tasks}
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({task["task_id"]: len(task["candidate_papers"]) for task in tasks},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
