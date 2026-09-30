"""回收人工标注 CSV：生成人工测试 qrels，并报告银标-人工一致性。

- human_relevance 只接受空、0、1、2、3；非法值报错并指出行号。
- 已填项进入 test qrels；留空项不计入测试集（但保留银标 dev 不变）。
- 同时输出逐主题的银标/人工分布、总体一致率、线性加权 kappa 和分歧清单，
  用于判断银标是否可靠。

用法：
  python scripts/import_human_review_sheet.py \
    --sheet data/retrieval_eval/human_review_sheet_filled.csv \
    --qrels-out data/retrieval_eval/human_test_qrels.jsonl \
    --report-out data/retrieval_eval/human_agreement.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABELS = {0, 1, 2, 3}


def weighted_kappa(pairs: list[tuple[int, int]]) -> float | None:
    """线性加权 Cohen's kappa（0-3 有序标签）。"""
    if not pairs:
        return None
    categories = sorted(LABELS)
    n = len(pairs)
    silver_counts = Counter(silver for silver, _ in pairs)
    human_counts = Counter(human for _, human in pairs)
    observed_disagreement = sum(abs(silver - human) / 3 for silver, human in pairs) / n
    expected_disagreement = 0.0
    for s_label in categories:
        for h_label in categories:
            expected_disagreement += (
                silver_counts[s_label] / n * human_counts[h_label] / n
                * abs(s_label - h_label) / 3
            )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return round(1 - observed_disagreement / expected_disagreement, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--qrels-out", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.sheet).read_text(encoding="utf-8-sig").splitlines()))
    test_records = []
    pairs = []
    by_topic: dict[str, dict] = defaultdict(lambda: {
        "filled": 0, "silver": Counter(), "human": Counter(),
        "exact_agree": 0, "within_one": 0, "pairs": [],
    })
    disagreements = []
    for line_number, row in enumerate(rows, start=2):
        task_id, paper_id = row["task_id"].strip(), row["paper_id"].strip()
        raw = (row.get("human_relevance") or "").strip()
        if raw == "":
            continue
        if not raw.isdigit() or int(raw) not in LABELS:
            raise SystemExit(f"第 {line_number} 行 human_relevance 非法：{raw!r}（只允许空或 0-3）")
        human = int(raw)
        silver = int(row["silver_relevance"])
        from app.tools.language_router import detect_paper_language
        language = detect_paper_language({"title": row.get("title"), "abstract": row.get("abstract")})
        test_records.append({
            "task_id": task_id, "paper_id": paper_id,
            "relevance": human, "language": language,
        })
        pairs.append((silver, human))
        topic = by_topic[task_id]
        topic["filled"] += 1
        topic["silver"][silver] += 1
        topic["human"][human] += 1
        topic["exact_agree"] += int(silver == human)
        topic["within_one"] += int(abs(silver - human) <= 1)
        topic["pairs"].append((silver, human))
        if silver != human:
            disagreements.append({
                "task_id": task_id, "paper_id": paper_id,
                "silver": silver, "human": human,
                "title": (row.get("title") or "")[:120],
            })

    Path(args.qrels_out).write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in test_records),
        encoding="utf-8")

    topic_report = {}
    for task_id, topic in by_topic.items():
        topic_report[task_id] = {
            "filled": topic["filled"],
            "exact_agreement": round(topic["exact_agree"] / topic["filled"], 4),
            "within_one_agreement": round(topic["within_one"] / topic["filled"], 4),
            "weighted_kappa": weighted_kappa(topic["pairs"]),
            "human_distribution": dict(sorted(topic["human"].items())),
        }
    report = {
        "rows_total": len(rows), "filled": len(test_records),
        "overall_exact_agreement": round(
            sum(s == h for s, h in pairs) / len(pairs), 4) if pairs else None,
        "overall_within_one": round(
            sum(abs(s - h) <= 1 for s, h in pairs) / len(pairs), 4) if pairs else None,
        "overall_weighted_kappa": weighted_kappa(pairs),
        "by_topic": topic_report,
        "disagreements": disagreements,
    }
    Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
