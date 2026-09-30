"""按 plan.md 第 11 节门槛汇总真实评测报告。

输出每个模式 × 主题 × 语言在 @72/@120 的 Recall/Precision/nDCG，
宏平均，以及与 rules 基线的对比和门槛判定。判定结果只是对当前标注集
（银标 dev / 人工 test）的结论，不替代最终验收。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODES = ("rules", "bm25", "dense", "rrf", "rrf_ce")
DEFAULT_CUTS = ("72", "120")


def _average(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None


def analyze(report: dict) -> dict:
    tasks = report.get("tasks") or []
    available_cuts = sorted({
        cut for task in tasks for mode in MODES
        for cut in task["rankings"][mode]["overall_by_k"]
    }, key=int)
    cuts = tuple(available_cuts) or DEFAULT_CUTS
    table = {}
    for mode in MODES:
        table[mode] = {}
        for cut in cuts:
            overall, zh, rec_all, prec_all, ndcg_all = [], [], [], [], []
            for task in tasks:
                metrics = task["rankings"][mode]
                overall_metrics = metrics["overall_by_k"][cut]
                rec_all.append(overall_metrics["recall"])
                prec_all.append(overall_metrics["precision"])
                ndcg_all.append(overall_metrics["ndcg"])
                zh_metrics = metrics["by_language"].get("zh", {}).get(cut)
                if zh_metrics:
                    zh.append(zh_metrics["recall"])
                    overall.append(overall_metrics["recall"])
            table[mode][cut] = {
                "macro_recall": _average(rec_all),
                "macro_precision": _average(prec_all),
                "macro_ndcg": _average(ndcg_all),
                "macro_recall_zh": _average(zh),
                "per_task_recall": {
                    task["task_id"]: task["rankings"][mode]["overall_by_k"][cut]["recall"]
                    for task in tasks
                },
            }

    rules, hybrid = table["rules"], table["rrf_ce"]

    def _ge(cut, key):
        return cut in hybrid and cut in rules and (hybrid[cut][key] or 0) >= (rules[cut][key] or 0)

    gates = {
        "precision_hybrid_ge_rules": {
            cut: _ge(cut, "macro_precision") for cut in cuts},
        "ndcg_hybrid_ge_rules": {
            cut: _ge(cut, "macro_ndcg") for cut in cuts},
        "recall_hybrid_ge_rules": {
            cut: _ge(cut, "macro_recall") for cut in cuts},
        "recall_hybrid_ge_rules_zh": {
            cut: _ge(cut, "macro_recall_zh") for cut in cuts},
    }
    if "120" in cuts:
        gates.update({
            "recall120_hybrid_ge_rules_overall":
                hybrid["120"]["macro_recall"] >= rules["120"]["macro_recall"],
            "recall120_hybrid_ge_rules_zh":
                (hybrid["120"]["macro_recall_zh"] or 0)
                >= (rules["120"]["macro_recall_zh"] or 0),
            "recall120_target_0.95_overall":
                (hybrid["120"]["macro_recall"] or 0) >= 0.95,
            "recall120_target_0.95_zh":
                (hybrid["120"]["macro_recall_zh"] or 0) >= 0.95,
        })
    gate_values = [value for value in gates.values() if isinstance(value, bool)]
    gate_values += [all(value.values()) for value in gates.values()
                    if isinstance(value, dict)]
    return {"cuts": cuts, "table": table, "gates": gates,
            "all_available_gates_pass": all(gate_values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    result = analyze(report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
