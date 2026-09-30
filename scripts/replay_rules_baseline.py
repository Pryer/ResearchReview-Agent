"""在冻结真实快照上重放 rules 硬过滤，核对分母与分语言排除原因（P0）。

不调用任何远程模型，只跑生产规则代码。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools.language_router import detect_paper_language, split_papers_by_language
from app.tools.paper_matching import compile_scope
from app.tools.rank_papers import evaluate_paper_hard_filters


def replay(snapshot_path: str) -> dict:
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    task = snapshot["tasks"][0]
    papers = task["candidate_papers"]

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

    zh, en = split_papers_by_language([dict(p) for p in papers])
    report = {"total": len(papers), "languages": {}}
    for label, group in (("zh", zh), ("en", en)):
        stages = Counter()
        passed_ids = []
        excluded = Counter()
        for paper in group:
            ok, stage, _reason = evaluate_paper_hard_filters(
                paper,
                topic=task.get("topic") or "",
                keywords=task.get("keywords") or [],
                required_concepts=task.get("required_concepts") or [],
                excluded_title_terms=task.get("excluded_title_terms") or [],
                scope=task.get("selected_scope") or {},
                search_branches=task.get("search_branches") or [],
                research_mode=str((task.get("research_semantic_frame") or {}).get("research_mode") or ""),
                screening_protocol=task.get("screening_protocol") or {},
                language_branch=label,
                compiled_scope=compiled,
                ranking_mode="rules",
            )
            stages[stage] += 1
            if ok:
                passed_ids.append(str(paper.get("paper_id") or ""))
            else:
                excluded[stage] += 1
        report["languages"][label] = {
            "candidates": len(group),
            "passed": len(passed_ids),
            "pass_rate": round(len(passed_ids) / len(group), 4) if group else None,
            "excluded_by_stage": dict(excluded),
        }

    # 不区分语言的确定性边界（年份/文献形态），与 hybrid 入口口径对照
    year_out = 0
    for paper in papers:
        year = paper.get("year")
        try:
            year = int(year) if year is not None else None
        except (TypeError, ValueError):
            year = None
        if year and task.get("start_year") and year < int(task["start_year"]):
            year_out += 1
        elif year and task.get("end_year") and year > int(task["end_year"]):
            year_out += 1
    report["year_out_of_range"] = year_out
    report["scope_fingerprint"] = compiled.get("fingerprint")
    report["note"] = (
        "冻结池为会话累积去重候选 889 篇；2026-09-25 23:14 日志中的 562/311→21 "
        "是某一中间轮次快照，日志样本被截断无法完整复原，按计划标记缺失。"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = replay(args.snapshot)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
