"""为评测采集新主题的真实候选（P0 三主题要求的第二、第三主题）。

只访问无需登录的开放来源（crossref/openalex/semantic_scholar）。
不使用 CNKI（需要机构登录，AGENTS.md 禁止绕过）；因此这两个主题以英文
候选为主，中文覆盖能力仍由真实课堂会话主题检验。

多组查询覆盖整体问题与不同侧面，召回噪声（书目章节、邻近主题）保留为
天然难负例，后续由独立标注判定，不在这里做相关性裁剪。

用法：
  python scripts/collect_live_eval_topic.py --config data/retrieval_eval/live_topics/rag.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools.search_papers import search_papers
from app.utils.deduplicate import deduplicate_papers

OPEN_SOURCES = ["crossref", "openalex", "semantic_scholar"]


def collect(config: dict) -> dict:
    queries = config["queries"]
    start_year, end_year = int(config["start_year"]), int(config["end_year"])
    per_query = int(config.get("max_results_per_query", 50))
    all_papers: list[dict] = []
    query_log = []
    for index, query in enumerate(queries):
        diagnostics = []
        found = search_papers(
            query=query, start_year=start_year, end_year=end_year,
            max_results=per_query, sources=OPEN_SOURCES, diagnostics=diagnostics,
        )
        papers = [p.model_dump() if hasattr(p, "model_dump") else dict(p) for p in found]
        query_log.append({
            "query": query, "returned": len(papers),
            "source_diagnostics": [
                d.model_dump() if hasattr(d, "model_dump") else d for d in diagnostics
            ],
        })
        all_papers.extend(papers)
        if index < len(queries) - 1:
            time.sleep(float(config.get("inter_query_seconds", 3)))

    deduped = deduplicate_papers([dict(p) for p in all_papers])
    task = {
        "task_id": config["task_id"],
        "topic": config["topic"],
        "keywords": config.get("keywords", []),
        "required_concepts": config.get("required_concepts", []),
        "topic_anchors": config.get("required_concepts", []),
        "excluded_title_terms": config.get("excluded_title_terms", []),
        "selected_scope": {},
        "search_branches": [],
        "screening_protocol": {},
        "start_year": start_year,
        "end_year": end_year,
        "candidate_papers": deduped,
    }
    return {
        "dataset_kind": "real_open_source_live_search_frozen",
        "exported_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": {
            "sources": OPEN_SOURCES,
            "queries": query_log,
            "raw_total": len(all_papers),
            "deduped_candidate_count": len(deduped),
            "note": "实时开放来源检索；arxiv 当日 406 未纳入；无 CNKI（需机构登录）。",
        },
        "tasks": [task],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    snapshot = collect(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    task = snapshot["tasks"][0]
    print(json.dumps({
        "output": str(output),
        "candidates": len(task["candidate_papers"]),
        "with_abstract": sum(1 for p in task["candidate_papers"] if str(p.get("abstract") or "").strip()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
