"""从研究资料库导出真实候选评测快照（P0）。

把指定会话外置在 research_artifacts 中的 raw_search_result 候选逐篇还原、
按生产口径去重，并附上当时运行态中的研究范围、筛选协议、语义框架、检索
分支、年份和关键词，生成 evaluate_hybrid_retrieval.py 可消费的冻结快照。

只导出本地数据库中已有的真实检索记录，不发起任何新检索。
候选论文是用户自己的研究资料，仅写入本地 data/ 目录（已被 git 忽略）。

用法：
  python scripts/export_real_eval_snapshot.py \
    --session 4a7275152ca549f8871c5573fa4bdd6c \
    --task-id real_classroom --output data/retrieval_eval/real_classroom_snapshot.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.deduplicate import deduplicate_papers

DB_PATH = "data/research_review.db"

# rank_candidates/evaluate 脚本读取的研究范围字段；全部来自当时运行态，
# 不用今天的默认值或 LLM 猜测重建（AGENTS.md：未知保持 None）。
SCOPE_FIELDS = (
    "topic", "canonical_topic", "keywords", "core_keywords",
    "required_concepts", "topic_anchors", "excluded_title_terms",
    "selected_scope", "screening_protocol", "research_semantic_frame",
    "search_branches", "start_year", "end_year",
    "required_reference_count", "retrieval_target", "max_papers",
)


def _load_session_artifacts(con, session_id: str) -> tuple[list[dict], dict]:
    rows = con.execute(
        "SELECT payload_json, provenance_json, created_at FROM research_artifacts "
        "WHERE session_id=? AND artifact_type='raw_search_result' ORDER BY created_at",
        (session_id,),
    ).fetchall()
    papers: list[dict] = []
    fetched = []
    for payload_json, provenance_json, created_at in rows:
        value = json.loads(payload_json).get("value")
        if isinstance(value, dict):
            papers.append(value)
            fetched.append(created_at)
    return papers, {
        "artifact_count": len(rows),
        "earliest_fetched_at": fetched[0] if fetched else None,
        "latest_fetched_at": fetched[-1] if fetched else None,
    }


def _load_runtime_state(con, session_id: str) -> dict:
    row = con.execute(
        "SELECT state_json FROM research_runtime WHERE session_id=?", (session_id,)
    ).fetchone()
    return json.loads(row[0]) if row else {}


def _load_detail_decisions(con, session_id: str) -> dict[str, dict]:
    """会话当时已做过详情补全的 60/120 篇及其真实 LLM 筛选结论。"""
    decisions: dict[str, dict] = {}
    rows = con.execute(
        "SELECT payload_json FROM research_artifacts "
        "WHERE session_id=? AND artifact_type='paper_metadata'",
        (session_id,),
    ).fetchall()
    for (payload_json,) in rows:
        value = json.loads(payload_json).get("value")
        if not isinstance(value, dict):
            continue
        paper_id = str(value.get("paper_id") or "")
        if paper_id:
            decisions[paper_id] = {
                "session_screening_decision": value.get("_screening_decision"),
                "session_topic_relation": value.get("_topic_relation"),
                "session_eligible_deliverables": value.get("_eligible_deliverables"),
                "session_screening_confidence": value.get("_screening_confidence"),
            }
    return decisions


def export_snapshot(session_id: str, task_id: str) -> dict:
    import sqlite3

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    session = con.execute(
        "SELECT original_query, created_at FROM research_sessions WHERE session_id=?",
        (session_id,),
    ).fetchone()
    if session is None:
        raise SystemExit(f"会话不存在: {session_id}")

    raw_papers, fetch_meta = _load_session_artifacts(con, session_id)
    runtime = _load_runtime_state(con, session_id)
    decisions = _load_detail_decisions(con, session_id)

    candidates = deduplicate_papers([dict(paper) for paper in raw_papers])
    # 仅标记会话当时的真实筛选结论；不改变候选文本与排序字段。
    for paper in candidates:
        prior = decisions.get(str(paper.get("paper_id") or ""))
        if prior:
            paper["_session_screening"] = prior

    task = {"task_id": task_id}
    for field in SCOPE_FIELDS:
        if field in runtime:
            task[field] = runtime[field]
    if not task.get("topic"):
        task["topic"] = session["original_query"]
    task["candidate_papers"] = candidates

    snapshot = {
        "dataset_kind": "real_session_candidates_frozen_from_local_research_db",
        "exported_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": {
            "db_path": DB_PATH,
            "session_id": session_id,
            "session_created_at": session["created_at"],
            "original_query": session["original_query"],
            "raw_artifact_papers": len(raw_papers),
            **fetch_meta,
            "deduped_candidate_count": len(candidates),
            "session_detail_decisions": len(decisions),
        },
        "tasks": [task],
    }
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    snapshot = export_snapshot(args.session, args.task_id)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    task = snapshot["tasks"][0]
    zh = sum(1 for p in task["candidate_papers"] if str(p.get("_language_branch") or "") == "zh")
    print(json.dumps({
        "output": str(output),
        "candidates": len(task["candidate_papers"]),
        "zh": zh, "en": len(task["candidate_papers"]) - zh,
        "with_abstract": sum(1 for p in task["candidate_papers"] if str(p.get("abstract") or "").strip()),
        "session_decisions_attached": sum(1 for p in task["candidate_papers"] if p.get("_session_screening")),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
