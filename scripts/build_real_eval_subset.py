"""从冻结真实候选池构造分层评测子集并生成 qrels 银标（P0）。

分层（不按任何新算法分数取样，避免评测循环论证）：
1. rules 重放判定每篇的通过/排除阶段；
2. 课堂主题优先纳入会话当时真实 LLM 已判过的论文（带 provenance）；
3. 其余在"规则通过 / 主题锚点排除"两个总体与中英文分层内按身份等距抽样；
4. 显式纳入少量年份/学位论文边界案例。

标签来源（写入 provenance，互不混用）：
- session_llm：会话当时的真实 LLM 筛选结论（include/direct→3 等）；
- deterministic：年份越界/学位论文 → 0；
- silver_llm：独立聊天 LLM（非 embedding/rerank 通道）按 0-3 标准重新标注；
  无摘要论文按标题标注并进入人工复核优先队列。

输出供 evaluate_hybrid_retrieval.py 消费的子集快照与 qrels，以及人工
留出测试集模板。银标只用于 dev 指标与管道验收，第 11 节正式门槛仍以
人工标注留出集为准。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.json_utils import parse_json_object
from app.services.llm_service import LLMService
from app.tools.language_router import detect_paper_language
from app.tools.paper_matching import compile_scope
from app.tools.rank_papers import evaluate_paper_hard_filters

LABEL_BATCH = 10

LABEL_PROMPT = """你是学术文献综述的独立标注员。请对每篇论文按其与研究主题的真实相关性打 0-3 分：

研究主题：{topic}
必要研究范围（逐篇应满足的核心概念）：{concepts}

评分标准：
3 = 论文核心研究内容就是该主题（直接研究证据）
2 = 明确相关的方法、应用或评测，可作为支撑文献
1 = 仅背景、邻接领域或泛泛提及，不能作为直接证据
0 = 不相关

只依据给出的标题和摘要判断；信息不足时宁给低分并在 reason 注明。
返回 JSON：{{"results":[{{"paper_id":"...","relevance":0到3的整数,"reason":"不超过30字"}}]}}

论文列表：
{items}"""


def _rules_stage(paper, task, compiled, language):
    ok, stage, _ = evaluate_paper_hard_filters(
        paper,
        topic=task.get("topic") or "",
        keywords=task.get("keywords") or [],
        required_concepts=task.get("required_concepts") or [],
        excluded_title_terms=task.get("excluded_title_terms") or [],
        scope=task.get("selected_scope") or {},
        search_branches=task.get("search_branches") or [],
        research_mode=str((task.get("research_semantic_frame") or {}).get("research_mode") or ""),
        screening_protocol=task.get("screening_protocol") or {},
        language_branch=language,
        compiled_scope=compiled,
        ranking_mode="rules",
    )
    return ok, stage


def _stratified_sample(papers, n):
    """按当前顺序等距抽样，尽量覆盖整个候选池而非只取头部。"""
    if n <= 0:
        return []
    if len(papers) <= n:
        return list(papers)
    step = len(papers) / n
    return [papers[int(index * step)] for index in range(n)]


def select_subset(task: dict, target: int) -> list[dict]:
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
    strata: dict[tuple, list[dict]] = defaultdict(list)
    session_papers = []
    for paper in sorted(task["candidate_papers"], key=lambda p: str(p.get("paper_id") or "")):
        language = detect_paper_language(paper)
        ok, stage = _rules_stage(paper, task, compiled, language)
        key = (language, "passed" if ok else stage)
        strata[key].append(paper)
        if paper.get("_session_screening"):
            session_papers.append(paper)

    chosen: list[dict] = []
    chosen_ids: set[str] = set()

    def add(papers):
        for paper in papers:
            pid = str(paper.get("paper_id") or "")
            if pid and pid not in chosen_ids:
                chosen.append(paper)
                chosen_ids.add(pid)

    # 1) 会话真实 LLM 判过的论文优先（最高质量的真实标签来源）。
    add(session_papers)

    # 2) 总体配额的一半给锚点排除群体（误杀检验的核心），一半给规则通过。
    remaining = target - len(chosen)
    anchor_groups = {key: value for key, value in strata.items()
                     if key[1] in {"topic_anchor_filter", "protocol_hard_filter", "scope_filter"}}
    passed_groups = {key: value for key, value in strata.items() if key[1] == "passed"}
    doc_type_groups = {key: value for key, value in strata.items() if key[1] == "document_type_filter"}

    def fill(groups, budget):
        groups = {key: [p for p in value if str(p.get("paper_id")) not in chosen_ids]
                  for key, value in groups.items() if value}
        total = sum(len(value) for value in groups.values())
        used = 0
        for key, value in sorted(groups.items()):
            share = round(budget * len(value) / total) if total else 0
            picked = _stratified_sample(value, min(share, len(value)))
            add(picked)
            used += len(picked)
        return used

    used = fill(anchor_groups, remaining // 2)
    remaining -= used
    used = fill(passed_groups, max(0, remaining - min(4, sum(len(v) for v in doc_type_groups.values()))))
    remaining -= used
    # 3) 少量确定性边界案例（学位论文等）。
    if remaining > 0:
        boundary = [p for value in doc_type_groups.values() for p in value
                    if str(p.get("paper_id")) not in chosen_ids]
        add(_stratified_sample(boundary, min(remaining, len(boundary))))
    # 4) 仍不足时从各层补足。
    rest = [p for value in strata.values() for p in value
            if str(p.get("paper_id")) not in chosen_ids]
    add(_stratified_sample(rest, max(0, target - len(chosen))))
    return chosen[:target]


def _session_label(paper):
    decision_paper = paper.get("_session_screening") or {}
    decision = decision_paper.get("session_screening_decision")
    relation = decision_paper.get("session_topic_relation")
    eligible = decision_paper.get("session_eligible_deliverables")
    confidence = decision_paper.get("session_screening_confidence") or 0
    if decision == "include" and eligible and relation == "direct" and float(confidence) >= 0.8:
        return 3
    if decision == "include" and eligible and relation == "near":
        return 2
    if decision == "exclude" and float(confidence) >= 0.8:
        return 0
    if relation in {"indirect", "unrelated"}:
        return 1
    return None


def label_papers(papers, task, llm):
    concepts = [" / ".join(group) for group in (task.get("required_concepts") or [])[:6]]
    labels = {}
    sources = {}
    pending = []
    for paper in papers:
        pid = str(paper.get("paper_id") or "")
        session_value = _session_label(paper)
        if session_value is not None:
            labels[pid] = session_value
            sources[pid] = "session_llm"
        else:
            pending.append(paper)

    for start in range(0, len(pending), LABEL_BATCH):
        batch = pending[start:start + LABEL_BATCH]
        items = "\n".join(
            f"{i+1}. paper_id={p.get('paper_id')} | 标题：{str(p.get('title') or '')[:160]} | "
            f"摘要：{str(p.get('abstract') or '')[:500] or '（无摘要）'}"
            for i, p in enumerate(batch)
        )
        prompt = LABEL_PROMPT.format(
            topic=task.get("topic") or "", concepts="；".join(concepts) or "（无额外限定）", items=items)
        response = llm.complete(prompt, response_format="json_object", temperature=0.0,
                                max_tokens=2000, operation="p0_silver_label")
        data = parse_json_object(response if isinstance(response, str) else str(response))
        fresh = {str(item.get("paper_id")): item for item in data.get("results", [])
                 if isinstance(item, dict)}
        for paper in batch:
            pid = str(paper.get("paper_id") or "")
            item = fresh.get(pid) or {}
            value = item.get("relevance")
            value = int(value) if isinstance(value, (int, float)) and value in (0, 1, 2, 3) else None
            if value is None:
                raise RuntimeError(f"银标返回缺项或非法分数：{pid}")
            labels[pid] = value
            sources[pid] = "silver_llm"
    return labels, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--target-size", type=int, default=150)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--output-snapshot", required=True)
    parser.add_argument("--output-qrels", required=True)
    parser.add_argument("--output-provenance", required=True)
    parser.add_argument("--output-human-template", required=True)
    args = parser.parse_args()

    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    task = snapshot["tasks"][0]
    subset = select_subset(task, args.target_size)
    llm = LLMService()
    labels, sources = label_papers(subset, task, llm)

    qrels = []
    provenance = []
    human_template = []
    for index, paper in enumerate(subset):
        pid = str(paper.get("paper_id") or "")
        language = detect_paper_language(paper)
        qrels.append({"task_id": task["task_id"], "paper_id": pid,
                      "relevance": labels[pid], "language": language})
        title_only = not bool(str(paper.get("abstract") or "").strip())
        provenance.append({
            "paper_id": pid, "label_source": sources[pid], "relevance": labels[pid],
            "title_only": title_only, "language": language,
        })
        # 留出测试集：等距抽取，且无摘要（标题标注）优先人工复核。
        is_test = title_only or (index % max(2, round(1 / args.test_ratio)) == 0)
        if is_test:
            human_template.append({
                "task_id": task["task_id"], "paper_id": pid,
                "silver_relevance": labels[pid], "human_relevance": None,
                "label_source": sources[pid],
                "title": paper.get("title"), "abstract": (paper.get("abstract") or "")[:800],
                "year": paper.get("year"), "venue": paper.get("venue"),
            })

    task_out = {key: value for key, value in task.items() if key != "candidate_papers"}
    task_out["candidate_papers"] = [
        {key: value for key, value in paper.items() if not key.startswith("_session_screening")}
        for paper in subset
    ]
    out_snapshot = {**{k: v for k, v in snapshot.items() if k != "tasks"},
                    "subset_of": snapshot.get("source", {}).get("deduped_candidate_count"),
                    "labeling": "silver_dev_pending_human_test",
                    "tasks": [task_out]}
    Path(args.output_snapshot).write_text(
        json.dumps(out_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_qrels).write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in qrels), encoding="utf-8")
    Path(args.output_provenance).write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in provenance), encoding="utf-8")
    Path(args.output_human_template).write_text(
        json.dumps(human_template, ensure_ascii=False, indent=2), encoding="utf-8")
    distribution = defaultdict(int)
    for item in qrels:
        distribution[item["relevance"]] += 1
    print(json.dumps({
        "subset": len(subset), "label_distribution": dict(sorted(distribution.items())),
        "human_review_queue": len(human_template),
        "session_labels": sum(1 for p in provenance if p["label_source"] == "session_llm"),
        "silver_labels": sum(1 for p in provenance if p["label_source"] == "silver_llm"),
        "title_only": sum(1 for p in provenance if p["title_only"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
