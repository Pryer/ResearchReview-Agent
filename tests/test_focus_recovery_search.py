"""重点质量缺口的增量检索必须在首轮使用门禁记录的缺失重点。"""

from app.agent import retrieval_loop
from app.agent.nodes.base import _select_batch_first_keywords


def test_focus_recovery_queries_are_dispatched_even_when_metadata_looks_covered(monkeypatch):
    paper = {"paper_id": "p1", "title": "主题与重点甲的研究", "abstract": "重点甲"}
    seen = []

    def search(state, should_cancel=None):
        seen.append({
            "keywords": list(state["keywords"]),
            "branch": dict(state["search_branches"][0]),
        })
        state["candidate_papers"] = [paper]
        state["ranked_papers"] = [paper]

    monkeypatch.setattr(retrieval_loop, "search_node", search)
    monkeypatch.setattr(retrieval_loop, "rank_node", lambda state, llm=None: None)
    state = {
        "topic": "主题", "keywords": ["主题"], "max_papers": 1,
        "required_reference_count": 1, "incremental_retrieval": True,
        "research_semantic_frame": {"evidence_requirements": [{
            "requirement_id": "focus-1", "label": "重点甲", "aliases": ["重点甲"],
            "evidence_role": "analytical_method", "explicit": True,
        }]},
        "active_quality_recovery": {"action": "TARGETED_SEARCH", "issues": [{
            "code": "required_focus_evidence_not_met",
            "details": {"missing_focuses": ["重点甲"]},
        }]},
        "steps": [], "errors": [],
    }

    retrieval_loop.search_rank_with_refinement(state, llm=None)

    assert len(seen) == 1
    assert any("重点甲" in keyword for keyword in seen[0]["keywords"])
    assert seen[0]["branch"]["constraint_level"] == "targeted_recovery"
    assert any("重点甲" in query for query in seen[0]["branch"]["queries"])
    # 即使普通 exact 批次占满名额，恢复分支仍预留至少一个重点查询。
    selected = _select_batch_first_keywords(
        ["普通查询", *seen[0]["branch"]["queries"]],
        [{"type": "exact", "keywords": ["普通查询"]}],
        [seen[0]["branch"]],
        limit=1,
    )
    assert "重点甲" in selected[0]


def test_no_focus_recovery_does_not_inject_focus_branch(monkeypatch):
    monkeypatch.setattr(retrieval_loop, "search_node", lambda state, should_cancel=None: None)
    monkeypatch.setattr(retrieval_loop, "rank_node", lambda state, llm=None: None)
    state = {"topic": "主题", "keywords": ["主题"], "max_papers": 1,
             "incremental_retrieval": True, "steps": [], "errors": []}

    retrieval_loop.search_rank_with_refinement(state, llm=None)

    assert state["keywords"] == ["主题"]
    assert not state.get("search_branches")
