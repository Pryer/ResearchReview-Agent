"""摘要预算及准入缓存回归；无真实模型调用。"""
import json

import pytest

from app.core.config import get_settings
from app.tools.screening_context import select_abstract_context
from app.tools.paper_rerank import llm_rerank_papers


@pytest.mark.parametrize("text", ["", "完整中文摘要。", "English abstract."])
def test_short_abstract_is_not_cut(text):
    selected, context = select_abstract_context(text, 500)
    assert selected == text
    assert context["truncated"] is False


@pytest.mark.parametrize("budget", [1, 20, 500, 1000])
def test_overflow_is_bounded_and_has_original_ranges(budget):
    text = "开头研究目标。" + "中间方法内容。" * 300 + "结尾：结果不显著，样本限制。"
    selected, context = select_abstract_context(text, budget)
    assert len(selected) <= budget
    assert context["truncated"] is True
    pieces = [text[start:end] for start, end in context["ranges"]]
    assert selected.startswith(pieces[0]) and selected.endswith(pieces[-1])
    if budget >= 500:
        assert "结果不显著" in selected
        assert "省略" in selected


class CaptureModel:
    def __init__(self):
        self.items = []

    def complete(self, prompt, **kwargs):
        items = json.loads(prompt.split("候选论文：\n", 1)[1].split("\n\n注意：", 1)[0])
        self.items.append(items)
        return json.dumps({"results": [{
            "paper_id": item["paper_id"], "topic_relevance": 9, "scope_alignment": 9,
            "method_alignment": 9, "decision": "include", "confidence": 0.95,
            "relation_type": "direct", "eligible_deliverables": ["research_background"],
            "required_condition_results": [],
        } for item in items]})


def test_screening_batch_budget_and_tail_are_applied_to_actual_prompt(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "paper_screening_abstract_chars", 1000)
    monkeypatch.setattr(settings, "paper_screening_batch_abstract_chars", 1500)
    text = "Research aims. " * 120 + "Results do not support the proposed improvement."
    papers = [{"paper_id": str(i), "title": "Retrieval", "abstract": text} for i in range(3)]
    model = CaptureModel()
    result = llm_rerank_papers(papers, "retrieval", llm=model, top_k=3, batch_size=3,
                               required_conditions=[])
    assert len(result) == 3
    items = model.items[0]
    assert sum(len(item["abstract"]) for item in items) <= 1500
    for item in items:
        assert "Results do not support" in item["abstract"]
        assert item["abstract_context"]["truncated"]
    assert all(paper["abstract"] == text for paper in papers)


def test_context_change_invalidates_screening_cache(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "paper_screening_abstract_chars", 1000)
    paper = {"paper_id": "p", "title": "Retrieval", "abstract": "Abstract. " * 600}
    model = CaptureModel()
    cache = {}

    def screen():
        llm_rerank_papers([paper.copy()], "retrieval", llm=model, top_k=1,
                          required_conditions=[], screening_cache=cache, cache_namespace="fixed")

    screen()
    screen()
    assert len(model.items) == 1
    monkeypatch.setattr(settings, "paper_screening_abstract_chars", 2000)
    screen()
    assert len(model.items) == 2
    assert len(model.items[-1][0]["abstract"]) == 2000
