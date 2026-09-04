"""防止编排层和运行时提示词把某个领域写死。

这些测试使用一个与教育无关的主题，确保候选路线和引用分配仍完全由
当前语义帧/论文决定，而不会因为提示词中的示例污染意图或研究路线。
"""

from __future__ import annotations

from app.agent.provisional_routes import _build_provisional_route_prompt
from app.agent.nodes.synthesis import _plan_citation_allocation
from app.agent.prompts import RESEARCH_SEMANTIC_PARSER_PROMPT


_DOMAIN_SPECIFIC_TOKENS = (
    "课堂行为",
    "课堂教学行为",
    "师生互动",
    "学生行为",
    "教师行为",
    "校园安防",
)


def test_provisional_route_prompt_is_topic_neutral():
    prompt = _build_provisional_route_prompt(
        topic="海洋微塑料迁移",
        user_query="调研海洋微塑料迁移的研究现状",
        semantic_frame={
            "application_domains": [{"label": "海洋环境"}],
            "research_objects": [{"label": "微塑料迁移"}],
            "methods": [],
        },
    )
    assert "海洋微塑料迁移" in prompt
    assert not any(token in prompt for token in _DOMAIN_SPECIFIC_TOKENS)


def test_semantic_parser_prompt_does_not_contain_domain_examples():
    prompt = RESEARCH_SEMANTIC_PARSER_PROMPT.format(
        user_query="分析电池老化机理",
        topic="电池老化机理",
        deliverables_json="[]",
        retrieved_examples_json="[]",
    )
    assert not any(token in prompt for token in _DOMAIN_SPECIFIC_TOKENS)


class _CaptureLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return '{"sections": [{"section_id": "body", "paper_ids": ["p1"]}]}'


def test_citation_allocation_prompt_is_domain_neutral():
    llm = _CaptureLLM()
    state = {
        "topic": "电池老化机理",
        "selected_scope": {"include_terms": ["固态电池", "循环寿命"]},
        "paper_cards": [{"paper_id": "p1", "title": "Battery ageing study", "quality_status": "valid"}],
        "paper_details": [{"paper_id": "p1", "title": "Battery ageing study", "year": 2025, "abstract": ""}],
    }
    _plan_citation_allocation(state, llm, state["paper_cards"], required_count=1)
    assert llm.prompts
    assert not any(token in llm.prompts[0] for token in _DOMAIN_SPECIFIC_TOKENS)

