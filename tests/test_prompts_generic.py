"""领域自适应提示词契约测试。"""

from __future__ import annotations

from pathlib import Path

from app.agent.prompts import (
    ASSIGNMENT_PROMPT,
    AXIS_INDUCTION_PROMPT,
    CITATION_CHECK_PROMPT,
    INTENT_RECOGNITION_PROMPT,
    INTRODUCTION_PROMPT,
    LITERATURE_REVIEW_PROMPT,
    PAPER_CARD_EXTRACTION_PROCTION_PROMPT,
    RELATED_WORK_PROMPT,
    RESEARCH_SEMANTIC_PARSER_PROMPT,
    SEARCH_KEYWORD_GENERATION_PROMPT,
    SEARCH_KEYWORD_REFINEMENT_PROMPT,
    SLOT_EXTRACTION_PROMPT,
    TOPIC_DISAMBIGUATION_PROMPT,
)
from app.tools.taxonomy_strategy import TaxonomyStrategyResolver
from app.tools.write_deliverable import WRITER_PROMPT


RUNTIME_PROMPTS = [
    RELATED_WORK_PROMPT,
    INTRODUCTION_PROMPT,
    INTENT_RECOGNITION_PROMPT,
    SLOT_EXTRACTION_PROMPT,
    SEARCH_KEYWORD_GENERATION_PROMPT,
    SEARCH_KEYWORD_REFINEMENT_PROMPT,
    PAPER_CARD_EXTRACTION_PROCTION_PROMPT,
    AXIS_INDUCTION_PROMPT,
    ASSIGNMENT_PROMPT,
    LITERATURE_REVIEW_PROMPT,
    CITATION_CHECK_PROMPT,
    TOPIC_DISAMBIGUATION_PROMPT,
    RESEARCH_SEMANTIC_PARSER_PROMPT,
]

BIASED_TERMS = (
    "计算机视觉",
    "目标检测",
    "少样本",
    "小样本",
    "few-shot",
    "yolo",
    "transformer",
    "cvpr",
    "iccv",
    "eccv",
    "aaai",
    "ijcai",
    "neurips",
    "icml",
)


def test_runtime_prompts_do_not_encode_a_specific_research_domain():
    combined = "\n".join(RUNTIME_PROMPTS).lower()
    for term in BIASED_TERMS:
        assert term.lower() not in combined


def test_writer_and_taxonomy_constraints_do_not_force_technical_dimensions():
    combined = WRITER_PROMPT.lower()
    strategies = "\n".join(
        TaxonomyStrategyResolver.resolve(mode)["axis_instruction"]
        for mode in (
            "technology_oriented",
            "domain_oriented",
            "technology_applied_to_domain",
            "mixed",
        )
    ).lower()

    for fixed_dimension in ("模型架构", "训练范式", "匹配策略", "trx", "otam", "clip"):
        assert fixed_dimension not in combined
        assert fixed_dimension not in strategies
    assert "comparison_dimensions" in WRITER_PROMPT
    assert all(
        TaxonomyStrategyResolver.resolve(mode)["example_axes"] == []
        for mode in ("technology_oriented", "domain_oriented", "mixed")
    )


def test_prompt_contracts_do_not_reintroduce_domain_examples():
    root = Path(__file__).resolve().parents[1]
    prompt_dir = root / "app" / "agent" / "prompt_templates"
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(prompt_dir.glob("*.md"))
    ).lower()

    for term in BIASED_TERMS:
        assert term.lower() not in combined
    assert "不得预设" in combined
    assert "输入" in combined


def test_query_prompt_uses_semantic_diversity_instead_of_fixed_taxonomy():
    assert "窄主题" in SEARCH_KEYWORD_GENERATION_PROMPT
    assert "宽主题" in SEARCH_KEYWORD_GENERATION_PROMPT
    assert "不得预设任何固定学科分类" in SEARCH_KEYWORD_GENERATION_PROMPT
    assert "互补召回价值" in SEARCH_KEYWORD_GENERATION_PROMPT


def test_all_runtime_prompt_templates_format_with_their_declared_inputs():
    RELATED_WORK_PROMPT.format(
        language="中文",
        topic="测试主题",
        target_length=1000,
        our_work_json="{}",
        papers_json="[]",
        clusters_json="[]",
    )
    INTRODUCTION_PROMPT.format(
        language="中文",
        topic="测试主题",
        target_length=1000,
        background_json="{}",
        existing_limitations_json="[]",
        our_work_json="{}",
        verified_results_json="[]",
        papers_json="[]",
    )
    INTENT_RECOGNITION_PROMPT.format(user_query="测试请求")
    SLOT_EXTRACTION_PROMPT.format(user_query="测试请求", current_year=2026)
    SEARCH_KEYWORD_GENERATION_PROMPT.format(user_query="测试请求", topic="测试主题")
    SEARCH_KEYWORD_REFINEMENT_PROMPT.format(
        user_query="测试请求",
        topic="测试主题",
        keywords_json="[]",
        feedback_json="{}",
    )
    PAPER_CARD_EXTRACTION_PROCTION_PROMPT.format(
        evidence_label="摘要证据",
        title="测试论文",
        full_text_or_json="正文",
        paper_id="P001",
        evidence_source="abstract",
    )
    AXIS_INDUCTION_PROMPT.format(
        strategy_instruction="根据研究问题分类",
        strategy_examples="- 示例类别",
        paper_cards_json="[]",
    )
    ASSIGNMENT_PROMPT.format(
        taxonomy_themes_json="[]",
        paper_cards_json="[]",
    )
    LITERATURE_REVIEW_PROMPT.format(
        topic="测试主题",
        language="中文",
        paper_cards_json="[]",
        output_structure="## 相关研究现状",
        section_guidance="根据输入归纳",
        required_reference_count=1,
    )
    CITATION_CHECK_PROMPT.format(review_text="正文", references_json="[]")
    TOPIC_DISAMBIGUATION_PROMPT.format(
        user_query="测试请求",
        topic="测试主题",
        constraints_json="{}",
    )
    RESEARCH_SEMANTIC_PARSER_PROMPT.format(
        user_query="测试请求",
        topic="测试主题",
        deliverables_json="[]",
        retrieved_examples_json="[]",
    )
