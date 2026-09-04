"""脱敏写作少样本的覆盖与污染隔离测试。"""

import re

from app.deliverables.few_shot_blueprints import (
    detect_blueprint_leakage,
    get_section_blueprint,
)
from app.schemas.deliverable_schema import (
    CoreDeliverableType,
    WritingPlan,
    WritingSection,
)
from app.tools.validate_deliverable import validate_deliverable
from app.deliverables.renderers import (
    _section_rewrite_prompt,
    _validate_rewritten_section,
    _write_sections_in_chinese,
)


def test_four_deliverables_have_section_level_blueprints():
    section_ids = {
        CoreDeliverableType.RESEARCH_BACKGROUND: [
            "problem_context", "importance", "existing_approaches", "research_need",
        ],
        CoreDeliverableType.RESEARCH_STATUS: [
            "scope_definition", "theme_T1", "cross_route_comparison", "research_gaps",
        ],
        CoreDeliverableType.RELATED_WORK: [
            "theme_T1", "gap_and_positioning",
        ],
        CoreDeliverableType.NARRATIVE_REVIEW: [
            "abstract", "introduction", "search_scope", "scope_definition",
            "theme_T1", "cross_route_comparison", "future_directions",
            "conclusion", "evidence_statement",
        ],
    }

    for deliverable_type, ids in section_ids.items():
        for section_id in ids:
            blueprint = get_section_blueprint(deliverable_type, section_id)
            assert blueprint["moves"]
            assert blueprint["example"]
            assert blueprint["evidence_role"] == "style_only_not_evidence"


def test_section_prompt_physically_separates_blueprint_and_real_evidence():
    prompt = _section_rewrite_prompt(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        section_id="theme_T1",
        title="自动行为识别",
        topic="课堂行为分析",
        original="## 自动行为识别\n\n研究采用视觉模型[p1]。",
        required_ids=["p1"],
    )

    assert "脱敏写作少样本" in prompt
    assert "只示范修辞结构，不是事实证据" in prompt
    assert "真实证据草稿——正文事实与引用的唯一来源" in prompt
    assert prompt.index("脱敏写作少样本") < prompt.index("真实证据草稿")


def test_blueprint_placeholders_are_rejected_by_section_gate():
    text = (
        "## 自动行为识别\n\n围绕〈示例主题〉，现有研究形成了共同路线"
        "〔证据A〕，真实证据见[p1]。"
    )

    assert detect_blueprint_leakage(text)
    errors = _validate_rewritten_section(text, "自动行为识别", ["p1"])
    assert any("少样本" in error for error in errors)


def test_blueprint_placeholders_are_rejected_by_deliverable_gate():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_BACKGROUND,
        purpose="测试",
        organizing_strategy="evidence_driven",
        sections=[
            WritingSection(
                id="problem_context",
                title="研究问题与场景",
                purpose="测试",
                supporting_paper_ids=["p1", "p2"],
            )
        ],
        citation_policy={"minimum_unique_references": 2},
    )
    state = {
        "topic": "课堂行为分析",
        "paper_cards": [{"paper_id": "p1"}, {"paper_id": "p2"}],
    }
    text = (
        "## 研究问题与场景\n\n课堂行为分析围绕〈示例主题〉形成了不同研究"
        "方向[p1][p2]。"
    )

    validation = validate_deliverable(text, plan, state)

    assert validation["valid"] is False
    assert any("少样本" in error for error in validation["errors"])
    assert validation["metrics"]["blueprint_leakage_count"] > 0


def test_section_retry_repairs_previous_candidate_instead_of_starting_over():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="测试",
        organizing_strategy="evidence_driven",
        sections=[
            WritingSection(
                id="theme_T1",
                title="自动识别",
                purpose="测试",
                supporting_paper_ids=["p1", "p2"],
            )
        ],
        citation_policy={"minimum_unique_references": 2},
    )
    state = {"topic": "课堂行为分析"}

    class RepairingLLM:
        prompts: list[str] = []
        kwargs_seen: list[dict] = []

        def complete(self, prompt: str, **kwargs) -> str:
            self.prompts.append(prompt)
            self.kwargs_seen.append(kwargs)
            if len(self.prompts) == 1:
                return "## 自动识别\n\n现有研究采用自动识别方法[p1]。"
            assert "上一次输出及机器检查结果" in prompt
            assert "缺少引用编号" in prompt
            assert "现有研究采用自动识别方法[p1]" in prompt
            return (
                "## 自动识别\n\n现有研究采用自动识别方法，并从不同数据条件"
                "验证其适用性[p1][p2]。"
            )

    llm = RepairingLLM()
    result = _write_sections_in_chinese(
        "## 自动识别\n\nMethod one is reported by one study[p1]. "
        "Method two is reported by another study[p2].",
        plan,
        state,
        [],
        llm,
    )

    assert len(llm.prompts) == 2
    assert all(item.get("retry_empty") is True for item in llm.kwargs_seen)
    assert "[p1][p2]" in result
    assert state["writer_section_diagnostics"][-1]["sections"][0]["status"] == "success"


def test_fullwidth_model_citations_are_normalized_by_section_writer():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="测试",
        organizing_strategy="evidence_driven",
        sections=[
            WritingSection(
                id="theme_T1",
                title="自动识别",
                purpose="测试",
                supporting_paper_ids=["p1", "p2"],
            )
        ],
        citation_policy={"minimum_unique_references": 2},
    )
    state = {"topic": "课堂行为分析"}

    class FullwidthCitationLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return (
                "## 自动识别\n\n现有研究分别讨论课堂行为的视觉识别"
                "与教育解释〔p1〕〔p2〕。"
            )

    result = _write_sections_in_chinese(
        "## 自动识别\n\n视觉识别研究[p1]，教育解释研究[p2]。",
        plan,
        state,
        [],
        FullwidthCitationLLM(),
    )

    assert "〔" not in result
    assert "[p1][p2]" in result
    assert state["writer_section_diagnostics"][-1]["sections"][0]["status"] == "success"


def test_structurally_valid_section_without_required_citations_is_quarantined():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="测试",
        organizing_strategy="evidence_driven",
        sections=[
            WritingSection(
                id="theme_T1",
                title="自动识别",
                purpose="测试",
                supporting_paper_ids=["p1", "p2"],
            )
        ],
        citation_policy={"minimum_unique_references": 2},
    )
    state = {"topic": "课堂行为分析"}

    class CitationDroppingLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return "## 自动识别\n\n现有研究形成了多条自动行为识别路线。"

    result = _write_sections_in_chinese(
        "## 自动识别\n\n视觉识别研究[p1]，教育解释研究[p2]。",
        plan,
        state,
        [],
        CitationDroppingLLM(),
    )

    # 所有改写均失败时不再简单返回空——若原始草稿经门禁剥离后
    # 仍可过基础校验（有引用、非纯英文、结构完整），则保留以保全文内引用。
    assert result != "" and "[p1]" in result and "[p2]" in result
    section_diags = state["writer_section_diagnostics"][-1]["sections"]
    statuses = [d["status"] for d in section_diags]
    assert "evidence_limited" in statuses
    assert "original_draft_retained" in statuses


def test_failed_section_never_returns_raw_english_evidence():
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="测试",
        organizing_strategy="evidence_driven",
        sections=[
            WritingSection(
                id="theme_T1",
                title="多模态分析",
                purpose="测试",
                supporting_paper_ids=["p1"],
            )
        ],
        citation_policy={"minimum_unique_references": 1},
    )
    state = {"topic": "课堂行为分析"}

    class InvalidLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return "This invalid response has no required heading or citation."

    result = _write_sections_in_chinese(
        "## 多模态分析\n\nThis study proposes a multimodal classroom model[p1].",
        plan,
        state,
        [],
        InvalidLLM(),
    )

    assert "This study proposes" not in result
    # 新契约：失败章节不再静默删空（那会让整章从交付物中消失），
    # 而是回退保守证据段——不含英文原文、不含未验证结论，结构保留。
    assert "## 多模态分析" in result
    assert "当前证据池中没有分配给本节的论文" in result
    assert not re.search(r"[A-Za-z]{20,}", result)
    assert state["writer_section_diagnostics"][-1]["sections"][0]["status"] == "evidence_limited"
