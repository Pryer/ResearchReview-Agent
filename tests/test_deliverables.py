"""四交付物收敛、规划和验证测试。"""

from app.agent.deliverable_router import (
    check_deliverable_readiness,
    extract_user_paper_profile,
    resolve_core_deliverables,
)
from app.agent.writing_plan import build_writing_plan
from app.schemas.deliverable_schema import CoreDeliverableType
from app.tools.synthesize_themes import synthesize_themes
from app.tools.validate_deliverable import validate_deliverable
from app.tools.write_deliverable import write_deliverable


def _card(paper_id: str, problem: str, method: str) -> dict:
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "year": 2024,
        "quality_status": "partial",
        "evidence_source": "abstract",
        "evidence_state": {"access_level": "abstract", "verification_status": "source_verified"},
        "unsupported_fields": ["limitations", "ablation_results"],
        "field_claims": {
            "research_problem": [{
                "claim": problem, "source_text": problem, "source_section": "abstract",
                "evidence_id": f"{paper_id}:problem", "evidence_level": "abstract",
                "confidence": 1.0, "explicitly_reported": True,
            }],
            "method": [{
                "claim": method, "source_text": method, "source_section": "abstract",
                "evidence_id": f"{paper_id}:method", "evidence_level": "abstract",
                "confidence": 1.0, "explicitly_reported": True,
            }],
        },
    }


def _state() -> dict:
    cards = [
        _card("p1", "研究课堂观察编码", "采用人工行为编码"),
        _card("p2", "研究课堂互动模式", "采用序列分析"),
        _card("p3", "识别课堂学生动作", "采用视觉检测模型"),
        _card("p4", "识别多模态课堂行为", "采用视频与语音融合"),
    ]
    taxonomy = {
        "organizing_principle": "研究问题与分析路线",
        "themes": [
            {"theme_id": "T1", "name": "课堂观察与互动编码"},
            {"theme_id": "T2", "name": "自动行为识别"},
        ],
        "assignments": [
            {"paper_id": "p1", "primary_theme_id": "T1"},
            {"paper_id": "p2", "primary_theme_id": "T1"},
            {"paper_id": "p3", "primary_theme_id": "T2"},
            {"paper_id": "p4", "primary_theme_id": "T2"},
        ],
    }
    synthesis = synthesize_themes(cards, taxonomy)
    return {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "dynamic_taxonomy": taxonomy,
        "theme_synthesis": synthesis,
        "search_report": {"start_year": 2023, "end_year": 2026, "sources": ["openalex"], "writing_pool_count": 4},
        "evidence_quality_report": {"limitations": ["4篇论文仅获得摘要。"]},
    }


def test_legacy_intents_converge_to_four_core_deliverables():
    assert resolve_core_deliverables("generate_review", ["background"])[0] == CoreDeliverableType.RESEARCH_BACKGROUND
    assert resolve_core_deliverables("generate_review", ["research_status"])[0] == CoreDeliverableType.RESEARCH_STATUS
    assert resolve_core_deliverables("generate_related_work", ["related_work"])[0] == CoreDeliverableType.RELATED_WORK
    assert resolve_core_deliverables("generate_review", ["narrative_review"])[0] == CoreDeliverableType.NARRATIVE_REVIEW


def test_related_work_requires_user_problem_and_method_before_retrieval():
    state = {"user_paper_profile": {}}
    result = check_deliverable_readiness("related_work", state, phase="pre_retrieval")
    assert result.ready is False
    assert result.clarification_question
    profile = extract_user_paper_profile("我的论文采用图神经网络检测金融欺诈，重点解决类别不平衡问题")
    assert profile.research_problem
    assert profile.proposed_method == "图神经网络"


def test_theme_synthesis_and_writing_plan_use_explicit_claims():
    state = _state()
    assert len(state["theme_synthesis"]) == 2
    assert all(item["reported_methods"] for item in state["theme_synthesis"])
    assert all(item["common_methods"] == [] for item in state["theme_synthesis"])
    plan = build_writing_plan("research_status", state)
    assert any(section.title == "（一）课堂观察与互动编码" for section in plan.sections)
    assert all(section.supporting_paper_ids for section in plan.sections if section.id.startswith("theme_"))
    text = write_deliverable(plan, state, llm=None)
    validation = validate_deliverable(text, plan, state)
    assert "### （一）课堂观察与互动编码" in text
    assert validation["valid"] is True


def test_writing_plan_never_emits_fallback_theme_as_formal_section():
    state = _state()
    state["theme_synthesis"].append({
        "theme_id": "T_OTHER",
        "theme_name": "其他相关研究",
        "paper_ids": ["p1"],
        "common_problems": [],
        "common_methods": [],
        "reported_findings": [],
    })

    plan = build_writing_plan("research_status", state)

    assert all(section.title != "其他相关研究" for section in plan.sections)


def test_research_status_validator_uses_actual_planned_route_count():
    state = _state()
    plan = build_writing_plan("research_status", state)
    plan.sections = [section for section in plan.sections if section.id != "theme_2"]
    text = "## 研究现状\n\n### （一）课堂观察与互动编码\n\n现有研究集中于此 [p1]。\n"

    validation = validate_deliverable(text, plan, state)

    assert not any("动态三级研究路线" in error for error in validation["errors"])


def test_background_claim_plan_does_not_fallback_to_first_five_papers():
    from app.agent.claim_plan import build_background_claim_plan

    outline = {"paragraph_goals": [{"id": "g1", "label": "无关目标", "goal": "量子计算"}]}
    cards = [_card(f"p{i}", "课堂行为研究", "课堂观察方法") for i in range(1, 7)]

    plans = build_background_claim_plan(outline, cards)

    assert plans[0]["total_evidence_papers"] == 0
    assert plans[0]["claims"] == []


def test_provisional_routes_accept_one_valid_route():
    from app.agent.provisional_routes import generate_provisional_routes

    class LLM:
        def complete(self, prompt, **kwargs):
            return '{"research_scope": {}, "background_outline": {}, "provisional_routes": [{"route_id": "R1", "name": "唯一路线", "core_concepts": ["概念一", "概念二", "概念三"]}]}'

    result = generate_provisional_routes({"topic": "测试主题", "user_query": "测试"}, LLM())

    assert [route["route_id"] for route in result["provisional_routes"]] == ["R1"]


def test_narrative_validator_rejects_fake_systematic_review_process():
    state = _state()
    # 补足叙述性综述最低论文数，仅用于验证结构规则。
    state["paper_cards"] = state["paper_cards"] * 2
    plan = build_writing_plan("narrative_review", state)
    text = write_deliverable(plan, state, llm=None) + "\n\n本系统综述遵循PRISMA流程。"
    validation = validate_deliverable(text, plan, state)
    assert validation["valid"] is False
    assert any("系统综述" in error for error in validation["errors"])


def test_background_writer_rejects_uncited_and_overreaching_llm_draft():
    state = _state()
    plan = build_writing_plan("research_background", state)

    class InvalidDraftLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            sections = "\n\n".join(
                f"## {section.title}\n本文提出一种全新方法并解决该问题。"
                for section in plan.sections
            )
            return "# 研究背景\n\n" + sections

    text = write_deliverable(plan, state, llm=InvalidDraftLLM())
    validation = validate_deliverable(text, plan, state)

    assert validation["valid"] is True
    assert "本文提出" not in text
    assert not text.startswith("# ")
    assert validation["metrics"]["unique_cited_papers"] >= 2


def test_research_status_fallback_meets_requested_unique_reference_target():
    cards = [
        _card(f"p{index}", f"课堂行为研究问题{index}", f"课堂行为分析方法{index}")
        for index in range(1, 13)
    ]
    taxonomy = {
        "organizing_principle": "研究路线",
        "themes": [
            {"theme_id": "T1", "name": "自动识别与编码"},
            {"theme_id": "T2", "name": "教育分析与解释"},
        ],
        "assignments": [
            {
                "paper_id": card["paper_id"],
                "primary_theme_id": "T1" if index < 6 else "T2",
            }
            for index, card in enumerate(cards)
        ],
    }
    state = {
        "topic": "课堂行为分析",
        "paper_cards": cards,
        "dynamic_taxonomy": taxonomy,
        "theme_synthesis": synthesize_themes(cards, taxonomy),
        "required_reference_count": 10,
    }
    plan = build_writing_plan("research_status", state)
    text = write_deliverable(plan, state, llm=None)
    validation = validate_deliverable(text, plan, state)

    assert plan.citation_policy["minimum_unique_references"] == 10
    assert validation["metrics"]["unique_cited_papers"] >= 10


def test_fallback_claim_dedup_merges_same_section_citations_but_not_other_sections():
    from app.deliverables.renderers.base_renderer import _deduplicate_fallback_claims
    from app.schemas.deliverable_schema import WritingPlan, WritingSection

    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="测试", organizing_strategy="按主题分节",
        sections=[
            WritingSection(id="theme_a", title="A", purpose="a", supporting_paper_ids=["p1", "p2"]),
            WritingSection(id="theme_b", title="B", purpose="b", supporting_paper_ids=["p2"]),
        ],
    )
    sections = [
        ("theme_a", "## A\n同一主张[ p1 ]。同一主张[p2]。"),
        ("theme_b", "## B\n同一主张[p2]。"),
    ]
    result = _deduplicate_fallback_claims(sections, plan, {})
    assert result[0][1].count("同一主张") == 1
    assert "p1" in result[0][1] and "p2" in result[0][1]
    assert result[1][1].count("同一主张") == 1


def test_write_deliverable_fallback_uses_sanitized_synthesis(monkeypatch):
    """回归：交付物级降级渲染不得经原始 state 泄漏未授权综合条目。

    背景：write_deliverable 的兜底路径原先把原始 state 传给
    render_fallback，theme_synthesis 未经过 allowed_claim_ids /
    prompt_paper_ids 过滤，claim 门禁未放行的声明会直接进入正文。
    """
    from app.deliverables.renderers.base_renderer import BaseRenderer
    from app.schemas.deliverable_schema import WritingPlan, WritingSection
    from app.tools import write_deliverable as wd_module

    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="测试",
        organizing_strategy="按主题分节",
        sections=[
            WritingSection(
                id="theme_a",
                title="主题A",
                purpose="验证主题节",
                supporting_paper_ids=["p1"],
                supporting_claim_ids=["c_ok"],
            ),
        ],
    )
    state = {
        "canonical_topic": "测试主题",
        "paper_cards": [{"paper_id": "p1", "title": "论文一", "field_claims": {}}],
        "theme_synthesis": [{
            "theme_id": "a",
            "theme_name": "主题A",
            "paper_ids": ["p1"],
            "reported_problems": [
                {"claim_text": "未授权声明内容", "paper_id": "p1", "claim_id": "c_bad"},
                {"claim_text": "已授权声明内容", "paper_id": "p1", "claim_id": "c_ok"},
            ],
        }],
    }

    class EmptyRenderRenderer(BaseRenderer):
        """模拟渲染器全链失败返回空文本，强制触发交付物级兜底。"""

        def render(self, **kwargs):
            return ""

    monkeypatch.setattr(
        wd_module, "get_renderer",
        lambda _t: EmptyRenderRenderer(CoreDeliverableType.RESEARCH_STATUS),
    )

    text = wd_module.write_deliverable(plan, state, llm=None)

    assert "已授权声明内容" in text
    assert "未授权声明内容" not in text


def test_writer_receives_authoritative_bibliographic_fields(monkeypatch):
    """Writer 提示词要求点名第一作者，卡片投影必须携带可核验书目字段。"""
    from app.deliverables.renderers.base_renderer import BaseRenderer
    from app.schemas.deliverable_schema import WritingPlan, WritingSection
    from app.tools import write_deliverable as wd_module

    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="测试",
        organizing_strategy="按主题分节",
        sections=[
            WritingSection(
                id="theme_a",
                title="主题A",
                purpose="验证主题节",
                supporting_paper_ids=["p1"],
                supporting_claim_ids=["p1:method"],
            ),
        ],
    )
    state = {
        "canonical_topic": "课堂行为分析",
        "paper_cards": [{
            "paper_id": "p1",
            "title": "课堂行为识别方法",
            "authors": ["Shuai Ma", "Jian Han"],
            "year": 2025,
            "venue": "ICACTE",
            "doi": "10.1109/ICACTE.2025.1",
            "publication_status": "published",
            "peer_review_status": "likely_peer_reviewed",
            "publication_type": "conference_paper",
            "evidence_state": {"access_level": "abstract"},
            "field_claims": {
                "method": [{
                    "claim": "采用 YOLOv10 检测课堂行为",
                    "source_text": "We adopt YOLOv10 for classroom behavior detection.",
                    "source_section": "abstract",
                    "evidence_id": "p1:method",
                    "evidence_level": "abstract",
                    "confidence": 1.0,
                    "explicitly_reported": True,
                }],
            },
        }],
        "theme_synthesis": [],
    }
    captured: list[dict] = []

    class CapturingRenderer(BaseRenderer):
        def render(self, **kwargs):
            captured.extend(kwargs.get("cards") or [])
            return "## 主题A\n\n采用 YOLOv10 检测课堂行为[p1]。"

    monkeypatch.setattr(
        wd_module, "get_renderer",
        lambda _t: CapturingRenderer(CoreDeliverableType.RESEARCH_STATUS),
    )

    wd_module.write_deliverable(plan, state, llm=None)

    assert captured
    card = captured[0]
    assert card["authors"] == ["Shuai Ma", "Jian Han"]
    assert card["title"] == "课堂行为识别方法"
    assert card["year"] == 2025
    assert card["doi"] == "10.1109/ICACTE.2025.1"
    assert card["publication_status"] == "published"
    assert card["claims"][0]["evidence_id"] == "p1:method"
    assert "YOLOv10" in card["claims"][0]["source_text"]


def test_citation_outside_plan_authorization_is_rejected():
    """引用了证据池中存在但计划未授权的论文，必须被拦截（M15）。"""
    from app.schemas.deliverable_schema import WritingPlan, WritingSection

    state = _state()
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_BACKGROUND,
        purpose="背景",
        organizing_strategy="递进",
        sections=[WritingSection(
            id="background_body",
            title="研究背景",
            purpose="x",
            supporting_paper_ids=["p1", "p2"],
            heading_level=2,
        )],
    )
    text = (
        "## 研究背景\n\n"
        "课堂行为分析近年受到持续关注 [p1]。相关证据也来自视觉检测路线 [p3]。\n"
    )
    validation = validate_deliverable(text, plan, state)
    assert validation["valid"] is False
    assert any("未授权" in error for error in validation["errors"])


def test_single_theme_synthesis_downgraded_to_warning():
    """证据只支持单一研究路线时降级为警告，不再硬性否决（M15）。"""
    from app.schemas.deliverable_schema import WritingPlan, WritingSection

    state = _state()
    state["theme_synthesis"] = state["theme_synthesis"][:1]
    plan = WritingPlan(
        deliverable_type=CoreDeliverableType.RESEARCH_STATUS,
        purpose="现状",
        organizing_strategy="单路线",
        sections=[WritingSection(
            id="theme_single",
            title="单一研究路线",
            purpose="x",
            supporting_paper_ids=["p1", "p2"],
            heading_level=3,
        )],
    )
    text = (
        "## 研究现状\n\n### 单一研究路线\n\n现有研究集中于此 [p1]。\n"
    )
    validation = validate_deliverable(text, plan, state)
    assert all("两条研究路线" not in error for error in validation["errors"])
    assert any("多路线对比" in warning for warning in validation["warnings"])
