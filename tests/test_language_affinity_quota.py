"""主题语言倾向驱动的中文分支配额，以及检索目标安全余量。"""

import pytest

from app.agent.planner import resolve_language_branch_zh_ratio
from app.agent.slot_extractor import _derive_count_targets
from app.core.config import get_settings
from app.schemas.research_plan_schema import LanguageAffinity, ResearchSemanticFrame
from app.tools.branch_merge import (
    build_language_coverage_contract,
    calculate_branch_targets,
)


def test_zh_dominant_raises_chinese_quota():
    ratio, reason = resolve_language_branch_zh_ratio({
        "language_affinity": "zh_dominant",
        "language_affinity_reason": "议题依附中文教育研究社区",
    })
    assert ratio == pytest.approx(get_settings().language_branch_zh_ratio_zh_dominant)
    assert "zh_dominant" in reason


def test_en_dominant_lowers_chinese_quota():
    ratio, _ = resolve_language_branch_zh_ratio({"language_affinity": "en_dominant"})
    assert ratio == pytest.approx(get_settings().language_branch_zh_ratio_en_dominant)


def test_balanced_keeps_configured_default():
    ratio, reason = resolve_language_branch_zh_ratio({"language_affinity": "balanced"})
    assert ratio == pytest.approx(get_settings().language_branch_zh_ratio)
    assert reason == "balanced"


@pytest.mark.parametrize("frame", [None, {}, {"language_affinity": "偏中文一点"}])
def test_missing_or_invalid_affinity_falls_back_to_default(frame):
    ratio, reason = resolve_language_branch_zh_ratio(frame)
    assert ratio == pytest.approx(get_settings().language_branch_zh_ratio)
    assert reason


def test_accepts_pydantic_frame_object():
    frame = ResearchSemanticFrame(
        canonical_topic="课堂行为分析",
        language_affinity=LanguageAffinity.ZH_DOMINANT,
        language_affinity_reason="中文教育研究议题",
    )
    ratio, reason = resolve_language_branch_zh_ratio(frame)
    assert ratio == pytest.approx(get_settings().language_branch_zh_ratio_zh_dominant)
    assert "zh_dominant" in reason


def test_ratio_is_clamped_within_safe_bounds(monkeypatch):
    """即使映射值被配成极端数，也必须钳制在区间内，防止某一分支失效。"""
    settings = get_settings()
    monkeypatch.setattr(settings, "language_branch_zh_ratio_zh_dominant", 0.98)
    monkeypatch.setattr(settings, "language_branch_zh_ratio_en_dominant", 0.01)

    high, _ = resolve_language_branch_zh_ratio({"language_affinity": "zh_dominant"})
    low, _ = resolve_language_branch_zh_ratio({"language_affinity": "en_dominant"})

    assert high == pytest.approx(settings.language_branch_zh_ratio_max)
    assert low == pytest.approx(settings.language_branch_zh_ratio_min)


def test_affinity_can_be_disabled(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "language_branch_affinity_enabled", False)
    ratio, reason = resolve_language_branch_zh_ratio({"language_affinity": "zh_dominant"})
    assert ratio == pytest.approx(settings.language_branch_zh_ratio)
    assert reason == "affinity_disabled"


def test_raised_ratio_actually_admits_more_chinese_papers():
    """配额提高必须真的转化为更多中文入选，且英文分支不被压到失效。"""
    zh_low, en_low = calculate_branch_targets(64, 0.40, zh_count=311, en_count=713)
    zh_high, en_high = calculate_branch_targets(64, 0.55, zh_count=311, en_count=713)

    assert zh_high > zh_low
    assert zh_high + en_high == zh_low + en_low == 64
    assert en_high >= get_settings().language_branch_min_en


def test_retrieval_margin_covers_downstream_attrition():
    """40 篇显式引用要求需要足够候选余量抵消证据链损耗。"""
    target, generation_limit = _derive_count_targets(40, True)
    assert target == generation_limit
    # 实测 51 篇池子只产出 38 篇引用（约 25% 损耗），余量须显著高于此。
    assert target >= 60
    assert target / 40 >= 1.5


def test_retrieval_margin_stays_bounded_for_large_requests():
    """余量有上限，不能把大额引用要求放大为无节制检索。"""
    target, _ = _derive_count_targets(200, True)
    assert target <= 240


def test_low_pass_protection_can_freeze_quota_transfer():
    zh_target, en_target = calculate_branch_targets(
        40, 0.40, zh_count=40, en_count=2, min_zh=8, min_en=12,
        allow_quota_transfer=False,
    )
    assert en_target < 12
    assert zh_target + en_target < 40


def test_language_coverage_contract_does_not_transfer_missing_english_minimum():
    contract = build_language_coverage_contract(
        40,
        0.55,
        min_zh=8,
        min_en=12,
        eligible_zh=40,
        eligible_en=2,
        affinity="zh_dominant",
    )

    assert contract["minimum_en"] == 12
    assert contract["deficits"] == {"zh": 0, "en": 10}
    assert contract["satisfied_at_screening"] is False


# ============================================================
# 澄清后的语义帧失效重解析：语言倾向必须看到已确认范围
# ============================================================
def _frame(affinity: LanguageAffinity, reason: str) -> ResearchSemanticFrame:
    return ResearchSemanticFrame(
        canonical_topic="课堂行为分析",
        language_affinity=affinity,
        language_affinity_reason=reason,
    )


def _plan_state(user_query: str, frame: ResearchSemanticFrame, source_query: str) -> dict:
    return {
        "user_query": user_query,
        "research_semantic_frame": frame.model_dump(mode="json"),
        "semantic_frame_source_query": source_query,
        "steps": [],
        "errors": [],
    }


def _install_parse_spy(monkeypatch, returned: ResearchSemanticFrame) -> list[str]:
    """记录 plan_node 的语义帧解析调用。

    三点必须注意：
    1. plan_node 在函数体内局部 import 该符号，补丁要打在源模块上。
    2. 间谍只记录并返回合法帧。plan_node 用 try/except 包住整个函数体，
       在替身里抛异常只会被吞掉并让节点整体失败，掩盖要验证的行为。
    3. ``_fallback_search_strategy``（LLM 不可用时的关键词兜底）也会调用同一
       函数，且按位置传参；只记录带 ``user_query`` 关键字的调用，才能把帧解析
       与该兜底路径区分开。
    """
    seen: list[str] = []

    def spy(*args, **kwargs):
        if "user_query" in kwargs:
            seen.append(kwargs["user_query"])
        return returned

    monkeypatch.setattr(
        "app.agent.research_semantic_parser.parse_research_semantics", spy,
    )
    return seen


def test_stale_frame_is_reparsed_after_clarification(monkeypatch):
    """澄清扩大了工作查询后，范围收窄前解析的帧必须重新解析。"""
    from app.agent.nodes.planning import plan_node

    scoped = _frame(
        LanguageAffinity.ZH_DOMINANT, "范围收窄到教育技术视角下的课堂教学实践",
    )
    seen = _install_parse_spy(monkeypatch, scoped)

    clarified = "调研近三年课堂行为分析论文\n用户澄清原文：侧重教育技术视角"
    state = _plan_state(
        user_query=clarified,
        frame=_frame(LanguageAffinity.BALANCED, "宽泛话题，双语产出"),
        source_query="调研近三年课堂行为分析论文",
    )

    plan_node(state, llm=None, current_year=2026)

    assert seen == [clarified]
    assert state["research_semantic_frame"]["language_affinity"] == "zh_dominant"
    assert state["language_branch_zh_ratio"] == pytest.approx(
        get_settings().language_branch_zh_ratio_zh_dominant
    )
    # 来源查询同步更新，后续轮次不再重复解析。
    assert state["semantic_frame_source_query"] == clarified


def test_fresh_frame_reused_without_reparsing(monkeypatch):
    """查询未变化时复用缓存帧，不为每次规划重复付解析成本。"""
    from app.agent.nodes.planning import plan_node

    cached = _frame(LanguageAffinity.ZH_DOMINANT, "教育技术课堂教学议题")
    # 间谍返回一个可区分的帧：一旦被调用，断言能指出配额来自重解析而非缓存。
    seen = _install_parse_spy(
        monkeypatch, _frame(LanguageAffinity.EN_DOMINANT, "不应被使用"),
    )

    query = "调研近三年课堂行为分析论文\n用户澄清原文：侧重教育技术视角"
    state = _plan_state(user_query=query, frame=cached, source_query=query)

    plan_node(state, llm=None, current_year=2026)

    assert seen == []
    assert state["research_semantic_frame"]["language_affinity"] == "zh_dominant"
    assert state["language_branch_zh_ratio"] == pytest.approx(
        get_settings().language_branch_zh_ratio_zh_dominant
    )


def test_missing_source_query_forces_reparse(monkeypatch):
    """旧会话没有该字段时按失效处理，避免沿用范围收窄前的判断。"""
    from app.agent.nodes.planning import plan_node

    seen = _install_parse_spy(
        monkeypatch, _frame(LanguageAffinity.ZH_DOMINANT, "重解析后的判断"),
    )
    state = _plan_state(
        user_query="调研近三年课堂行为分析论文",
        frame=_frame(LanguageAffinity.BALANCED, "旧帧"),
        source_query="",
    )

    plan_node(state, llm=None, current_year=2026)

    assert len(seen) == 1
    assert state["language_branch_zh_ratio"] == pytest.approx(
        get_settings().language_branch_zh_ratio_zh_dominant
    )


def test_relative_time_window_refreshes_stale_semantic_frame_text(monkeypatch):
    """最终槽位年份必须覆盖 semantic frame 中 LLM 生成的旧绝对区间。"""
    from app.agent.nodes.planning import plan_node
    from app.schemas.research_plan_schema import TerminalGoal

    stale = _frame(LanguageAffinity.BALANCED, "双语研究主题").model_copy(update={
        "terminal_goal": TerminalGoal(
            type="review",
            description="总结2022-2024年的课堂互动研究",
        ),
        "assumptions": ["近三年按2022—2024年处理"],
    })
    seen = _install_parse_spy(monkeypatch, stale)
    query = "调研近三年课堂行为分析论文"
    state = _plan_state(query, stale, source_query="")

    plan_node(state, llm=None, current_year=2026)

    assert seen == [query]
    frame = state["research_semantic_frame"]
    assert "2024-2026" in frame["terminal_goal"]["description"]
    assert "2024-2026" in frame["assumptions"][0]
    assert "2022" not in frame["terminal_goal"]["description"]
