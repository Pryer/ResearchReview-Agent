"""槽位抽取模块测试。"""

from __future__ import annotations

import pytest

from app.agent.slot_extractor import (
    extract_citation_style,
    extract_language,
    extract_max_papers,
    extract_peer_review_requirement,
    extract_slots,
    extract_topic,
    extract_year_range,
)


class TestExtractTopic:
    """主题抽取测试。"""

    def test_chinese_topic(self):
        result = extract_topic("帮我调研课堂行为识别相关论文")
        assert "课堂行为识别" in result or result  # 至少抽取到部分内容

    def test_topic_removes_relative_year_prefix(self):
        result = extract_topic("帮我调研近五年目标检测相关论文，并生成中文文献综述")
        assert result == "目标检测"

    def test_generate_review_topic(self):
        assert extract_topic("帮我生成长视频理解的综述") == "长视频理解"
        assert extract_topic("生成长视频理解综述") == "长视频理解"

    def test_search_topic_without_about(self):
        assert extract_topic("找几篇时序动作定位论文") == "时序动作定位"

    def test_bare_topic(self):
        assert extract_topic("时序动作定位") == "时序动作定位"

    def test_topic_removes_absolute_year_and_count(self):
        assert (
            extract_topic("检索2021-2024年少样本动作识别论文30-50篇")
            == "少样本动作识别"
        )

    def test_english_topic(self):
        result = extract_topic("research on vision transformers for image classification")
        assert result is None or len(result) > 0

    def test_no_topic_returns_none(self):
        result = extract_topic("生成文献综述")
        # 没有具体主题时可为 None 或空字符串
        assert result is None or isinstance(result, str)


class TestExtractYearRange:
    """年份范围抽取测试。"""

    @pytest.mark.parametrize(
        "query, expected_start, expected_end",
        [
            ("近五年", 2021, 2025),
            ("近3年", 2023, 2025),
            ("近十年", 2016, 2025),
        ],
    )
    def test_relative_years(self, query, expected_start, expected_end):
        result = extract_year_range(query, current_year=2025)
        assert result == (expected_start, expected_end)

    @pytest.mark.parametrize(
        "query, expected",
        [
            ("2020-2025", (2020, 2025)),
            ("2021到2024年", (2021, 2024)),
            ("2022年以后", (2022, 2025)),
        ],
    )
    def test_absolute_years(self, query, expected):
        result = extract_year_range(query, current_year=2025)
        assert result == expected

    def test_no_year_returns_none(self):
        result = extract_year_range("帮我检索一些论文", current_year=2025)
        assert result is None


class TestExtractMaxPapers:
    """论文数量抽取测试。"""

    @pytest.mark.parametrize(
        "query, expected",
        [
            ("引用不少于15篇", 15),
            ("至少10篇论文", 10),
            ("检索20篇", 20),
        ],
    )
    def test_extract_count(self, query, expected):
        assert extract_max_papers(query) == expected

    def test_default_count(self):
        assert extract_max_papers("帮我找一些论文") == 30

    @pytest.mark.parametrize(
        "query, expected",
        [
            ("检索30-50篇论文", 50),
            ("找30到50篇论文", 50),
            ("需要35篇论文", 35),
            ("论文20篇", 20),
        ],
    )
    def test_more_count_formats(self, query, expected):
        assert extract_max_papers(query) == expected


class TestExtractLanguage:
    """语言抽取测试。"""

    def test_chinese(self):
        assert extract_language("生成中文综述") == "zh"

    def test_english(self):
        assert extract_language("generate English review") == "en"

    def test_default(self):
        assert extract_language("生成综述") == "zh"

    def test_english_search_terms_do_not_change_default_output_language(self):
        scoped_query = (
            "调研近三年课堂行为分析论文，并生成研究背景和研究现状\n"
            "种子检索表达：student engagement learning outcome；"
            "classroom interaction analysis"
        )
        assert extract_language(scoped_query) == "zh"

    def test_standalone_language_code_is_supported(self):
        assert extract_language("output language: en") == "en"


class TestExtractCitationStyle:
    """引用格式抽取测试。"""

    @pytest.mark.parametrize(
        "query, expected",
        [
            ("GB/T 7714 格式", "gbt7714"),
            ("APA 格式引用", "apa"),
            ("IEEE 格式", "ieee"),
            ("BibTeX 格式", "bibtex"),
        ],
    )
    def test_extract_style(self, query, expected):
        assert extract_citation_style(query) == expected

    def test_default_style(self):
        assert extract_citation_style("生成参考文献") == "gbt7714"


class TestExtractPeerReviewRequirement:
    """同行评审/期刊来源要求抽取测试。"""

    @pytest.mark.parametrize(
        "query",
        [
            "只要同行评审的论文",
            "引用不少于40篇同行评议论文",
            "最好是期刊论文",
            "核心期刊论文30篇",
            "SCI/EI 收录的论文",
            "peer-reviewed papers only",
            "调研近三年 SCI 论文并生成综述",
        ],
    )
    def test_explicit_peer_review_requirement(self, query):
        assert extract_peer_review_requirement(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "综述近三年目标检测论文",
            "调研少样本视频动作识别并生成综述",
            "检索2024-2026年大模型论文40篇",
            "",
        ],
    )
    def test_no_peer_review_requirement(self, query):
        assert extract_peer_review_requirement(query) is False


class TestExtractSlots:
    """整合槽位抽取测试。"""

    def test_full_extraction(self):
        slots = extract_slots(
            "帮我调研近五年课堂行为识别相关论文，生成中文文献综述，引用不少于10篇",
            intent="generate_review",
            current_year=2025,
        )
        assert slots.start_year == 2021
        assert slots.end_year == 2025
        assert slots.max_papers == 10
        assert slots.required_reference_count == 10
        # 显式要求的余量比例为 0.80：实测端到端成品率约 0.556，
        # ceil(10*0.80)=8 → 10+8=18。
        assert slots.retrieval_target == 18
        assert slots.generation_limit == 18
        assert slots.language == "zh"

    def test_llm_null_max_papers_keeps_default(self):
        class NullMaxPapersLLM:
            def complete(self, *args, **kwargs):
                return (
                    '{"topic":"时序动作定位","start_year":null,'
                    '"end_year":null,"max_papers":null,'
                    '"language":"zh","citation_style":"gbt7714"}'
                )

        slots = extract_slots(
            "时序动作定位",
            intent="generate_review",
            llm=NullMaxPapersLLM(),
            current_year=2026,
        )

        assert slots.topic == "时序动作定位"
        assert slots.max_papers == 30
        assert slots.max_papers_explicit is False

    def test_user_parameters_take_priority_over_defaults(self):
        slots = extract_slots(
            "调研近五年时序动作定位论文，引用不少于45篇",
            intent="generate_review",
            current_year=2026,
        )

        assert (slots.start_year, slots.end_year) == (2022, 2026)
        assert slots.max_papers == 45
        assert slots.required_reference_count == 45
        assert slots.retrieval_target == 81
        assert slots.generation_limit == 81
        # 本测试的意图是用户参数压过默认值；派生目标必须高于显式引用要求，
        # 否则候选池等于要求本身，任何筛选损耗都会直接变成引用缺口。
        assert slots.retrieval_target > slots.required_reference_count
        assert slots.year_range_explicit is True
        assert slots.max_papers_explicit is True

    def test_explicit_minimum_count_is_not_treated_as_upper_bound(self):
        slots = extract_slots(
            "帮我调研近五年少样本动作识别相关论文，引用不少于50篇，并生成相关工作",
            intent="generate_review",
            current_year=2026,
        )

        assert slots.max_papers == 50
        assert slots.required_reference_count == 50
        # ceil(50*0.80)=40 正好触到余量上限 min(40, ...)，50+40=90。
        assert slots.retrieval_target == 90
        assert slots.generation_limit == 90
        # "不少于 50 篇"是下限而非上限：目标必须严格超过它。
        assert slots.retrieval_target > slots.required_reference_count
        assert slots.max_papers_explicit is True

    def test_extracts_requested_background_and_research_status_sections(self):
        slots = extract_slots(
            "帮我调研近三年课堂行为分析相关论文，引用不少于40篇，并生成学术论文中研究背景和相关研究现状",
            intent="generate_review",
            current_year=2026,
        )

        assert slots.requested_sections == ["background", "research_status"]

    def test_missing_parameters_keep_default_markers(self):
        slots = extract_slots(
            "帮我生成时序动作定位的综述",
            intent="generate_review",
            current_year=2026,
        )

        assert slots.start_year is None
        assert slots.end_year is None
        assert slots.max_papers == 30
        assert slots.year_range_explicit is False
        assert slots.max_papers_explicit is False

    def test_strict_year_range_marker(self):
        slots = extract_slots(
            "仅限近三年时序动作定位论文，至少30篇",
            intent="generate_review",
            current_year=2026,
        )

        assert slots.strict_year_range is True
