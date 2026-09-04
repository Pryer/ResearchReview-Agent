"""任务规划测试。"""

from __future__ import annotations

from app.agent.planner import (
    build_screening_protocol,
    generate_search_keywords,
    generate_search_strategy,
)


class FakeKeywordLLM:
    def complete(self, prompt: str, **kwargs) -> str:
        assert "图神经网络异常检测" in prompt
        op = str(kwargs.get("operation", ""))
        if "generate_search_keywords" in op:
            return """
            {
              "zh": [{"keyword": "图神经网络异常检测", "type": "exact"}],
              "en": [
                {"keyword": "graph neural network anomaly detection", "type": "exact"},
                {"keyword": "GNN anomaly detection", "type": "variant"}
              ]
            }
            """
        return """
        {
          "keywords": [
            "图神经网络异常检测",
            "graph neural network anomaly detection",
            "GNN anomaly detection",
            "graph outlier detection",
            "anomaly detection on attributed networks"
          ]
        }
        """


class FakeFewShotActionLLM:
    def complete(self, prompt: str, **kwargs) -> str:
        assert "少样本动作识别" in prompt
        # 区分搜索策略生成 vs 检索关键词生成 tool
        op = str(kwargs.get("operation", ""))
        if "generate_search_keywords" in op:
            return """
            {
              "zh": [
                {"keyword": "少样本视频动作识别", "type": "exact"},
                {"keyword": "少样本动作识别", "type": "broader"},
                {"keyword": "小样本动作识别", "type": "variant"}
              ],
              "en": [
                {"keyword": "few-shot video action recognition", "type": "exact"},
                {"keyword": "few-shot action recognition", "type": "broader"}
              ]
            }
            """
        return """
        {
          "keywords": [
            "少样本动作识别",
            "few-shot action recognition",
            "few-shot video action recognition",
            "few-shot human activity recognition"
          ],
          "required_concepts": [
            {"concept": "少样本", "terms": ["few-shot", "few shot", "one-shot"]},
            {"concept": "动作识别", "terms": ["action recognition", "activity recognition"]}
          ],
          "excluded_title_terms": ["zero-shot action recognition"]
        }
        """


def test_generate_search_keywords_uses_llm_for_arbitrary_topic():
    keywords = generate_search_keywords(
        "图神经网络异常检测",
        llm=FakeKeywordLLM(),
        user_query="帮我调研图神经网络异常检测相关论文",
    )

    assert keywords[0] == "图神经网络异常检测"
    assert "graph neural network anomaly detection" in keywords
    assert "GNN anomaly detection" in keywords


def test_known_topic_uses_llm_before_fallback_strategy():
    strategy = generate_search_strategy(
        "少样本动作识别",
        llm=FakeFewShotActionLLM(),
        user_query="帮我调研少样本动作识别相关论文",
    )

    assert strategy["keywords"][0] == "少样本动作识别"
    # 中文同义展开后，中文变体可能排在英文关键词前面
    assert "小样本动作识别" in strategy["keywords"]
    assert "few-shot action recognition" in strategy["keywords"]
    assert "few-shot video action recognition" in strategy["keywords"]
    assert "few-shot human activity recognition" in strategy["keywords"]
    assert "FSAR" not in strategy["keywords"]


def test_search_strategy_preserves_keyword_type_batches():
    strategy = generate_search_strategy(
        "少样本动作识别",
        llm=FakeFewShotActionLLM(),
        user_query="帮我调研少样本动作识别相关论文",
    )

    # 批次按 exact→broader→variant 组装，type 来自关键词生成工具本身
    batches = strategy["keyword_batches"]
    assert [batch["type"] for batch in batches] == ["exact", "broader", "variant"]
    assert "少样本视频动作识别" in batches[0]["keywords"]
    assert "few-shot video action recognition" in batches[0]["keywords"]
    assert "少样本动作识别" in batches[1]["keywords"]
    assert "小样本动作识别" in batches[2]["keywords"]
    # 扁平关键词池保持兼容：所有批次词都进池
    for batch in batches:
        for keyword in batch["keywords"]:
            assert keyword in strategy["keywords"]


def test_keyword_batches_are_driven_by_tool_types_not_domain_lists():
    # 图神经网络领域只有 exact 与 variant 两类 → 只产出两批；
    # 批次结构完全跟随工具返回的 type，不存在领域专属分支。
    strategy = generate_search_strategy(
        "图神经网络异常检测",
        llm=FakeKeywordLLM(),
        user_query="帮我调研图神经网络异常检测相关论文",
    )

    batches = strategy["keyword_batches"]
    assert [batch["type"] for batch in batches] == ["exact", "variant"]
    assert "图神经网络异常检测" in batches[0]["keywords"]
    assert "graph neural network anomaly detection" in batches[0]["keywords"]
    assert "GNN anomaly detection" in batches[1]["keywords"]


def test_refine_strategy_merges_new_typed_keywords_into_existing_batches():
    from app.agent.planner import refine_search_strategy

    existing = [
        {"type": "exact", "keywords": ["课堂行为分析"]},
        {"type": "variant", "keywords": ["教室行为分析"]},
    ]

    class RefineKeywordLLM(FakeKeywordLLM):
        def complete(self, prompt: str, **kwargs) -> str:
            if "generate_search_keywords" in str(kwargs.get("operation", "")):
                return """
                {
                  "zh": [
                    {"keyword": "课堂行为分析", "type": "exact"},
                    {"keyword": "课堂行为观察", "type": "variant"}
                  ],
                  "en": []
                }
                """
            return '{"keywords": ["课堂行为分析 new query"], "required_concepts": []}'

    result = refine_search_strategy(
        topic="课堂行为分析",
        user_query="调研课堂行为分析论文",
        current_keywords=["课堂行为分析", "教室行为分析"],
        feedback={"target": 10, "candidate_count": 3},
        llm=RefineKeywordLLM(),
        existing_batches=existing,
    )

    batches = {batch["type"]: batch["keywords"] for batch in result["keyword_batches"]}
    # 既有 exact 词保序在前，新 variant 词并入对应批次而不是散落词池
    assert batches["exact"] == ["课堂行为分析"]
    assert batches["variant"] == ["教室行为分析", "课堂行为观察"]


def test_generate_search_keywords_falls_back_without_llm():
    keywords = generate_search_keywords("少样本动作识别")

    assert keywords == ["少样本动作识别"]


def test_no_llm_strategy_does_not_invent_translations_or_domain_terms():
    strategy = generate_search_strategy("少样本动作识别")

    assert strategy["keywords"] == ["少样本动作识别"]
    assert strategy["topic_anchors"] == []


def test_screening_protocol_uses_multiturn_context_and_separates_corpus_routes():
    class ProtocolLLM:
        def __init__(self):
            self.prompt = ""

        def complete(self, prompt: str, **kwargs) -> str:
            self.prompt = prompt
            assert kwargs["operation"] == "screening_protocol_planning"
            return """
            {
              "corpus_goal": "技术识别、自动编码和教育解释由文献集合共同覆盖",
              "hard_include_criteria": [
                {
                  "criterion_id": "classroom_context",
                  "label": "课堂场景",
                  "terms": ["课堂", "classroom"],
                  "source": "user_explicit",
                  "applies_to_each_paper": true,
                  "rationale": "用户研究对象是课堂行为"
                }
              ],
              "soft_include_criteria": [
                {
                  "criterion_id": "education_preference",
                  "label": "教育学分析偏好",
                  "terms": ["教育学分析", "educational analysis"],
                  "source": "confirmed_scope",
                  "applies_to_each_paper": false,
                  "rationale": "用户要求偏向教育学"
                }
              ],
              "hard_exclude_title_terms": [],
              "routes": [
                {
                  "route_id": "automatic_recognition",
                  "label": "自动识别",
                  "terms": ["behavior recognition"],
                  "weight": 0.3,
                  "rationale": "上游技术路线"
                },
                {
                  "route_id": "educational_analysis",
                  "label": "教育学分析",
                  "terms": ["teacher-student interaction"],
                  "weight": 0.7,
                  "rationale": "用户偏重路线"
                }
              ],
              "notes": []
            }
            """

    llm = ProtocolLLM()
    protocol = build_screening_protocol(
        original_query="调研课堂行为分析论文",
        user_query="调研课堂行为分析论文，技术识别后做教育学分析",
        topic="课堂行为分析",
        conversation_history=[
            {"role": "user", "content": "调研课堂行为分析论文"},
            {"role": "assistant", "content": "希望侧重哪个方向？"},
            {"role": "user", "content": "先自动识别和编码，然后教育学分析，偏教育学"},
        ],
        selected_scope={
            "label": "人工智能识别与教育学分析交叉",
            "include_terms": ["课堂行为分析", "教育学分析"],
        },
        semantic_frame={"research_mode": "technology_assisted_domain_analysis"},
        search_branches=[],
        llm=llm,
    )

    assert "偏教育学" in llm.prompt
    assert protocol["generated_by"] == "llm"
    assert protocol["hard_include_criteria"][0]["criterion_id"] == "classroom_context"
    assert protocol["soft_include_criteria"][0]["applies_to_each_paper"] is False
    assert [route["weight"] for route in protocol["routes"]] == [0.3, 0.7]


def test_screening_protocol_demotes_year_window_from_text_hard_filter():
    class TemporalProtocolLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return """{
              "corpus_goal": "研究现状",
              "hard_include_criteria": [{
                "criterion_id": "publication_time_window",
                "label": "近三年发表时间",
                "terms": ["2024-2026", "近三年", "recent three years"],
                "source": "user_explicit",
                "applies_to_each_paper": true,
                "rationale": "年份要求"
              }],
              "soft_include_criteria": [],
              "hard_exclude_title_terms": [],
              "routes": [],
              "notes": []
            }"""

    protocol = build_screening_protocol(
        original_query="调研近三年目标检测论文",
        user_query="调研近三年目标检测论文",
        topic="目标检测",
        conversation_history=[{"role": "user", "content": "调研近三年目标检测论文"}],
        selected_scope={"label": "目标检测研究", "include_terms": ["目标检测"]},
        semantic_frame={},
        search_branches=[],
        llm=TemporalProtocolLLM(),
    )

    assert protocol["hard_include_criteria"] == []
    temporal = next(item for item in protocol["soft_include_criteria"] if item["criterion_id"] == "publication_time_window")
    assert temporal["applies_to_each_paper"] is False


def test_screening_protocol_coerces_dict_exclusions_into_strings():
    class ExclusionProtocolLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return """{
              "corpus_goal": "课堂行为分析",
              "hard_include_criteria": [],
              "soft_include_criteria": [],
              "hard_exclude_title_terms": [{
                "term_id": "computer_vision",
                "label": "计算机视觉",
                "description": "排除偏离主题的视觉识别路线"
              }],
              "routes": [],
              "notes": []
            }"""

    protocol = build_screening_protocol(
        original_query="调研课堂行为分析论文",
        user_query="调研课堂行为分析论文",
        topic="课堂行为分析",
        conversation_history=[{"role": "user", "content": "调研课堂行为分析论文"}],
        selected_scope={"label": "课堂行为分析", "include_terms": ["课堂行为分析"]},
        semantic_frame={},
        search_branches=[],
        llm=ExclusionProtocolLLM(),
    )

    assert protocol["generated_by"] == "llm"
    assert protocol["hard_exclude_title_terms"] == []


class FakeMonolingualAnchorLLM:
    """首轮与再生成均返回纯英文概念组（中文主题）。"""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        return """
        {
          "keywords": ["少样本动作识别", "few-shot action recognition"],
          "topic_anchors": [
            {"concept": "少样本", "terms": ["few-shot", "one-shot"]},
            {"concept": "动作识别", "terms": ["action recognition"]}
          ]
        }
        """


class FakeBilingualRetryLLM:
    """首轮返回纯英文概念组，定向再生成后返回合规双语组。"""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        if self.calls == 1:
            return """
            {
              "keywords": ["少样本动作识别", "few-shot action recognition"],
              "topic_anchors": [
                {"concept": "少样本", "terms": ["few-shot", "one-shot"]}
              ]
            }
            """
        assert "缺少中文术语" in prompt
        return """
        {
          "keywords": ["少样本动作识别", "few-shot action recognition"],
          "topic_anchors": [
            {"concept": "少样本", "terms": ["few-shot", "少样本", "小样本"]},
            {"concept": "动作识别", "terms": ["action recognition", "动作识别"]}
          ]
        }
        """


def test_generate_search_strategy_drops_monolingual_anchors_after_retry():
    """再生成仍不合规的单语概念组被显式丢弃并记录诊断，不静默流入打分链路。"""
    llm = FakeMonolingualAnchorLLM()
    strategy = generate_search_strategy(
        "少样本动作识别", llm=llm, user_query="调研少样本动作识别论文",
    )

    assert llm.calls >= 2  # 触发了一次定向再生成
    assert strategy["topic_anchors"] == []
    assert len(strategy["dropped_monolingual_groups"]) == 2
    assert "few-shot action recognition" in strategy["keywords"]


def test_generate_search_strategy_adopts_bilingual_retry_anchors():
    """定向再生成产出双语组后整体采用，首轮丢弃仍保留诊断。"""
    llm = FakeBilingualRetryLLM()
    strategy = generate_search_strategy(
        "少样本动作识别", llm=llm, user_query="调研少样本动作识别论文",
    )

    assert llm.calls >= 2
    flat_terms = [t for group in strategy["topic_anchors"] for t in group]
    assert "少样本" in flat_terms
    assert "few-shot" in flat_terms
    assert len(strategy["dropped_monolingual_groups"]) == 1  # 首轮丢弃被记录


def test_generate_search_strategy_consumes_list_form_anchors():
    """list 形态概念组不再被 dict 过滤器静默丢弃（管道断点回归测试）。"""

    class ListAnchorLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return """
            {
              "keywords": ["少样本动作识别", "few-shot action recognition"],
              "topic_anchors": [
                {"concept": "少样本", "terms": ["few-shot", "少样本"]},
                {"concept": "动作识别", "terms": ["action recognition", "动作识别"]}
              ]
            }
            """

    strategy = generate_search_strategy(
        "少样本动作识别", llm=ListAnchorLLM(), user_query="调研少样本动作识别论文",
    )

    assert len(strategy["topic_anchors"]) == 2
    assert strategy["dropped_monolingual_groups"] == []


def test_clean_keyword_pool_collapses_mixed_garbage():
    """混杂垃圾词在词池合并前拆出中文段，含锚点的词保持原样。"""
    from app.agent.planner import _clean_keyword_pool

    cleaned = _clean_keyword_pool(
        [
            "少样本动作识别",
            "少样本学习 few-shot learning human action",
            "少样本学习 human action",
            "survey 近三年少样本动作识别研究",
            "few-shot action recognition",
        ],
        topic="少样本动作识别",
    )
    assert cleaned == [
        "少样本动作识别",
        "少样本学习",
        "少样本学习",
        # 含锚点的词不在此层清洗，留给派发层，避免锚点判定失配
        "survey 近三年少样本动作识别研究",
        "few-shot action recognition",
    ]


def test_generate_search_strategy_dedupes_cleaned_garbage():
    """垃圾词清洗后与既有词重复的，应在词池去重阶段被合并。"""

    class GarbageLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            op = str(kwargs.get("operation", ""))
            if "generate_search_keywords" in op:
                return '{"zh": [], "en": []}'
            return """
            {
              "keywords": [
                "少样本学习",
                "少样本学习 few-shot learning human action",
                "few-shot action recognition",
                "few-shot learning action recognition"
              ]
            }
            """

    strategy = generate_search_strategy(
        "少样本动作识别", llm=GarbageLLM(), user_query="调研少样本动作识别论文",
    )
    keywords = strategy["keywords"]

    assert "少样本动作识别" in keywords
    # 混杂词拆出的"少样本学习"与既有词合并，只保留一份
    assert keywords.count("少样本学习") == 1
    assert not any("few-shot learning human action" in kw for kw in keywords)
