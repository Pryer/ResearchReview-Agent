"""意图识别模块测试。"""

from __future__ import annotations

import pytest

from app.agent.intent import (
    llm_based_intent_recognition,
    recognize_intent,
    rule_based_intent_recognition,
)
from app.schemas.agent_schema import IntentType


class TestRuleBasedIntentRecognition:
    """规则意图识别测试。"""

    @pytest.mark.parametrize(
        "query, expected_intent",
        [
            ("帮我写一篇关于图像分类的文献综述", IntentType.GENERATE_REVIEW),
            ("请帮我调研近五年 Transformer 的研究现状", IntentType.GENERATE_REVIEW),
            ("生成计算机视觉领域的 survey", IntentType.GENERATE_REVIEW),
            ("帮我找几篇关于目标检测的论文", IntentType.SEARCH_PAPERS),
            ("搜索 Vision Transformer 相关论文", IntentType.SEARCH_PAPERS),
            ("推荐一些 action recognition 的论文", IntentType.SEARCH_PAPERS),
            ("总结这篇论文的主要内容", IntentType.READ_PAPER),
            ("帮我读一下这篇 CVPR 论文", IntentType.READ_PAPER),
            ("对比 CNN 和 Transformer 的区别", IntentType.COMPARE_PAPERS),
            ("比较两者的性能差异", IntentType.COMPARE_PAPERS),
            ("生成 bibtex 格式的参考文献", IntentType.GENERATE_REFERENCES),
            ("请给出 GB/T 7714 引用格式", IntentType.GENERATE_REFERENCES),
            ("有哪些常用的图像分割数据集", IntentType.FIND_DATASETS),
            ("当前的研究热点是什么", IntentType.FIND_TRENDS),
        ],
    )
    def test_intent_recognition(self, query: str, expected_intent: IntentType):
        """测试规则是否能正确匹配各类意图。"""
        result = rule_based_intent_recognition(query)
        assert result.intent == expected_intent.value, (
            f"Query '{query}' → expected {expected_intent.value}, got {result.intent}"
        )
        assert result.confidence >= 0.5

    def test_empty_query_returns_general_qa(self):
        """空请求应返回 general_qa 意图。"""
        result = rule_based_intent_recognition("")
        assert result.intent == IntentType.GENERAL_QA.value

    def test_recognize_intent_fallback(self):
        """recognize_intent 应遵循规则优先策略。"""
        result = recognize_intent("帮我写一篇文献综述")
        assert result.intent == IntentType.GENERATE_REVIEW.value

    @pytest.mark.parametrize(
        "query",
        [
            "few-shot action recognition",
            "retrieval augmented generation hallucination",
            "RAG",
            "课堂行为分析",
        ],
    )
    def test_bare_research_topic_defaults_to_review(self, query: str):
        result = rule_based_intent_recognition(query)
        assert result.intent == IntentType.GENERATE_REVIEW.value
        assert result.confidence >= 0.7

    @pytest.mark.parametrize("query", ["你好", "谢谢", "什么是RAG？", "这是一段测试文本"])
    def test_conversation_is_not_treated_as_bare_topic(self, query: str):
        result = rule_based_intent_recognition(query)
        assert result.intent == IntentType.GENERAL_QA.value

    @pytest.mark.parametrize(
        "query",
        [
            "课堂行为数据分析",
            "多模态学习中视频数据的综述研究",
            "大数据环境下的舆情治理",
        ],
    )
    def test_generic_data_word_does_not_trigger_dataset_intent(self, query: str):
        """裸词“数据”不得以高置信度把研究主题误判为数据集查找。"""
        result = rule_based_intent_recognition(query)
        assert result.intent != IntentType.FIND_DATASETS.value, query

    def test_explicit_dataset_request_still_matches(self):
        result = rule_based_intent_recognition("有哪些常用的图像分割数据集")
        assert result.intent == IntentType.FIND_DATASETS.value

    def test_confidence_in_valid_range(self):
        """置信度必须在 0~1 范围内。"""
        result = rule_based_intent_recognition("这是一段测试文本")
        assert 0 <= result.confidence <= 1.0


class TestConversationAwareIntentRecognition:
    def test_clarification_answer_inherits_prior_intent_without_domain_rules(self):
        result = recognize_intent(
            "从实证视角出发，重点看编码和模式解释",
            conversation_role="clarification_answer",
            previous_intent=IntentType.GENERATE_REVIEW,
            original_query="调研某个研究主题并生成综述",
        )

        assert result.intent == IntentType.GENERATE_REVIEW.value
        assert result.confidence == 1.0

    def test_internal_working_query_is_not_reclassified_by_appended_text(self):
        result = recognize_intent(
            "生成某个主题的综述\n范围说明：查找数据集与趋势证据",
            conversation_role="working_query",
            previous_intent=IntentType.GENERATE_REVIEW.value,
            original_query="生成某个主题的综述",
        )

        assert result.intent == IntentType.GENERATE_REVIEW.value

    def test_invalid_llm_intent_is_bounded_to_supported_enum(self):
        class InvalidIntentLLM:
            def complete(self, prompt, **kwargs):
                return '{"intent":"invented_domain_task","confidence":0.99,"reason":"x"}'

        result = llm_based_intent_recognition("请处理这个学术请求", InvalidIntentLLM())

        assert result.intent == IntentType.GENERAL_QA.value

    def test_intent_prompt_marks_turn_role_and_original_request(self):
        class CaptureLLM:
            prompt = ""

            def complete(self, prompt, **kwargs):
                self.prompt = prompt
                return '{"intent":"generate_review","confidence":0.8,"reason":"context"}'

        llm = CaptureLLM()
        llm_based_intent_recognition(
            "补充的范围约束",
            llm,
            conversation_role="working_query",
            previous_intent=IntentType.GENERATE_REVIEW,
            original_query="原始调研请求",
        )

        assert "当前文本角色：working_query" in llm.prompt
        assert "原始用户请求：原始调研请求" in llm.prompt
