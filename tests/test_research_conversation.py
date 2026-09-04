"""主题消歧与多轮研究会话测试。"""

from __future__ import annotations

import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.topic_disambiguation import (
    analyze_topic_ambiguity,
    build_scoped_query,
    reconcile_selected_scope_from_history,
    resolve_scope,
    resolve_scope_conversational,
)
from app.database.models import Base
from app.database.repositories import ResearchSessionRepository
from app.schemas.agent_schema import AgentRequest
from app.services.research_conversation_service import ResearchConversationService


class AmbiguousLLM:
    def complete(self, prompt: str, **kwargs) -> str:
        assert "结构化约束" in prompt
        return """
        {
          "ambiguous": true,
          "confidence": 0.93,
          "reason": "该主题在两个研究共同体中具有不同分析单位",
          "recommended_strategy": "ask_user",
          "default_scope_id": "scope_a",
          "question": "请选择研究范围",
          "scopes": [
            {
              "scope_id": "scope_a",
              "label": "范围甲",
              "description": "关注研究对象甲",
              "include_terms": ["概念甲"],
              "exclude_terms": ["概念乙"],
              "seed_queries": ["topic perspective a"]
            },
            {
              "scope_id": "scope_b",
              "label": "范围乙",
              "description": "关注研究对象乙",
              "include_terms": ["概念乙"],
              "exclude_terms": ["概念甲"],
              "seed_queries": ["topic perspective b"]
            }
          ]
        }
        """


class ConversationalScopeLLM:
    def complete(self, prompt: str, **kwargs) -> str:
        if "结构化约束" in prompt:
            return """
            {
              "ambiguous": true,
              "confidence": 0.96,
              "reason": "技术识别与教育解释使用不同语料",
              "recommended_strategy": "ask_user",
              "default_scope_id": "technical",
              "question": "你希望课堂行为分析偏向技术识别、教育研究，还是二者结合的交叉分析？",
              "scopes": [
                {
                  "scope_id": "technical",
                  "label": "技术识别",
                  "description": "使用视觉或多模态算法识别课堂行为",
                  "include_terms": ["行为识别", "计算机视觉"],
                  "exclude_terms": ["纯课堂话语分析"],
                  "seed_queries": ["classroom behavior recognition"]
                },
                {
                  "scope_id": "education",
                  "label": "教育研究",
                  "description": "研究课堂观察、互动与教学行为",
                  "include_terms": ["课堂观察", "师生互动"],
                  "exclude_terms": ["纯算法检测"],
                  "seed_queries": ["classroom interaction analysis"]
                },
                {
                  "scope_id": "interdisciplinary",
                  "label": "技术与教育交叉",
                  "description": "以教育构念定义任务并用技术自动分析",
                  "include_terms": ["多模态学习分析", "自动行为编码"],
                  "exclude_terms": [],
                  "seed_queries": ["technology assisted classroom behavior analysis"]
                }
              ]
            }
            """
        if "用户回答：我还没想好" in prompt:
            return """
            {"matched_scope_ids": [], "needs_clarification": true,
             "question": "你更希望论文重点回答如何自动识别行为，还是这些行为如何影响教学？",
             "reason": "回答未提供范围信息"}
            """
        if any(term in prompt for term in ("自动编码", "教育学分析", "S-T", "滞后")):
            return """
            {"matched_scope_ids": ["interdisciplinary"], "needs_clarification": false,
             "question": null, "reason": "回答同时包含技术处理和下游解释"}
            """
        return """
        {"matched_scope_ids": ["technical"], "needs_clarification": false,
         "question": null, "reason": "用户提到摄像头和动作识别"}
        """


class ScopeSelectionLLM:
    def __init__(self, scope_id: str):
        self.scope_id = scope_id

    def complete(self, prompt: str, **kwargs) -> str:
        return json.dumps({
            "matched_scope_ids": [self.scope_id],
            "needs_clarification": False,
            "question": None,
            "reason": "测试中的回答与该候选范围最匹配",
        }, ensure_ascii=False)


class PipelineConversationLLM:
    """为会话编排提供动态语义与范围响应，避免单元测试访问真实 API。"""

    def complete(self, prompt: str, **kwargs) -> str:
        operation = str(kwargs.get("operation") or "")
        if operation == "research_semantic_parsing":
            if "S-T分析法" not in prompt:
                return json.dumps({
                    "canonical_topic": "课堂行为分析",
                    "application_domains": [], "research_objects": [], "methods": [],
                    "research_actions": [], "analysis_targets": [],
                    "terminal_goal": {"type": "unspecified"},
                    "clarification_needed": True,
                    "scope_ambiguities": ["技术识别与教育解释范围未确定"],
                    "confidence": {"overall": 0.6},
                }, ensure_ascii=False)
            return json.dumps({
                "canonical_topic": "课堂行为分析",
                "application_domains": [{
                    "id": "education", "label": "education", "surface_text": "课堂",
                    "explicit": True, "confidence": 0.95,
                }],
                "research_objects": [{
                    "id": "teacher_student_behavior", "label": "teacher-student behavior",
                    "surface_text": "教师和学生行为", "explicit": True, "confidence": 0.95,
                }],
                "methods": [
                    {"id": "action_recognition", "label": "action recognition", "surface_text": "自动识别", "category": "technical", "role": "intermediate_step", "explicit": True, "confidence": 0.95},
                    {"id": "st_analysis", "label": "S-T analysis", "surface_text": "S-T分析法", "category": "analytical", "explicit": True, "confidence": 0.95},
                    {"id": "lag_sequential_analysis", "label": "lag sequential analysis", "surface_text": "滞后序列分析法", "category": "analytical", "explicit": True, "confidence": 0.95},
                ],
                "research_actions": [{
                    "id": "automatic_behavior_coding", "label": "automatic behavior coding",
                    "surface_text": "自动行为编码", "explicit": True, "confidence": 0.95,
                }],
                "analysis_targets": [{
                    "id": "teaching_structure_and_interaction", "label": "teaching structure and interaction",
                    "surface_text": "教学结构与师生互动", "explicit": True, "confidence": 0.95,
                }],
                "terminal_goal": {"type": "domain_analysis", "target": "teaching_structure_and_interaction"},
                "task_chain": ["behavior_recognition", "automatic_behavior_coding", "st_or_lag_sequential_analysis", "teaching_structure_and_interaction_interpretation"],
                "required_focuses": ["教师与学生行为自动识别", "自动行为编码", "S-T分析法或滞后序列分析法", "教学结构与师生互动解释"],
                "confidence": {"overall": 0.95},
            }, ensure_ascii=False)
        if operation == "topic_disambiguation":
            return ConversationalScopeLLM().complete(prompt, **kwargs)
        if operation == "scope_answer_resolution":
            return ScopeSelectionLLM("interdisciplinary").complete(prompt, **kwargs)
        return '{"intent":"generate_review","confidence":0.95,"reason":"研究请求"}'


class ClearTopicLLM:
    def __init__(self):
        self.called = False

    def complete(self, prompt: str, **kwargs) -> str:
        self.called = True
        return """
        {
          "ambiguous": false,
          "confidence": 0.96,
          "reason": "这是边界明确的标准动作识别任务",
          "recommended_strategy": "single_scope",
          "default_scope_id": null,
          "question": null,
          "scopes": []
        }
        """


def _db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_ambiguity_analysis_preserves_hard_research_constraints():
    result = analyze_topic_ambiguity(
        "调研近三年课堂行为分析论文，引用不少于40篇，并生成研究背景和研究现状",
        llm=AmbiguousLLM(),
        current_year=2026,
    )

    request = result["research_request"]
    assert result["needs_clarification"] is True
    assert (request["start_year"], request["end_year"]) == (2024, 2026)
    assert request["required_reference_count"] == 40
    assert request["requested_sections"] == ["background", "research_status"]


def test_clear_bare_research_topic_is_analyzed_without_clarification():
    llm = ClearTopicLLM()
    result = analyze_topic_ambiguity(
        "few-shot action recognition",
        llm=llm,
        current_year=2026,
    )
    assert llm.called is True
    assert result["research_request"]["task_type"] == "generate_review"
    assert result["needs_clarification"] is False


def test_no_llm_does_not_fabricate_scope_options():
    result = analyze_topic_ambiguity(
        "调研近三年课堂行为分析论文，并生成研究背景和研究现状",
        llm=None,
        current_year=2026,
    )

    assert result["needs_clarification"] is False
    assert result["ambiguity"]["recommended_strategy"] == "single_scope"
    assert result["ambiguity"]["scopes"] == []


def test_llm_resolves_technology_assisted_domain_answer():
    clarification = analyze_topic_ambiguity(
        "调研近三年课堂行为分析论文",
        llm=ConversationalScopeLLM(),
        current_year=2026,
    )["ambiguity"]
    result = resolve_scope_conversational(
        clarification,
        "基于人工智能技术自动识别行为并自动编码，然后基于教育学分析编码结果",
        llm=ConversationalScopeLLM(),
    )

    assert result["needs_clarification"] is False
    assert result["selected_scope"]["scope_id"] == "interdisciplinary"


def test_llm_treats_st_and_lag_analysis_as_cross_domain_goal():
    clarification = analyze_topic_ambiguity(
        "调研近三年课堂行为分析论文",
        llm=ConversationalScopeLLM(),
        current_year=2026,
    )["ambiguity"]
    result = resolve_scope_conversational(
        clarification,
        "先用人工智能自动识别和行为编码，然后用S-T分析法或滞后分析法进行分析",
        llm=ConversationalScopeLLM(),
    )

    assert result["needs_clarification"] is False
    assert result["selected_scope"]["scope_id"] == "interdisciplinary"
    assert "自动行为编码" in result["selected_scope"]["include_terms"]


def test_saved_technical_scope_is_reconciled_from_original_clarification_answer():
    clarification = analyze_topic_ambiguity(
        "调研近三年课堂行为分析论文",
        llm=ConversationalScopeLLM(),
        current_year=2026,
    )["ambiguity"]
    repaired = reconcile_selected_scope_from_history(
        {"scope_id": "automatic_recognition", "label": "人工智能自动识别"},
        clarification["scopes"],
        [{
            "role": "user",
            "type": "clarification_answer",
            "content": "先自动识别和编码，再使用S-T分析法或滞后分析法",
        }],
    )

    assert repaired["scope_id"] == "interdisciplinary"


def test_mixed_answer_does_not_choose_first_scope_when_all_modes_are_mislabeled():
    clarification = {
        "question": "请选择技术路线",
        "scopes": [
            {
                "scope_id": "vision_based",
                "label": "基于计算机视觉的课堂行为分析",
                "description": "利用视频识别课堂动作",
                "include_terms": ["动作识别"],
                "exclude_terms": ["课堂观察"],
                "seed_queries": ["classroom action recognition"],
                "research_mode": "mixed",
            },
            {
                "scope_id": "educational_data_mining",
                "label": "基于教育数据挖掘的课堂行为分析",
                "description": "利用日志分析学习效果",
                "include_terms": ["学习分析"],
                "exclude_terms": ["视频"],
                "seed_queries": ["learning analytics"],
                "research_mode": "mixed",
            },
            {
                "scope_id": "comprehensive",
                "label": "涵盖所有技术路线",
                "description": "综合计算机视觉、多模态和日志挖掘",
                "include_terms": ["课堂行为分析"],
                "exclude_terms": [],
                "seed_queries": ["classroom behavior analysis"],
                "research_mode": "mixed",
            },
        ],
    }

    result = resolve_scope_conversational(
        clarification,
        "先基于人工智能技术进行动作识别和编码，然后基于教育学进行分析，偏教育",
        llm=ScopeSelectionLLM("comprehensive"),
    )

    selected = result["selected_scope"]
    assert selected["scope_id"] == "comprehensive"
    assert "课堂行为分析" in selected["include_terms"]
    assert selected["exclude_terms"] == []


def test_mixed_pipeline_selects_cross_scope_instead_of_pure_education_scope():
    clarification = {
        "scopes": [
            {
                "scope_id": "education_focused",
                "label": "教育视角",
                "description": "基于教育学和课堂观察研究师生互动",
                "include_terms": ["课堂互动", "学生参与"],
                "exclude_terms": ["计算机视觉", "动作识别"],
                "seed_queries": ["classroom interaction"],
                "research_mode": "mixed",
            },
            {
                "scope_id": "technology_focused",
                "label": "技术视角",
                "description": "利用计算机视觉自动识别课堂行为并评价算法性能",
                "include_terms": ["动作识别", "计算机视觉"],
                "exclude_terms": ["师生互动"],
                "seed_queries": ["classroom action recognition"],
                "research_mode": "mixed",
            },
            {
                "scope_id": "cross_disciplinary",
                "label": "跨学科综合",
                "description": "以自动化行为识别衔接教育理论、教学结构和师生互动分析",
                "include_terms": ["行为识别", "师生互动", "教学结构"],
                "exclude_terms": [],
                "seed_queries": ["automatic behavior recognition educational analysis"],
                "research_mode": "mixed",
            },
        ]
    }
    result = resolve_scope_conversational(
        clarification,
        "先用人工智能自动识别和编码，再采用S-T或滞后序列分析法进行分析",
        llm=ScopeSelectionLLM("cross_disciplinary"),
    )

    assert result["selected_scope"]["scope_id"] == "cross_disciplinary"
    assert result["selected_scope"]["exclude_terms"] == []


def test_scope_resolution_accepts_id_number_and_label():
    clarification = analyze_topic_ambiguity(
        "调研近三年课堂行为分析论文",
        llm=AmbiguousLLM(),
        current_year=2026,
    )["ambiguity"]

    assert resolve_scope(clarification, "scope_b")["label"] == "范围乙"
    assert resolve_scope(clarification, "2")["scope_id"] == "scope_b"
    assert resolve_scope(clarification, "范围甲")["scope_id"] == "scope_a"
    scoped = build_scoped_query("原始请求", resolve_scope(clarification, "scope_a"))
    assert "原始请求" in scoped
    assert "优先纳入：概念甲" in scoped
    assert "排除相邻含义：概念乙" in scoped


def test_conversational_scope_resolution_understands_semantic_answer():
    clarification = analyze_topic_ambiguity(
        "调研课堂行为分析论文",
        llm=ConversationalScopeLLM(),
        current_year=2026,
    )["ambiguity"]
    result = resolve_scope_conversational(
        clarification,
        "我主要想研究怎么通过摄像头自动识别学生动作",
        llm=ConversationalScopeLLM(),
    )
    assert result["needs_clarification"] is False
    assert result["selected_scope"]["scope_id"] == "technical"


def test_research_session_repository_round_trip():
    db = _db_session()
    repo = ResearchSessionRepository(db)
    repo.save(
        session_id="session-1",
        status="needs_clarification",
        original_query="原始请求",
        state={"required_reference_count": 40},
        clarification={"question": "请选择"},
    )
    db.commit()

    saved = repo.get("session-1")
    assert saved["status"] == "needs_clarification"
    assert saved["state"]["required_reference_count"] == 40
    assert saved["clarification"]["question"] == "请选择"


def test_conversation_service_resumes_original_request_after_scope_selection():
    db = _db_session()
    calls = []

    def fake_runner(query, db=None, initial_state=None):
        calls.append({"query": query, "state": initial_state})
        return {
            "answer": "完成",
            "intent": "generate_review",
            "topic": "课堂行为分析",
            "steps": [],
            "references": [],
            "paper_cards": [],
            "clusters": [],
            "errors": [],
        }

    service = ResearchConversationService(
        db,
        llm=AmbiguousLLM(),
        agent_runner=fake_runner,
    )
    first = service.handle(
        AgentRequest(
            user_query="调研近三年课堂行为分析论文，引用不少于40篇，并生成研究背景和研究现状",
            session_id="session-2",
        )
    )

    assert first["status"] == "needs_clarification"
    assert calls == []

    resumed = service.handle(
        AgentRequest(
            user_query="范围甲",
            session_id="session-2",
            clarification_answer="scope_a",
        )
    )

    assert resumed["status"] == "completed"
    assert len(calls) == 1
    assert "调研近三年课堂行为分析" in calls[0]["query"]
    assert "研究范围确认：范围甲" in calls[0]["query"]
    assert calls[0]["state"]["research_request"]["required_reference_count"] == 40
    assert calls[0]["state"]["selected_scope"]["scope_id"] == "scope_a"
    assert ResearchSessionRepository(db).get("session-2")["status"] == "completed"


def test_waiting_session_treats_next_message_as_clarification_without_flag():
    db = _db_session()
    calls = []

    def fake_runner(query, db=None, initial_state=None):
        calls.append(initial_state)
        return {
            "answer": "完成",
            "intent": "generate_review",
            "topic": "课堂行为分析",
            "steps": [],
            "references": [],
            "paper_cards": [],
            "clusters": [],
            "errors": [],
        }

    service = ResearchConversationService(db, llm=AmbiguousLLM(), agent_runner=fake_runner)
    first = service.handle(AgentRequest(
        user_query="调研近三年课堂行为分析论文，引用不少于40篇",
        session_id="implicit-clarification",
    ))
    assert first["status"] == "needs_clarification"

    resumed = service.handle(AgentRequest(
        user_query="scope_a",
        session_id="implicit-clarification",
    ))

    assert resumed["status"] == "completed"
    assert len(calls) == 1
    assert calls[0]["selected_scope"]["scope_id"] == "scope_a"


def test_new_turn_reuses_one_session_and_preserves_conversation_history():
    db = _db_session()
    calls = []

    def fake_runner(query, db=None, initial_state=None):
        calls.append(initial_state)
        return {
            "answer": "完成",
            "intent": "generate_review",
            "topic": "课堂行为分析",
            "steps": [],
            "references": [],
            "paper_cards": [],
            "clusters": [],
            "errors": [],
        }

    service = ResearchConversationService(db, llm=ClearTopicLLM(), agent_runner=fake_runner)
    first = service.handle(AgentRequest(
        user_query="基于YOLO的课堂行为识别论文",
        session_id="one-conversation",
    ))
    assert first["status"] == "completed"

    second = service.handle(AgentRequest(
        user_query="继续调研师生互动序列分析",
        session_id="one-conversation",
    ))
    assert second["status"] == "completed"
    assert len(calls) == 2
    history = calls[1]["conversation_history"]
    assert [item["content"] for item in history if item["role"] == "user"] == [
        "基于YOLO的课堂行为识别论文",
        "继续调研师生互动序列分析",
    ]
    assert ResearchSessionRepository(db).get("one-conversation")["original_query"] == (
        "继续调研师生互动序列分析"
    )


def test_clarification_methods_are_preserved_in_resumed_semantic_frame():
    db = _db_session()
    calls = []

    def fake_runner(query, db=None, initial_state=None):
        calls.append({"query": query, "state": initial_state})
        return {
            "answer": "完成", "intent": "generate_review", "topic": "课堂行为分析",
            "steps": [], "references": [], "paper_cards": [], "clusters": [], "errors": [],
        }

    service = ResearchConversationService(
        db, llm=PipelineConversationLLM(), agent_runner=fake_runner
    )
    first = service.handle(AgentRequest(
        user_query="调研近三年课堂行为分析论文，并生成研究背景和研究现状",
        session_id="preserve-methods",
    ))
    assert first["status"] == "needs_clarification"

    answer = "先自动识别教师和学生行为并自动行为编码，再用S-T分析法或滞后序列分析法解释教学结构与师生互动"
    service.handle(AgentRequest(
        user_query=answer,
        session_id="preserve-methods",
        clarification_answer=answer,
    ))

    assert answer in calls[0]["query"]
    frame = calls[0]["state"]["research_semantic_frame"]
    assert {item["id"] for item in frame["methods"]} >= {
        "st_analysis", "lag_sequential_analysis",
    }
    assert frame["task_chain"][-2:] == [
        "st_or_lag_sequential_analysis",
        "teaching_structure_and_interaction_interpretation",
    ]


def test_conversation_service_keeps_asking_one_question_until_scope_is_clear():
    db = _db_session()
    calls = []

    def fake_runner(query, db=None, initial_state=None):
        calls.append(query)
        return {
            "answer": "完成", "intent": "generate_review", "topic": "课堂行为分析",
            "steps": [], "references": [], "paper_cards": [], "clusters": [], "errors": [],
        }

    service = ResearchConversationService(
        db,
        llm=ConversationalScopeLLM(),
        agent_runner=fake_runner,
    )
    first = service.handle(
        AgentRequest(user_query="调研课堂行为分析论文", session_id="free-chat")
    )
    assert first["status"] == "needs_clarification"

    second = service.handle(
        AgentRequest(
            user_query="我还没想好",
            session_id="free-chat",
            clarification_answer="我还没想好",
        )
    )
    assert second["status"] == "needs_clarification"
    assert second["answer"].count("？") == 1
    assert calls == []

    final = service.handle(
        AgentRequest(
            user_query="我想用摄像头识别学生动作",
            session_id="free-chat",
            clarification_answer="我想用摄像头识别学生动作",
        )
    )
    assert final["status"] == "completed"
    assert len(calls) == 1
    saved = ResearchSessionRepository(db).get("free-chat")
    assert saved["state"]["selected_scope"]["scope_id"] == "technical"


def test_related_work_clarifies_user_paper_profile_before_search():
    db = _db_session()
    calls = []

    def fake_runner(query, db=None, initial_state=None):
        calls.append(initial_state)
        return {
            "answer": "完成", "intent": "generate_related_work", "topic": "金融欺诈检测",
            "steps": [], "references": [], "paper_cards": [], "clusters": [], "errors": [],
        }

    service = ResearchConversationService(db, llm=ClearTopicLLM(), agent_runner=fake_runner)
    first = service.handle(AgentRequest(
        user_query="帮我生成图神经网络论文的相关工作",
        session_id="related-profile",
    ))
    assert first["status"] == "needs_clarification"
    assert first["clarification"]["kind"] == "user_paper_profile"
    assert calls == []

    second = service.handle(AgentRequest(
        user_query="补充论文信息",
        session_id="related-profile",
        clarification_answer="我的论文采用图神经网络检测金融交易欺诈，重点解决类别不平衡问题",
    ))
    assert second["status"] == "completed"
    assert len(calls) == 1
    assert calls[0]["user_paper_profile"]["research_problem"]
    assert calls[0]["user_paper_profile"]["proposed_method"] == "图神经网络"


def _quality_blocked_result(required: int = 3, available: int = 2) -> dict:
    cards = [
        {"paper_id": f"p{index}", "title": f"Paper {index}"}
        for index in range(1, available + 1)
    ]
    research_state = {
        "intent": "generate_introduction",
        "topic": "课堂行为分析",
        "canonical_topic": "课堂行为分析",
        "core_deliverables": ["research_background"],
        "requested_sections": ["background"],
        "required_reference_count": required,
        "max_papers": required,
        "max_papers_explicit": True,
        "generation_limit": required * 2,
        "paper_details": cards,
        "paper_cards": cards,
        "research_request": {
            "task_type": "generate_introduction",
            "topic": "课堂行为分析",
            "start_year": 2023,
            "end_year": 2026,
            "required_reference_count": required,
            "requested_sections": ["background"],
        },
    }
    return {
        "answer": "## 正文生成已阻止",
        "intent": "generate_introduction",
        "topic": "课堂行为分析",
        "quality_gate": {
            "passed": False,
            "phase": "pre_generation",
            "blocking_issues": [{
                "code": "minimum_references_not_met",
                "message": f"要求至少引用 {required} 篇，但只有 {available} 篇可用证据",
                "requested": required,
                "available": available,
            }],
            "recovery_options": ["扩大检索年份范围", "确认接受当前篇数"],
        },
        "generation_readiness": {
            "ready": False,
            "requested_minimum_references": required,
            "usable_reference_count": available,
        },
        "generation_blocked": True,
        "steps": [],
        "references": [],
        "paper_cards": cards,
        "clusters": [],
        "errors": [],
        "research_state": research_state,
    }


def test_quality_gate_failure_enters_multi_turn_decision():
    db = _db_session()
    calls = []

    def fake_runner(query, db=None, initial_state=None):
        calls.append(query)
        return _quality_blocked_result()

    service = ResearchConversationService(db, llm=ClearTopicLLM(), agent_runner=fake_runner)
    result = service.handle(AgentRequest(
        user_query="生成课堂行为分析研究背景，引用不少于3篇",
        session_id="quality-decision",
    ))

    assert result["status"] == "needs_clarification"
    assert result["clarification"]["kind"] == "quality_decision"
    assert "接受当前篇数" in result["answer"]
    assert len(calls) == 1
    saved = ResearchSessionRepository(db).get("quality-decision")
    assert saved["status"] == "needs_clarification"
    assert len(saved["state"]["editable_research_state"]["paper_cards"]) == 2


def test_quality_decision_accepts_available_and_regenerates_without_search(monkeypatch):
    db = _db_session()
    runner_calls = []
    regenerated = []

    def fake_runner(query, db=None, initial_state=None):
        runner_calls.append(query)
        return _quality_blocked_result()

    def fake_regenerate(editable, **kwargs):
        regenerated.append(editable)
        return {
            "answer": "已基于2篇论文重新生成研究背景",
            "intent": "generate_introduction",
            "topic": "课堂行为分析",
            "quality_gate": {"passed": True, "phase": "post_generation"},
            "generation_blocked": False,
            "steps": [], "references": [], "paper_cards": editable["paper_cards"],
            "clusters": [], "errors": [], "research_state": editable,
        }

    monkeypatch.setattr("app.agent.graph.regenerate_research_agent", fake_regenerate)
    service = ResearchConversationService(db, llm=ClearTopicLLM(), agent_runner=fake_runner)
    service.handle(AgentRequest(
        user_query="生成课堂行为分析研究背景，引用不少于3篇",
        session_id="accept-available",
    ))
    result = service.handle(AgentRequest(
        user_query="接受当前2篇继续生成",
        session_id="accept-available",
        clarification_answer="接受当前2篇继续生成",
    ))

    assert result["status"] == "completed"
    assert len(runner_calls) == 1
    assert len(regenerated) == 1
    assert regenerated[0]["required_reference_count"] == 2
    assert regenerated[0]["research_request"]["required_reference_count"] == 2


def test_quality_decision_expands_time_range_incrementally(monkeypatch):
    db = _db_session()
    calls = []
    continued = []

    def fake_runner(query, db=None, initial_state=None):
        calls.append({"query": query, "state": initial_state})
        return _quality_blocked_result()

    def fake_continue(editable, **kwargs):
        continued.append(editable)
        return {
            "answer": "扩大年份后完成",
            "intent": "generate_introduction",
            "topic": "课堂行为分析",
            "quality_gate": {"passed": True, "phase": "post_generation"},
            "steps": [], "references": [], "paper_cards": editable["paper_cards"],
            "clusters": [], "errors": [], "research_state": editable,
        }

    monkeypatch.setattr("app.agent.graph.continue_research_agent", fake_continue)
    service = ResearchConversationService(db, llm=ClearTopicLLM(), agent_runner=fake_runner)
    service.handle(AgentRequest(
        user_query="生成课堂行为分析研究背景，引用不少于3篇",
        session_id="expand-years",
    ))
    result = service.handle(AgentRequest(
        user_query="扩大到近五年继续检索",
        session_id="expand-years",
        clarification_answer="扩大到近五年继续检索",
    ))

    assert result["status"] == "completed"
    assert len(calls) == 1
    assert len(continued) == 1
    resumed_request = continued[0]["research_request"]
    assert resumed_request["start_year"] == 2022
    assert resumed_request["end_year"] == 2026
    assert continued[0]["incremental_search_window"] == {
        "start_year": 2022,
        "end_year": 2022,
    }
    assert len(continued[0]["paper_cards"]) == 2
    assert len(continued[0]["paper_details"]) == 2


def test_quality_decision_recognizes_taxonomy_repair_without_research_rerun():
    decision = ResearchConversationService._parse_quality_decision(
        "自动排除低相关论文并重新分类",
        {"phase": "pre_generation"},
    )
    assert decision == {"action": "repair_taxonomy"}


def test_negated_stop_phrases_are_not_parsed_as_stop():
    """“不要取消”等否定式表达不得误判为停止指令。"""
    for answer in ("不要取消", "先别停止", "我不想取消任务", "不要终止"):
        decision = ResearchConversationService._parse_quality_decision(
            answer, {"phase": "pre_generation"}
        )
        assert decision is None or decision.get("action") != "stop", answer
    # 独立停止短语仍然生效
    assert ResearchConversationService._parse_quality_decision(
        "取消任务", {"phase": "pre_generation"}
    ) == {"action": "stop"}
    assert ResearchConversationService._parse_quality_decision(
        "不想做了", {"phase": "pre_generation"}
    ) == {"action": "stop"}


def test_quality_decision_recognizes_supplemental_search_and_direct_generation():
    taxonomy_gate = {
        "phase": "pre_generation",
        "blocking_issues": [{"code": "taxonomy_not_ready"}],
    }

    assert ResearchConversationService._parse_quality_decision(
        "补充检索", taxonomy_gate
    ) == {"action": "retry_search"}
    assert ResearchConversationService._parse_quality_decision(
        "直接生成", taxonomy_gate
    ) == {"action": "force_generate"}
    assert ResearchConversationService._parse_quality_decision(
        "接受当前篇数", taxonomy_gate
    ) == {"action": "force_generate"}


def _post_generation_citation_shortfall(required: int = 3, actual: int = 2) -> dict:
    blocked = _quality_blocked_result(required=required, available=actual)
    blocked["quality_gate"] = {
        "passed": False,
        "phase": "post_generation",
        "blocking_issues": [{
            "code": "minimum_cited_references_not_met",
            "message": f"要求正文至少引用 {required} 篇，实际有效引用 {actual} 篇",
            "requested": required,
            "actual": actual,
        }],
        "recovery_options": ["基于现有证据保守重写"],
    }
    return blocked


def test_force_generate_keeps_original_reference_requirement(monkeypatch):
    """force_generate 只标记降级，不得把用户显式的引用篇数要求改成当前水平。"""
    db = _db_session()
    regenerated = []
    blocked = _post_generation_citation_shortfall()

    def fake_regenerate(editable, **kwargs):
        regenerated.append(editable)
        return {
            "answer": "已生成当前最佳可用草稿",
            "intent": "generate_introduction",
            "topic": "课堂行为分析",
            "quality_gate": {
                "passed": False,
                "draft_available": True,
                "draft_released": True,
                "draft_disposition": "released_best_effort",
                "partial_success": True,
                "phase": "post_generation",
                "blocking_issues": [
                    {"code": "user_accepted_best_effort_generation"},
                    {"code": "minimum_cited_references_not_met", "requested": 3, "actual": 2},
                ],
            },
            "steps": [], "references": [], "paper_cards": editable["paper_cards"],
            "clusters": [], "errors": [], "research_state": editable,
        }

    monkeypatch.setattr("app.agent.graph.regenerate_research_agent", fake_regenerate)
    service = ResearchConversationService(
        db, llm=ClearTopicLLM(), agent_runner=lambda *args, **kwargs: {}
    )
    service._persist_or_pause_result(
        "force-keeps-target", "生成课堂行为分析研究背景，引用不少于3篇", {}, blocked
    )
    result = service.handle(AgentRequest(
        user_query="直接基于现有证据生成最佳可用草稿",
        session_id="force-keeps-target",
        clarification_answer="直接基于现有证据生成最佳可用草稿",
    ))

    assert len(regenerated) == 1
    assert regenerated[0]["best_effort_generation"] is True
    assert regenerated[0]["required_reference_count"] == 3
    assert regenerated[0]["max_papers"] == 3
    assert regenerated[0]["research_request"]["required_reference_count"] == 3
    # 要求没被下调，所以门禁必须如实报告缺口（AGENTS.md 规则 5）。
    assert result["status"] == "partial"
    codes = {issue["code"] for issue in result["quality_gate"]["blocking_issues"]}
    assert "minimum_cited_references_not_met" in codes


def test_quality_decision_answers_map_to_intended_actions():
    """系统提供的每种答复都必须落到语义相符的分支。"""
    post_generation = {
        "phase": "post_generation",
        "available": 2,
        "requested": 3,
        "blocking_issues": [{
            "code": "minimum_cited_references_not_met", "requested": 3, "actual": 2
        }],
    }
    pre_generation_count = {
        "phase": "pre_generation",
        "available": 2,
        "requested": 3,
        "blocking_issues": [{
            "code": "minimum_references_not_met", "requested": 3, "available": 2
        }],
    }
    taxonomy_gate = {
        "phase": "pre_generation",
        "blocking_issues": [{"code": "taxonomy_not_ready"}],
    }
    cases = [
        # 保守重写：曾全部落到 force_generate，跑完整 LLM 重写后支持率反而下降。
        ("基于现有证据保守重写", post_generation, "regenerate_existing"),
        ("基于现有证据重新保守生成", post_generation, "regenerate_existing"),
        ("保守重写", post_generation, "regenerate_existing"),
        ("重写", post_generation, "regenerate_existing"),
        # 无"保守/重写"字样的直接生成仍是 force_generate。
        ("基于现有证据直接写", post_generation, "force_generate"),
        ("直接基于现有证据生成最佳可用草稿", taxonomy_gate, "force_generate"),
        ("直接生成", taxonomy_gate, "force_generate"),
        ("接受当前篇数", pre_generation_count, "accept_available"),
        ("接受当前2篇继续生成", pre_generation_count, "accept_available"),
        ("扩大时间范围", pre_generation_count, "expand_time_range"),
        ("纳入更多文献类型", pre_generation_count, "include_more_types"),
        ("放宽主题范围", pre_generation_count, "broaden_scope"),
        ("自动重新分类", taxonomy_gate, "repair_taxonomy"),
        ("补充检索", pre_generation_count, "retry_search"),
        ("保持条件继续检索", post_generation, "retry_search"),
        ("结束本次任务", post_generation, "stop"),
    ]
    for answer, clarification, expected in cases:
        decision = ResearchConversationService._parse_quality_decision(
            answer, clarification
        )
        assert decision is not None, answer
        assert decision["action"] == expected, f"{answer} → {decision}"


def test_every_clarification_option_label_is_parsable():
    """结构护栏：问句里出现的每个选项原文都必须能被解析器解析。"""
    from app.services.research_conversation_service import _QUALITY_DECISION_OPTIONS

    cards = [{"paper_id": f"p{index}", "title": f"Paper {index}"} for index in range(1, 5)]
    gate_variants = [
        # 生成后引用不足且证据池已够 → 保守重写/补充检索/结束
        {
            "passed": False,
            "phase": "post_generation",
            "blocking_issues": [{
                "code": "minimum_cited_references_not_met", "requested": 3, "actual": 2
            }],
        },
        # 生成前可用篇数不足 → 接受当前篇数/扩大时间范围/纳入更多文献类型/放宽主题范围/继续检索
        {
            "passed": False,
            "phase": "pre_generation",
            "blocking_issues": [{
                "code": "minimum_references_not_met", "requested": 3, "available": 2
            }],
        },
        # 篇数足够但分类未通过 → 直接生成最佳可用草稿/自动重新分类/…
        {
            "passed": False,
            "phase": "pre_generation",
            "blocking_issues": [{"code": "taxonomy_not_ready"}],
        },
        # 生成前其他结构未达标
        {
            "passed": False,
            "phase": "pre_generation",
            "blocking_issues": [{"code": "route_validation_failed"}],
        },
        # 生成后主张证据未达标
        {
            "passed": False,
            "phase": "post_generation",
            "blocking_issues": [{"code": "claim_evidence_quality_not_met"}],
        },
    ]
    checked = 0
    for gate in gate_variants:
        clarification = ResearchConversationService._quality_clarification({
            "paper_cards": cards,
            "quality_gate": gate,
            "generation_readiness": {"usable_reference_count": 2},
        })
        assert clarification is not None, gate
        question = clarification["question"]
        for action, label, _pattern in _QUALITY_DECISION_OPTIONS.values():
            if label not in question:
                continue
            decision = ResearchConversationService._parse_quality_decision(
                label, clarification
            )
            assert decision is not None, f"问句选项无法解析：{label}"
            assert decision["action"] == action, f"{label} → {decision}"
            checked += 1
    # 五个问句分支共出现 19 次选项；数量骤降说明问句不再复用共享词表。
    assert checked >= 15, checked


def test_direct_generation_bypasses_taxonomy_gate_without_restarting_search(monkeypatch):
    db = _db_session()
    runner_calls = []
    regenerated = []

    blocked = _quality_blocked_result(required=3, available=3)
    blocked["quality_gate"] = {
        "passed": False,
        "phase": "pre_generation",
        "blocking_issues": [{
            "code": "taxonomy_not_ready",
            "message": "动态分类未通过验证",
        }],
        "recovery_options": ["直接生成", "补充检索"],
    }
    blocked["generation_readiness"] = {
        "ready": True,
        "usable_reference_count": 3,
    }

    def fake_runner(query, **kwargs):
        runner_calls.append(query)
        return blocked

    def fake_regenerate(editable, **kwargs):
        regenerated.append(editable)
        return {
            "answer": "已直接生成当前最佳可用草稿",
            "intent": "generate_introduction",
            "topic": "课堂行为分析",
            "quality_gate": {
                "passed": False,
                "draft_available": True,
                "draft_released": True,
                "draft_disposition": "released_best_effort",
                "partial_success": True,
                "phase": "post_generation",
                "blocking_issues": [{"code": "user_accepted_best_effort_generation"}],
            },
            "steps": [], "references": [], "paper_cards": editable["paper_cards"],
            "clusters": [], "errors": [], "research_state": editable,
        }

    monkeypatch.setattr("app.agent.graph.regenerate_research_agent", fake_regenerate)
    service = ResearchConversationService(db, llm=ClearTopicLLM(), agent_runner=fake_runner)
    service.handle(AgentRequest(
        user_query="生成课堂行为分析研究背景和研究现状，引用不少于3篇",
        session_id="direct-generate",
    ))
    result = service.handle(AgentRequest(
        user_query="直接生成",
        session_id="direct-generate",
        clarification_answer="直接生成",
    ))

    assert result["status"] == "partial"
    assert len(runner_calls) == 1
    assert len(regenerated) == 1
    assert regenerated[0]["best_effort_generation"] is True
    assert regenerated[0]["allow_unvalidated_taxonomy"] is True


def test_third_failed_recovery_offers_only_direct_generation_or_stop():
    db = _db_session()
    service = ResearchConversationService(
        db, llm=ClearTopicLLM(), agent_runner=lambda *args, **kwargs: {}
    )
    blocked = _quality_blocked_result(required=3, available=3)
    blocked["quality_gate"] = {
        "passed": False,
        "phase": "pre_generation",
        "blocking_issues": [{"code": "taxonomy_not_ready", "message": "分类仍未通过"}],
        "recovery_options": ["继续检索"],
    }
    blocked["research_state"]["quality_recovery_attempts"] = 3

    result = service._persist_or_pause_result(
        "recovery-limit", "生成研究现状", {}, blocked
    )

    assert result["status"] == "needs_clarification"
    assert "连续尝试3次" in result["answer"]
    assert result["clarification"]["recovery_options"] == [
        "直接生成当前最佳可用草稿", "结束任务"
    ]


def test_quality_resume_migrates_old_strict_frame_for_open_method_alternatives():
    class OpenAlternativeLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return json.dumps({
                "canonical_topic": "课堂行为分析",
                "application_domains": [],
                "research_objects": [{
                    "id": "classroom_behavior", "label": "classroom behavior",
                    "surface_text": "课堂行为", "explicit": True, "confidence": 0.95,
                }],
                "methods": [
                    {"id": "st_analysis", "label": "S-T analysis", "surface_text": "S-T分析法", "category": "analytical", "explicit": True, "confidence": 0.95},
                    {"id": "lag_sequential_analysis", "label": "lag sequential analysis", "surface_text": "滞后序列分析法", "category": "analytical", "explicit": True, "confidence": 0.95},
                ],
                "research_actions": [], "analysis_targets": [],
                "evidence_requirements": [{
                    "requirement_id": "analytical_method:open_alternatives",
                    "label": "适用的课堂行为分析方法",
                    "evidence_role": "analytical_method",
                    "aliases": ["S-T分析法", "S-T analysis", "滞后序列分析法", "lag sequential analysis"],
                    "context_aliases": ["课堂行为", "classroom behavior"],
                    "source_ids": ["st_analysis", "lag_sequential_analysis"],
                    "exact_method_required": False,
                    "selection_mode": "open_any",
                }],
            }, ensure_ascii=False)

    old_frame = {
        "evidence_requirements": [
            {"requirement_id": "analytical_method:st", "selection_mode": "all"},
            {"requirement_id": "analytical_method:lag", "selection_mode": "all"},
        ]
    }
    refreshed = ResearchConversationService._refresh_open_alternative_semantics(
        "调研课堂行为分析",
        [{
            "type": "clarification_answer",
            "content": (
                "先自动识别和自动编码，再采用S-T分析法或者"
                "滞后序列分析法等分析法进行分析"
            ),
        }],
        old_frame,
        "课堂行为分析",
        ["research_background", "research_status"],
        llm=OpenAlternativeLLM(),
    )

    analytical = [
        item for item in refreshed["evidence_requirements"]
        if item["evidence_role"] == "analytical_method"
    ]
    assert len(analytical) == 1
    assert analytical[0]["selection_mode"] == "open_any"


def test_post_generation_retry_uses_existing_evidence_pool_before_research(monkeypatch):
    db = _db_session()
    regenerated = []

    def fake_regenerate(editable, **kwargs):
        regenerated.append(editable)
        return {
            "answer": "已基于现有证据池保守重写",
            "intent": "generate_review",
            "topic": "课堂行为分析",
            "quality_gate": {"passed": True, "phase": "post_generation"},
            "steps": [], "references": [], "paper_cards": editable["paper_cards"],
            "clusters": [], "errors": [], "research_state": editable,
        }

    monkeypatch.setattr("app.agent.graph.regenerate_research_agent", fake_regenerate)
    service = ResearchConversationService(db, llm=ClearTopicLLM(), agent_runner=lambda *a, **k: {})
    blocked = _quality_blocked_result(required=3, available=4)
    blocked["quality_gate"] = {
        "passed": False,
        "phase": "post_generation",
        "blocking_issues": [{
            "code": "minimum_cited_references_not_met",
            "message": "正文只引用2篇",
            "requested": 3,
            "actual": 2,
        }],
        "recovery_options": ["基于现有证据重写"],
    }
    service._persist_or_pause_result("reuse-evidence", "原始请求", {}, blocked)

    result = service.handle(AgentRequest(
        user_query="保持条件继续检索",
        session_id="reuse-evidence",
        clarification_answer="保持条件继续检索",
    ))

    assert result["status"] == "completed"
    assert len(regenerated) == 1
    assert regenerated[0]["conservative_regeneration"] is True


def test_released_best_effort_draft_is_persisted_with_partial_status():
    db = _db_session()
    service = ResearchConversationService(
        db,
        llm=ClearTopicLLM(),
        agent_runner=lambda *args, **kwargs: {},
    )
    result = _quality_blocked_result(required=3, available=3)
    result["answer"] = "> 质量门禁提示\n\n## 研究背景\n\n未完全达标草稿"
    result["quality_gate"] = {
        "passed": False,
        "draft_available": True,
        "draft_released": True,
        "draft_disposition": "released_best_effort",
        "partial_success": True,
        "phase": "post_generation",
        "blocking_issues": [{
            "code": "claim_evidence_quality_not_met",
            "message": "主张证据支持率未达标",
        }],
        "recovery_options": ["补充证据后重新生成"],
    }

    persisted = service._persist_or_pause_result(
        "partial-result",
        "生成课堂行为分析研究背景",
        {},
        result,
    )

    assert persisted["status"] == "partial"
    assert ResearchSessionRepository(db).get("partial-result")["status"] == "partial"


def test_unauthorized_failed_draft_enters_quality_decision_instead_of_partial():
    """未获授权的失败草稿必须进入质量决策，不能作为 partial 终态直接返回。"""
    db = _db_session()
    service = ResearchConversationService(
        db,
        llm=ClearTopicLLM(),
        agent_runner=lambda *args, **kwargs: {},
    )
    result = _quality_blocked_result(required=3, available=3)
    result["answer"] = "## 研究背景\n\nQUARANTINE_SENTINEL 未经验证的正文"
    result["quality_gate"] = {
        "passed": False,
        "draft_available": True,
        "draft_released": False,
        "draft_disposition": "quarantined",
        "partial_success": False,
        "phase": "post_generation",
        "blocking_issues": [{
            "code": "claim_evidence_quality_not_met",
            "message": "主张证据支持率未达标",
        }],
        "recovery_options": ["补充证据后重新生成"],
    }

    persisted = service._persist_or_pause_result(
        "quarantined-result",
        "生成课堂行为分析研究背景",
        {},
        result,
    )

    assert persisted["status"] == "needs_clarification"
    assert "QUARANTINE_SENTINEL" not in persisted["answer"]
    assert ResearchSessionRepository(db).get("quarantined-result")["status"] == "needs_clarification"
