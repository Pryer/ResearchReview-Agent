"""检索前 UnsupportedTaskGuard 测试。"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.graph import run_research_agent
from app.agent.unsupported_task_guard import check_unsupported_task
from app.database.models import Base
from app.database.repositories import ResearchSessionRepository
from app.schemas.agent_schema import AgentRequest
from app.services.research_conversation_service import ResearchConversationService


@pytest.mark.parametrize(
    ("query", "deliverable"),
    [
        ("生成联邦学习的研究背景", "research_background"),
        ("生成联邦学习的研究现状", "research_status"),
        ("我的论文采用联邦学习保护隐私，请生成相关工作", "related_work"),
        ("生成联邦学习的叙述性综述初稿", "narrative_review"),
    ],
)
def test_guard_allows_only_four_core_deliverables(query, deliverable):
    result = check_unsupported_task(query)

    assert result.allowed is True
    assert deliverable in [item.value for item in result.supported_deliverables]


@pytest.mark.parametrize(
    "query",
    [
        "帮我写联邦学习论文的引言",
        "请按PRISMA生成系统综述",
        "对这些研究做元分析",
        "给我一些方法设计建议",
        "帮我设计消融实验",
        "帮我写开题报告",
    ],
)
def test_guard_rejects_tasks_outside_four_deliverables(query):
    result = check_unsupported_task(query)

    assert result.allowed is False
    assert "目前我没有" in result.message
    assert "本次没有执行论文检索" in result.message


def test_guard_allows_standalone_paper_search():
    """独立论文检索与 graph 的 search_papers 早退分支对齐，直接放行。"""
    for query in ("帮我找联邦学习论文", "帮我找几篇关于目标检测的论文"):
        result = check_unsupported_task(query)
        assert result.allowed is True, query
        assert result.message == ""


def test_guard_rejects_mixed_request_instead_of_ignoring_extra_task():
    result = check_unsupported_task("生成联邦学习研究现状，并提供方法设计建议")

    assert result.allowed is False
    assert "research_status" in [item.value for item in result.supported_deliverables]
    assert "研究方法或方案设计" in result.unsupported_requests


def test_direct_agent_stops_before_planning_and_retrieval(monkeypatch):
    def forbidden_llm():
        raise AssertionError("不支持任务不应创建 LLM")

    monkeypatch.setattr("app.agent.graph._get_llm", forbidden_llm)
    result = run_research_agent("生成RAG研究现状，并帮我设计实验")

    assert result["generation_blocked"] is True
    assert result["references"] == []
    assert result["unsupported_task_guard"]["allowed"] is False
    assert [step["step_name"] for step in result["steps"]] == ["unsupported_task_guard"]


def test_conversation_guard_runs_before_disambiguation_and_runner():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    calls: list[str] = []

    class ForbiddenLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            raise AssertionError("不支持任务不应进入主题消歧")

    def forbidden_runner(query, **kwargs):
        calls.append(query)
        raise AssertionError("不支持任务不应进入 Agent")

    service = ResearchConversationService(
        db,
        llm=ForbiddenLLM(),
        agent_runner=forbidden_runner,
    )
    result = service.handle(
        AgentRequest(user_query="帮我写开题报告", session_id="unsupported-task")
    )

    assert result["status"] == "blocked"
    assert result["unsupported_task_guard"]["allowed"] is False
    assert result["references"] == []
    assert calls == []
    saved = ResearchSessionRepository(db).get("unsupported-task")
    assert saved["status"] == "blocked"
    assert saved["state"]["unsupported_task_guard"]["allowed"] is False
