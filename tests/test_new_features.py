"""测试新增功能：相关工作和引言生成。

本测试文件涵盖 v1.3.0 新增的功能：
1. IntentType 新增类型
2. generate_related_work 工具
3. generate_introduction 工具
4. citation_count_by_source 字段
5. SourceDiagnostic 模型
"""

import pytest
from app.schemas.paper_schema import PaperMetadata, SourceDiagnostic
from app.schemas.agent_schema import IntentType
from app.agent.intent import recognize_intent


class TestNewIntents:
    """测试新增的意图类型。"""

    def test_generate_related_work_intent(self):
        """测试相关工作意图识别。"""
        queries = [
            "帮我生成少样本学习的相关工作章节",
            "写一个关于动作识别的 related work",
            "生成已有研究综述的相关工作部分",
        ]
        for query in queries:
            result = recognize_intent(query, llm=None)
            assert result.intent == IntentType.GENERATE_RELATED_WORK.value
            assert result.confidence >= 0.9

    def test_generate_introduction_intent(self):
        """测试引言意图识别。"""
        queries = [
            "写一个关于动作识别的引言",
            "生成少样本学习的 introduction",
            "帮我写研究背景和引言部分",
        ]
        for query in queries:
            result = recognize_intent(query, llm=None)
            assert result.intent == IntentType.GENERATE_INTRODUCTION.value
            assert result.confidence >= 0.9

    def test_intent_priority(self):
        """测试意图优先级（related work 不应误判为 search_papers）。"""
        result = recognize_intent("找一些 related work 论文", llm=None)
        # "related work" 关键词应优先匹配 generate_related_work（置信度 0.95）
        # 而非 search_papers（置信度 0.9）
        assert result.intent == IntentType.GENERATE_RELATED_WORK.value


class TestCitationCountBySource:
    """测试按源记录引用量功能。"""

    def test_schema_field(self):
        """测试 PaperMetadata 支持 citation_count_by_source 字段。"""
        paper = PaperMetadata(
            paper_id="test:001",
            title="Test Paper",
            authors=["Author A"],
            year=2024,
            citation_count=100,
            citation_count_by_source={"semantic_scholar": 100, "openalex": 95},
            source="semantic_scholar",
        )
        assert paper.citation_count == 100
        assert paper.citation_count_by_source == {"semantic_scholar": 100, "openalex": 95}

    def test_schema_backward_compatibility(self):
        """测试向后兼容性：citation_count_by_source 可为 None。"""
        paper = PaperMetadata(
            paper_id="test:002",
            title="Test Paper 2",
            authors=["Author B"],
            year=2023,
            citation_count=50,
            citation_count_by_source=None,
            source="arxiv",
        )
        assert paper.citation_count == 50
        assert paper.citation_count_by_source is None

    def test_merge_enrichment_logic(self):
        """测试 fetch_metadata 的 _merge_enrichment 逻辑。"""
        from app.tools.fetch_metadata import _merge_enrichment

        paper = {
            "paper_id": "test:003",
            "title": "Test Paper 3",
            "source": "cnki",
            "citation_count": 10,
            "citation_count_by_source": {"cnki": 10},
        }

        detail = {
            "title": "Test Paper 3",
            "citation_count": 100,
            "source": "semantic_scholar",
            "venue": "CVPR 2023",
        }

        merged = _merge_enrichment(paper, detail)
        
        assert merged["citation_count_by_source"] == {"cnki": 10, "semantic_scholar": 100}
        assert merged["citation_count"] == 100  # max(10, 100)
        assert merged.get("venue") == "CVPR 2023"


class TestSourceDiagnostic:
    """测试检索失败诊断功能。"""

    def test_source_diagnostic_model(self):
        """测试 SourceDiagnostic 模型。"""
        diag = SourceDiagnostic(
            source="cnki",
            status="failed",
            returned_count=0,
            error_code="TIMEOUT",
            message="网络超时",
        )
        assert diag.source == "cnki"
        assert diag.status == "failed"
        assert diag.returned_count == 0
        assert diag.error_code == "TIMEOUT"

    def test_source_diagnostic_success(self):
        """测试成功状态的诊断。"""
        diag = SourceDiagnostic(
            source="semantic_scholar",
            status="success",
            returned_count=20,
            error_code=None,
            message=None,
        )
        assert diag.status == "success"
        assert diag.returned_count == 20


class TestAgentIntegration:
    """测试 Agent 主流程集成。"""

    def test_state_new_fields(self):
        """测试 state 支持新字段。"""
        from app.agent.state import ResearchAgentState

        state: ResearchAgentState = {
            "user_query": "test",
            "our_work": {"research_problem": "...", "method_name": "...", "method_summary": "...", "innovations": []},
            "background": {"task_definition": "...", "importance": "...", "application_scenarios": []},
            "existing_limitations": [],
            "verified_results": [],
            "target_length": 1500,
            "related_work": "",
            "related_work_data": {},
            "introduction": "",
            "introduction_data": {},
            "source_diagnostics": [],
        }

        assert "our_work" in state
        assert "background" in state
        assert "related_work" in state
        assert "introduction" in state
        assert "source_diagnostics" in state

    def test_nodes_import(self):
        """测试主链节点可以正常从包入口导入。"""
        from app.agent.nodes import (
            generate_deliverables_node,
            final_answer_node,
        )

        assert callable(generate_deliverables_node)
        assert callable(final_answer_node)

    def test_graph_import(self):
        """测试 graph 可以正常导入（含新节点）。"""
        from app.agent.graph import run_research_agent

        assert callable(run_research_agent)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
