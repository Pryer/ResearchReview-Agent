"""v1.3.0-rc2 修复验证测试。

测试所有 P0 和 P1 修复的功能。
"""

from app.agent.intent import recognize_intent
from app.schemas.agent_schema import AgentRequest, IntentType
from app.schemas.paper_schema import PaperMetadata, SourceDiagnostic


def test_agent_request_has_state_field():
    """测试 AgentRequest 新增 state 字段。"""
    request = AgentRequest(
        user_query="生成相关工作",
        state={
            "our_work": {
                "research_problem": "测试问题",
                "method_name": "TestMethod",
                "method_summary": "测试摘要",
                "innovations": ["创新1"]
            }
        }
    )
    assert request.state is not None
    assert "our_work" in request.state
    assert request.state["our_work"]["method_name"] == "TestMethod"


def test_agent_request_state_is_optional():
    """测试 state 字段是可选的。"""
    request = AgentRequest(user_query="帮我找论文")
    assert request.state is None


def test_intent_related_work_strict_keywords():
    """测试 Related Work 意图识别规则收紧。"""
    # 应该识别为 generate_related_work
    result1 = recognize_intent("生成少样本学习的相关工作")
    assert result1.intent == IntentType.GENERATE_RELATED_WORK.value
    assert result1.confidence >= 0.9
    
    result2 = recognize_intent("帮我写 related work 章节")
    assert result2.intent == IntentType.GENERATE_RELATED_WORK.value
    
    # 应该识别为 search_papers（包含明确搜索关键词）
    result3 = recognize_intent("搜索少样本学习的论文")
    assert result3.intent == IntentType.SEARCH_PAPERS.value
    
    result4 = recognize_intent("帮我找几篇目标检测的论文")
    assert result4.intent == IntentType.SEARCH_PAPERS.value
    
    # 没有明确关键词的会被判定为 general_qa（这是合理的）
    result5 = recognize_intent("有什么好的研究方向")
    assert result5.intent in [IntentType.GENERAL_QA.value, IntentType.FIND_TRENDS.value]


def test_source_diagnostic_has_human_action_required():
    """测试 SourceDiagnostic 新增 human_action_required 状态。"""
    diag = SourceDiagnostic(
        source="cnki",
        status="human_action_required",
        returned_count=0,
        error_code="CAPTCHA_REQUIRED",
        message="检测到验证码"
    )
    assert diag.status == "human_action_required"
    assert diag.error_code == "CAPTCHA_REQUIRED"


def test_source_diagnostic_old_statuses_still_work():
    """测试旧的诊断状态仍然可用。"""
    diag1 = SourceDiagnostic(source="arxiv", status="success", returned_count=10)
    assert diag1.status == "success"
    
    diag2 = SourceDiagnostic(source="cnki", status="empty", returned_count=0)
    assert diag2.status == "empty"
    
    diag3 = SourceDiagnostic(
        source="semantic_scholar",
        status="failed",
        returned_count=0,
        error_code="TIMEOUT"
    )
    assert diag3.status == "failed"


def test_paper_metadata_citation_count_by_source():
    """测试 citation_count_by_source 字段。"""
    paper = PaperMetadata(
        paper_id="test:001",
        title="Test Paper",
        authors=["Alice"],
        year=2024,
        citation_count=100,
        citation_count_by_source={
            "semantic_scholar": 100,
            "openalex": 95
        }
    )
    assert paper.citation_count == 100
    assert paper.citation_count_by_source["semantic_scholar"] == 100
    assert paper.citation_count_by_source["openalex"] == 95


def test_paper_metadata_citation_count_by_source_is_optional():
    """测试 citation_count_by_source 是可选字段。"""
    paper = PaperMetadata(
        paper_id="cnki:001",
        title="CNKI Paper",
        authors=["Bob"],
        year=2024,
        citation_count=None,
        citation_count_by_source=None
    )
    assert paper.citation_count is None
    assert paper.citation_count_by_source is None


def test_cnki_paper_has_no_citation_count():
    """测试 CNKI 论文没有引用量（符合实际）。"""
    from app.clients.cnki_client import _to_paper_metadata
    
    raw = {
        "title": "测试论文",
        "authors": "张三 李四",
        "year": "2024",
        "abstract": "摘要",
        "venue": "期刊名",
        "doi": "",
        "url": "https://kns.cnki.net/test",
        "pdf_url": "",
        "keywords": "关键词1;关键词2"
    }
    
    paper = _to_paper_metadata(raw)
    assert paper.citation_count is None
    assert paper.citation_count_by_source is None
    assert paper.source == "cnki"


def test_cnki_result_collection_retries_after_dynamic_dom_refresh(monkeypatch):
    """结果列表在等待后被重绘时，应重新采集而不是终止整个 CNKI 来源。"""
    import pytest
    pytest.importorskip("selenium")
    from selenium.common.exceptions import StaleElementReferenceException
    from app.clients import cnki_client

    class ImmediateWait:
        def __init__(self, driver, timeout):
            self.driver = driver

        def until(self, condition):
            return [object()]

    class DynamicResultDriver:
        calls = 0

        def execute_script(self, script):
            self.calls += 1
            if self.calls == 1:
                raise StaleElementReferenceException("result list refreshed")
            return [
                {"href": "https://kns.cnki.net/a", "title": "论文 A"},
                {"href": "https://kns.cnki.net/a", "title": "论文 A"},
                {"href": "https://kns.cnki.net/b", "title": "论文 B"},
            ]

    driver = DynamicResultDriver()
    monkeypatch.setattr(cnki_client, "WebDriverWait", ImmediateWait)
    monkeypatch.setattr(cnki_client, "STALE_RETRY_DELAY_SECONDS", 0)

    urls = cnki_client.collect_result_urls(driver)

    assert driver.calls == 2
    assert urls == ["https://kns.cnki.net/a", "https://kns.cnki.net/b"]


def test_cnki_search_box_is_relocated_after_becoming_stale(monkeypatch):
    """知网替换搜索组件后，不再复用已经失效的输入框对象。"""
    import pytest
    pytest.importorskip("selenium")
    from selenium.common.exceptions import StaleElementReferenceException
    from app.clients import cnki_client

    class StaleBox:
        def clear(self):
            raise StaleElementReferenceException("search component refreshed")

    class FreshBox:
        value = ""

        def clear(self):
            self.value = ""

        def send_keys(self, value):
            self.value = value

    fresh = FreshBox()
    boxes = iter([StaleBox(), fresh])
    monkeypatch.setattr(cnki_client, "wait_first", lambda *args, **kwargs: next(boxes))
    monkeypatch.setattr(cnki_client, "STALE_RETRY_DELAY_SECONDS", 0)

    result = cnki_client._fill_search_box(
        object(), [("id", "txt_SearchText")], "课堂行为分析", timeout=1
    )

    assert result is fresh
    assert fresh.value == "课堂行为分析"


def test_cnki_keeps_result_page_metadata_when_detail_enrichment_fails(monkeypatch):
    """详情页卡死不能再把已经采集到的 CNKI 标题结果清空。"""
    import pytest
    pytest.importorskip("selenium")
    from selenium.common.exceptions import WebDriverException
    from app.clients import cnki_client

    class FakeDriver:
        def quit(self):
            pass

    monkeypatch.setattr(cnki_client, "build_driver", lambda **kwargs: FakeDriver())
    monkeypatch.setattr(cnki_client, "search", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cnki_client,
        "collect_result_records",
        lambda driver: [{
            "url": "https://kns.cnki.net/kcms/detail/1",
            "title": "课堂行为分析研究",
            "row_text": "作者甲 2024 教育研究",
        }],
    )
    monkeypatch.setattr(
        cnki_client,
        "parse_detail",
        lambda *args, **kwargs: (_ for _ in ()).throw(WebDriverException("timeout")),
    )
    monkeypatch.setattr(cnki_client.settings, "cnki_detail_enrichment_limit", 1)
    from app.core.circuit_breaker import get_circuit_breaker
    get_circuit_breaker("cnki").reset()

    papers = cnki_client.search_cnki("课堂行为分析", 2023, 2026, max_results=1)

    assert len(papers) == 1
    assert papers[0].title == "课堂行为分析研究"
    assert papers[0].year == 2024
    assert papers[0].source == "cnki"


def test_semantic_scholar_429_is_reported_as_rate_limit(monkeypatch):
    """429 必须是明确失败状态，不能伪装成“检索结果为空”。"""
    from app.clients import semantic_scholar_client as client

    class Response:
        status_code = 429
        headers = {}
        url = "https://api.semanticscholar.org/test"
        text = "rate limited"

    monkeypatch.setattr(client.requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(client.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(client.settings, "semantic_scholar_max_retries", 1)
    monkeypatch.setattr(client.settings, "semantic_scholar_max_retry_wait_seconds", 2)
    monkeypatch.setattr(client.settings, "semantic_scholar_cooldown_seconds", 1)
    monkeypatch.setattr(client, "_S2_COOLDOWN_UNTIL", 0.0)

    try:
        client.search_semantic_scholar("test", 2023, 2026, 5)
        assert False, "expected rate-limit exception"
    except client.SemanticScholarRateLimitError as exc:
        assert exc.error_code == "RATE_LIMITED"
    finally:
        client._S2_COOLDOWN_UNTIL = 0.0


def test_cnki_query_strips_generic_review_suffix(monkeypatch):
    """CNKI 站内检索对“综述”后缀敏感，送入前应剥离泛化后缀。"""
    import app.tools.search_papers as sp

    seen: list[str] = []

    def fake_client(query, start_year, end_year, max_results):
        seen.append(query)
        return []

    monkeypatch.setattr("app.clients.cnki_client.search_cnki", fake_client)
    sp._search_cnki("少样本动作识别综述", 2022, 2026, 30)
    sp._search_cnki("少样本动作识别研究综述", 2022, 2026, 30)
    sp._search_cnki("少样本动作识别", 2022, 2026, 30)
    # 整个查询就是后缀时不得清空
    sp._search_cnki("综述", 2022, 2026, 30)

    assert seen == [
        "少样本动作识别",
        "少样本动作识别",
        "少样本动作识别",
        "综述",
    ]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
