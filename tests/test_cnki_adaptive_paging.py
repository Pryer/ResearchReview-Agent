"""知网自适应翻页：按年份窗口下界收敛，兼顾页数/条数/时长硬上限。"""

import pytest

from app.clients import cnki_client
from app.core.circuit_breaker import get_circuit_breaker


class _FakeDriver:
    def quit(self):
        pass


class _PermissiveLimiter:
    """测试用限流器：不引入 0.5qps 的真实等待。"""

    rate = 1000.0
    capacity = 1000.0

    def acquire(self, tokens: float = 1.0, timeout: float | None = None) -> bool:
        return True


def _pin_detail_limit(monkeypatch, limit: int) -> None:
    """把详情增强额度钉死为 limit。

    额度现在按 demand_factor x max_results 自适应，并受上下限钳制；这些用例
    要验证"额度花在哪些记录上"，需要一个确定的额度，故同时钉住三个旋钮。
    """
    monkeypatch.setattr(cnki_client.settings, "cnki_detail_enrichment_limit", limit)
    monkeypatch.setattr(cnki_client.settings, "cnki_detail_enrichment_max", limit)
    monkeypatch.setattr(cnki_client.settings, "cnki_detail_enrichment_demand_factor", 1.0)


def _record(title: str, year: int) -> dict:
    return {
        "title": title,
        "url": f"https://kns.cnki.net/kcms2/article/abstract?v={title}",
        "row_text": f"{title} 某期刊 {year}-03-15",
    }


def _install_pages(monkeypatch, pages: list[list[dict]]):
    """把若干页结果依次喂给 collect_result_records，并记录翻页次数。"""
    state = {"page": 0, "next_calls": 0}

    def fake_collect(_driver):
        idx = state["page"]
        if idx >= len(pages):
            return []
        return pages[idx]

    def fake_next(_driver):
        state["next_calls"] += 1
        state["page"] += 1
        return state["page"] < len(pages)

    monkeypatch.setattr(cnki_client, "build_driver", lambda **kwargs: _FakeDriver())
    monkeypatch.setattr(cnki_client, "search", lambda *a, **k: None)
    monkeypatch.setattr(cnki_client, "collect_result_records", fake_collect)
    monkeypatch.setattr(cnki_client, "go_next_page", fake_next)
    return state


@pytest.fixture(autouse=True)
def _reset_cnki_guards(monkeypatch):
    get_circuit_breaker("cnki", failure_threshold=2, recovery_timeout=120.0).reset()
    # 本机测试环境未安装 selenium；翻页逻辑本身与驱动实现无关。
    monkeypatch.setattr(cnki_client, "SELENIUM_AVAILABLE", True)
    monkeypatch.setattr(
        "app.core.rate_limiter.get_rate_limiter", lambda *a, **k: _PermissiveLimiter(),
    )
    # 详情页增强与本用例无关，关闭以免触发 parse_detail。
    monkeypatch.setattr(cnki_client.settings, "cnki_detail_enrichment_limit", 0)
    monkeypatch.setattr(cnki_client.settings, "cnki_adaptive_year_paging", True)
    monkeypatch.setattr(cnki_client.settings, "cnki_year_boundary_pages", 1)
    monkeypatch.setattr(cnki_client.settings, "cnki_max_pages", 15)
    monkeypatch.setattr(cnki_client.settings, "cnki_max_results_ceiling", 300)
    monkeypatch.setattr(cnki_client.settings, "cnki_paging_time_budget_seconds", 600.0)
    yield
    get_circuit_breaker("cnki", failure_threshold=2, recovery_timeout=120.0).reset()


def test_adaptive_paging_exceeds_max_results_within_year_window(monkeypatch):
    """窗口内文献多于 max_results 时应继续翻页，而不是停在固定页数。"""
    pages = [
        [_record(f"p{page}-{i}", 2025) for i in range(20)]
        for page in range(4)
    ]
    pages.append([_record("old-0", 2019)])  # 越界页
    _install_pages(monkeypatch, pages)

    papers = cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=20)

    # 固定页数模式下只会抓 1 页 20 条；自适应模式抓完窗口内 4 页 80 条。
    assert len(papers) == 80
    assert all(p.year == 2025 for p in papers)


def test_stops_at_year_boundary(monkeypatch):
    """整页早于 start_year 即停止，不再无谓深翻。"""
    pages = [
        [_record(f"new-{i}", 2025) for i in range(20)],
        [_record(f"old-{i}", 2018) for i in range(20)],
        [_record(f"older-{i}", 2010) for i in range(20)],
    ]
    state = _install_pages(monkeypatch, pages)

    papers = cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=200)

    # 第 2 页整页越界 → 停止，第 3 页不应被访问。
    assert state["next_calls"] == 1
    # 越界记录在年份过滤阶段被剔除。
    assert len(papers) == 20
    assert all(p.year == 2025 for p in papers)


def test_boundary_requires_configured_consecutive_pages(monkeypatch):
    """boundary_pages=2 时单页抖动不触发停止。"""
    monkeypatch.setattr(cnki_client.settings, "cnki_year_boundary_pages", 2)
    pages = [
        [_record(f"a-{i}", 2025) for i in range(20)],
        [_record(f"b-{i}", 2019) for i in range(20)],   # 抖动页
        [_record(f"c-{i}", 2025) for i in range(20)],   # 又回到窗口内
        [_record(f"d-{i}", 2018) for i in range(20)],
        [_record(f"e-{i}", 2017) for i in range(20)],   # 连续第 2 页越界 → 停
        [_record(f"f-{i}", 2016) for i in range(20)],
    ]
    state = _install_pages(monkeypatch, pages)

    papers = cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=200)

    assert state["next_calls"] == 4  # 抓到第 5 页后停止
    assert len(papers) == 40  # 两页 2025 的记录通过年份过滤


def test_result_cap_ceiling_stops_paging(monkeypatch):
    """总条数上限触顶即停止，防止宽泛检索式无限深翻。"""
    monkeypatch.setattr(cnki_client.settings, "cnki_max_results_ceiling", 45)
    pages = [[_record(f"p{page}-{i}", 2025) for i in range(20)] for page in range(10)]
    _install_pages(monkeypatch, pages)

    papers = cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=20)

    assert len(papers) == 45


def test_page_limit_caps_adaptive_paging(monkeypatch):
    """页数硬上限生效。"""
    monkeypatch.setattr(cnki_client.settings, "cnki_max_pages", 2)
    pages = [[_record(f"p{page}-{i}", 2025) for i in range(20)] for page in range(10)]
    _install_pages(monkeypatch, pages)

    papers = cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=200)

    assert len(papers) == 40


def test_time_budget_stops_paging(monkeypatch):
    """时长预算耗尽即停止。"""
    monkeypatch.setattr(cnki_client.settings, "cnki_paging_time_budget_seconds", 1.0)
    pages = [[_record(f"p{page}-{i}", 2025) for i in range(20)] for page in range(10)]
    state = _install_pages(monkeypatch, pages)

    # 时钟由页面采集回调驱动：读完第 1 页即把时间推到远超预算处。
    # 直接 patch time.monotonic 会连带影响熔断器/限流器的取值，故用共享 holder。
    clock = {"now": 0.0}
    real_collect = cnki_client.collect_result_records

    def collect_and_advance(driver):
        records = real_collect(driver)
        clock["now"] += 100.0
        return records

    monkeypatch.setattr(cnki_client, "collect_result_records", collect_and_advance)
    monkeypatch.setattr(cnki_client.time, "monotonic", lambda: clock["now"])

    papers = cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=200)

    assert state["next_calls"] == 0  # 第 1 页后预算即耗尽，未再翻页
    assert len(papers) == 20


def test_unparsable_years_do_not_stop_paging(monkeypatch):
    """整页年份都解析不出时不应误判越界。"""
    no_year_page = [
        {"title": f"x{i}", "url": f"u{i}", "row_text": "无年份信息"} for i in range(20)
    ]
    pages = [
        no_year_page,
        [_record(f"y-{i}", 2025) for i in range(20)],
        [_record(f"z-{i}", 2015) for i in range(20)],
    ]
    state = _install_pages(monkeypatch, pages)

    papers = cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=200)

    assert state["next_calls"] == 2  # 继续翻到第 3 页才因越界停止
    # 无年份记录按 year is None 放行，2025 的 20 条通过过滤。
    assert len(papers) == 40


def test_fixed_paging_mode_preserved(monkeypatch):
    """关闭自适应时行为回到原固定页数逻辑。"""
    monkeypatch.setattr(cnki_client.settings, "cnki_adaptive_year_paging", False)
    pages = [[_record(f"p{page}-{i}", 2025) for i in range(20)] for page in range(10)]
    _install_pages(monkeypatch, pages)

    papers = cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=20)

    assert len(papers) == 20


# ============================================================
# 详情页增强：先排序再增强
# ============================================================
def test_detail_enrichment_targets_most_relevant_records(monkeypatch):
    """额度落在相关度最高的记录上，而非结果页靠前的条目。"""
    _pin_detail_limit(monkeypatch, 2)
    records = [
        _record("与主题无关的园艺栽培技术", 2025),
        _record("智慧课堂环境下的课堂行为分析研究", 2025),
        _record("另一篇不相关的机械加工工艺", 2025),
        _record("课堂行为分析与教学互动评价", 2025),
    ]
    _install_pages(monkeypatch, [records])

    enriched_urls = []

    def fake_parse_detail(_driver, url):
        enriched_urls.append(url)
        return {"abstract": "摘要内容", "keywords": "课堂行为;教学互动"}

    monkeypatch.setattr(cnki_client, "parse_detail", fake_parse_detail)

    papers = cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=10)

    assert len(enriched_urls) == 2
    # 命中主题词的两篇拿到详情增强。
    enriched_titles = {p.title for p in papers if p.abstract}
    assert enriched_titles == {
        "智慧课堂环境下的课堂行为分析研究",
        "课堂行为分析与教学互动评价",
    }
    # 其余条目仍保留结果页元数据，不因未增强而丢失。
    assert len(papers) == 4


def test_detail_enrichment_skips_out_of_window_records(monkeypatch):
    """年份越界的记录不该消耗详情额度——它们最终会被年份过滤剔除。"""
    _pin_detail_limit(monkeypatch, 5)
    records = [
        _record("课堂行为分析早期探索", 2016),
        _record("课堂行为分析新进展", 2025),
        _record("课堂行为分析历史回顾", 2015),
    ]
    _install_pages(monkeypatch, [records])

    enriched_urls = []

    def fake_parse_detail(_driver, url):
        enriched_urls.append(url)
        return {"abstract": "摘要"}

    monkeypatch.setattr(cnki_client, "parse_detail", fake_parse_detail)

    cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=10)

    # 仅窗口内那一篇被增强。
    assert len(enriched_urls) == 1
    assert "课堂行为分析新进展" in enriched_urls[0]


def test_detail_failures_stop_enrichment_but_keep_results(monkeypatch):
    """连续详情失败仅停止增强，不让整次检索退化为 0 篇。"""
    _pin_detail_limit(monkeypatch, 10)
    monkeypatch.setattr(cnki_client.settings, "cnki_max_consecutive_detail_failures", 2)
    records = [_record(f"课堂行为分析研究{i}", 2025) for i in range(6)]
    _install_pages(monkeypatch, [records])

    attempts = {"n": 0}

    def failing_parse_detail(_driver, _url):
        attempts["n"] += 1
        raise cnki_client.TimeoutException("detail timeout")

    monkeypatch.setattr(cnki_client, "parse_detail", failing_parse_detail)

    papers = cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=10)

    assert attempts["n"] == 2  # 连续 2 次失败后停止
    assert len(papers) == 6  # 结果页元数据完整保留
    assert all(not p.abstract for p in papers)


def test_legacy_enrichment_order_when_ranking_disabled(monkeypatch):
    """关闭开关时回到"按结果页顺序取前 N 条"的旧行为。"""
    monkeypatch.setattr(cnki_client.settings, "cnki_detail_rank_before_enrichment", False)
    _pin_detail_limit(monkeypatch, 2)
    records = [
        _record("与主题无关的园艺栽培技术", 2025),
        _record("另一篇不相关的机械加工工艺", 2025),
        _record("课堂行为分析与教学互动评价", 2025),
    ]
    _install_pages(monkeypatch, [records])

    enriched_urls = []

    def fake_parse_detail(_driver, url):
        enriched_urls.append(url)
        return {"abstract": "摘要"}

    monkeypatch.setattr(cnki_client, "parse_detail", fake_parse_detail)

    cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=10)

    assert len(enriched_urls) == 2
    # 旧行为：取前两条（均不相关）。
    assert "园艺栽培技术" in enriched_urls[0]
    assert "机械加工工艺" in enriched_urls[1]


def test_select_records_for_detail_is_order_stable():
    """选中下标按原始顺序返回，且同分不打乱原顺序。"""
    records = [
        {"title": "课堂行为分析 A", "row_text": "刊物 2025"},
        {"title": "无关主题", "row_text": "刊物 2025"},
        {"title": "课堂行为分析 B", "row_text": "刊物 2025"},
    ]

    indices = cnki_client.select_records_for_detail(
        records, "课堂行为分析", 2024, 2026, limit=2,
    )

    assert indices == [0, 2]


def test_select_records_for_detail_respects_zero_limit():
    """额度为 0 时不选任何记录。"""
    records = [{"title": "课堂行为分析", "row_text": "2025"}]

    assert cnki_client.select_records_for_detail(records, "课堂行为分析", 2024, 2026, 0) == []


# ============================================================
# 详情额度自适应：随下游需求缩放，独立时长预算
# ============================================================
def test_detail_limit_scales_with_downstream_demand(monkeypatch):
    """额度随 max_results 增长，避免大结果池里多数文献只有标题年份。"""
    # autouse fixture 关闭了增强；这里恢复默认下限以验证缩放行为。
    monkeypatch.setattr(cnki_client.settings, "cnki_detail_enrichment_limit", 60)
    small = cnki_client.resolve_detail_enrichment_limit(30)
    large = cnki_client.resolve_detail_enrichment_limit(64)
    assert large > small
    # 需求倍数 >1：粗排选中集与最终入选集不完全重合，须留重叠损耗。
    assert large >= 64


def test_detail_limit_respects_floor_and_ceiling(monkeypatch):
    settings = cnki_client.settings
    monkeypatch.setattr(settings, "cnki_detail_enrichment_limit", 60)
    monkeypatch.setattr(settings, "cnki_detail_enrichment_max", 120)
    monkeypatch.setattr(settings, "cnki_detail_enrichment_demand_factor", 1.6)

    # 小请求不低于下限，避免小样本任务连基本摘要都拿不到。
    assert cnki_client.resolve_detail_enrichment_limit(5) == 60
    # 大请求被上限兜住，防止时间成本失控。
    assert cnki_client.resolve_detail_enrichment_limit(500) == 120


def test_detail_limit_zero_disables_enrichment(monkeypatch):
    """额度配 0 时整体关闭增强，沿用旧语义。"""
    monkeypatch.setattr(cnki_client.settings, "cnki_detail_enrichment_limit", 0)
    assert cnki_client.resolve_detail_enrichment_limit(100) == 0


def test_detail_budget_is_independent_from_paging_budget(monkeypatch):
    """翻页耗时长时增强仍应拿到自己的时间预算。"""
    monkeypatch.setattr(cnki_client.settings, "cnki_paging_time_budget_seconds", 1000.0)
    monkeypatch.setattr(cnki_client.settings, "cnki_detail_time_budget_seconds", 1000.0)
    _pin_detail_limit(monkeypatch, 3)
    records = [_record(f"课堂行为分析研究{i}", 2025) for i in range(3)]
    _install_pages(monkeypatch, [records])

    # 让翻页阶段"消耗"远超翻页预算之外的时间，仍不应影响增强额度。
    clock = {"now": 0.0}
    real_collect = cnki_client.collect_result_records

    def collect_and_advance(driver):
        out = real_collect(driver)
        clock["now"] += 900.0
        return out

    enriched = []
    monkeypatch.setattr(cnki_client, "collect_result_records", collect_and_advance)
    monkeypatch.setattr(cnki_client.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        cnki_client, "parse_detail",
        lambda _d, url: enriched.append(url) or {"abstract": "摘要"},
    )

    cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=10)

    assert len(enriched) == 3


def test_detail_budget_exhaustion_stops_enrichment_only(monkeypatch):
    """增强预算耗尽只停止增强，已抓取的文献仍全部交付。"""
    monkeypatch.setattr(cnki_client.settings, "cnki_detail_time_budget_seconds", 1.0)
    _pin_detail_limit(monkeypatch, 5)
    records = [_record(f"课堂行为分析研究{i}", 2025) for i in range(5)]
    _install_pages(monkeypatch, [records])

    clock = {"now": 0.0}
    enriched = []

    def parse_and_advance(_driver, url):
        enriched.append(url)
        clock["now"] += 100.0
        return {"abstract": "摘要"}

    monkeypatch.setattr(cnki_client, "parse_detail", parse_and_advance)
    monkeypatch.setattr(cnki_client.time, "monotonic", lambda: clock["now"])

    papers = cnki_client.search_cnki("课堂行为分析", 2024, 2026, max_results=10)

    assert len(enriched) == 1          # 第一篇后预算即耗尽
    assert len(papers) == 5            # 结果页元数据完整交付
    assert sum(1 for p in papers if p.abstract) == 1


def test_detail_title_excludes_page_control_text_inside_heading(monkeypatch):
    """详情页标题只取 h1 的直接文本，忽略嵌在标题里的操作控件文案。

    回归实测缺陷：CNKI 把「题录」等控件放在标题 h1 内部，``element.text``
    把它一并带出，参考文献里出现《基于课堂行为分析的医学理论教学质量评估
    模型 题录》。判据取 DOM 结构（直接文本节点 vs 后代元素），不依赖控件
    文案词表，因此知网改文案或加按钮都不会让它失效。

    不依赖 selenium：``find_own_text`` 只用到被替身接管的 ``wait_first``
    和 driver 的 ``execute_script``。
    """

    class _Heading:
        text = "基于课堂行为分析的医学理论教学质量评估模型 题录"

    class _OwnTextDriver:
        def execute_script(self, script, *args):
            # 只有标题的直接文本节点，控件文案属于后代元素
            return "基于课堂行为分析的医学理论教学质量评估模型"

    monkeypatch.setattr(
        cnki_client, "wait_first", lambda *args, **kwargs: _Heading()
    )

    title = cnki_client.find_own_text(
        _OwnTextDriver(), [("css selector", "div.wx-tit h1")]
    )

    assert title == "基于课堂行为分析的医学理论教学质量评估模型"


def test_detail_title_falls_back_to_element_text_when_script_unavailable(monkeypatch):
    """取不到直接文本节点时退回 element.text：宁可带控件文案也不丢标题。"""

    class _Heading:
        text = "一种基于实时网络的学生课堂行为分析方法"

    class _BrokenScriptDriver:
        def execute_script(self, script, *args):
            raise RuntimeError("javascript disabled")

    monkeypatch.setattr(
        cnki_client, "wait_first", lambda *args, **kwargs: _Heading()
    )

    title = cnki_client.find_own_text(
        _BrokenScriptDriver(), [("css selector", "h1")]
    )

    assert title == "一种基于实时网络的学生课堂行为分析方法"
