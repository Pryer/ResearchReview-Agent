# -*- coding: utf-8 -*-
"""精准短语检索与标题清洗单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.clients.arxiv_client import _format_arxiv_query, search_arxiv
from app.schemas.paper_schema import PaperMetadata
from app.utils.deduplicate import deduplicate_papers, normalize_title, title_similarity
from app.utils.title_cleaner import clean_abstract, clean_title, clean_venue


class TestTitleCleaner:
    """测试标题与摘要文本清洗工具。"""

    def test_clean_html_tags(self):
        raw = "<i>Deep Learning</i> for <jats:italic>Action Recognition</jats:italic>"
        assert clean_title(raw) == "Deep Learning for Action Recognition"

    def test_clean_latex_math_formulas(self):
        raw = r"$$\text {A}^2\text {M}^2$$-Net: Adaptively Aligned Multi-Scale Moment"
        cleaned = clean_title(raw)
        assert "A2M2-Net" in cleaned
        assert r"\text" not in cleaned
        assert "$$" not in cleaned

    def test_clean_html_entities(self):
        raw = "Robotics &amp; Autonomous Systems: Part&#x2013;1"
        assert clean_title(raw) == "Robotics & Autonomous Systems: Part–1"

    def test_font_commands_outside_the_old_whitelist_are_stripped(self):
        r"""回归：词表只列了 \text/\textbf/\textit/\mathrm，\mathbf 整条漏进标题。

        真实参考文献条目渲染出 "(CLIP-CP\mathbf{M2C)"——反斜杠命令直接见刊。
        """
        cleaned = clean_title(
            r"Multi-Modal Prompt Learning (CLIP-CP$\mathbf{M^2C}$) for Recognition"
        )
        assert "\\" not in cleaned
        assert "mathbf" not in cleaned
        assert "M2C" in cleaned

        for command in [r"\mathcal{X}", r"\bm{v}", r"\boldsymbol{w}",
                        r"\emph{Net}", r"\operatorname{softmax}"]:
            out = clean_title(f"A Study of {command} Models")
            assert "\\" not in out, command

    def test_brace_truncated_font_command_leaves_no_backslash(self):
        """上游截断花括号时脱壳规则匹配不到，命令名会裸奔进标题。"""
        cleaned = clean_title(r"Prompt Learning (CLIP-CP\mathbf{M2C) for Few-Shot")
        assert "\\" not in cleaned
        assert "mathbf" not in cleaned
        assert "M2C)" in cleaned

    def test_symbol_commands_keep_their_name(self):
        r"""\alpha 这类命令名即内容，不能与样式命令一起删掉。"""
        cleaned = clean_title(r"On $\alpha$-Divergence Minimization")
        assert "alpha" in cleaned
        assert "Divergence Minimization" in cleaned

    def test_nested_font_commands_are_unwrapped(self):
        cleaned = clean_title(r"\textbf{\emph{Deep}} Metric Learning")
        assert cleaned == "Deep Metric Learning"

    def test_clean_abstract_strips_font_commands(self):
        cleaned = clean_abstract(r"We propose $\mathbf{M^2C}$, a new module.")
        assert "\\" not in cleaned
        assert "mathbf" not in cleaned

    def test_clean_abstract_tags(self):
        raw = "<p>In this paper, we propose a novel <b>framework</b> for few-shot learning.</p>"
        assert clean_abstract(raw) == "In this paper, we propose a novel framework for few-shot learning."

    def test_clean_empty_and_none(self):
        assert clean_title(None) == ""
        assert clean_title("") == ""
        assert clean_abstract(None) == ""


class TestVenueCleaner:
    """测试 venue 清洗：行政区划标签与机构名/刊名的边界。"""

    def test_administrative_tags_do_not_eat_institution_names(self):
        """回归：省份正则的"省"字曾是可选的且无锚定，省名被从机构名内部剥掉。

        实测落地样本 logs/app.log:9088 的 "[J]. 大学, 2022"——参考文献里
        出现了《大学》这样的假刊名。市 / 自治区 后缀此前也整个漏过。
        """
        assert clean_venue("浙江大学浙江省211工程院校985工程院校教育部直属院校一流大学") == "浙江大学"
        assert clean_venue("河南大学河南省") == "河南大学"
        assert clean_venue("北京邮电大学北京市211工程院校教育部直属院校") == "北京邮电大学"
        assert clean_venue("桂林电子科技大学广西壮族自治区") == "桂林电子科技大学"
        assert clean_venue("合肥工业大学安徽省211工程院校教育部直属院校") == "合肥工业大学"
        assert clean_venue("湖南理工学院湖南省") == "湖南理工学院"

    def test_province_prefixed_journal_names_are_preserved(self):
        """受害者不止学位论文：省名开头的中文刊名同样被砍。"""
        assert clean_venue("浙江社会科学") == "浙江社会科学"
        assert clean_venue("江苏高教") == "江苏高教"
        assert clean_venue("北京大学学报(哲学社会科学版)") == "北京大学学报(哲学社会科学版)"
        assert clean_venue(
            "天津大学学报(自然科学与工程技术版) . 2025 ,58 (01) : 91-100 查看该刊数据库收录来源"
        ).startswith("天津大学学报(自然科学与工程技术版)")

    def test_ssrn_fake_journal_name_is_normalized(self):
        """Crossref 给所有 SSRN 预印本套同一个假刊名，不能当期刊渲染。"""
        assert clean_venue("SSRN Electronic Journal") == "SSRN"

    def test_volume_issue_page_tail_is_stripped(self):
        """回归：CNKI 把年/卷/期/页拼在刊名后，整串进入 venue 位导致年份重复。

        参考文献曾渲染成 "…智能物联技术 . 2026 ,58 (03) : 93-98, 2026." ——
        PaperCard 没有 volume/issue/pages 字段，这段残留无处安放。
        """
        assert clean_venue("智能物联技术 . 2026 ,58 (03) : 93-98 查看该刊数据库收录来源") == "智能物联技术"
        assert clean_venue("中文信息学报 . 2026 ,40 (01) : 1-29") == "中文信息学报"
        assert clean_venue("传感技术学报 . 2024 ,37 (11) : 1921-1930 查看该刊数据库收录来源") == "传感技术学报"
        # 无卷号、增刊号（Z1）形态
        assert clean_venue("教育信息技术 . 2026 (Z1) : 13-17 下载 HTML阅读 CNKI AI阅读") == "教育信息技术"
        # 刊名自带括号：必须锚定 " . YYYY" 而不是按括号切
        assert clean_venue(
            "天津大学学报(自然科学与工程技术版) . 2025 ,58 (01) : 91-100 查看该刊数据库收录来源"
        ) == "天津大学学报(自然科学与工程技术版)"

    def test_parenthesized_journal_names_are_not_truncated(self):
        """带括号或以年份结尾的正当刊名不得被卷期页正则误伤。"""
        for venue in [
            "Journal of Shanghai Jiaotong University (Science)",
            "ACM Computing Surveys (CSUR)",
            "南京邮电大学学报(自然科学版)",
            # 裸年份结尾没有卷/期/页，不构成 CNKI 尾巴
            "Proceedings of SPIE . 2024",
        ]:
            assert clean_venue(venue) == venue, f"Failed for {venue}"

    def test_clean_empty_venue(self):
        assert clean_venue(None) == ""
        assert clean_venue("") == ""


class TestArxivQueryFormatting:
    """测试 arXiv 查询构造器。"""

    def test_multi_word_phrase_is_quoted(self):
        q = "few-shot action recognition"
        formatted = _format_arxiv_query(q)
        assert formatted == 'all:"few-shot action recognition"'

    def test_single_word_remains_unquoted(self):
        q = "fewshot"
        formatted = _format_arxiv_query(q)
        assert formatted == "all:fewshot"

    def test_already_quoted_query_preserved(self):
        q = '"few-shot learning"'
        formatted = _format_arxiv_query(q)
        assert formatted == '"few-shot learning"'

    def test_field_prefixed_query_preserved(self):
        q = 'ti:"action recognition"'
        formatted = _format_arxiv_query(q)
        assert formatted == 'ti:"action recognition"'

    def test_boolean_query_preserved(self):
        q = "few-shot AND action"
        formatted = _format_arxiv_query(q)
        assert formatted == "all:few-shot AND action"

    def test_date_sort_forces_phrase(self):
        q = "few shot"
        formatted = _format_arxiv_query(q, force_phrase=True)
        assert formatted == 'all:"few shot"'


class TestArxivSearchFallback:
    """测试 arXiv 检索执行与平滑回退。"""

    @patch("app.clients.arxiv_client._arxiv_get")
    def test_phrase_search_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2508.03695v1</id>
                <title>Trokens: Semantic-Aware Relational Trajectory Tokens</title>
                <summary>Test abstract</summary>
                <published>2025-08-01T00:00:00Z</published>
            </entry>
        </feed>"""
        mock_get.return_value = mock_resp

        papers = search_arxiv("few-shot action recognition", 2024, 2026, max_results=10)
        assert len(papers) == 1
        assert papers[0].arxiv_id == "2508.03695"
        assert "Trokens" in papers[0].title

    @patch("app.clients.arxiv_client._arxiv_get")
    def test_phrase_search_fallback_when_empty(self, mock_get):
        empty_resp = MagicMock()
        empty_resp.text = """<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""

        fallback_resp = MagicMock()
        fallback_resp.text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2401.00001v1</id>
                <title>Fallback Paper</title>
                <summary>Abstract</summary>
                <published>2024-01-01T00:00:00Z</published>
            </entry>
        </feed>"""

        # 第一次精准检索返回空，第二次松散检索返回 1 篇
        mock_get.side_effect = [empty_resp, fallback_resp]

        papers = search_arxiv("very specific phrase query", 2024, 2026, max_results=10)
        assert len(papers) == 1
        assert papers[0].title == "Fallback Paper"
        assert mock_get.call_count == 2


class TestLatexTitleDeduplication:
    """测试包含 LaTeX 数学公式标题的去重能力。"""

    def test_latex_title_matches_plain_title(self):
        p1 = {
            "paper_id": "doi:10.1007/s11263-025-02432-4",
            "title": r"$$\text {A}^2\text {M}^2$$-Net: Adaptively Aligned Multi-scale Moment for Few-Shot Action Recognition",
            "year": 2025,
            "doi": "10.1007/s11263-025-02432-4",
        }
        p2 = {
            "paper_id": "arxiv:2509.17638",
            "title": "A2M2-Net: Adaptively Aligned Multi-Scale Moment for Few-Shot Action Recognition",
            "year": 2025,
            "doi": None,
        }

        # 标题相似度应足够高
        sim = title_similarity(p1["title"], p2["title"])
        assert sim >= 0.85

        deduped = deduplicate_papers([p1, p2], similarity_threshold=0.85)
        assert len(deduped) == 1


class TestCrossKeyIdentityClosure:
    """DOI 与 arXiv 键空间必须有跨键闭包（M14）。"""

    def test_same_paper_via_doi_only_and_arxiv_only_merges(self):
        p_openalex = {
            "paper_id": "openalex:W1",
            "title": "Attention Is All You Need",
            "doi": "10.5555/attention",
            "arxiv_id": None,
        }
        p_arxiv = {
            "paper_id": "arxiv:2401.12345",
            "title": "Attention Is All You Need",
            "doi": None,
            "arxiv_id": "2401.12345",
            "pdf_url": "https://arxiv.org/pdf/2401.12345",
        }
        # 无桥接记录时依靠标题相似度合并；合并后双键回填到存活记录，
        # 后续身份判定看到同一组键。
        for papers in ([p_openalex, p_arxiv], [p_arxiv, p_openalex]):
            result = deduplicate_papers([dict(p) for p in papers])
            assert len(result) == 1
            merged = result[0]
            assert merged.get("doi") == "10.5555/attention"
            assert merged.get("arxiv_id") == "2401.12345"

    def test_divergent_titles_without_bridge_stay_separate(self):
        # 确定性规则的保守边界：既无共享键又无桥接记录、标题差异超阈值，
        # 不允许凭空判定为同一论文。
        r0 = {"paper_id": "a", "title": "One Title", "doi": "10.1/x", "arxiv_id": None}
        r1 = {
            "paper_id": "b", "title": "A Completely Different Study",
            "doi": None, "arxiv_id": "2401.9",
        }
        assert len(deduplicate_papers([r0, r1])) == 2

    def test_third_record_bridging_both_keys_folds_earlier_survivors(self):
        r0 = {"paper_id": "a", "title": "One Title", "doi": "10.1/x", "arxiv_id": None}
        r1 = {"paper_id": "b", "title": "Another Title", "doi": None, "arxiv_id": "2401.9"}
        bridge = {
            "paper_id": "c", "title": "Yet Another",
            "doi": "10.1/x", "arxiv_id": "2401.9",
        }
        result = deduplicate_papers([r0, r1, bridge])
        assert len(result) == 1

    def test_plain_title_dedup_behavior_preserved(self):
        papers = [
            {"paper_id": "a", "title": "Same Title Here"},
            {"paper_id": "b", "title": "Same Title Here"},
            {"paper_id": "c", "title": "Different Topic Paper"},
        ]
        assert len(deduplicate_papers(papers)) == 2


# ---------- CNKI 卷期页码尾巴：含续接页「+」形态 ----------

def test_cnki_venue_tail_with_continuation_pages_is_stripped():
    """CNKI 用 "+" 表示续接页，原正则只允许「数字-数字」，整段未被剥除。

    实测参考文献残留 "中国教育技术装备 . 2026 (05) : 46-52+62"。
    """
    from app.utils.title_cleaner import clean_venue

    assert clean_venue("中国教育技术装备 . 2026 (05) : 46-52+62", "cnki") == "中国教育技术装备"
    assert clean_venue("电脑知识与技术 . 2026 ,22 (01) : 26-28+55", "cnki") == "电脑知识与技术"
    assert clean_venue("中国医学教育技术 . 2024 ,38 (06) : 741-745+761", "cnki") == "中国医学教育技术"
    # 增刊号与普通页码范围仍正常
    assert clean_venue("计算机科学 . 2026 ,53 (Z1) : 1-8", "cnki") == "计算机科学"


def test_clean_venue_does_not_truncate_legitimate_names():
    """带括号的正当刊名与以年份结尾的刊名不得被误砍。"""
    from app.utils.title_cleaner import clean_venue

    assert clean_venue("现代教育技术", "cnki") == "现代教育技术"
    assert clean_venue("天津大学学报(自然科学与工程技术版)", "cnki") == "天津大学学报(自然科学与工程技术版)"
    assert (
        clean_venue("Journal of Shanghai Jiaotong University (Science)", "crossref")
        == "Journal of Shanghai Jiaotong University (Science)"
    )
    # 年份后没有卷/期/页时不剥除
    assert clean_venue("某刊 . 2024", "cnki") == "某刊 . 2024"


# ---------- 作者名规范化：只做确定性清洗，不猜姓名顺序 ----------

def test_author_normalization_strips_affiliation_superscript():
    from app.utils.title_cleaner import normalize_author_name, normalize_author_names

    assert normalize_author_name("王猛2") == "王猛"
    assert normalize_author_name("李雷 1") == "李雷"
    assert normalize_author_names(["王猛2", "", "李雷 1"]) == ["王猛", "李雷"]
    assert normalize_author_names(None) == []


def test_author_normalization_never_reorders_names():
    """不得猜测中文姓名顺序。

    误写的 "英哲 武"（名2+姓1）与正当写法 "欧阳 修"（复姓2+名1）字形完全一致，
    没有姓氏表无法区分；把正确姓名改错比保留原样更有害。
    """
    from app.utils.title_cleaner import normalize_author_name

    assert normalize_author_name("欧阳 修") == "欧阳 修"
    assert normalize_author_name("英哲 武") == "英哲 武"
    assert normalize_author_name("武英哲") == "武英哲"
    assert normalize_author_name("Zhang Wei") == "Zhang Wei"
    assert normalize_author_name("Smith, John") == "Smith, John"
    assert normalize_author_name(None) == ""


def test_cnki_author_split_applies_normalization():
    from app.clients.cnki_client import _split_authors

    assert _split_authors("孙洋杰1 王猛2") == ["孙洋杰", "王猛"]
    assert _split_authors("") == []
