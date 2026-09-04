"""出版 venue 权威分层词表与分类器。

顶会顶刊 / 中文核心（CSSCI/CSCD/卓越计划）/ 预印本平台的确定性识别，
供质量打分使用。同时提供文献形态判据（预印本 / 会议 / 学位论文），
供文献类型推断与证据池硬过滤复用同一份 venue 词表。纯数据 + 纯函数。
"""

from __future__ import annotations

import re


_TOP_TIER_PHRASES = (
    # Top Multidisciplinary & Comprehensive International Journals
    "nature communications", "science advances", "proceedings of the national academy of sciences",
    # Top Medicine & Life Sciences
    "the lancet", "lancet", "new england journal of medicine", "nature medicine",
    "nature biotechnology", "nature methods", "nature genetics", "nature neuroscience",
    # Top Physics, Chemistry & Materials
    "physical review letters", "journal of the american chemical society", "angewandte chemie",
    "advanced materials",
    # Top CV / PR / Multimodal Conferences & Journals
    "tpami", "pattern analysis and machine intelligence",
    "cvpr", "computer vision and pattern recognition",
    "iccv", "international conference on computer vision",
    "eccv", "european conference on computer vision",
    "ijcv", "international journal of computer vision",
    "circuits and systems for video technology", "tcsvt",
    "neural networks and learning systems", "tnnls",
    "acm multimedia", "acm mm",
    "pattern recognition",
    # Top AI / ML Conferences & Journals
    "neurips", "neural information processing systems",
    "icml", "international conference on machine learning",
    "iclr", "international conference on learning representations",
    "aaai",
    "ijcai", "international joint conference on artificial intelligence",
    "journal of machine learning research", "jmlr",
    "artificial intelligence",
    # Top NLP / Data Mining / Systems / Comprehensive CS
    "association for computational linguistics", "emnlp",
    "sigkdd", "sigir", "thewebconf", "world wide web conference",
    "proceedings of the ieee",
    # ── 中文顶刊 / 权威 C 刊（CSSCI / CSCD 核心 / 卓越行动计划） ──
    # 计算机、信息、自动化与电子 (CSCD/CCF-T1)
    "计算机学报", "软件学报", "自动化学报", "电子学报", "通信学报", "模式识别与人工智能",
    "计算机研究与发展", "控制与决策", "计算机辅助设计与图形学学报", "控制理论与应用",
    "信息安全学报", "中文信息学报", "数据分析与知识发现", "系统工程理论与实践",
    # 综合与自然科学 (卓越计划 / CSCD核心)
    "中国科学", "科学通报", "物理学报", "化学学报", "力学学报", "光学学报",
    "生物物理学报", "地球物理学报", "机械工程学报", "航空学报", "仪器仪表学报",
    # 人文社科、教育、经管与心理 (权威 CSSCI / C 刊)
    "中国社会科学", "经济研究", "管理世界", "教育研究", "电化教育研究",
    "中国电化教育", "开放教育研究", "现代远程教育研究", "现代教育技术",
    "心理学报", "心理科学", "心理发展与教育", "社会学研究", "哲学研究",
    "法学研究", "新闻与传播研究", "管理科学学报", "中国软科学", "科研管理",
    "科学学研究", "情报学报", "数量经济技术经济研究",
    # 医学与生命科学中文顶刊
    "中华医学杂志", "中华内科杂志", "中华外科杂志", "药学学报", "生理学报",
    "生物化学与生物物理进展",
)

_TOP_TIER_EXACT_WORDS = {
    "nejm", "pnas", "prl", "jacs",
    "nips", "kdd", "acl", "naacl", "tip", "tmm", "pr", "aij",
    "cssci", "cscd",  # 中文 C 刊 / CSCD 核心明确标识
}

# 既是顶刊名又是普通英文词的刊名（Science / Nature / Cell）：只允许
# 作为 venue 的唯一实义词时才算顶刊（"Science"、"Science (New York,
# N.Y.)"），否则 "Computer Science"、"Stem Cell Research"、"Cell
# Biology" 这类名称会借单 token 命中被误判 +0.35。Lancet 家族由短语表
# 覆盖（"the lancet"/"lancet" 子串），不在此列。
_AMBIGUOUS_TOP_JOURNAL_WORDS = {"nature", "science", "cell"}
# 结构词白名单：出版地、括号注释等非实义词，允许与刊名同现。
_STRUCTURAL_VENUE_TOKENS = {
    "the", "new", "york", "ny", "n", "y", "london", "cambridge",
    "berlin", "boston", "usa", "uk", "england", "press", "inc", "ltd",
}

# 中文核心显式标识关键词（含 CSSCI / CSCD / 北大核心 / 卓越行动计划）
_CHINESE_CORE_TAGS = (
    "cssci", "cscd", "南大核心", "c刊", "北大核心", "中文核心", "卓越计划", "卓越期刊",
)

_PREPRINT_VENUES = (
    "arxiv", "biorxiv", "medrxiv", "ssrn", "research square", "techrxiv",
    "preprints.org", "chemrxiv", "osf.io", "open mind",
)

# 预印本平台自有 DOI 前缀：这些前缀本身就证明该条目是平台托管的预印本，
# 与 venue 文本是否规范无关（Crossref 给 SSRN 预印本统一套假刊名
# "SSRN Electronic Journal"，只看 venue 会被当成期刊）。
_PREPRINT_DOI_PREFIXES = (
    "10.48550/arxiv",  # arXiv
    "10.2139/ssrn",    # SSRN
    "10.1101/",        # bioRxiv / medRxiv
    "10.21203/rs.",    # Research Square
    "10.26434/chemrxiv",
)

# 学位论文标识。CNKI 学位论文 DOI 的 "/d.cnki." 段即"学位论文"，
# 10.7666/d. 是其早期形式。
_THESIS_DOI_MARKERS = ("/d.cnki.", "10.7666/d.")

_THESIS_VENUE_MARKERS = (
    "硕士学位论文", "博士学位论文", "硕士专业学位论文", "博士专业学位论文",
    "学位论文", "优秀硕士论文", "毕业论文",
    "master's thesis", "masters thesis", "master thesis",
    "doctoral dissertation", "doctoral thesis", "phd thesis", "ph.d. thesis",
    "proquest dissertations",
)

# 标题里的学位论文标记只在"文献类型后缀"位置才算判据：括注或结尾。
# 裸子串匹配会把《研究生学位论文质量评价研究》这类研究学位论文的期刊论文
# 一并排除——标题里提到学位论文说的是主题，不是该文自身的形态。
_THESIS_TITLE_SUFFIX = re.compile(
    r"[（(\[【]\s*(?:硕士|博士)?(?:专业)?(?:学位)?论文\s*[)）\]】]\s*$"
    r"|(?:硕士|博士)(?:专业)?学位论文\s*$"
    r"|(?:master(?:'s)?\s+thesis|doctoral\s+(?:thesis|dissertation)|ph\.?d\.?\s+thesis)\s*$",
    re.IGNORECASE,
)

# CNKI 详情页把培养单位属性标签直接拼在机构名后面
# （"合肥工业大学安徽省211工程院校教育部直属院校"）。判定机构型 venue 前
# 先剥掉这些标签，否则尾部不是机构后缀。
_INSTITUTION_ATTRIBUTE_TAGS = re.compile(
    r"(?:211工程(?:院校|大学)?|985工程(?:院校|大学)?|一流大学(?:建设高校)?"
    r"|一流学科(?:建设高校)?|双一流(?:建设高校)?|省部共建(?:协同创新中心)?"
    r"|教育部直属院校|中央部委院校"
    r"|(?:安徽|江苏|浙江|北京|上海|广东|山东|河南|湖北|湖南|四川|陕西|河北"
    r"|山西|辽宁|吉林|黑龙江|江西|福建|广西|贵州|云南|西藏|海南|内蒙古|宁夏"
    r"|新疆|青海|甘肃|重庆|天津)"
    r"(?:省|市|(?:壮族|回族|维吾尔)?自治区))+"
)

# 机构名后缀（培养单位）与刊物后缀。刊物后缀优先：
# "南京邮电大学学报" 是期刊而不是培养单位。
_INSTITUTION_SUFFIXES = ("大学", "学院", "研究院", "研究所", "科学院", "学校")
_PERIODICAL_MARKERS = ("学报", "学刊", "期刊", "论坛", "评论", "通报", "月刊", "季刊")

# 裸会议名：S2 / OpenAlex 常把会议 venue 记成不含 "conference" 字面词的
# 简称或全称（"Computer Vision and Pattern Recognition"、"ACM Multimedia"），
# 只靠字面词判定会把这些会议论文误标成期刊论文。
_CONFERENCE_NAME_PHRASES = (
    "computer vision and pattern recognition",
    "international conference on computer vision",
    "european conference on computer vision",
    "winter conference on applications of computer vision",
    "acm multimedia", "acm mm",
    "neural information processing systems",
    "international conference on machine learning",
    "international conference on learning representations",
    "conference on artificial intelligence",
    "international joint conference on artificial intelligence",
    "association for computational linguistics",
    "empirical methods in natural language processing",
    "knowledge discovery and data mining",
    "international conference on robotics and automation",
    "interspeech", "cvpr", "iccv", "eccv", "wacv", "neurips", "icml",
    "iclr", "ijcai", "emnlp", "naacl", "sigkdd", "sigir", "icassp", "icra",
)

# 会议出版方 DOI 段：venue 文本缺失或含混时的兜底判据。
_CONFERENCE_DOI_MARKERS = (
    "10.1109/cvpr", "10.1109/iccv", "10.1109/wacv", "10.1109/icassp",
    "10.1109/icra", "10.1609/aaai", "10.24963/ijcai", "10.18653/v1",
)

# 会议字面词：原有判据，保留作为通用规则。
_CONFERENCE_KEYWORDS = ("conference", "proceedings", "workshop", "symposium")


def has_publisher_doi(doi: str) -> bool:
    """判断 DOI 是否为正式出版方分配（而非预印本平台自有前缀）。"""
    doi_clean = str(doi or "").lower().strip()
    if not doi_clean:
        return False
    return not any(doi_clean.startswith(prefix) for prefix in _PREPRINT_DOI_PREFIXES)


def is_preprint_record(
    venue: str = "", doi: str = "", source: str = "", arxiv_id: str = "",
) -> bool:
    """判断条目是否为预印本。

    判据优先级：平台自有 DOI 前缀 > 出版方 DOI > 平台 venue / arxiv_id。
    出版方 DOI 必须压过平台信号，而不只是压过 ``arxiv_id``——arXiv 客户端
    对每条记录硬编码 ``venue="arXiv"``（arxiv_client.py），同时又从
    ``<link title="doi">`` 取回出版方 DOI。若让平台 venue 短路判定，
    CVPR / ICCV / AAAI / IJCV / TCSVT 的正式论文会照旧被标成 [EB/OL]。
    """
    doi_clean = str(doi or "").lower().strip()
    if any(doi_clean.startswith(prefix) for prefix in _PREPRINT_DOI_PREFIXES):
        return True
    if has_publisher_doi(doi):
        return False
    venue_clean = str(venue or "").lower().strip()
    if any(platform in venue_clean for platform in _PREPRINT_VENUES):
        return True
    return str(source or "").lower().strip() == "arxiv" or bool(arxiv_id)


# 平台占位 venue：整个 venue 就是托管平台名本身，不含任何刊物/会议信息。
# 只匹配"纯平台名"，避免把 "Open Mind" 这类正当刊名或含平台名的长串误伤。
_PLATFORM_ONLY_VENUE = re.compile(
    r"^(?:the\s+)?(?:arxiv|biorxiv|medrxiv|chemrxiv|techrxiv|ssrn|research\s+square)"
    r"(?:[\s.]*(?:org|com|preprints?|e-?prints?|server|repository))*$"
)


def is_platform_placeholder_venue(venue: str = "", doi: str = "") -> bool:
    """venue 只是托管平台名，而该条目已有出版方 DOI —— 这个 venue 是错的。

    渲染成 "[J]. arXiv, 2024" 等于断言该文发表在一本名为 arXiv 的期刊上。
    宁可不写 venue（DOI 可解析到正式出处），也不写一个已知错误的刊名。
    """
    venue_clean = str(venue or "").lower().strip()
    if not venue_clean:
        return False
    return bool(_PLATFORM_ONLY_VENUE.match(venue_clean)) and has_publisher_doi(doi)


def is_conference_venue(venue: str = "", doi: str = "") -> bool:
    """判断 venue 是否为学术会议（区别于期刊）。"""
    venue_clean = str(venue or "").lower().strip()
    if venue_clean:
        if any(keyword in venue_clean for keyword in _CONFERENCE_KEYWORDS):
            return True
        # 子串匹配方向很关键："pattern recognition"（Elsevier 期刊）不含
        # "computer vision and pattern recognition"，因此不会被误判为 CVPR。
        if any(phrase in venue_clean for phrase in _CONFERENCE_NAME_PHRASES):
            return True
    doi_clean = str(doi or "").lower().strip()
    return any(marker in doi_clean for marker in _CONFERENCE_DOI_MARKERS)


def is_degree_thesis(venue: str = "", doi: str = "", title: str = "") -> bool:
    """判断条目是否为学位论文（硕士/博士）。

    三类判据任一命中即成立：学位论文 DOI 段、venue/标题中的显式学位论文
    标记、venue 是纯培养单位名称。第三类是 CNKI 的常态——详情页的来源字段
    就是培养单位加属性标签（"合肥工业大学安徽省211工程院校教育部直属院校"）。
    """
    doi_clean = str(doi or "").lower().strip()
    if any(marker in doi_clean for marker in _THESIS_DOI_MARKERS):
        return True
    venue_clean = str(venue or "").lower()
    if any(marker in venue_clean for marker in _THESIS_VENUE_MARKERS):
        return True
    if _THESIS_TITLE_SUFFIX.search(str(title or "").strip()):
        return True

    venue_text = str(venue or "").strip()
    if not venue_text:
        return False
    # 刊物后缀优先：南京邮电大学学报 是期刊，不是培养单位。
    if any(marker in venue_text for marker in _PERIODICAL_MARKERS):
        return False
    stripped = _INSTITUTION_ATTRIBUTE_TAGS.sub("", venue_text).strip()
    return bool(stripped) and stripped.endswith(_INSTITUTION_SUFFIXES)


def classify_venue_tier(venue: str, doi: str = "", source: str = "") -> tuple[str, float]:
    """对学术出版物进行权威分层。

    分层规则（用户显式规则）：
    1. 顶会顶刊 / 中文C刊（Top-tier published conferences/journals, CSSCI/CSCD）：最高质量评分 (+0.35)。
    2. 普通已发表同行评审会议/期刊（Standard peer-reviewed published）：第二档 (+0.25)。
    3. 预印本平台（Preprints, 如 arXiv / bioRxiv / SSRN 等）：第三档 (+0.10)，评分严格低于已发表论文。
    4. 无出处 / 未知（Unknown）：基线分 (+0.02)。
    """
    venue_clean = str(venue or "").lower().strip()
    source_clean = str(source or "").lower().strip()
    doi_clean = str(doi or "").lower().strip()

    # 1. 检查是否为预印本
    is_preprint = any(p in venue_clean for p in _PREPRINT_VENUES)
    if is_preprint or (source_clean == "arxiv" and (not venue_clean or venue_clean == "arxiv")):
        return "preprint", 0.10

    # 2. 检查是否为顶会顶刊 / 中文 C 刊（含 CSSCI、CSCD、卓越期刊）
    tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", venue_clean))
    is_top_tier = False

    # (a) 精确词或核心标签（如 cssci, cscd, 物理学报, 软件学报 等）
    if tokens & _TOP_TIER_EXACT_WORDS:
        is_top_tier = True
    elif (
        tokens & _AMBIGUOUS_TOP_JOURNAL_WORDS
        and tokens <= _AMBIGUOUS_TOP_JOURNAL_WORDS | _STRUCTURAL_VENUE_TOKENS
    ):
        # "Science"、"Nature (London)" 算顶刊本身；"Computer Science"
        # 等含领域实义词的名称不算。
        is_top_tier = True
    elif any(tag in venue_clean for tag in _CHINESE_CORE_TAGS):
        is_top_tier = True
    elif any(phrase in venue_clean for phrase in _TOP_TIER_PHRASES):
        is_top_tier = True
    elif doi_clean and any(p in doi_clean for p in ("10.1609/aaai", "10.1109/tpami", "10.1007/s11263")):
        is_top_tier = True

    if is_top_tier:
        return "top_tier_published", 0.35

    # 3. 检查是否为普通已发表学术出版物（有正式会议/期刊名，或来自 crossref / cnki 且有出版信息）
    if venue_clean and len(venue_clean) >= 3 and not is_preprint:
        return "standard_published", 0.25

    if (source_clean in ("cnki", "crossref", "openalex") and (venue_clean or doi_clean)) and not is_preprint:
        return "standard_published", 0.25

    if source_clean == "arxiv":
        return "preprint", 0.10

    return "unknown", 0.02
