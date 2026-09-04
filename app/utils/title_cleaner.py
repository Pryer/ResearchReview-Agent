# -*- coding: utf-8 -*-
"""论文标题与文本清洗工具。

用于清洗来自不同学术数据源（Crossref, arXiv, OpenAlex 等）中的
HTML/XML 标签、LaTeX/MathJax 数学公式标记以及非规范字符，
确保标题规范化与去重/匹配的准确性。
"""

from __future__ import annotations

import html
import re
from typing import Optional


# LaTeX 字体/样式命令名：命令本身不携带信息，只包裹内容。此处必须用一条通配
# 规则而不是逐条列举——真实抓取样本 "…(CLIP-CP$\mathbf{M^2C}$)" 就因为词表里
# 只有 \text / \textbf / \textit / \mathrm，\mathbf 整条漏过，反斜杠命令直接
# 渲进了参考文献标题。
_LATEX_STYLE_NAMES = (
    r"math(?:bf|rm|it|cal|bb|sf|tt|frak)"
    r"|text(?:bf|it|rm|sf|tt|sc|up|md)?"
    r"|bm|boldsymbol|emph|mbox|operatorname|underline"
)

# 带花括号的规范形态：脱壳保留花括号里的内容。
_LATEX_STYLE_COMMAND = re.compile(
    r"\\(?:" + _LATEX_STYLE_NAMES + r")\s*\{([^{}]*)\}"
)

# 花括号被上游截断时（真实样本 "(CLIP-CP\mathbf{M2C)" 即是）上面的脱壳规则匹配
# 不到，命令名会裸奔进标题。只删样式命令名与残缺的左括号，保留其后正文；不碰
# \alpha 这类"命令名即内容"的符号命令，否则 "$\alpha$-divergence" 会丢掉 alpha。
_LATEX_ORPHAN_STYLE_COMMAND = re.compile(
    r"\\(?:" + _LATEX_STYLE_NAMES + r")\s*\{?"
)


def _strip_latex_style_commands(text: str) -> str:
    """反复脱壳直到稳定，以处理 \\textbf{\\emph{x}} 这类嵌套。"""
    for _ in range(3):
        text, replaced = _LATEX_STYLE_COMMAND.subn(r"\1", text)
        if not replaced:
            break
    return text


def clean_title(title: Optional[str]) -> str:
    """清洗并规范化论文标题。

    处理：
    1. HTML/JATS XML 标签（如 <i>, <b>, <jats:italic>, <sub>, <sup> 等）
    2. LaTeX 数学公式标记（如 $$\\text{A}^2\\text{M}^2$$, $\\alpha$, \\textbf{...} 等）
    3. HTML 转义字符（如 &amp;, &quot;, &#x2013; 等）
    4. 规范化换行与连续空白字符

    Args:
        title: 原始标题字符串。

    Returns:
        清洗后的标题字符串。
    """
    if not title:
        return ""

    text = str(title).strip()

    # 1. HTML 实体反转义
    text = html.unescape(text)

    # 2. 清除 HTML/XML 标签
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. 脱掉 LaTeX 字体/样式命令外壳，保留其中内容
    # 例如: $$\text {A}^2\text {M}^2$$ -> A^2M^2 或 A2M2
    text = _strip_latex_style_commands(text)

    # 4. 去除多余的公式包裹符号 $$ 或 $
    text = re.sub(r"\$\$([^\$]+)\$\$", r"\1", text)
    text = re.sub(r"\$([^\$]+)\$", r"\1", text)

    # 5. 规范化 LaTeX 上标/下标字母数字组合（如 A^2 -> A2, M_1 -> M1, A^{2} -> A2）
    text = re.sub(r"([A-Za-z])\^\{?([0-9]+)\}?", r"\1\2", text)
    text = re.sub(r"([A-Za-z])_\{?([0-9]+)\}?", r"\1\2", text)

    # 6. 清除多余的反斜杠符号（如 \documentclass, \usepackage 等偶尔嵌入在元数据里的残留）
    # 必须排在第 7 步之前：这些命令的花括号内容要整段丢掉，而第 7 步是保留内容的。
    text = re.sub(r"\\(documentclass|usepackage|setlength|oddsidemargin|begin|end)(\[[^\]]*\])?\{[^}]*\}", " ", text)

    # 7. 清除花括号残缺而在第 3 步漏过的裸样式命令，保留其后的正文
    text = _LATEX_ORPHAN_STYLE_COMMAND.sub(" ", text)

    # 8. 规范化换行与连续空白
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_abstract(abstract: Optional[str]) -> str:
    """清洗并规范化论文摘要文本。

    Args:
        abstract: 原始摘要字符串。

    Returns:
        清洗后的摘要文本。
    """
    if not abstract:
        return ""

    text = str(abstract).strip()
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = _strip_latex_style_commands(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_CNKI_VENUE_NOISE = [
    r"211工程(?:院校|大学)?",
    r"985工程(?:院校|大学)?",
    r"一流大学(?:建设高校)?",
    r"一流学科(?:建设高校)?",
    r"双一流(?:建设高校)?",
    r"省部共建(?:协同创新中心)?",
    r"教育部直属院校",
    r"中央部委院校",
    r"硕士(?:专业)?学位论文",
    r"博士(?:专业)?学位论文",
    r"学位论文",
    r"优秀硕士论文",
    # 行政区划标签必须带"省/市/自治区"后缀：CNKI 把培养单位所在地拼在
    # 机构名后（"浙江大学浙江省"）。若后缀可选，省名会被从机构名和刊名
    # 内部剥掉——"浙江大学浙江省…" 曾被清成"大学"、"天津大学学报" 曾被
    # 清成"大学学报"，最终参考文献里出现书名号《大学》式的假刊名。
    r"(?:安徽|江苏|浙江|北京|上海|广东|山东|河南|湖北|湖南|四川|陕西|河北"
    r"|山西|辽宁|吉林|黑龙江|江西|福建|广西|贵州|云南|西藏|海南|内蒙古|宁夏"
    r"|新疆|青海|甘肃|重庆|天津)(?:省|市|(?:壮族|回族|维吾尔)?自治区)",
]

# 平台假刊名归一化：Crossref 给所有 SSRN 预印本套同一个"期刊"名，
# 直接渲染成参考文献会把预印本伪装成期刊论文。
_VENUE_CANONICAL_NAMES = {
    "ssrn electronic journal": "SSRN",
}

# CNKI 把年/卷/期/页拼在刊名后面（"智能物联技术 . 2026 ,58 (03) : 93-98"）。
# PaperCard 没有 volume/issue/pages 字段，这段残留会整串进入 venue 位，
# 渲染成 "…智能物联技术 . 2026 ,58 (03) : 93-98, 2026." ——年份重复且标点错乱。
# 必须锚定在 " . YYYY" 分隔符上：刊名自身可能带括号
# （"Journal of Shanghai Jiaotong University (Science)"、
# "天津大学学报(自然科学与工程技术版)"），按括号切会砍掉正当刊名。
# 年份后要求紧跟卷/期/页起始符，避免误伤 "Foo . 2024" 这类以年份结尾的刊名。
_CNKI_VOLUME_ISSUE_TAIL = re.compile(
    r"\s*[.．]\s*(?:19|20)\d{2}\s*"              # . YYYY
    r"(?=[,，(（:：])"                            # 后面必须还有卷/期/页
    r"(?:[,，]\s*\d+\s*)?"                       # ,卷
    r"(?:[(（]\s*[0-9A-Za-z]{1,8}\s*[)）]\s*)?"   # (期)，含 Z1/S1 增刊号
    # : 页码。CNKI 用 "+" 表示续接页（"46-52+62"），也可能给出多段页码，
    # 原先只允许「数字-数字」，因此 "46-52+62" 整段未被剥除，刊名残留
    # 卷期页尾巴（实测 "中国教育技术装备 . 2026 (05) : 46-52+62"）。
    r"(?:[:：]\s*\d+(?:\s*[-–—~+,，]\s*\d+)*\s*)?$"
)


def clean_venue(venue: Optional[str], source: str = "") -> str:
    """清洗学术出版物/期刊名（venue），去除 CNKI 抓取时嵌入的院校标签及非规范残留。

    Args:
        venue: 原始 venue 文本。
        source: 数据源标识（如 cnki, crossref 等）。

    Returns:
        清洗后的 venue 字符串。
    """
    if not venue:
        return ""

    text = str(venue).strip()
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)

    # 移除检索页面控件文案
    text = re.sub(
        r"(?:\s*[-|｜]?\s*(?:下载|HTML阅读|AI阅读|CNKI AI阅读|阅读))+$",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\s*(?:下载\s*HTML阅读|HTML阅读\s*下载|CNKI\s*AI\s*阅读|AI\s*阅读)\s*$",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s*查看该刊数据库收录来源.*$", "", text).strip()
    if re.search(r"检索\s*CNKI\s*AI\s*出版来源|我的CNKI", text, re.I):
        return ""

    # 清除拼在刊名后的年/卷/期/页（页面控件文案已在上面剥完，此处才是纯尾巴）
    text = _CNKI_VOLUME_ISSUE_TAIL.sub("", text).strip()

    # 清除 "(Print)" 或 "(Online)" 残留
    text = re.sub(r"\s*\((?:Print|Online|Electronic)\)\s*", " ", text, flags=re.I).strip()

    # 清除 CNKI 院校/属性标签
    for pattern in _CNKI_VENUE_NOISE:
        text = re.sub(pattern, "", text)

    # 清除尾部无意义的点号、连字符或标点残留
    text = re.sub(r"[\s\.,;\-，。；]+$", "", text).strip()
    text = re.sub(r"^\s*[\.,;\-，。；]+", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()

    return _VENUE_CANONICAL_NAMES.get(text.lower(), text)



# 中文姓名被空格拆成两段的形态。CNKI 部分条目按 "given family" 输出中文作者
# （实测参考文献出现 "英哲 武, 景荣 沙, 敬豪 周"，正确形态是 "武英哲, 沙景荣,
# 周敬豪"）。**这里不做倒置还原**：该判据在结构上不可判定——误写的 "英哲 武"
# （名2+姓1）与正当写法 "欧阳 修"（复姓2+名1）字形完全一致，没有姓氏表无法
# 区分，而姓氏表既不完整（复姓、少数民族姓名、音译名）又会把正确数据改错。
# 把正确姓名改错比保留原样更有害，倒置问题须在解析源头按字段结构修复。


def normalize_author_name(author: Optional[str]) -> str:
    """规范化单个作者名：去机构角标、压缩空白、剥离首尾标点。

    只做确定性清洗，不猜测姓名顺序（见上方注释）。
    """
    text = re.sub(r"\s+", " ", str(author or "")).strip(" ,.;，。；")
    if not text:
        return ""
    # 去掉紧跟姓名的机构角标（"王猛2"、"李雷 1"）
    text = re.sub(r"\s*\d+\s*$", "", text).strip()
    return text


def normalize_author_names(authors: Optional[list]) -> list:
    """批量规范化作者名，保持顺序并去掉空项。"""
    if not authors:
        return []
    normalized = [normalize_author_name(item) for item in authors]
    return [item for item in normalized if item]
