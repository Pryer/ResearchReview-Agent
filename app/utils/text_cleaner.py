"""文本清洗工具。

提供 PDF 解析后文本、HTML 标签、多余空白等清洗功能。
"""

from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """将任意空白符（含换行、制表符）压缩为单个空格。

    Args:
        text: 原始文本。

    Returns:
        空白归一化后的文本。
    """
    if not text:
        return ""
    # 统一全角/半角空格
    text = text.replace("　", " ")
    # 压缩连续空白
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_html_tags(text: str) -> str:
    """移除 HTML/XML 标签。

    Args:
        text: 可能含标签的文本。

    Returns:
        纯文本。
    """
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text)


def remove_reference_noise(text: str) -> str:
    """移除 PDF 解析后常见的参考文献区域噪声。

    匹配以 "References" / "REFERENCES" 开头的尾部段落并截断。
    """
    if not text:
        return ""
    patterns = [
        r"\n\s*References\s*\n.*$",
        r"\n\s*REFERENCES\s*\n.*$",
        r"\n\s*Bibliography\s*\n.*$",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    """截断文本到指定字符数。

    Args:
        text: 原始文本。
        max_chars: 最大字符数。

    Returns:
        超长则截断并追加 ``...`` 后缀。
    """
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def clean_parsed_text(text: str) -> str:
    """PDF 解析文本综合清洗。

    依次执行：HTML 去标签、控制字符移除、多余空行压缩。

    Args:
        text: PDF 原始解析文本。

    Returns:
        清洗后文本。
    """
    if not text:
        return ""
    # 去标签
    text = remove_html_tags(text)
    # 移除控制字符（保留换行）
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t"
    )
    # 压缩 3+ 连续空行为 2 个换行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 行内多余空格
    lines = [normalize_whitespace(line) for line in text.split("\n")]
    return "\n".join(lines).strip()


def split_sentences(text: str, max_len: int = 512) -> list[str]:
    """按中英文句号分句，并按 max_len 合并。

    用于将长文本切分为适合 LLM 处理的片段。
    """
    if not text:
        return []
    # 按句号、问号、感叹号分句
    raw = re.split(r"(?<=[。！？!?\.])\s*", text)
    chunks: list[str] = []
    buf = ""
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) > max_len:
            if buf:
                chunks.append(buf)
            buf = s
        else:
            buf = buf + s if buf else s
    if buf:
        chunks.append(buf)
    return chunks
