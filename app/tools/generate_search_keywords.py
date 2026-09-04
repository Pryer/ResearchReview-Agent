"""
学术文献检索关键词生成工具。

根据用户输入的自然语言研究主题，由 LLM 动态生成适合论文数据库检索的
中英文关键词。

设计目标：
1. 不依赖硬编码领域词表；
2. 同时覆盖中文和英文常见学术表达；
3. 允许有限度的"安全放宽"，提升论文召回率；
4. 严格限制语义漂移，不把相关研究方向误当成检索关键词；
5. 输出可直接供 CNKI、Google Scholar、Semantic Scholar、
   Crossref、OpenAlex 等检索模块使用。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_GENERATE_KEYWORDS_PROMPT = """
你是学术文献检索系统中的"检索关键词生成器"。

你的任务是：

根据用户输入的研究主题，生成一组适合用于检索学术论文的
中文关键词和英文关键词。

你的目标不是穷举相关概念，也不是扩展研究方向，
而是找出学术论文中真实、常见、能够用于检索该主题文献的表达方式。


用户研究主题：
{topic}


====================
一、核心原则
====================

生成的关键词必须始终围绕用户研究主题中的"核心研究问题"。

你可以提高检索召回率，但不能改变研究问题本身。

关键词应该来源于以下三种情况：

A. 完整主题表达
B. 安全的简化表达
C. 常见的同义或书写变体


====================
二、允许生成的关键词
====================

【1. 完整学术表达】

首先生成与用户研究主题直接对应的标准学术术语。

完整表达必须保留用户主题中的核心对象、任务和限定条件，并使用对应语言中
真实常见的学术术语。


【2. 安全的简化表达】

允许删除"非核心限定词"，前提是删除以后仍然检索的是
同一个核心研究任务，而不是另一个研究方向。

模型必须先识别主题中的核心对象、核心任务和不可删除的研究限定；broader
表达只能删除修饰性成分，不能删除任何会改变研究设定的核心限定。


【3. 常见同义表达】

如果学术文献中确实存在广泛使用的同义表达，可以生成。

只有在学术语境中确实常见时才能生成同义表达。

禁止为了增加关键词数量而创造生僻表达。

中文主题的核心修饰词若存在中文学术界并用的同义形式
（例如“少样本”与“小样本”），必须为每种形式至少生成一个
关键词，不得只保留其中一种。


【4. 常见书写变体】

如果英文术语存在常见的连字符或空格书写形式，可以同时保留。

可以保留常见的连字符、空格、缩写或全称变体，但不要生成明显不规范或
极少出现的拼写。

英文核心术语若同时存在连字符与空格两种常用书写形式
（例如 few-shot 与 few shot、zero-shot 与 zero shot），
必须在主形式之外再生成一个变体形式。


====================
三、严格禁止的扩展
====================

以下情况不得生成。


【1. 不得改变核心学习设定】

不得把相邻但不同的学习设定、任务定义、数据条件或监督方式当作表达变体。


【2. 不得加入用户没有提出的方法】

除非方法本身明确出现在用户主题中，否则不得自动加入具体模型、架构、
训练策略或算法名称。


【3. 不得加入其他研究方向】

不得因为领域相关就加入用户未提出的迁移方向、模态方向、监督范式或应用场景。


【4. 不得把相关概念当成同义词】

概念相关不等于同义；模型必须根据当前主题逐项判断，不能默认相邻概念可互换。


【5. 不得过度泛化】

任何删除核心限制条件、把具体研究问题退化为上位领域名称的表达都不得输出。


====================
四、中英文规则
====================

中文关键词必须符合中文学术论文中的常见表达。

英文关键词必须符合英文学术论文标题、摘要和关键词中的常见表达。

不要机械逐字翻译。

应该根据两个学术社区实际常用的表达分别生成关键词。

因此：

中文关键词和英文关键词不要求严格一一对应。


====================
五、数量边界
====================

中文关键词最多 6 个。
英文关键词最多 6 个。

不要为了凑数量生成低质量关键词。

如果只有 2～4 个可靠表达，就只返回这些。


====================
六、关键词分类
====================

每个关键词必须标记来源类型：

exact
    与研究主题直接对应的完整学术表达。

broader
    删除非核心限定词后的安全简化表达。
    必须仍保持核心研究问题不变。

variant
    常见同义表达、术语变体或书写变体。


====================
七、输出格式
====================

只返回严格 JSON：

{{
  "zh": [
    {{
      "keyword": "中文关键词",
      "type": "exact"
    }}
  ],
  "en": [
    {{
      "keyword": "English keyword",
      "type": "exact"
    }}
  ]
}}

type 只能是：

exact
broader
variant

不要输出解释。
不要输出 Markdown。
不要输出 AND / OR。
不要添加引号。
不要生成完整布尔检索式。
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_search_keywords(
    topic: str,
    llm: Any | None = None,
    *,
    max_zh: int = 6,
    max_en: int = 6,
    include_metadata: bool = False,
) -> dict[str, list]:
    """
    根据研究主题生成适合学术论文检索的中英文关键词。

    Args:
        topic:
            用户输入的研究主题。

        llm:
            LLM 客户端。
            需要实现 complete(prompt, ...) 方法。

        max_zh:
            最多返回多少个中文关键词。
            默认 6。

        max_en:
            最多返回多少个英文关键词。
            默认 6。

        include_metadata:
            False:
                {
                    "zh": ["中文检索表达", ...],
                    "en": ["English search expression", ...]
                }

            True:
                {
                    "zh": [
                        {
                            "keyword": "中文检索表达",
                            "type": "exact"
                        }
                    ],
                    "en": [...]
                }

    Returns:
        中英文检索关键词。

    Notes:
        本工具只负责生成"基础检索关键词"。

        不负责：
        - 生成 Boolean Query
        - 添加 AND / OR
        - 数据库字段限制
        - 年份限制
        - 作者过滤
        - 方法词扩展
        - citation expansion
        - query rewriting

        这些应由下游检索模块单独处理。
    """

    topic = _normalize_topic(topic)

    if not topic:
        return {
            "zh": [],
            "en": [],
        }

    if llm is None:
        logger.warning(
            "generate_search_keywords called without LLM client"
        )
        return {
            "zh": [],
            "en": [],
        }

    # 防止异常参数导致无限扩展
    max_zh = _clamp(max_zh, 1, 10)
    max_en = _clamp(max_en, 1, 10)

    prompt = _GENERATE_KEYWORDS_PROMPT.format(
        topic=topic,
    )

    try:
        response = llm.complete(
            prompt,
            temperature=0.0,
            retry_empty=True,
            thinking_enabled=False,
            operation="generate_search_keywords",
        )

        data = _safe_parse_json(response)

        zh_items = _parse_keyword_items(
            data.get("zh"),
            language="zh",
            limit=max_zh,
        )

        en_items = _parse_keyword_items(
            data.get("en"),
            language="en",
            limit=max_en,
        )

        if not zh_items and not en_items:
            logger.warning(
                "Keyword generator returned no usable keywords "
                "for topic: %s",
                topic,
            )

            return {
                "zh": [],
                "en": [],
            }

        logger.debug(
            "Generated search keywords for %r: zh=%d, en=%d",
            topic,
            len(zh_items),
            len(en_items),
        )

        if include_metadata:
            return {
                "zh": zh_items,
                "en": en_items,
            }

        return {
            "zh": [
                item["keyword"]
                for item in zh_items
            ],
            "en": [
                item["keyword"]
                for item in en_items
            ],
        }

    except Exception as exc:
        logger.warning(
            "Search keyword generation failed for %r: %s",
            topic,
            exc,
        )

        return {
            "zh": [],
            "en": [],
        }


# ---------------------------------------------------------------------------
# Parsing / Validation
# ---------------------------------------------------------------------------

_ALLOWED_TYPES = {
    "exact",
    "broader",
    "variant",
}


def _parse_keyword_items(
    raw: Any,
    *,
    language: str,
    limit: int,
) -> list[dict[str, str]]:
    """
    清洗并验证 LLM 返回的关键词。

    这里只做"结构边界"和明显异常过滤，
    不使用硬编码领域词表判断语义。
    """

    if not isinstance(raw, list):
        return []

    result: list[dict[str, str]] = []
    seen: set[str] = set()

    for obj in raw:
        if len(result) >= limit:
            break

        if not isinstance(obj, dict):
            continue

        keyword = obj.get("keyword")
        keyword_type = obj.get("type")

        if not isinstance(keyword, str):
            continue

        if not isinstance(keyword_type, str):
            continue

        keyword = _normalize_keyword(keyword)
        keyword_type = keyword_type.strip().lower()

        if not keyword:
            continue

        if keyword_type not in _ALLOWED_TYPES:
            continue

        # ---------------------------------------------------------------
        # 基础长度边界
        # ---------------------------------------------------------------

        # 一个词太长通常意味着模型生成了句子，而不是检索短语
        if len(keyword) > 120:
            continue

        # ---------------------------------------------------------------
        # 禁止 Boolean Query / 搜索语法泄漏
        # ---------------------------------------------------------------

        if _contains_query_syntax(keyword):
            continue

        # ---------------------------------------------------------------
        # 语言边界
        # ---------------------------------------------------------------

        if language == "zh":
            if not _looks_like_chinese_keyword(keyword):
                continue

        elif language == "en":
            if not _looks_like_english_keyword(keyword):
                continue

        # ---------------------------------------------------------------
        # 去重
        # ---------------------------------------------------------------

        normalized_key = _dedup_key(
            keyword,
            language=language,
        )

        if normalized_key in seen:
            continue

        seen.add(normalized_key)

        result.append(
            {
                "keyword": keyword,
                "type": keyword_type,
            }
        )

    return result


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_topic(topic: Any) -> str:
    """清洗用户研究主题。"""

    if not isinstance(topic, str):
        return ""

    topic = " ".join(topic.strip().split())

    # 防止异常超长输入直接进入关键词生成 Prompt
    if len(topic) > 1000:
        topic = topic[:1000]

    return topic


def _normalize_keyword(keyword: str) -> str:
    """规范关键词中的空白字符。"""

    return " ".join(keyword.strip().split())


def _dedup_key(
    keyword: str,
    *,
    language: str,
) -> str:
    """
    用于去重的标准形式。

    注意：
    不把 few-shot 和 few shot 合并，
    因为它们是用户希望保留的真实检索书写变体。
    """

    keyword = keyword.strip().casefold()

    if language == "zh":
        # 中文中普通空格没有检索意义
        keyword = keyword.replace(" ", "")

    return keyword


# ---------------------------------------------------------------------------
# Boundary Checks
# ---------------------------------------------------------------------------

def _contains_query_syntax(keyword: str) -> bool:
    """
    防止生成器越权生成完整检索表达式。

    关键词生成和 Boolean Query 构造必须分离。
    """

    upper = f" {keyword.upper()} "

    forbidden_tokens = (
        " AND ",
        " OR ",
        " NOT ",
    )

    if any(token in upper for token in forbidden_tokens):
        return True

    # 常见数据库字段查询语法
    forbidden_patterns = (
        "TITLE:",
        "ABSTRACT:",
        "AUTHOR:",
        "KEYWORD:",
        "PUBYEAR:",
        "SITE:",
    )

    upper_keyword = keyword.upper()

    return any(
        pattern in upper_keyword
        for pattern in forbidden_patterns
    )


def _looks_like_chinese_keyword(keyword: str) -> bool:
    """
    中文支线至少应包含中文字符。

    允许包含：
    CLIP
    GNN
    3D
    Transformer
    等嵌入式英文术语。
    """

    return any(
        "一" <= char <= "鿿"
        for char in keyword
    )


def _looks_like_english_keyword(keyword: str) -> bool:
    """
    英文关键词需要包含拉丁字母。

    同时避免模型把完整中文表达放进英文支线。
    """

    has_latin = any(
        ("a" <= char.lower() <= "z")
        for char in keyword
    )

    if not has_latin:
        return False

    chinese_chars = sum(
        1
        for char in keyword
        if "一" <= char <= "鿿"
    )

    # 英文关键词里偶尔可能出现数据集中文名，
    # 但大量中文显然是语言支线错误。
    if chinese_chars > 2:
        return False

    return True


def _clamp(
    value: int,
    minimum: int,
    maximum: int,
) -> int:
    """限制整数参数范围。"""

    try:
        value = int(value)
    except (TypeError, ValueError):
        return minimum

    return max(
        minimum,
        min(value, maximum),
    )


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

from app.core.json_utils import parse_json_object as _safe_parse_json  # noqa: E402
