"""旧 Prompt 入口的惰性兼容层。

新代码应直接从 :mod:`app.prompt` 下的具体任务模块导入。这里仅用于兼容历史
调用，并通过模块级 ``__getattr__`` 确保访问某个常量时只加载对应模块。
"""

from __future__ import annotations

from importlib import import_module
from typing import Final

_PROMPT_MODULES: Final[dict[str, str]] = {
    "RELATED_WORK_PROMPT": "app.prompt.writing.related_work",
    "INTRODUCTION_PROMPT": "app.prompt.writing.introduction",
    "INTENT_RECOGNITION_PROMPT": "app.prompt.intent",
    "RESEARCH_SEMANTIC_PARSER_PROMPT": "app.prompt.research_semantics",
    "TOPIC_DISAMBIGUATION_PROMPT": "app.prompt.topic_disambiguation",
    "SCOPE_ANSWER_RESOLUTION_PROMPT": "app.prompt.topic_disambiguation",
    "SLOT_EXTRACTION_PROMPT": "app.prompt.slots",
    "SEARCH_KEYWORD_GENERATION_PROMPT": "app.prompt.search",
    "SEARCH_KEYWORD_REFINEMENT_PROMPT": "app.prompt.search",
    "FUTURE_TREND_PROMPT": "app.prompt.future_trend",
    "SCREENING_PROTOCOL_GENERATION_PROMPT": "app.prompt.screening",
    "PAPER_CARD_EXTRACTION_PROCTION_PROMPT": "app.prompt.paper_card",
    "AXIS_INDUCTION_PROMPT": "app.prompt.taxonomy",
    "ASSIGNMENT_PROMPT": "app.prompt.taxonomy",
    "LITERATURE_REVIEW_PROMPT": "app.prompt.writing.literature_review",
    "CITATION_CHECK_PROMPT": "app.prompt.citation",
}

__all__ = tuple(_PROMPT_MODULES)


def __getattr__(name: str):
    module_name = _PROMPT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
