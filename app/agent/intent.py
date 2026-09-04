"""意图识别模块。

采用「规则优先、LLM 兜底」的两阶段策略：
1. 先通过关键词规则快速判断，高置信度直接返回。
2. 规则置信度低时调用 LLM 做精确判断。
"""

from __future__ import annotations

import re

from app.core.exceptions import IntentRecognitionError
from app.core.json_utils import parse_json_object as _parse_json_object_robust
from app.core.logger import get_logger
from app.schemas.agent_schema import IntentResult, IntentType

logger = get_logger(__name__)

# ---------- 规则表 ----------
# (关键词列表, 对应意图, 置信度)
_INTENT_RULES: list[tuple[list[str], IntentType, float]] = [
    (["综述", "文献综述", "研究现状", "survey", "review", "调研", "发展脉络"], IntentType.GENERATE_REVIEW, 0.9),
    (["相关工作", "related work"], IntentType.GENERATE_RELATED_WORK, 0.95),
    (["引言", "introduction", "研究背景", "写引言", "生成引言"], IntentType.GENERATE_INTRODUCTION, 0.95),
    (["找论文", "搜索", "检索", "推荐论文", "papers", "find papers", "找几篇", "推荐一些"], IntentType.SEARCH_PAPERS, 0.9),
    (["总结这篇", "读一下", "解析论文", "paper summary", "summarize"], IntentType.READ_PAPER, 0.85),
    (["对比", "比较", "区别", "compare", "comparison", "versus", "vs"], IntentType.COMPARE_PAPERS, 0.85),
    (["参考文献", "bibtex", "apa", "gb/t", "引用格式", "reference format"], IntentType.GENERATE_REFERENCES, 0.9),
    # 裸词“数据”会以 0.8 置信度压过裸主题兜底（如“课堂行为数据分析”
    # 被误判为数据集查找并在检索前被拦截），只保留明确的“数据集”类线索。
    (["数据集", "dataset", "benchmark"], IntentType.FIND_DATASETS, 0.8),
    (["趋势", "发展方向", "未来方向", "研究热点", "trend", "future"], IntentType.FIND_TRENDS, 0.8),
]


def recognize_intent(
    user_query: str,
    llm=None,
    *,
    conversation_role: str = "request",
    previous_intent: str | IntentType | None = None,
    original_query: str | None = None,
) -> IntentResult:
    """识别用户意图。

    先走规则，低置信度时用 LLM 兜底。

    Args:
        user_query: 用户原始请求。
        llm: 可选的 LLM 客户端。
        conversation_role: ``request`` / ``clarification_answer`` /
            ``working_query``。由会话编排层显式标注文本的作用。
        previous_intent: 当前会话已确认的顶层意图。
        original_query: 未附加澄清文本的原始请求。

    Returns:
        意图识别结果。
    """
    role = str(conversation_role or "request").strip().lower()
    if role not in {"request", "clarification_answer", "working_query"}:
        role = "request"
    prior = _normalize_intent(previous_intent)

    # 澄清回答与内部工作查询不是新的顶层请求。这一判断依赖
    # 编排层的轮次角色，而不是某个学科的关键词。
    if role in {"clarification_answer", "working_query"} and prior:
        return IntentResult(
            intent=prior,
            confidence=1.0,
            reason=f"根据对话角色 {role} 继承已确认的顶层意图",
        )

    classification_query = (
        str(original_query or "").strip()
        if role == "working_query" and str(original_query or "").strip()
        else user_query
    )

    # 阶段 1：规则
    result = rule_based_intent_recognition(classification_query)
    if result.confidence >= 0.7:
        logger.info("Rule-based intent: %s (%.2f)", result.intent, result.confidence)
        return result

    # 阶段 2：LLM
    if llm is not None:
        try:
            result = llm_based_intent_recognition(
                user_query,
                llm,
                conversation_role=role,
                previous_intent=prior,
                original_query=original_query or classification_query,
            )
            logger.info("LLM-based intent: %s (%.2f)", result.intent, result.confidence)
            return result
        except Exception as e:
            logger.warning("LLM intent recognition failed: %s, fallback to rule", e)

    logger.info("Fallback to rule intent: %s (%.2f)", result.intent, result.confidence)
    return result


def rule_based_intent_recognition(user_query: str) -> IntentResult:
    """基于关键词规则判断意图。

    遍历规则表，命中关键词最多的意图获胜。
    """
    if not user_query.strip():
        return IntentResult(
            intent=IntentType.GENERAL_QA.value,
            confidence=0.3,
            reason="空请求",
        )

    query_lower = user_query.lower()
    best_intent = IntentType.GENERAL_QA
    best_score = 0.0
    best_reason = "无明确匹配规则"

    for keywords, intent, base_conf in _INTENT_RULES:
        matched = [kw for kw in keywords if kw.lower() in query_lower]
        if matched:
            # 匹配到关键词即给 base_conf，多关键词匹配再加分
            score = base_conf + 0.1 * (len(matched) - 1)
            score = min(score, 1.0)
            if score > best_score:
                best_score = score
                best_intent = intent
                best_reason = f"命中关键词: {matched}"

    # 本项目默认服务于学术调研。用户只输入一个边界明确的研究主题时，
    # 不能因为缺少“调研/综述”等动作词就降级为 general_qa。
    if best_score == 0.0 and _looks_like_bare_research_topic(user_query):
        best_intent = IntentType.GENERATE_REVIEW
        best_score = 0.72
        best_reason = "未包含动作词，按裸学术主题默认执行文献调研"

    return IntentResult(
        intent=best_intent.value,
        confidence=round(best_score, 2),
        reason=best_reason,
    )


def _looks_like_bare_research_topic(user_query: str) -> bool:
    """保守判断输入是否是一个裸学术主题，而不是问句或闲聊。"""
    query = re.sub(r"\s+", " ", str(user_query or "")).strip()
    if not query or len(query) > 120 or "\n" in query:
        return False

    lower = query.lower()
    if re.search(r"[？?！!]", query):
        return False
    if lower in {
        "hello", "hi", "hey", "thanks", "thank you",
        "你好", "您好", "谢谢", "在吗", "测试",
    }:
        return False
    if re.match(
        r"^(什么|如何|为什么|为何|怎么|怎样|谁|哪里|请问|能否|可以|有什么|有哪些|有没有|是否|哪(?:些|个|种)?|what\b|why\b|how\b|who\b|where\b|when\b|can\b|could\b|would\b)",
        lower,
    ):
        return False

    chinese = bool(re.search(r"[\u4e00-\u9fff]", query))
    english_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+.-]*", query)
    if chinese:
        # 中文没有空格分词，因此只利用句法外形做保守判断：排除对话句、
        # 问句和常见语气句，剩余的短名词性文本在本学术助手中视为裸主题。
        # 不枚举学科、对象、方法或应用场景词表。
        compact = re.sub(r"\s+", "", query)
        if re.match(r"^(?:我|你|您|他|她|它|我们|你们|他们|这|那|咱们)", compact):
            return False
        if re.search(r"(?:吗|呢|吧|呀|啊|啦|了)$", compact):
            return False
        return 2 <= len(compact) <= 50

    if not english_tokens or len(english_tokens) > 12:
        return False
    if len(english_tokens) >= 2:
        return True
    token = english_tokens[0]
    # 单个常用缩写或专名也可作为研究主题，如 RAG、BERT、Transformer。
    return bool(re.fullmatch(r"[A-Z][A-Z0-9-]{1,11}", token) or token[:1].isupper())


def llm_based_intent_recognition(
    user_query: str,
    llm,
    *,
    conversation_role: str = "request",
    previous_intent: str | IntentType | None = None,
    original_query: str | None = None,
) -> IntentResult:
    """调用 LLM 输出 JSON 格式意图结果。

    Args:
        user_query: 用户请求。
        llm: LLM 客户端（需实现 ``complete(prompt) -> str``）。

    Returns:
        意图识别结果。
    """
    from app.prompt.intent import INTENT_RECOGNITION_PROMPT

    prior = _normalize_intent(previous_intent)
    # 仅保留 ``{user_query}`` 为 format 占位，保持旧调用方可以直接对 prompt 进行 format。
    # 会话元数据使用不会与 format 冲突的标记后处理。
    prompt = INTENT_RECOGNITION_PROMPT.format(user_query=user_query)
    prompt = (
        prompt.replace("__CONVERSATION_ROLE__", str(conversation_role or "request"))
        .replace("__PREVIOUS_INTENT__", prior or "none")
        .replace("__ORIGINAL_QUERY__", str(original_query or user_query))
    )
    try:
        response = llm.complete(prompt, response_format="json")
        data = _safe_parse_json(response)
        intent = _normalize_intent(data.get("intent")) or IntentType.GENERAL_QA.value
        return IntentResult(
            intent=intent,
            confidence=float(data.get("confidence", 0.5)),
            reason=data.get("reason", ""),
        )
    except Exception as e:
        raise IntentRecognitionError(f"LLM 意图识别失败: {e}")


def _normalize_intent(value: str | IntentType | None) -> str | None:
    """把外部会话状态和模型输出收敛到受支持的意图集。"""
    raw = value.value if isinstance(value, IntentType) else str(value or "").strip()
    valid = {item.value for item in IntentType}
    return raw if raw in valid else None



from app.core.json_utils import parse_json_object as _safe_parse_json  # noqa: E402
