"""证据约束的文献分类工具。

LLM 根据当前研究请求与论文样本动态归纳分类轴；不可用时，只按 PaperCard
实际存在的结构化字段或样本内区分词回退，不维护领域方法词表。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger
from app.schemas.taxonomy_schema import (
    DynamicTaxonomy,
    PaperThemeAssignment,
    ResearchTheme,
    TaxonomyStatus,
    TaxonomyValidationResult,
)
from app.tools.paper_rerank import RULE_SCREENED_RESERVE

logger = get_logger(__name__)

TOKEN_STOPWORDS = {
    "study", "research", "analysis", "method", "methods", "paper",
    # 学术写作通用词：任何领域都不构成区分点；曾因此把拆分子路线的
    # 兜底名选成“度量学习与特征对齐：learning/approach”。
    "learning", "approach", "approaches", "model", "models", "based",
    "framework", "novel", "task", "tasks", "using", "via", "towards",
    "survey", "review", "deep", "neural", "we", "our",
    "propose", "proposed", "proposes",
    "研究", "分析", "方法", "论文",
}

_GENERIC_CLUSTER_LABELS = {
    "", "unknown", "unavailable", "n/a", "na", "none", "null",
    "未明确报告", "未命名研究方向", "real", "true", "真实",
}

_FIELD_LABELS = {
    "study_design": "研究设计",
    "publication_type": "发表类型",
    "data_modalities": "数据模态",
    "dataset": "数据集",
    "research_problem": "研究问题",
    "method": "方法",
}

# ---------- 无领域预设的确定性回退分类 ----------
# publication_type 已被移除：按"期刊论文/会议论文"分类不是研究路线
_FALLBACK_AXIS_FIELDS = (
    "study_design", "data_modalities",
    "dataset", "research_problem", "method",
)


def _axis_value(card: Dict[str, Any], field: str) -> str:
    value = card.get(field)
    if isinstance(value, (list, tuple, set)):
        value = "、".join(str(item).strip() for item in value if str(item).strip())
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if text.lower() in {"", "unknown", "none", "null"}:
        return ""
    return text[:100]


def _fallback_cluster_label(field: str) -> str:
    label = _FIELD_LABELS.get(field, field or "证据")
    return f"{label}待补充证据"


def _normalize_cluster_label(field: str, value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;，。:：")
    if not cleaned:
        return _fallback_cluster_label(field)

    lowered = cleaned.casefold()
    if lowered in _GENERIC_CLUSTER_LABELS:
        return _fallback_cluster_label(field)

    if field == "study_design":
        design_aliases = {
            "experiment": "实验研究",
            "experiments": "实验研究",
            "randomized_experiment": "随机实验",
            "survey": "调查研究",
            "case_study": "案例研究",
            "observational": "观察研究",
        }
        if lowered in design_aliases:
            return design_aliases[lowered]
    if field == "publication_type":
        type_aliases = {
            "journal_article": "期刊论文",
            "conference_paper": "会议论文",
            "conference_short_paper": "会议短文",
            "preprint": "预印本",
            "systematic_review": "系统综述",
            "meta_analysis": "元分析",
        }
        if lowered in type_aliases:
            return type_aliases[lowered]

    if lowered in {
        "image", "images", "video", "videos", "text", "pose", "skeleton",
        "rgb", "visual", "experiment", "real", "true", "false", "unknown",
    }:
        return _fallback_cluster_label(field)

    if len(cleaned) <= 2 and not re.search(r"[\u4e00-\u9fff]{2,}", cleaned):
        return _fallback_cluster_label(field)
    return cleaned


def _is_low_signal_route_name(name: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(name or "")).strip()
    if not cleaned:
        return True
    lowered = cleaned.casefold()
    if lowered in _GENERIC_CLUSTER_LABELS:
        return True
    if lowered in {
        "image", "images", "video", "videos", "text", "audio", "speech", "sensor", "sensors",
        "signal", "signals", "pose", "skeleton", "rgb", "visual", "experiment", "dataset",
        "method", "methods", "task", "topic", "cluster", "route", "dna", "rna", "gene", "protein",
        "tabular", "molecule",
        # 禁止按出版类型作为研究路线名称
        "期刊论文", "会议论文", "会议短文", "预印本", "学位论文",
        "journal article", "conference paper", "preprint", "thesis",
        "journal_article", "conference_paper", "conference_short_paper",
    }:
        return True
    if re.fullmatch(r"[a-z_\- ]{1,18}", lowered) and len(lowered.split()) <= 2:
        return True
    if len(cleaned) <= 2 and not re.search(r"[\u4e00-\u9fff]{2,}", cleaned):
        return True
    if re.search(r"\u5f85\u8865\u5145\u8bc1\u636e|\u672a\u547d\u540d|\u672a\u77e5|unknown|other|misc", lowered):
        return True
    return False


def semantic_route_label(
    cards: List[Dict[str, Any]],
    raw_name: str,
    *,
    fallback_field: str = "research_problem",
) -> str:
    """纯数据驱动路线命名：从当前聚类论文的共同词项生成名称，不维护领域方法词表。"""
    cleaned = re.sub(r"\s+", " ", str(raw_name or "")).strip()
    if cleaned and not _is_low_signal_route_name(cleaned):
        return cleaned

    # 提取每篇论文的词项，找跨论文共同词
    per_paper_tokens: list[set[str]] = []
    for card in cards:
        text = " ".join(
            str(card.get(field) or "")
            for field in ("title", "research_problem", "method")
        )
        per_paper_tokens.append(taxonomy_tokens(text))

    # 统计词项在多少篇论文中出现（每篇只计一次）
    paper_counts: dict[str, int] = {}
    for tokens in per_paper_tokens:
        for token in set(tokens):
            paper_counts[token] = paper_counts.get(token, 0) + 1

    # 至少在 2 篇论文中出现的词才是"共同词"
    shared = {
        token for token, count in paper_counts.items()
        if count >= 2 and token not in TOKEN_STOPWORDS
    }

    if shared:
        # 优先长词（3-4字中文词 > 2字词 > 英文词）
        def _term_priority(token: str) -> tuple[int, int, int]:
            is_cjk = 1 if re.search(r"[一-鿿]", token) else 0
            return (-is_cjk, -len(token), -paper_counts.get(token, 0))

        ranked = sorted(shared, key=_term_priority)
        top = ranked[:2]
        if len(top) >= 2:
            return f"{top[0]}与{top[1]}研究"
        if len(top) == 1:
            return f"{top[0]}研究"

    # 无共同词 → 回退到 fallback_field 的全部词项
    fallback_text = " ".join(
        str(card.get(fallback_field) or "")
        for card in cards
        if str(card.get(fallback_field) or "").strip()
    ).strip()
    if fallback_text:
        fallback_tokens = taxonomy_tokens(fallback_text)
        salient = sorted(
            [t for t in fallback_tokens if t not in TOKEN_STOPWORDS and len(t) > 1],
            key=lambda t: -len(t),
        )[:2]
        if len(salient) >= 2:
            return f"{salient[0]}与{salient[1]}研究"
        if len(salient) == 1:
            return f"{salient[0]}研究"

    return _fallback_cluster_label(fallback_field)


def _select_fallback_axis(cards: List[Dict[str, Any]]) -> str | None:
    """从当前卡片字段分布选择覆盖充分且不过度碎片化的分类轴。"""
    if len(cards) < 2:
        return None
    best: tuple[float, str] | None = None
    for field in _FALLBACK_AXIS_FIELDS:
        values = [_axis_value(card, field) for card in cards]
        present = [value for value in values if value]
        unique = set(present)
        if len(unique) < 2 or len(unique) > min(8, max(2, len(cards) - 1)):
            continue
        coverage = len(present) / len(cards)
        largest = max(present.count(value) for value in unique) / len(present)
        score = coverage + (1.0 - largest)
        if best is None or score > best[0]:
            best = (score, field)
    return best[1] if best else None


def _cluster_by_evidence_axis(
    cards: List[Dict[str, Any]],
    *,
    forced_axis: str | None = None,
) -> tuple[str, List[Dict[str, Any]]]:
    axis = forced_axis or _select_fallback_axis(cards)
    if not axis:
        return "available_evidence", [{
            "cluster_name": "当前研究证据",
            "description": "现有结构化字段不足以形成可靠的细分轴",
            "paper_ids": [str(card.get("paper_id") or "") for card in cards if card.get("paper_id")],
            "representative_papers": [str(card.get("paper_id") or "") for card in cards[:3] if card.get("paper_id")],
        }]

    buckets: Dict[str, List[str]] = {}
    for card in cards:
        value = _normalize_cluster_label(axis, _axis_value(card, axis))
        paper_id = str(card.get("paper_id") or "")
        if paper_id:
            buckets.setdefault(value, []).append(paper_id)
    clusters = [
        {
            "cluster_name": value,
            "description": f"依据论文显式字段 {axis} 归纳的研究分组：{value}",
            "paper_ids": paper_ids,
            "representative_papers": paper_ids[:3],
        }
        for value, paper_ids in buckets.items()
    ]
    return axis, clusters


def _cluster_by_salient_terms(
    cards: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    """从当前标题/问题/方法证据中寻找可复现的二分词项。

    该回退只使用输入数据中实际出现的词，不维护任何领域方法表。候选词必须
    同时出现在足够多、但不是全部论文中，避免按论文编号或孤立词制造碎片类。
    """
    if len(cards) < 4:
        return "", []
    token_rows: list[set[str]] = []
    frequencies: Dict[str, int] = {}
    for card in cards:
        tokens = taxonomy_tokens(" ".join(
            str(card.get(field) or "")
            for field in ("title", "research_problem", "method")
        ))
        token_rows.append(tokens)
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1

    minimum = max(2, round(len(cards) * 0.2))
    maximum = len(cards) - minimum
    candidates = [
        token for token, count in frequencies.items()
        if minimum <= count <= maximum
    ]
    if not candidates:
        return "", []
    pivot = min(
        candidates,
        key=lambda token: (abs(frequencies[token] / len(cards) - 0.5), token),
    )
    present = [
        str(card.get("paper_id") or "")
        for card, tokens in zip(cards, token_rows)
        if pivot in tokens and card.get("paper_id")
    ]
    absent = [
        str(card.get("paper_id") or "")
        for card, tokens in zip(cards, token_rows)
        if pivot not in tokens and card.get("paper_id")
    ]
    if not present or not absent:
        return "", []
    axis = f"salient_term:{pivot}"
    return axis, [
        {
            "cluster_name": f"包含“{pivot}”的研究路线",
            "description": f"标题、研究问题或方法字段明确包含区分词“{pivot}”",
            "paper_ids": present,
            "representative_papers": present[:3],
        },
        {
            "cluster_name": f"不包含“{pivot}”的研究路线",
            "description": f"当前结构化证据未出现区分词“{pivot}”",
            "paper_ids": absent,
            "representative_papers": absent[:3],
        },
    ]


def cluster_papers_by_method(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 PaperCard 显式报告的方法分组，不使用领域方法词表。"""
    return _cluster_by_evidence_axis(cards, forced_axis="method")[1]


def cluster_papers_by_task(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按卡片中显式抽取的研究问题分组，不用数据集冒充研究任务。"""
    task_clusters: Dict[str, Dict[str, Any]] = {}
    for card in cards:
        task = str(card.get("research_problem") or "").strip() or "Unknown"
        normalized = re.sub(r"\W+", "", task.casefold())[:160] or "unknown"
        if normalized not in task_clusters:
            task_clusters[normalized] = {
                "cluster_name": f"Task: {task}",
                "description": f"研究问题：{task}" if task != "Unknown" else "研究问题尚未明确抽取",
                "paper_ids": [],
                "representative_papers": [],
            }
        task_clusters[normalized]["paper_ids"].append(card.get("paper_id", ""))

    for c in task_clusters.values():
        c["representative_papers"] = c["paper_ids"][:3]
    return list(task_clusters.values())


def cluster_papers_by_dataset(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按卡片中显式抽取的数据集或样本来源分组。"""
    clusters: Dict[str, Dict[str, Any]] = {}
    for card in cards:
        dataset = str(card.get("dataset") or "").strip() or "Unknown"
        if dataset not in clusters:
            clusters[dataset] = {
                "cluster_name": f"Dataset: {dataset}",
                "description": (
                    f"使用 {dataset} 数据或样本的研究"
                    if dataset != "Unknown" else "数据或样本来源尚未明确抽取"
                ),
                "paper_ids": [],
                "representative_papers": [],
            }
        clusters[dataset]["paper_ids"].append(card.get("paper_id", ""))
    for cluster in clusters.values():
        cluster["representative_papers"] = cluster["paper_ids"][:3]
    return list(clusters.values())


def cluster_papers_by_year(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按年份分类。"""
    year_groups: Dict[int, List[str]] = {}
    for card in cards:
        year = card.get("year") or 0
        year_groups.setdefault(year, []).append(card.get("paper_id", ""))

    clusters = []
    for year in sorted(year_groups.keys(), reverse=True):
        clusters.append({
            "cluster_name": f"{year}" if year else "Unknown",
            "description": f"{year} 年发表的论文" if year else "年份未知",
            "paper_ids": year_groups[year],
            "representative_papers": year_groups[year][:3],
        })
    return clusters


def generate_cluster_summary(clusters: List[Dict[str, Any]], llm=None, topic: str = "") -> str:
    """总结每类文献特点；提供 LLM 时根据当前聚类动态综合。"""
    if not clusters:
        return ""

    if llm is not None:
        prompt = f"""你是学术聚类摘要器。请只依据给定聚类结构，为每个类别生成简洁中文摘要。
不得补充外部事实、预设学科维度或逐篇虚构论文内容。保留类别名称，并说明该类别
由输入中哪些共同研究问题或证据特征形成。

研究主题：{topic}
聚类结构：{json.dumps(clusters, ensure_ascii=False)}

直接输出 Markdown，不要输出分析过程。"""
        try:
            response = llm.complete(
                prompt,
                temperature=0.0,
                operation="generate_cluster_summary",
            )
            if str(response or "").strip():
                return str(response).strip()
        except Exception as exc:
            logger.warning("LLM cluster summary failed, using deterministic fallback: %s", exc)

    # 规则生成摘要
    summaries = [f"# {topic}\n" if topic else ""]
    for c in clusters:
        name = c.get("cluster_name", "")
        paper_ids = c.get("paper_ids", [])
        desc = c.get("description", "")
        summaries.append(f"\n## {name}（{len(paper_ids)} 篇）\n{desc}")
    return "\n".join(summaries)


def cluster_papers(
    cards: List[Dict[str, Any]],
    llm=None,
    topic: str = "",
    scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """编译并验证动态分类体系，始终返回同一种结构化结果。"""
    taxonomy, validation = compile_dynamic_taxonomy(cards, llm=llm, topic=topic, scope=scope)
    clusters = taxonomy_to_clusters(taxonomy)
    return {
        "clusters": clusters,
        "dynamic_taxonomy": taxonomy.model_dump(mode="json"),
        "taxonomy_validation": validation.model_dump(mode="json"),
    }


# ---------- 证据角色推断 ----------
_SURVEY_TITLE_KEYWORDS = [
    "survey", "review", "comprehensive", "systematic", "bibliometric",
    "meta-analysis", "meta analysis", "综述", "系统评价", "元分析", "文献计量",
]
_BENCHMARK_TITLE_KEYWORDS = [
    "dataset", "benchmark", "corpus", "数据集", "基准", "语料",
]


def infer_evidence_role(paper: Dict[str, Any]) -> str:
    """推断单篇论文在综述论证中的证据角色。

    用于约束写作 LLM 的引用权限：survey 才能支撑宏观断言，
    method/benchmark/application 只能支撑自身边界内的陈述。

    Returns:
        "survey" | "method" | "benchmark" | "application"
    """
    title = str(paper.get("title") or "").lower()
    if any(kw in title for kw in _SURVEY_TITLE_KEYWORDS):
        return "survey"
    if any(kw in title for kw in _BENCHMARK_TITLE_KEYWORDS):
        return "benchmark"
    screening_decision = str(paper.get("_screening_decision") or "")
    if screening_decision in ("uncertain", RULE_SCREENED_RESERVE):
        return "application"
    return "method"


def enrich_cards_with_evidence_roles(cards: List[Dict[str, Any]]) -> None:
    """为每张卡片写入 ``evidence_role``（已有值保持不变）。

    就地修改传入列表中的卡片 dict，使后续写作阶段（含实时
    ``write_deliverable`` 路径与旧版 ``generate_review`` 路径）
    都能读取统一的证据角色。
    """
    for card in cards:
        if str(card.get("evidence_role") or "") not in (
            "survey", "method", "benchmark", "application"
        ):
            card["evidence_role"] = infer_evidence_role(card)


def compile_dynamic_taxonomy(
    cards: List[Dict[str, Any]],
    llm=None,
    topic: str = "",
    scope: Optional[Dict[str, Any]] = None,
) -> tuple[DynamicTaxonomy, TaxonomyValidationResult]:
    """执行有界分类闭环：LLM 抽样定轴 -> 确定性全量分配 -> 验证。

    LLM 只负责一次高价值的主题归纳。逐批调用 LLM 分配几十篇论文既慢，
    又会令任一批次超时拖垮整个分类节点；全量归属改由可复现的证据词项完成。
    """
    scope = scope or {}
    # 证据角色是写作阶段引用权限的约束输入，先于分类写入卡片
    enrich_cards_with_evidence_roles(cards)
    if llm and cards:
        try:
            from app.core.config import get_settings
            from app.prompt.taxonomy import AXIS_INDUCTION_PROMPT
            from app.tools.taxonomy_strategy import TaxonomyStrategyResolver

            settings = get_settings()
            # 阶段 0：均匀取样，供定轴 + 领域检测共用
            induction_cards = _representative_cards(
                cards, settings.taxonomy_induction_limit
            )
            research_mode = scope.get("research_mode", "mixed")
            # 研究模式来自上游 LLM 语义规划；分类器只根据当前论文样本归纳
            # 分类轴，不再通过领域词表把主题强制改写为某个预设技术领域。
            strategy = TaxonomyStrategyResolver.resolve(research_mode)

            # 阶段 1：定轴 (Axis Induction)
            induction_payload = _taxonomy_payload(induction_cards, topic, scope)
            induction_payload["total_paper_count"] = len(cards)
            
            axis_raw = llm.complete(
                AXIS_INDUCTION_PROMPT.format(
                    strategy_instruction=strategy["axis_instruction"],
                    strategy_examples="\\n".join(f"- {e}" for e in strategy["example_axes"]),
                    paper_cards_json=json.dumps(induction_payload, ensure_ascii=False),
                ),
                response_format="text",
                temperature=0.0,
                timeout=settings.taxonomy_induction_timeout,
                retry_empty=False,
                operation="taxonomy_axis_induction",
            )
            axis_data = _parse_json(axis_raw)
            taxonomy = _normalize_llm_taxonomy(axis_data, cards, topic, scope)
            if not taxonomy.themes:
                raise ValueError("分类定轴未返回可用主题")
            taxonomy = _complete_sampled_assignments(taxonomy, cards)
            taxonomy = _normalize_taxonomy_theme_names(taxonomy, cards)
            validation = validate_taxonomy(taxonomy, cards)
            if validation.concentration_requires_split:
                taxonomy, validation = _refine_dominant_theme(
                    taxonomy, validation, cards, topic, scope
                )
            if taxonomy.themes:
                logger.info(
                    "Dynamic taxonomy compiled: papers=%d themes=%d coverage=%.2f status=%s",
                    len(cards), len(taxonomy.themes), validation.paper_coverage,
                    validation.status,
                )
                return taxonomy, validation
        except Exception as exc:
            logger.warning(
                "LLM taxonomy induction failed, using evidence-field fallback: %s", exc
            )

    fallback_axis, fallback_clusters = _cluster_by_evidence_axis(cards)
    taxonomy = _taxonomy_from_legacy_clusters(
        fallback_clusters, cards, topic, scope, source="rule_fallback"
    )
    taxonomy.organizing_principle = fallback_axis
    validation = validate_taxonomy(taxonomy, cards)
    if validation.concentration_requires_split:
        taxonomy, validation = _refine_dominant_theme(
            taxonomy, validation, cards, topic, scope
        )
    logger.info(
        "Fallback taxonomy: papers=%d themes=%d coverage=%.2f status=%s",
        len(cards), len(taxonomy.themes), validation.paper_coverage, validation.status,
    )
    return taxonomy, validation


def _normalize_taxonomy_theme_names(
    taxonomy: DynamicTaxonomy,
    cards: List[Dict[str, Any]],
) -> DynamicTaxonomy:
    """用实际成员的共同术语替换悬空、泛化或异常英文主题名。"""
    card_map = {
        str(card.get("paper_id") or ""): card for card in cards if card.get("paper_id")
    }
    assignments = {
        theme.theme_id: [
            item.paper_id for item in taxonomy.assignments
            if item.primary_theme_id == theme.theme_id
        ]
        for theme in taxonomy.themes
    }
    for theme in taxonomy.themes:
        name = str(theme.name or "").strip()
        paper_cards = [
            card_map[pid] for pid in assignments.get(theme.theme_id, []) if pid in card_map
        ]
        is_fallback_theme = bool(
            re.search(r"其他|其它|未分类|other|misc", name, re.IGNORECASE)
        )
        if paper_cards and not is_fallback_theme and (
            _is_low_signal_route_name(name) or _is_abnormal_theme_name(name, paper_cards)
        ):
            theme.name = semantic_route_label(paper_cards, "", fallback_field="method")
    return taxonomy


def _is_abnormal_theme_name(name: str, cards: List[Dict[str, Any]]) -> bool:
    lowered = str(name or "").casefold()
    if re.search(r"[a-z]", lowered) and not re.search(r"[\u4e00-\u9fff]", lowered):
        return True
    evidence = taxonomy_tokens(" ".join(
        str(card.get(field) or "")
        for card in cards
        for field in ("title", "research_problem", "method")
    ))
    name_tokens = taxonomy_tokens(name)
    return bool(name_tokens) and not (name_tokens & evidence)


def _representative_cards(
    cards: List[Dict[str, Any]], limit: int
) -> List[Dict[str, Any]]:
    """从有序候选中均匀取样，避免只观察排名前部。"""
    if len(cards) <= limit:
        return list(cards)
    if limit <= 1:
        return [cards[0]]
    indexes = {
        round(index * (len(cards) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [cards[index] for index in sorted(indexes)]


def _complete_sampled_assignments(
    taxonomy: DynamicTaxonomy,
    cards: List[Dict[str, Any]],
) -> DynamicTaxonomy:
    """依据主题定义词项为 LLM 未分配的论文补齐唯一主主题。"""
    valid_ids = {str(card.get("paper_id") or "") for card in cards if card.get("paper_id")}
    taxonomy.assignments = [
        assignment for assignment in taxonomy.assignments
        if assignment.paper_id in valid_ids
        and any(theme.theme_id == assignment.primary_theme_id for theme in taxonomy.themes)
    ]
    assigned_ids = {assignment.paper_id for assignment in taxonomy.assignments}
    theme_tokens = {
        theme.theme_id: taxonomy_tokens(" ".join([
            theme.name,
            theme.description,
            *theme.inclusion_criteria,
        ]))
        for theme in taxonomy.themes
    }
    for card in cards:
        paper_id = str(card.get("paper_id") or "")
        if not paper_id or paper_id in assigned_ids:
            continue
        card_text = " ".join(
            str(card.get(field) or "")
            for field in (
                "title", "research_problem", "method", "study_design", "dataset",
                "publication_type",
            )
        )
        card_tokens = taxonomy_tokens(card_text)
        scores = {
            theme_id: len(card_tokens & tokens) / max(1, len(tokens))
            for theme_id, tokens in theme_tokens.items()
        }
        if scores:
            theme_id, score = max(scores.items(), key=lambda item: item[1])
        else:
            theme_id, score = "", 0.0
        if not theme_id or score <= 0:
            fallback = next(
                (
                    theme for theme in taxonomy.themes
                    if "其他" in theme.name or "未分类" in theme.name
                ),
                None,
            )
            if fallback is None:
                fallback = ResearchTheme(
                    theme_id="T_OTHER",
                    name="其他相关研究",
                    description="当前结构化证据不足以归入主要主题",
                    inclusion_criteria=["与研究主题相关但缺少主题归属证据"],
                )
                taxonomy.themes.append(fallback)
                theme_tokens[fallback.theme_id] = set()
            theme_id = fallback.theme_id
            score = 0.2
        taxonomy.assignments.append(PaperThemeAssignment(
            paper_id=paper_id,
            primary_theme_id=theme_id,
            confidence=max(0.2, min(1.0, score)),
            rationale="依据主题定义与 PaperCard 显式字段的词项重合补齐归属",
            evidence_fields=["title", "research_problem", "method", "study_design"],
        ))
        assigned_ids.add(paper_id)
    return taxonomy


def taxonomy_tokens(text: str) -> set[str]:
    lowered = str(text or "").lower()
    english = {
        token for token in re.findall(r"[a-z][a-z0-9_-]{2,}", lowered)
        if token not in TOKEN_STOPWORDS
    }
    chinese: set[str] = set()
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
        for size in (2, 3, 4):
            chinese.update(
                sequence[index:index + size]
                for index in range(max(0, len(sequence) - size + 1))
            )
    return english | {token for token in chinese if token not in TOKEN_STOPWORDS}


def _taxonomy_payload(
    cards: List[Dict[str, Any]], topic: str, scope: Dict[str, Any]
) -> Dict[str, Any]:
    def compact(value: Any, limit: int) -> Any:
        if isinstance(value, list):
            return [str(item)[:80] for item in value[:6]]
        return str(value or "")[:limit]

    return {
        "topic": topic,
        "scope": scope,
        "paper_count": len(cards),
        "papers": [
            {
                "paper_id": card.get("paper_id"),
                "title": compact(card.get("title"), 220),
                "research_problem": compact(card.get("research_problem"), 260),
                "method": compact(card.get("method"), 220),
                "study_design": compact(card.get("study_design"), 100),
                "data_modalities": compact(card.get("data_modalities"), 0),
                "behavior_categories": compact(card.get("behavior_categories"), 0),
                "publication_type": compact(card.get("publication_type"), 60),
                "dataset": compact(card.get("dataset"), 100),
                "quality_status": compact(card.get("quality_status"), 40),
            }
            for card in cards
        ],
    }


def _normalize_llm_taxonomy(
    data: Dict[str, Any],
    cards: List[Dict[str, Any]],
    topic: str,
    scope: Dict[str, Any],
) -> DynamicTaxonomy:
    """同时接受新版 themes/assignments 和旧版 clusters 响应。"""
    if data.get("themes"):
        valid_ids = {str(card.get("paper_id")) for card in cards if card.get("paper_id")}
        themes: list[ResearchTheme] = []
        theme_ids: set[str] = set()
        for index, raw_theme in enumerate((data.get("themes") or [])[:8], start=1):
            theme_id = str(raw_theme.get("theme_id") or f"T{index}")[:40]
            if theme_id in theme_ids:
                theme_id = f"T{index}"
            theme_ids.add(theme_id)
            themes.append(ResearchTheme(
                theme_id=theme_id,
                name=str(raw_theme.get("name") or "未命名研究方向")[:100],
                description=str(raw_theme.get("description") or "")[:500],
                inclusion_criteria=[str(x)[:200] for x in raw_theme.get("inclusion_criteria") or []][:6],
                exclusion_criteria=[str(x)[:200] for x in raw_theme.get("exclusion_criteria") or []][:6],
                representative_papers=[
                    str(x) for x in raw_theme.get("representative_papers") or []
                    if str(x) in valid_ids
                ][:3],
            ))
        assignments: list[PaperThemeAssignment] = []
        assigned: set[str] = set()
        for raw_assignment in data.get("assignments") or []:
            paper_id = str(raw_assignment.get("paper_id") or "")
            primary = str(raw_assignment.get("primary_theme_id") or "")
            if paper_id not in valid_ids or paper_id in assigned or primary not in theme_ids:
                continue
            assigned.add(paper_id)
            secondary = [
                str(x) for x in raw_assignment.get("secondary_theme_ids") or []
                if str(x) in theme_ids and str(x) != primary
            ][:3]
            assignments.append(PaperThemeAssignment(
                paper_id=paper_id,
                primary_theme_id=primary,
                secondary_theme_ids=list(dict.fromkeys(secondary)),
                confidence=_safe_confidence(raw_assignment.get("confidence")),
                rationale=str(raw_assignment.get("rationale") or "")[:500],
                evidence_fields=[str(x)[:60] for x in raw_assignment.get("evidence_fields") or []][:8],
            ))
        return DynamicTaxonomy(
            topic=topic,
            scope=scope,
            organizing_principle=str(data.get("organizing_principle") or "research_problem_or_route")[:120],
            rationale=str(data.get("rationale") or "")[:800],
            themes=themes,
            assignments=assignments,
        )
    normalized = _normalize_llm_clusters(data.get("clusters") or [], cards)
    return _taxonomy_from_legacy_clusters(normalized, cards, topic, scope, source="llm_legacy")


def _taxonomy_from_legacy_clusters(
    clusters: List[Dict[str, Any]],
    cards: List[Dict[str, Any]],
    topic: str,
    scope: Dict[str, Any],
    source: str,
) -> DynamicTaxonomy:
    card_by_id = {
        str(card.get("paper_id") or ""): card for card in cards if card.get("paper_id")
    }
    themes: list[ResearchTheme] = []
    assignments: list[PaperThemeAssignment] = []
    for index, cluster in enumerate(clusters, start=1):
        theme_id = f"T{index}"
        paper_ids = [str(x) for x in cluster.get("paper_ids") or [] if x]
        cluster_cards = [card_by_id.get(paper_id, {}) for paper_id in paper_ids]
        raw_name = str(cluster.get("cluster_name") or "")
        theme_name = (
            raw_name
            if source == "rule_fallback" and raw_name
            else semantic_route_label(
                cluster_cards,
                raw_name,
                fallback_field="research_problem",
            )
        )
        themes.append(ResearchTheme(
            theme_id=theme_id,
            name=theme_name,
            description=str(cluster.get("description") or ""),
            representative_papers=[str(x) for x in cluster.get("representative_papers") or []][:3],
        ))
        assignments.extend(
            PaperThemeAssignment(
                paper_id=paper_id,
                primary_theme_id=theme_id,
                confidence=0.5 if source == "rule_fallback" else 0.7,
                rationale="由兼容分类结果映射",
                evidence_fields=["title", "research_problem", "method"],
            )
            for paper_id in paper_ids
        )
    return DynamicTaxonomy(
        topic=topic,
        scope=scope,
        organizing_principle="evidence_aware_fallback" if source == "rule_fallback" else "llm_selected",
        themes=themes,
        assignments=assignments,
        source=source,
    )


def validate_taxonomy(
    taxonomy: DynamicTaxonomy,
    cards: List[Dict[str, Any]],
) -> TaxonomyValidationResult:
    """使用确定性指标验证覆盖、唯一归属、主题规模和兜底比例。"""
    paper_ids = {str(card.get("paper_id")) for card in cards if card.get("paper_id")}
    theme_ids = {theme.theme_id for theme in taxonomy.themes}
    counts: Dict[str, int] = {theme_id: 0 for theme_id in theme_ids}
    assigned: set[str] = set()
    duplicate_count = 0
    invalid_count = 0
    for assignment in taxonomy.assignments:
        if assignment.paper_id not in paper_ids or assignment.primary_theme_id not in theme_ids:
            invalid_count += 1
            continue
        if assignment.paper_id in assigned:
            duplicate_count += 1
            continue
        assigned.add(assignment.paper_id)
        counts[assignment.primary_theme_id] += 1

    total = len(paper_ids)
    coverage = len(assigned) / total if total else 1.0
    largest = max(counts.values(), default=0) / total if total else 0.0
    other_ids = {
        theme.theme_id for theme in taxonomy.themes
        if re.search(r"其他|其它|未分类|other|misc", theme.name, re.IGNORECASE)
    }
    other_count = sum(counts.get(theme_id, 0) for theme_id in other_ids)
    unassigned_ratio = (total - len(assigned) + other_count) / total if total else 0.0
    min_size = 2 if total >= 8 else 1
    undersized = [theme_id for theme_id, count in counts.items() if 0 < count < min_size]
    errors: list[str] = []
    warnings: list[str] = []
    if coverage < 0.90:
        errors.append(f"论文主主题覆盖率仅为 {coverage:.1%}")
    if duplicate_count:
        errors.append(f"存在 {duplicate_count} 个重复主主题归属")
    if invalid_count:
        errors.append(f"存在 {invalid_count} 个无效论文或主题引用")
    if not taxonomy.themes and total:
        errors.append("未生成可用主题")
    if unassigned_ratio > 0.30:
        errors.append(f"未分配或兜底论文比例为 {unassigned_ratio:.1%}")
    if largest > 0.50:
        warnings.append(f"最大主题包含 {largest:.1%} 的论文，请检查是否需要细分")

    # 集中度触发器：不再作为硬错误，而是触发二级细分评估
    concentration_requires_split = False
    dominant_theme_id_for_split: str | None = None
    if largest > 0.50 and total >= 20:
        concentration_requires_split = True
        dominant_theme_id_for_split = max(counts, key=lambda k: counts[k])
        warnings.append(
            f"单一主题占比 {largest:.1%}，分类粒度可能过粗，将尝试按方法路线细分"
        )

    if undersized:
        warnings.append("存在仅含一篇论文的碎片主题")
    # 语义一致性检测：主题内部论文是否真的属于同一研究路线
    incoherent_themes = _detect_incoherent_themes(taxonomy, cards)
    if incoherent_themes:
        for theme_id, score in incoherent_themes:
            warnings.append(
                f"主题 '{theme_id}' 内部语义一致性过低 (coherence={score:.2f})，"
                f"可能混杂了不同研究路线的论文"
            )
    expected_min = 3 if total >= 8 else (2 if total >= 3 else 1)
    if total and len([x for x in counts.values() if x]) < expected_min:
        warnings.append("有效主题数量与当前论文规模不匹配")

    # 只有结构性错误或确实需要二级细分时才要求修订。单篇碎片主题属于
    # valid_with_warning：Writer 可以忽略/合并这类长尾主题，不能因此把
    # 整份研究现状降级为研究背景。
    requires_revision = bool(errors or concentration_requires_split)

    # 精确状态：invalid > refinement_required > valid_with_warning > valid
    if errors:
        status = TaxonomyStatus.INVALID
    elif concentration_requires_split:
        status = TaxonomyStatus.REFINEMENT_REQUIRED
    elif warnings:
        status = TaxonomyStatus.VALID_WITH_WARNING
    else:
        status = TaxonomyStatus.VALID

    return TaxonomyValidationResult(
        valid=not errors,
        requires_revision=requires_revision,
        status=status,
        concentration_requires_split=concentration_requires_split,
        dominant_theme_id=dominant_theme_id_for_split,
        paper_count=total,
        assigned_count=len(assigned),
        theme_count=len([x for x in counts.values() if x]),
        paper_coverage=coverage,
        primary_assignment_ratio=coverage,
        largest_theme_ratio=largest,
        unassigned_ratio=unassigned_ratio,
        undersized_theme_ids=undersized,
        errors=errors,
        warnings=warnings,
    )



def _detect_incoherent_themes(
    taxonomy: DynamicTaxonomy,
    cards: List[Dict[str, Any]],
    min_coherence: float = 0.25,
) -> list[tuple[str, float]]:
    """检测主题内部论文是否语义一致。

    计算主题内每对论文的标题/研究问题/方法 token overlap 均值。
    低于阈值的主题可能混杂了不同研究路线，应触发警告或细分。
    """
    card_map = {str(card.get("paper_id") or ""): card for card in cards if card.get("paper_id")}
    incoherent: list[tuple[str, float]] = []
    for theme in taxonomy.themes:
        paper_ids = [
            a.paper_id for a in taxonomy.assignments
            if a.primary_theme_id == theme.theme_id and a.paper_id in card_map
        ]
        if len(paper_ids) < 3:
            continue  # 太少论文不检测
        texts = [
            taxonomy_tokens(" ".join(
                str(card_map[paper_id].get(field) or "")
                for field in ("title", "research_problem", "method")
            ))
            for paper_id in paper_ids
        ]
        if not texts:
            continue
        scores: list[float] = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                union = texts[i] | texts[j]
                if not union:
                    continue
                scores.append(len(texts[i] & texts[j]) / len(union))
        if not scores:
            continue
        avg_coherence = sum(scores) / len(scores)
        if avg_coherence < min_coherence:
            incoherent.append((theme.theme_id, avg_coherence))
    return incoherent


def taxonomy_to_clusters(taxonomy: DynamicTaxonomy, flatten: bool = True) -> List[Dict[str, Any]]:
    """转换为既有生成器使用的 clusters 结构。"""
    assignments: Dict[str, list[str]] = {theme.theme_id: [] for theme in taxonomy.themes}
    for assignment in taxonomy.assignments:
        if assignment.primary_theme_id in assignments:
            assignments[assignment.primary_theme_id].append(assignment.paper_id)
            
    if flatten:
        # 扁平模式：只保留实际分配了论文的主题（过滤掉作为抽象类的父主题）
        return [
            {
                "cluster_name": theme.name,
                "description": theme.description,
                "paper_ids": assignments[theme.theme_id],
                "representative_papers": [
                    paper_id for paper_id in theme.representative_papers
                    if paper_id in assignments[theme.theme_id]
                ][:3] or assignments[theme.theme_id][:3],
                "theme_id": theme.theme_id,
                "inclusion_criteria": theme.inclusion_criteria,
                "exclusion_criteria": theme.exclusion_criteria,
                "level": theme.level,
                "parent_theme_id": theme.parent_theme_id,
            }
            for theme in taxonomy.themes
            if assignments[theme.theme_id]
        ]
    else:
        # 层级模式：保留所有主题（包括父主题），由外部处理嵌套关系
        return [
            {
                "cluster_name": theme.name,
                "description": theme.description,
                "paper_ids": assignments[theme.theme_id],
                "representative_papers": [
                    paper_id for paper_id in theme.representative_papers
                    if paper_id in assignments[theme.theme_id]
                ][:3] or assignments[theme.theme_id][:3],
                "theme_id": theme.theme_id,
                "inclusion_criteria": theme.inclusion_criteria,
                "exclusion_criteria": theme.exclusion_criteria,
                "level": theme.level,
                "parent_theme_id": theme.parent_theme_id,
                "child_theme_ids": theme.child_theme_ids,
            }
            for theme in taxonomy.themes
        ]


def _validation_score(result: TaxonomyValidationResult) -> tuple[int, float, float, int]:
    return (-len(result.errors), result.paper_coverage, -result.unassigned_ratio, -len(result.undersized_theme_ids))


def _safe_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _parse_json(text: str) -> Dict[str, Any]:
    """健壮解析 JSON（委托 json_utils 三段式策略）。"""
    from app.core.json_utils import parse_json_object
    return parse_json_object(text)


def _normalize_llm_clusters(
    clusters: List[Dict[str, Any]],
    cards: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """保证分类互斥、完整，拒绝模型编造的 paper_id。"""
    valid_ids = [str(card.get("paper_id") or "") for card in cards if card.get("paper_id")]
    valid_set = set(valid_ids)
    assigned: set[str] = set()
    result: List[Dict[str, Any]] = []
    for cluster in clusters[:8]:
        paper_ids = []
        for paper_id in cluster.get("paper_ids") or []:
            paper_id = str(paper_id)
            if paper_id in valid_set and paper_id not in assigned:
                assigned.add(paper_id)
                paper_ids.append(paper_id)
        if not paper_ids:
            continue
        representatives = [
            str(value) for value in (cluster.get("representative_papers") or [])
            if str(value) in set(paper_ids)
        ][:3]
        cluster_cards = [
            card for card in cards
            if str(card.get("paper_id") or "") in set(paper_ids)
        ]
        result.append({
            "cluster_name": _normalize_cluster_label(
                "research_problem",
                semantic_route_label(
                    cluster_cards,
                    str(cluster.get("cluster_name") or ""),
                    fallback_field="research_problem",
                ),
            )[:100],
            "description": str(cluster.get("description") or "")[:500],
            "paper_ids": paper_ids,
            "representative_papers": representatives or paper_ids[:3],
        })
    missing = [paper_id for paper_id in valid_ids if paper_id not in assigned]
    if missing:
        result.append({
            "cluster_name": "其他相关研究",
            "description": "现有证据不足以可靠归入上述主类的相关研究",
            "paper_ids": missing,
            "representative_papers": missing[:3],
        })
    return result


def taxonomy_fingerprint(taxonomy: DynamicTaxonomy) -> str:
    """生成分类结果的稳定哈希，用于检测重试是否产生变化。"""
    theme_signatures = []
    for theme in sorted(taxonomy.themes, key=lambda t: t.theme_id):
        # 使用主题包含的论文ID的排序列表作为签名
        paper_ids = sorted(
            [a.paper_id for a in taxonomy.assignments if a.primary_theme_id == theme.theme_id]
        )
        theme_signatures.append(f"{theme.theme_id}:{','.join(paper_ids)}")
    
    combined = "|".join(theme_signatures)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def _refine_dominant_theme(
    taxonomy: DynamicTaxonomy,
    validation: TaxonomyValidationResult,
    cards: List[Dict[str, Any]],
    topic: str,
    scope: Dict[str, Any],
) -> tuple[DynamicTaxonomy, TaxonomyValidationResult]:
    """用当前卡片字段分布细分主导主题，不使用预置技术分面。"""
    dominant_theme_id = validation.dominant_theme_id
    if not dominant_theme_id:
        return taxonomy, validation
    target_ids = {
        assignment.paper_id
        for assignment in taxonomy.assignments
        if assignment.primary_theme_id == dominant_theme_id
    }
    target_cards = [card for card in cards if str(card.get("paper_id") or "") in target_ids]
    axis, clusters = _cluster_by_evidence_axis(target_cards)
    if len(clusters) < 2:
        axis, clusters = _cluster_by_salient_terms(target_cards)
        if len(clusters) < 2:
            logger.info("Dominant theme %s lacks evidence diversity for splitting.", dominant_theme_id)
            return taxonomy, validation

    new_themes = [theme for theme in taxonomy.themes if theme.theme_id != dominant_theme_id]
    new_assignments = [
        assignment for assignment in taxonomy.assignments
        if assignment.primary_theme_id != dominant_theme_id
    ]
    for index, cluster in enumerate(clusters, 1):
        theme_id = f"{dominant_theme_id}_S{index}"
        paper_ids = [str(value) for value in cluster.get("paper_ids") or []]
        cluster_cards = [card for card in cards if str(card.get("paper_id") or "") in set(paper_ids)]
        new_themes.append(ResearchTheme(
            theme_id=theme_id,
            name=semantic_route_label(
                cluster_cards,
                str(cluster.get("cluster_name") or f"子主题 {index}"),
                fallback_field="research_problem",
            ),
            description=str(cluster.get("description") or ""),
            representative_papers=paper_ids[:3],
            inclusion_criteria=[f"PaperCard.{axis} 与该分组值一致"],
            exclusion_criteria=[],
        ))
        new_assignments.extend(
            PaperThemeAssignment(
                paper_id=paper_id,
                primary_theme_id=theme_id,
                confidence=0.8,
                rationale=f"依据显式字段 {axis} 归入该主题",
                evidence_fields=[axis],
            )
            for paper_id in paper_ids
        )

    refined = DynamicTaxonomy(
        topic=taxonomy.topic or topic,
        scope=taxonomy.scope or scope,
        organizing_principle=axis,
        rationale=taxonomy.rationale,
        themes=new_themes,
        assignments=new_assignments,
        version=taxonomy.version + 1,
        source="evidence_axis_refined",
    )
    return refined, validate_taxonomy(refined, cards)
