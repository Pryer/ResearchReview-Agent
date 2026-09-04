"""论文排序工具。

去重委托、硬规则过滤（研究范围与检索分支）、相关性/质量打分、排序与
来源软配额。术语匹配原语在 ``paper_matching``，出版分层词表在
``venue_tiers``，LLM 语义重排在 ``paper_rerank``；本模块按原导入路径
再导出这些 API，既有调用方无需修改。
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, List, Sequence

from app.core.logger import get_logger
from app.tools.paper_matching import (
    compile_scope,
    term_matches_haystack as _term_matches_haystack,
)
from app.tools.paper_matching import (
    excluded_term_matches_title,
    hard_anchor_matches_haystack,
)
from app.tools.paper_matching import matches_seed_context as _matches_seed_context
from app.tools.paper_rerank import llm_rerank_papers
from app.tools.paper_rerank import (
    best_protocol_route as _best_protocol_route,
    concept_coverage as _concept_coverage,
)
from app.tools.venue_tiers import classify_venue_tier as _classify_venue_tier
from app.tools.venue_tiers import is_degree_thesis
from app.utils.deduplicate import deduplicate_papers

logger = get_logger(__name__)

LEGACY_RELEVANCE_WEIGHT = 0.70
LEGACY_QUALITY_WEIGHT = 0.30
PROTOCOL_RELEVANCE_WEIGHT = 0.85
PROTOCOL_QUALITY_WEIGHT = 0.15


def _safe_int(value: Any, default: int = 0) -> int:
    """安全的整数转换 helper，防止 TypeError/ValueError 导致的运行时异常。"""
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def normalize_title(title: str) -> str:
    """标题归一化。"""
    from app.utils.deduplicate import normalize_title as _impl
    return _impl(title)


def title_similarity(a: str, b: str) -> float:
    """计算标题相似度。"""
    from app.utils.deduplicate import title_similarity as _impl
    return _impl(a, b)


def _extract_topic_synonyms(topic: str, keywords: Sequence[str] | None = None) -> list[str]:
    """从检索关键词中提取与 topic 跨语言或同义的变体词。"""
    if not topic or not keywords:
        return []
    topic_clean = str(topic).strip()
    synonyms: list[str] = []
    seen = {topic_clean.lower()}
    for kw in keywords:
        kw_clean = re.sub(r"\s+", " ", str(kw or "")).strip()
        if not kw_clean or len(kw_clean) < 2 or kw_clean.lower() in seen:
            continue
        seen.add(kw_clean.lower())
        # 跨语言与同语言变体同样收录（历史上此处曾按 CJK 差异分两个
        # 分支，但两分支实现完全相同；保持既有行为：按关键词顺序收集）。
        synonyms.append(kw_clean)
    # WHY: 旧实现直接截取前 8 个关键词。规划器通常先产出中文扩展词，英文
    # 别名排在其后，中文主题的英文论文因此在 topic anchor 阶段整支误杀。
    # 保持各语言内部顺序，但为两种语言分别保留名额。
    cjk = [item for item in synonyms if re.search(r"[\u4e00-\u9fff]", item)]
    latin = [item for item in synonyms if not re.search(r"[\u4e00-\u9fff]", item)]
    if cjk and latin:
        return [*cjk[:6], *latin[:6]]
    return synonyms[:12]

# ── 动态任务偏移惩罚 ──────────────────────────────────────────────────────
# 从 required_concepts（概念组）自动派生惩罚规则，不维护领域专属硬编码表。
# 每个概念组定义一个"核心维度"（如 ["few-shot", "少样本"] 和
# ["action recognition", "动作识别"]），论文标题缺失某个核心维度
# 时给予惩罚——缺失维度越多，惩罚越大。


def _apply_task_mismatch_penalty(
    score: float,
    topic: str,
    title: str,
    abstract: str,
    required_concepts: Sequence[Sequence[str]] | None = None,
    topic_synonyms: Sequence[str] | None = None,
) -> float:
    """从 required_concepts 动态派生惩罚，无需硬编码规则表。

    逻辑：
    1. 遍历每个概念组（如 ["few-shot", "少样本"]）。
    2. 如果主题或任一同义词（含跨语言变体）包含该组的某个词
       （说明该维度对用户有意义），但论文标题不包含该组的任何词，
       则视为"标题偏离该维度"。同义词锚点保证纯中文主题下由英文
       关键词召回的英文概念组同样生效，不再静默跳过。
    3. 偏离一个核心维度 → 惩罚 0.25；偏离两个及以上 → 额外 +0.15。
    4. 若论文摘要包含该维度的词，部分豁免（该维度惩罚减半）。
    5. 维度命中判定与打分路径使用同一个归一化匹配器
       ``_term_matches_haystack``（兼容连字符/空格变体与词边界），
       避免 "Few Shot" 式标题在加分路径命中概念组、却在惩罚路径
       被误判缺失维度而双重处置。

    这使得系统对任意研究主题自动生效：
    - "少样本动作识别" → 缺 "few-shot" 或 "action" 都会被惩罚
    - "图像分割" → 缺 "segmentation"/"分割" 会被惩罚
    - 无需为每个领域手动编写规则
    """
    concept_groups = [group for group in (required_concepts or []) if group]
    if not concept_groups:
        return score

    anchors = [
        anchor for anchor in (
            str(topic or ""), *(str(s) for s in topic_synonyms or []),
        )
        if anchor.strip()
    ]
    if not anchors:
        return score

    title_text = str(title or "")
    abstract_text = str(abstract or "")

    missed_dimensions = 0
    total_penalty = 0.0

    for group in concept_groups:
        terms = [str(t).strip().lower() for t in group if str(t).strip()]
        if not terms:
            continue

        # 只对主题或同义词中也出现的维度做检查（规划未关联该维度时跳过）
        topic_has_dimension = any(
            _term_matches_haystack(term, anchor)
            for term in terms
            for anchor in anchors
        )
        if not topic_has_dimension:
            continue

        # 标题是否包含该维度的任何词
        if any(_term_matches_haystack(term, title_text) for term in terms):
            continue

        # 标题缺失该维度 → 计算惩罚
        missed_dimensions += 1
        dimension_penalty = 0.25

        # 摘要包含该维度 → 部分豁免
        if any(_term_matches_haystack(term, abstract_text) for term in terms):
            dimension_penalty *= 0.5

        total_penalty += dimension_penalty

    # 缺失多个维度时额外加重（缺 2 个比缺 1 个严重得多）
    if missed_dimensions >= 2:
        total_penalty += 0.15

    return max(0.0, score - total_penalty)


def compute_relevance_score(
    paper: Dict[str, Any],
    topic: str,
    required_concepts: Sequence[Sequence[str]] | None = None,
    topic_synonyms: Sequence[str] | None = None,
) -> float:
    """计算论文与主题的相关性分数（0-1）。

    匹配层级规则（用户显式规则）：
    1. 标题中出现完整关键词/主题：赋予最高相关分（0.85 ~ 1.0）。
    2. 仅在摘要中出现完整关键词/主题：赋予第二高分（0.55 ~ 0.75）。
    3. 仅部分词汇重叠/泛词共现：赋予基础重叠分（0.20 ~ 0.45）。
    支持中英文跨语言主题同义词加权匹配。
    """
    topic_clean = (topic or "").strip()
    concept_groups = [group for group in (required_concepts or []) if group]
    synonyms = [s.strip() for s in (topic_synonyms or []) if s and s.strip()]

    if not topic_clean and not concept_groups and not synonyms:
        return 0.5

    title = str(paper.get("title") or "").lower()
    abstract = str(paper.get("abstract") or "").lower()
    haystack = f"{title} {abstract}".lower()

    score = 0.0
    matched_title = False
    matched_abstract = False

    if topic_clean:
        topic_lower = topic_clean.lower()
        # 1. 标题出现完整主题关键词 → 最高分
        if _term_matches_haystack(topic_lower, title) or topic_lower in title:
            score += 0.55
            matched_title = True
        # 2. 仅在摘要出现完整主题关键词 → 第二高分
        elif _term_matches_haystack(topic_lower, abstract) or topic_lower in abstract:
            score += 0.30
            matched_abstract = True

        # 3. 词汇重叠度 (按重叠比例补充)
        topic_words = set(topic_lower.split())
        if topic_words:
            title_words = set(title.split())
            abstract_words = set(abstract.split())
            overlap = topic_words & (title_words | abstract_words)
            score += 0.15 * (len(overlap) / len(topic_words))

    # 跨语言同义词匹配（如中文 topic 匹配英文关键词，或英文 topic 匹配中文关键词）
    if synonyms:
        syn_title_hit = False
        syn_abstract_hit = False
        best_syn_overlap = 0.0
        for syn in synonyms:
            syn_lower = syn.lower()
            if not syn_title_hit and (_term_matches_haystack(syn_lower, title) or syn_lower in title):
                syn_title_hit = True
            elif not syn_abstract_hit and (_term_matches_haystack(syn_lower, abstract) or syn_lower in abstract):
                syn_abstract_hit = True
            syn_words = set(syn_lower.split())
            if syn_words:
                title_words = set(title.split())
                abstract_words = set(abstract.split())
                overlap = syn_words & (title_words | abstract_words)
                best_syn_overlap = max(best_syn_overlap, len(overlap) / len(syn_words))

        # 同义词标题命中最高分，摘要命中第二高分
        if syn_title_hit and not matched_title:
            score += 0.50
            matched_title = True
        elif syn_abstract_hit and not matched_abstract:
            score += 0.28
            matched_abstract = True

        if best_syn_overlap > 0 and not topic_clean:
            score += 0.15 * best_syn_overlap
        elif best_syn_overlap > 0 and score < 0.3:
            score += 0.10 * best_syn_overlap

    if concept_groups:
        field_hits = 0
        title_hits = 0
        for group in concept_groups:
            if any(_term_matches_haystack(term, haystack) for term in group):
                field_hits += 1
            if any(_term_matches_haystack(term, title) for term in group):
                title_hits += 1
        score += 0.30 * (field_hits / len(concept_groups))
        score += 0.25 * (title_hits / len(concept_groups))

    # ── 任务偏移惩罚 ──
    # 从 required_concepts 自动派生核心维度，标题缺失核心维度的论文被惩罚。
    # 对任意研究主题自动生效，无需维护领域专属规则表。
    score = _apply_task_mismatch_penalty(
        score, topic_clean, title, abstract, required_concepts=required_concepts,
        topic_synonyms=synonyms,
    )

    return round(min(score, 1.0), 4)


def passes_topic_filter(
    paper: Dict[str, Any],
    topic: str,
    keywords: Sequence[str] | None = None,
    required_concepts: Sequence[Sequence[str]] | None = None,
    excluded_title_terms: Sequence[str] | None = None,
    compiled_scope: Dict[str, Any] | None = None,
) -> bool:
    """返回论文是否通过主题过滤。"""
    passed, _ = evaluate_topic_filter(
        paper,
        topic,
        keywords=keywords,
        required_concepts=required_concepts,
        excluded_title_terms=excluded_title_terms,
        compiled_scope=compiled_scope,
    )
    return passed


# 主题锚点过滤中属于“降级放行”的原因关键词：这些论文未经词法硬确认，
# 相关性只由后续 LLM 语义筛选担保，需要标记为低置信并提示可能偏题。
_LOW_CONFIDENCE_PASS_REASONS = (
    "宽松命中",
    "交由语义筛选",
)


def _fallback_cjk_bigram_match(term: str, haystack: str, threshold: float = 0.75) -> bool:
    """兜底中文锚点的宽松匹配：二元词组覆盖率达标即视为命中。

    裸主题串（如「少样本动作识别」）很少被中文论文逐字包含，
    但主题的核心二元词组（少样本/样本动作/动作识别…）通常会出现。
    仅在退化兜底路径使用，避免误杀中文分支；正常概念组仍走精确规则。
    """
    chars = re.findall(r"[\u4e00-\u9fff]", str(term or ""))
    if len(chars) < 3:
        return False
    bigrams = {"".join(chars[i:i + 2]) for i in range(len(chars) - 1)}
    if not bigrams:
        return False
    haystack_clean = re.sub(r"\s+", "", str(haystack or ""))
    matched = sum(1 for gram in bigrams if gram in haystack_clean)
    return matched / len(bigrams) >= threshold


def _compiled_aliases(compiled_scope: Dict[str, Any] | None, role: str) -> list[str]:
    if not compiled_scope:
        return []
    aliases = compiled_scope.get("aliases") or {}
    return [str(value) for value in aliases.get(role) or [] if str(value).strip()]


def _compiled_groups(compiled_scope: Dict[str, Any] | None, roles: set[str]) -> list[list[str]]:
    if not compiled_scope:
        return []
    return [
        [str(value) for value in group.get("aliases") or [] if str(value).strip()]
        for group in compiled_scope.get("groups") or []
        if str(group.get("role") or "") in roles and group.get("aliases")
    ]


def evaluate_topic_anchor_filter(
    paper: Dict[str, Any],
    topic: str,
    keywords: Sequence[str] | None = None,
    required_concepts: Sequence[Sequence[str]] | None = None,
    topic_synonyms: Sequence[str] | None = None,
    compiled_scope: Dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """主题锚点硬下限：核心概念维度一个都不命中的论文直接排除。

    概念组可用时按组判定（命中任一词即算该组命中）；概念组缺失时用
    主题及其同义词构成单一锚点组兜底。该约束在任何降级模式
    （best-effort、保守重生成）下都不豁免——放宽的应该是篇数，
    不是相关性。标题与摘要均无信息时放行，交由后续语义筛选。
    """
    if compiled_scope is None:
        compiled_scope = compile_scope(
            required_concepts=required_concepts, topic_anchors=None,
            topic=topic, research_mode="",
        )
    del keywords  # 保留签名兼容；兜底锚点使用统一编译结果
    if compiled_scope:
        compiled_groups = _compiled_groups(compiled_scope, {"topic_anchor"})
        if compiled_groups:
            required_concepts = compiled_groups
            topic_synonyms = _compiled_aliases(compiled_scope, "topic")
            topic = ""
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")
    haystack = f"{title} {abstract}".strip()
    if not haystack:
        return True, "无标题摘要信息，交由语义筛选"

    anchor_groups = [group for group in (required_concepts or []) if group]
    fallback_single_group = False
    if not anchor_groups:
        anchors = [str(topic or "").strip(), *(str(s).strip() for s in topic_synonyms or [])]
        anchors = [a for a in anchors if a]
        if not anchors:
            return True, "无可用主题锚点，放行"
        if len(anchors) == 1:
            # 仅剩裸主题串（无概念组、无同义词的退化规划）时词法硬杀过于
            # 脆弱：论文极少逐字包含主题串。硬下限只在有概念组或双语
            # 同义词锚点时执行，其余交由 LLM 语义筛选。
            return True, "仅有裸主题串锚点，交由语义筛选"
        anchor_groups = [anchors]
        fallback_single_group = True

    # 锚点与论文语言不匹配时不做硬判：纯中文锚点（缺英文关键词/概念组的
    # 退化规划）对纯英文论文永远无法词法命中，硬杀会清空整个英文池。
    # 此时交由 LLM 语义筛选；生产路径的关键词/概念组总是双语的。
    all_terms = [str(term) for group in anchor_groups for term in group]
    anchor_all_cjk = all(re.search(r"[\u4e00-\u9fff]", t) for t in all_terms)
    if anchor_all_cjk and not re.search(r"[\u4e00-\u9fff]", haystack):
        return True, "锚点语言与论文不匹配，交由语义筛选"
    # 对称护栏：锚点无任何中文词时，对中文论文的词法匹配永远无法命中，
    # 硬杀会清空整个中文分支（如 CNKI 召回池），同样交由语义筛选。
    anchor_has_cjk = any(re.search(r"[\u4e00-\u9fff]", t) for t in all_terms)
    if not anchor_has_cjk and re.search(r"[\u4e00-\u9fff]", haystack):
        return True, "锚点缺少中文术语，中文论文交由语义筛选"

    hits = sum(
        1 for group in anchor_groups
        if any(_term_matches_haystack(str(term), haystack) for term in group)
    )
    if hits == 0 and fallback_single_group:
        # 兜底单组对中文论文追加宽松的二元词组匹配，避免要求逐字
        # 包含主题串而清空中文分支；仍不命中才硬杀。
        if any(
            _fallback_cjk_bigram_match(str(term), haystack)
            for term in anchor_groups[0]
        ):
            return True, "主题锚点宽松命中（中文二元词组覆盖）"
    if hits == 0:
        return False, (
            f"未命中任何主题概念维度（{len(anchor_groups)} 组锚点均未命中标题+摘要）"
        )
    return True, f"主题锚点命中 {hits}/{len(anchor_groups)} 组"


def evaluate_topic_filter(
    paper: Dict[str, Any],
    topic: str,
    keywords: Sequence[str] | None = None,
    required_concepts: Sequence[Sequence[str]] | None = None,
    excluded_title_terms: Sequence[str] | None = None,
    compiled_scope: Dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """词法打分——不负责论文死刑。

    概念组未命中不再返回 False。语义判断交由 LLM screening 完成。
    唯一保留的确定性排除：用户明确要求的标题排除词（含 hard_exclude_title_terms）。
    """
    if compiled_scope is None:
        compiled_scope = compile_scope(
            required_concepts=required_concepts,
            topic=topic,
            excluded_title_terms=excluded_title_terms,
        )
    topic = (topic or "").strip()
    if compiled_scope is not None:
        required_concepts = _compiled_groups(compiled_scope, {"topic_anchor"}) or required_concepts
        excluded_title_terms = (
            _compiled_aliases(compiled_scope, "title_exclude")
            or excluded_title_terms
        )

    title = str(paper.get("title") or "").lower()
    abstract = str(paper.get("abstract") or "").lower()
    content_haystack = f"{title} {abstract}".strip()

    # 唯一保留的确定性排除：用户明确指定的标题排除词
    for term in (excluded_title_terms or []):
        if excluded_term_matches_title(term, title):
            return False, f"命中用户明确排除词: {term}"

    # 概念组只做词法打分标记，不杀论文
    concept_groups = [group for group in (required_concepts or []) if group]
    if concept_groups:
        has_abstract = bool(abstract.strip())
        search_field = content_haystack if has_abstract else title

        hits = 0
        title_hits = 0
        missed_groups: list[str] = []
        for group in concept_groups:
            title_matched = any(_term_matches_haystack(term, title) for term in group)
            field_matched = any(_term_matches_haystack(term, search_field) for term in group)
            if field_matched:
                hits += 1
                if title_matched:
                    title_hits += 1
            else:
                missed_groups.append(' / '.join(group[:3]))
        scope_str = "标题" if not has_abstract else "标题+摘要"
        return True, (
            f"topic_anchor: {hits}/{len(concept_groups)} matched ({scope_str})"
            + (f"; missed: {'; '.join(missed_groups[:2])}" if missed_groups else "")
        )

    # 其他词法检查也只做标记
    dynamic_keywords = _meaningful_filter_keywords(keywords or [], topic)
    if dynamic_keywords:
        hit = any(_term_matches_haystack(keyword, content_haystack) for keyword in dynamic_keywords)
        return True, f"dynamic_kw: {'hit' if hit else 'miss'}"

    # 无特定规则时放行
    return True, "放行——无词法规则"


def evaluate_scope_filter(
    paper: Dict[str, Any],
    scope: Dict[str, Any] | None,
    compiled_scope: Dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """执行用户已确认研究范围的硬过滤；无范围时保持兼容放行。"""
    if compiled_scope is None:
        compiled_scope = compile_scope(selected_scope=scope)
    if compiled_scope is not None:
        scope = {
            "exclude_terms": _compiled_aliases(compiled_scope, "scope_exclude"),
            "include_terms": _compiled_aliases(compiled_scope, "scope_include"),
            "seed_queries": _compiled_aliases(compiled_scope, "scope_seed"),
            "branches": [
                {
                    "label": group.get("label"),
                    "scope_id": group.get("scope_id"),
                    "seed_queries": group.get("seed_queries") or group.get("aliases") or [],
                }
                for group in compiled_scope.get("branches") or []
            ],
        }
    scope = scope or {}
    if not scope:
        return True, "未指定消歧范围"
    haystack = f"{str(paper.get('title') or '')} {str(paper.get('abstract') or '')} {str(paper.get('venue') or '')}".lower()

    for raw_term in scope.get("exclude_terms") or []:
        terms = [str(raw_term)]
        if any(_term_matches_haystack(term, haystack) for term in terms if str(term).strip()):
            return False, f"命中范围排除概念: {raw_term}"

    branches = [item for item in (scope.get("branches") or []) if isinstance(item, dict)]
    if branches:
        for branch in branches:
            matched, reason = _matches_seed_context(
                haystack,
                [str(x) for x in branch.get("seed_queries") or []],
            )
            if matched:
                return True, f"命中组合范围分支 {branch.get('label') or branch.get('scope_id')}: {reason}"
        return False, "未命中任何组合范围分支的完整语境"

    include_terms = [str(term) for term in scope.get("include_terms") or [] if str(term).strip()]
    for raw_term in include_terms:
        terms = [raw_term]
        if any(_term_matches_haystack(term, haystack) for term in terms):
            return True, f"命中范围纳入词: {raw_term}"

    seed_queries = [str(value) for value in scope.get("seed_queries") or [] if str(value).strip()]
    seed_matched, seed_reason = _matches_seed_context(haystack, seed_queries)
    if seed_matched:
        return True, seed_reason

    if include_terms or seed_queries:
        return False, "未命中已确认范围的纳入语境"
    return True, "范围未提供可执行纳入条件"


def evaluate_search_branch_filter(
    paper: Dict[str, Any],
    search_branches: Sequence[Dict[str, Any]] | None,
    research_mode: str = "",
    topic: str = "",
    required_concepts: Sequence[Sequence[str]] | None = None,
    compiled_scope: Dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """校验论文是否满足其召回分支。"""
    if compiled_scope is None:
        compiled_scope = compile_scope(
            required_concepts=required_concepts,
            search_branches=search_branches,
            topic=topic,
            research_mode=research_mode,
        )
    if compiled_scope is not None:
        search_branches = [
            {
                "branch_type": group.get("branch_type") or "",
                "required_concepts": group.get("required_concepts") or [],
                "constraint_level": group.get("constraint_level") or "soft",
            }
            for group in compiled_scope.get("groups") or []
            if group.get("role") == "search_branch"
        ]
        required_concepts = _compiled_groups(compiled_scope, {"topic_anchor"}) or required_concepts
    branches = {str(item.get("branch_type") or ""): item for item in (search_branches or [])}
    matched_types = [str(x) for x in paper.get("_search_branches") or [] if str(x)]
    if not branches or not matched_types:
        return True, "无分支约束"

    domain_applied = research_mode in {
        "technology_applied_to_domain", "technology_assisted_domain_analysis",
    }
    if domain_applied and set(matched_types) <= {"technical_method"}:
        return False, "仅命中通用技术方法分支，缺少应用领域锚点"

    failure_reasons = []
    for branch_type in matched_types:
        if branch_type == "topic_core":
            # refine_search 产生的新查询会被标记为 topic_core。检索源的
            # 语义召回并不保证结果真的包含主题对象，因此不能仅凭来源标签放行。
            domain_groups = [
                group
                for branch in branches.values()
                if str(branch.get("branch_type") or "") == "domain_foundation"
                for group in (branch.get("required_concepts") or [])
                if group
            ]
            passed, reason = evaluate_topic_filter(
                paper,
                topic,
                required_concepts=domain_groups or required_concepts,
            )
            if passed:
                return True, f"通过主题核心锚点: {reason}"
            failure_reasons.append(f"topic_core: {reason}")
            continue
        branch = branches.get(branch_type)
        if not branch:
            continue
        concepts = branch.get("required_concepts") or []
        if not concepts:
            if branch.get("constraint_level") != "exploratory":
                return True, f"通过分支 {branch_type}"
            failure_reasons.append(f"{branch_type}: 探索分支无领域约束")
            continue
        passed, reason = evaluate_topic_filter(
            paper,
            "",
            required_concepts=concepts,
        )
        if passed:
            return True, f"通过分支 {branch_type}: {reason}"
        failure_reasons.append(f"{branch_type}: {reason}")
    if failure_reasons:
        return False, "；".join(failure_reasons[:3])
    return True, "召回分支未提供可执行约束"


def evaluate_screening_protocol_hard_filter(
    paper: Dict[str, Any],
    screening_protocol: Dict[str, Any] | None,
    compiled_scope: Dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """只执行协议中明确标记为逐篇必需的条件。"""
    if compiled_scope is None:
        compiled_scope = compile_scope(screening_protocol=screening_protocol)
    protocol = screening_protocol or {}
    if compiled_scope is not None:
        criteria = []
        for group in compiled_scope.get("groups") or []:
            if group.get("role") != "hard_include_criteria":
                continue
            criteria.append({
                "terms": group.get("aliases") or [],
                "applies_to_each_paper": group.get("applies_to_each_paper", True),
                "label": group.get("label"),
            })
        protocol = {
            "hard_exclude_title_terms": _compiled_aliases(compiled_scope, "protocol_exclude"),
            "hard_include_criteria": criteria,
        }
    title = str(paper.get("title") or "")
    haystack = " ".join([
        title,
        str(paper.get("abstract") or ""),
        str(paper.get("venue") or ""),
    ]).lower()
    for term in protocol.get("hard_exclude_title_terms") or []:
        if excluded_term_matches_title(str(term), title):
            return False, f"标题命中协议排除词: {term}"

    checked = 0
    for criterion in protocol.get("hard_include_criteria") or []:
        if not isinstance(criterion, dict) or not criterion.get("applies_to_each_paper", True):
            continue
        terms = [str(term) for term in criterion.get("terms") or [] if str(term).strip()]
        if not terms:
            terms = [
                str(term)
                for key in ("terms_zh", "terms_en")
                for term in criterion.get(key) or []
                if str(term).strip()
            ]
        if not terms:
            continue
        checked += 1
        if not any(hard_anchor_matches_haystack(term, haystack) for term in terms):
            return False, f"未满足逐篇硬条件: {criterion.get('label') or criterion.get('criterion_id')}"
    return True, "通过上下文硬条件" if checked else "协议无逐篇硬条件"


def evaluate_document_type_filter(paper: Dict[str, Any]) -> tuple[bool, str]:
    """文献形态硬边界：学位论文不计入证据池。

    学位论文未经期刊/会议同行评审，与综述引用的已发表文献不同级；
    放行会让 ``global_evidence_gate`` 的同行评审占比虚高。

    注意判定信号的时序：CNKI 检索结果页只有 ``row_text``，学位论文的
    培养单位 venue 与 ``/d.cnki.`` DOI 只有补全详情后才出现，因此
    ``retrieval`` 节点在详情补全后还要复核一次——那里才是主拦截点。
    """
    if is_degree_thesis(
        venue=str(paper.get("venue") or ""),
        doi=str(paper.get("doi") or ""),
        title=str(paper.get("title") or ""),
    ):
        return False, "学位论文不计入证据池"
    return True, "文献形态可用"


def evaluate_paper_hard_filters(
    paper: Dict[str, Any],
    *,
    topic: str,
    keywords: Sequence[str] | None = None,
    required_concepts: Sequence[Sequence[str]] | None = None,
    excluded_title_terms: Sequence[str] | None = None,
    scope: Dict[str, Any] | None = None,
    search_branches: Sequence[Dict[str, Any]] | None = None,
    research_mode: str = "",
    screening_protocol: Dict[str, Any] | None = None,
    language_branch: str = "",
    topic_synonyms: Sequence[str] | None = None,
    compiled_scope: Dict[str, Any] | None = None,
) -> tuple[bool, str, str]:
    """统一执行排序前与详情补全后的确定性硬过滤。

    返回 ``(passed, stage, reason)``，让调用方可以保留阶段化诊断。
    详情补全可能新增摘要、venue 和 DOI，因此必须用与首次排序相同的
    规则再次检查，不能在存在 screening protocol 时跳过 scope/topic。
    """
    if compiled_scope is None:
        compiled_scope = compile_scope(
            selected_scope=scope,
            semantic_frame=None,
            screening_protocol=screening_protocol,
            required_concepts=required_concepts,
            search_branches=search_branches,
            excluded_title_terms=excluded_title_terms,
            topic=topic,
            research_mode=research_mode,
        )
    protocol = screening_protocol or {}
    if protocol or compiled_scope is not None:
        if language_branch in {"zh", "en"}:
            from app.tools.language_filter import evaluate_language_hard_filter

            passed, reason = evaluate_language_hard_filter(
                paper, protocol, language_branch, compiled_scope=compiled_scope,
            )
        else:
            passed, reason = evaluate_screening_protocol_hard_filter(
                paper, protocol, compiled_scope=compiled_scope,
            )
        if not passed:
            return False, "protocol_hard_filter", reason

    checks = (
        ("document_type_filter", lambda: evaluate_document_type_filter(paper)),
        ("scope_filter", lambda: evaluate_scope_filter(
            paper, scope, compiled_scope=compiled_scope,
        )),
        (
            "branch_filter",
            lambda: evaluate_search_branch_filter(
                paper, search_branches, research_mode, topic, required_concepts,
                compiled_scope=compiled_scope,
            ),
        ),
        (
            "topic_anchor_filter",
            lambda: evaluate_topic_anchor_filter(
                paper,
                topic,
                keywords=keywords,
                required_concepts=required_concepts,
                topic_synonyms=topic_synonyms,
                compiled_scope=compiled_scope,
            ),
        ),
        (
            "topic_filter",
            lambda: evaluate_topic_filter(
                paper,
                topic,
                keywords=keywords,
                required_concepts=required_concepts,
                excluded_title_terms=excluded_title_terms,
                compiled_scope=compiled_scope,
            ),
        )
    )
    for stage, check in checks:
        passed, reason = check()
        if not passed:
            return False, stage, reason
        if stage == "topic_anchor_filter" and any(
            marker in reason for marker in _LOW_CONFIDENCE_PASS_REASONS
        ):
            # WHY: 语言护栏/宽松锚点只是允许进入语义筛选，不等于已经证实相关。
            paper["_anchor_low_confidence"] = True
            paper["_anchor_pass_reason"] = reason
    return True, "hard_filters", "通过全部确定性硬过滤"


def compute_protocol_relevance_score(
    paper: Dict[str, Any],
    topic: str,
    screening_protocol: Dict[str, Any] | None,
    language: str = "",
    topic_synonyms: Sequence[str] | None = None,
) -> float:
    """把协议软条件与动态路线覆盖加入通用相关性分数。"""
    haystack = " ".join([
        str(paper.get("title") or ""), str(paper.get("abstract") or ""),
        str(paper.get("venue") or ""),
    ]).lower()
    base = compute_relevance_score(
        paper, topic, topic_synonyms=topic_synonyms
    )
    soft = _concept_coverage(
        (screening_protocol or {}).get("soft_include_criteria") or [],
        haystack,
        str(paper.get("title") or ""),
        language,
    )
    route_id, route_score = _best_protocol_route(paper, screening_protocol)
    if route_id:
        paper["_protocol_route_id"] = route_id
    return max(0.0, min(1.0, base * 0.65 + soft * 0.20 + min(1.0, route_score) * 0.15))


def _record_filter_diagnostic(
    diagnostics: Dict[str, Any] | None,
    *,
    stage: str,
    paper: Dict[str, Any],
    reason: str,
) -> None:
    if diagnostics is None:
        return
    diagnostics["filtered_count"] = int(diagnostics.get("filtered_count") or 0) + 1
    by_stage = diagnostics.setdefault("filtered_by_stage", {})
    by_stage[stage] = int(by_stage.get(stage) or 0) + 1
    # 阶段粒度不足以定位问题：同一阶段下"哪一组概念/哪一条排除词"筛掉了多少篇，
    # 只有按原因聚合才看得出来。
    by_reason = diagnostics.setdefault("filter_reasons", {})
    by_reason[reason] = int(by_reason.get(reason) or 0) + 1
    examples = diagnostics.setdefault("filtered_examples", [])
    if len(examples) < 12:
        examples.append({
            "paper_id": paper.get("paper_id"),
            "title": paper.get("title"),
            "stage": stage,
            "reason": reason,
        })


def _meaningful_filter_keywords(keywords: Sequence[str], topic: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in [topic, *keywords]:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        key = text.casefold()
        if len(text) < 2 or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def compute_quality_score(paper: Dict[str, Any], current_year: int | None = None) -> float:
    """依据出版级别（顶会顶刊最高、已发表高于预印本）、元数据完整度、时效性和引用量计算质量分。"""
    current_year = int(current_year or datetime.now().year)
    score = 0.0

    # 1. 出版层级与同行评审 (最高 0.35: 顶会顶刊 0.35, 普通已发表 0.25, 预印本 0.10, 未知 0.02)
    venue = str(paper.get("venue") or "").strip()
    doi = str(paper.get("doi") or "").strip()
    source = str(paper.get("source") or "").strip()
    tier, venue_score = _classify_venue_tier(venue, doi=doi, source=source)
    score += venue_score

    # 2. 摘要完整度 (0.22)
    score += 0.22 if str(paper.get("abstract") or "").strip() else 0.0

    # 3. 作者列表 (0.10)
    score += 0.10 if paper.get("authors") else 0.0

    # 4. 标识符 / 可溯源链接 (0.08)
    score += 0.08 if paper.get("doi") or paper.get("arxiv_id") or paper.get("url") else 0.0

    # 5. 时效性 (0.15)
    year = _safe_int(paper.get("year"))
    if year:
        age = max(0, current_year - year)
        score += max(0.0, 0.15 - min(age, 15) * 0.01)

    # 引用数可能来自不同来源，口径和覆盖范围不可比；保留元数据但不直接评分。
    return round(max(0.0, min(1.0, score)), 4)


def rank_papers(
    papers: List[Dict[str, Any]],
    topic: str,
    top_k: int = 20,
    keywords: Sequence[str] | None = None,
    required_concepts: Sequence[Sequence[str]] | None = None,
    excluded_title_terms: Sequence[str] | None = None,
    scope: Dict[str, Any] | None = None,
    search_branches: Sequence[Dict[str, Any]] | None = None,
    research_mode: str = "",
    current_year: int | None = None,
    screening_protocol: Dict[str, Any] | None = None,
    filter_diagnostics: Dict[str, Any] | None = None,
    source_minimums: Dict[str, int] | None = None,
    branch_minimums: Dict[str, int] | None = None,
    language_branch: str = "",
    start_year: int | None = None,
    end_year: int | None = None,
    reserve_k: int = 0,
    compiled_scope: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """执行确定性硬边界、相关性/质量评分、排序和来源软配额。

    Args:
        reserve_k: 在 ``top_k`` 主窗口之外额外保留的规则合格尾部数量。
            尾部论文打 ``_rule_screened_reserve`` 标记，只作为下游 LLM
            语义重排的加深筛选材料，不参与来源/分支软配额竞争。
    """
    top_k = max(0, int(top_k))
    reserve_k = max(0, int(reserve_k))
    if compiled_scope is None:
        compiled_scope = compile_scope(
            selected_scope=scope,
            screening_protocol=screening_protocol,
            required_concepts=required_concepts,
            topic=topic,
            search_branches=search_branches,
            excluded_title_terms=excluded_title_terms,
            research_mode=research_mode,
        )
    topic_synonyms = _extract_topic_synonyms(topic, keywords)
    scored: list[tuple[int, Dict[str, Any]]] = []
    for index, source_paper in enumerate(papers):
        paper = dict(source_paper)
        passed, stage, reason = evaluate_paper_hard_filters(
            paper,
            topic=topic,
            keywords=keywords,
            required_concepts=required_concepts,
            excluded_title_terms=excluded_title_terms,
            scope=scope,
            search_branches=search_branches,
            research_mode=research_mode,
            screening_protocol=screening_protocol,
            language_branch=language_branch,
            topic_synonyms=topic_synonyms,
            compiled_scope=compiled_scope,
        )
        if not passed:
            _record_filter_diagnostic(
                filter_diagnostics, stage=stage, paper=paper, reason=reason,
            )
            continue

        relevance = (
            compute_protocol_relevance_score(
                paper, topic, screening_protocol, language_branch,
                topic_synonyms=topic_synonyms,
            )
            if screening_protocol else
            compute_relevance_score(
                paper, topic, required_concepts,
                topic_synonyms=topic_synonyms,
            )
        )
        quality = compute_quality_score(paper, current_year)
        paper["_relevance_score"] = relevance
        paper["_quality_score"] = quality
        paper["_rank_score"] = round(
            relevance * (PROTOCOL_RELEVANCE_WEIGHT if screening_protocol else LEGACY_RELEVANCE_WEIGHT)
            + quality * (PROTOCOL_QUALITY_WEIGHT if screening_protocol else LEGACY_QUALITY_WEIGHT),
            4,
        )
        scored.append((index, paper))

    scored.sort(
        key=lambda item: (
            item[1].get("_rank_score", 0),
            item[1].get("_relevance_score", 0),
            item[1].get("_quality_score", 0),
            _safe_int(item[1].get("year")),
            -item[0],
        ),
        reverse=True,
    )
    result = [paper for _, paper in scored[:top_k]]

    # 年份多样性只在请求明确的时间窗内生效，并按窗口中的年份数动态选择。
    diversity_start = start_year
    diversity_end = end_year
    if diversity_start is None or diversity_end is None:
        time_window = (screening_protocol or {}).get("time_window") or {}
        diversity_start = diversity_start or time_window.get("start_year")
        diversity_end = diversity_end or time_window.get("end_year")
    try:
        diversity_start = int(diversity_start) if diversity_start is not None else None
        diversity_end = int(diversity_end) if diversity_end is not None else None
    except (TypeError, ValueError):
        diversity_start = diversity_end = None
    if diversity_start is not None and diversity_end is not None and diversity_start <= diversity_end:
        window_years = list(range(diversity_start, diversity_end + 1))
        active_years = [year for year in window_years if any(
            _safe_int(paper.get("year")) == year for _, paper in scored
        )]
        if top_k >= len(active_years) and len(active_years) >= 2:
            selected: list[Dict[str, Any]] = []
            selected_ids: set[str] = set()
            for year in active_years:
                for paper in (paper for _, paper in scored if _safe_int(paper.get("year")) == year):
                    paper_id = str(paper.get("paper_id") or "")
                    if paper_id and paper_id not in selected_ids:
                        selected.append(paper)
                        selected_ids.add(paper_id)
                        break
            for _, paper in scored:
                paper_id = str(paper.get("paper_id") or "")
                if paper_id and paper_id not in selected_ids:
                    selected.append(paper)
                    selected_ids.add(paper_id)
                if len(selected) >= top_k:
                    break
            result = selected[:top_k]

    if source_minimums and top_k:
        selected_ids = {str(paper.get("paper_id") or "") for paper in result}
        for source, minimum in source_minimums.items():
            source_name = str(source).strip().lower()
            quota = max(0, min(int(minimum or 0), top_k))
            selected_count = sum(
                str(paper.get("source") or "").strip().lower() == source_name
                for paper in result
            )
            if selected_count >= quota:
                continue
            reserves = [
                paper for _, paper in scored[top_k:]
                if str(paper.get("source") or "").strip().lower() == source_name
                and str(paper.get("paper_id") or "") not in selected_ids
            ]
            for reserve in reserves[: quota - selected_count]:
                replacement_index = next(
                    (
                        idx for idx in range(len(result) - 1, -1, -1)
                        if str(result[idx].get("source") or "").strip().lower() != source_name
                    ),
                    None,
                )
                if replacement_index is None:
                    break
                selected_ids.discard(str(result[replacement_index].get("paper_id") or ""))
                result[replacement_index] = reserve
                selected_ids.add(str(reserve.get("paper_id") or ""))
            result.sort(key=lambda paper: paper.get("_rank_score", 0), reverse=True)
        if filter_diagnostics is not None:
            filter_diagnostics["source_minimums"] = {
                str(source): int(minimum or 0)
                for source, minimum in source_minimums.items()
            }

    # 定向恢复召回的软配额。只保证这些论文能进入 top_k 视野，不放宽任何
    # 硬过滤：它们必须已经通过 scored 阶段的全部边界与语义筛选。
    if branch_minimums and top_k:
        selected_ids = {str(paper.get("paper_id") or "") for paper in result}
        for branch, minimum in branch_minimums.items():
            branch_name = str(branch).strip()
            quota = max(0, min(int(minimum or 0), top_k))
            if not branch_name or not quota:
                continue

            def _in_branch(paper: Dict[str, Any]) -> bool:
                return branch_name in {
                    str(item).strip()
                    for item in paper.get("_search_branches") or []
                }

            selected_count = sum(_in_branch(paper) for paper in result)
            if selected_count >= quota:
                continue
            reserves = [
                paper for _, paper in scored[top_k:]
                if _in_branch(paper)
                and str(paper.get("paper_id") or "") not in selected_ids
            ]
            for reserve in reserves[: quota - selected_count]:
                replacement_index = next(
                    (
                        idx for idx in range(len(result) - 1, -1, -1)
                        if not _in_branch(result[idx])
                    ),
                    None,
                )
                if replacement_index is None:
                    break
                selected_ids.discard(str(result[replacement_index].get("paper_id") or ""))
                result[replacement_index] = reserve
                selected_ids.add(str(reserve.get("paper_id") or ""))
            result.sort(key=lambda paper: paper.get("_rank_score", 0), reverse=True)
        if filter_diagnostics is not None:
            filter_diagnostics["branch_minimums"] = {
                str(branch): int(minimum or 0)
                for branch, minimum in branch_minimums.items()
            }
    primary_count = len(result)
    reserve_tail: List[Dict[str, Any]] = []
    if reserve_k and top_k:
        # WHY: LLM 语义重排只对本函数的输出打分，尾部此前被直接丢弃；排除率
        # 偏高时（实测 64 篇中排除 34 篇）重排的回填池结构性为空，引用缺口
        # 再也补不回来。这段尾部只提供"继续向深处筛"的材料，不放宽任何硬过滤。
        selected_ids = {str(paper.get("paper_id") or "") for paper in result}
        for _, paper in scored[top_k:]:
            if len(reserve_tail) >= reserve_k:
                break
            paper_id = str(paper.get("paper_id") or "")
            if paper_id and paper_id in selected_ids:
                continue
            paper["_rule_screened_reserve"] = True
            reserve_tail.append(paper)
            if paper_id:
                selected_ids.add(paper_id)
        result = result + reserve_tail

    if filter_diagnostics is not None:
        filter_diagnostics["passed_hard_filters"] = len(scored)
        filter_diagnostics["selected_count"] = primary_count
        filter_diagnostics["reserve_selected_count"] = len(reserve_tail)
        filter_diagnostics["truncated_by_top_k"] = max(0, len(scored) - len(result))
    return result


def explain_ranking_reason(paper: Dict[str, Any], topic: str) -> str:
    """生成论文入选理由的简短说明。"""
    score = paper.get("_rank_score", 0)
    rel = paper.get("_relevance_score", 0)
    qual = paper.get("_quality_score", 0)
    return (
        f"综合得分 {score:.2f}（相关性 {rel:.2f}，质量 {qual:.2f}）；"
        f"与主题「{topic}」高度相关。"
    )


def deduplicate_and_rank(
    papers: List[Dict[str, Any]],
    topic: str,
    top_k: int = 20,
    keywords: Sequence[str] | None = None,
    required_concepts: Sequence[Sequence[str]] | None = None,
    excluded_title_terms: Sequence[str] | None = None,
    scope: Dict[str, Any] | None = None,
    search_branches: Sequence[Dict[str, Any]] | None = None,
    research_mode: str = "",
    current_year: int | None = None,
    screening_protocol: Dict[str, Any] | None = None,
    filter_diagnostics: Dict[str, Any] | None = None,
    source_minimums: Dict[str, int] | None = None,
    branch_minimums: Dict[str, int] | None = None,
    language_branch: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    reserve_k: int = 0,
    compiled_scope: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """去重 + 排序一站式入口。

    Args:
        language_branch: ``"zh"`` 或 ``"en"``，指定后使用语言感知过滤和评分。
        reserve_k: 透传给 ``rank_papers``，在 ``top_k`` 之外附加一段规则合格
            的后备尾部，供 LLM 语义重排在高排除率下继续加深筛选。
    """
    deduped = deduplicate_papers(papers)
    result = rank_papers(
        deduped,
        topic,
        top_k=top_k,
        keywords=keywords,
        required_concepts=required_concepts,
        excluded_title_terms=excluded_title_terms,
        scope=scope,
        search_branches=search_branches,
        research_mode=research_mode,
        current_year=current_year,
        screening_protocol=screening_protocol,
        filter_diagnostics=filter_diagnostics,
        source_minimums=source_minimums,
        branch_minimums=branch_minimums,
        language_branch=language_branch,
        start_year=start_year,
        end_year=end_year,
        reserve_k=reserve_k,
        compiled_scope=compiled_scope,
    )
    if filter_diagnostics is not None:
        filter_diagnostics["pre_dedup_count"] = len(papers)
        filter_diagnostics["deduplicated_count"] = len(deduped)
        filter_diagnostics["duplicate_removed_count"] = len(papers) - len(deduped)
    return result


