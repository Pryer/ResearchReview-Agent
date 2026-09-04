"""分支分数归一化、配额合并与跨语言去重。

中英文分支独立排序后的论文需要通过百分位归一化对齐分数，
再按可配置配额合并为最终论文池。

提供两阶段去重：
- ``identifier_level_cross_dedup`` — 评分前，基于 DOI/ID 精确去重
- ``global_cross_language_deduplicate`` — 合并后，基于 identity key 去重，
  胜者按元数据完备度选择，与语言和分数归一化方式无关
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

from app.core.logger import get_logger

logger = get_logger(__name__)

# 小分支阈值：低于此数量的分支不使用百分位归一化
_SMALL_BRANCH_THRESHOLD = 10


def _identity_key_for_paper(paper: Dict[str, Any]) -> str:
    """稳定的论文身份标识（DOI → arXiv ID → 标准化标题 → paper_id）。"""
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    arxiv_id = (paper.get("arxiv_id") or "").strip().lower()
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    from app.utils.deduplicate import normalize_title
    norm_title = normalize_title(str(paper.get("title") or ""))
    if norm_title:
        return f"title:{norm_title}"
    paper_id = str(paper.get("paper_id") or "").strip()
    if paper_id:
        return f"paper_id:{paper_id}"
    stable_payload = json.dumps(
        {
            key: value for key, value in paper.items()
            if not str(key).startswith("_")
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()[:16]
    return f"anonymous:{digest}"


# ============================================================
# 第一阶段去重：硬过滤后、评分前（仅精确 identifier 匹配）
# ============================================================
def identifier_level_cross_dedup(
    zh_papers: List[Dict[str, Any]],
    en_papers: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """硬过滤后、评分前的轻量跨语言去重。

    仅使用高置信度字段：DOI、arXiv ID、OpenAlex ID。
    不做标题语义匹配，避免评分前误删不同论文。

    Returns:
        ``(clean_zh, clean_en)`` 元组。
    """
    winners: Dict[str, Tuple[Dict[str, Any], str]] = {}
    anonymous: List[Tuple[Dict[str, Any], str]] = []
    for branch, papers in (("zh", zh_papers), ("en", en_papers)):
        for paper in papers:
            identifier = _extract_strong_identifier(paper)
            if not identifier:
                anonymous.append((paper, branch))
                continue
            current = winners.get(identifier)
            if current is None or _metadata_completeness(paper) > _metadata_completeness(current[0]):
                winners[identifier] = (paper, branch)

    selected = [*winners.values(), *anonymous]
    clean_zh = [paper for paper, branch in selected if branch == "zh"]
    clean_en = [paper for paper, branch in selected if branch == "en"]
    removed = len(zh_papers) + len(en_papers) - len(selected)
    if removed:
        logger.info("Identifier-level cross dedup: removed %d duplicate record(s)", removed)
    return clean_zh, clean_en


def _metadata_completeness(paper: Dict[str, Any]) -> Tuple[int, float]:
    """以字段完整度和已有质量分选择跨语言重复记录，不偏好某种语言。"""
    fields = (
        "title", "authors", "year", "venue", "abstract", "doi", "arxiv_id",
        "url", "pdf_url", "citation_count",
    )
    complete = sum(paper.get(field) not in (None, "", [], {}) for field in fields)
    quality = float(paper.get("_quality_score") or paper.get("_rank_score") or 0.0)
    return complete, quality


def _extract_strong_identifier(paper: Dict[str, Any]) -> str | None:
    """提取高置信度论文标识符，规则与跨源去重共用（``utils.deduplicate``）。"""
    from app.utils.deduplicate import strong_paper_identifier
    return strong_paper_identifier(paper)


# ============================================================
# 分数归一化（含小分支保护）
# ============================================================
def normalize_scores_by_percentile(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """在分支内将相关性分与质量分分别转换为百分位排名。

    小分支（< 10 篇）使用校准原始分而非百分位，
    避免"矮子里拔将军"导致低质论文获得高排名。

    质量分同样做分支内归一化：绝对质量分依赖元数据完备性（摘要完整度、
    引用量等），中英文数据库的元数据密度不同，跨分支直接比较会系统性
    压低元数据稀疏一侧（典型如 CNKI 缺摘要/引用量）的论文顺位。

    写入 ``_branch_percentile``、``_quality_percentile`` 和
    ``_branch_reliability`` 字段。
    """
    if not papers:
        return papers

    branch_size = len(papers)
    reliability = min(branch_size / _SMALL_BRANCH_THRESHOLD, 1.0)

    def _assign_percentiles(source_field: str, output_field: str) -> None:
        sorted_papers = sorted(
            papers,
            key=lambda p: float(p.get(source_field, 0)),
        )
        for index, paper in enumerate(sorted_papers):
            if branch_size <= 1:
                percentile = 1.0
            else:
                percentile = index / (branch_size - 1)
            paper[output_field] = round(percentile, 4)

    _assign_percentiles("_branch_final_score", "_branch_percentile")
    _assign_percentiles("_quality_score", "_quality_percentile")
    for paper in papers:
        paper["_branch_reliability"] = round(reliability, 4)

    return papers


def _calibrated_raw_score(paper: Dict[str, Any], field: str = "_branch_final_score") -> float:
    """将原始分数校准到近似百分位空间（用于小分支）。"""
    raw = float(paper.get(field, 0))
    # 将 [0, 1] 原始分映射到 [0.15, 0.85] 的伪百分位
    return round(0.15 + raw * 0.70, 4)


# ============================================================
# 配额计算（目标比例 + 动态补位）
# ============================================================
def calculate_branch_targets(
    top_k: int,
    zh_ratio: float,
    zh_count: int,
    en_count: int,
    min_zh: int = 8,
    min_en: int = 12,
    *,
    allow_quota_transfer: bool = True,
) -> Tuple[int, int]:
    """计算中英文分支的目标数量（软配额）。

    优先满足目标比例；某一分支可用量不足时配额自动转移。
    无论最低配额如何配置，两个目标之和都不会超过 ``top_k``。

    Returns:
        ``(zh_target, en_target)``。
    """
    top_k = max(0, int(top_k))
    zh_count = max(0, int(zh_count))
    en_count = max(0, int(en_count))
    if top_k == 0 or zh_count + en_count == 0:
        return 0, 0

    capacity = min(top_k, zh_count + en_count)
    ratio = min(1.0, max(0.0, float(zh_ratio)))
    zh_ideal = round(capacity * ratio)

    # 当总容量足以同时满足最低配额时才应用最低配额。容量不足时最低配额
    # 自动退化为比例目标，避免出现 zh_target + en_target > top_k。
    min_zh = max(0, int(min_zh))
    min_en = max(0, int(min_en))
    if capacity >= min_zh + min_en:
        zh_ideal = min(max(zh_ideal, min_zh), capacity - min_en)
    elif zh_count and en_count and capacity >= 2:
        zh_ideal = min(max(zh_ideal, 1), capacity - 1)
    elif not zh_count:
        zh_ideal = 0
    elif not en_count:
        zh_ideal = capacity

    en_ideal = capacity - zh_ideal
    zh_target = min(zh_ideal, zh_count)
    en_target = min(en_ideal, en_count)

    # 异常低通过率或显式语言契约缺口时，禁止把缺失语言的名额静默
    # 转给另一分支；最终门禁必须看到真实缺口并触发补检索/降级。
    if not allow_quota_transfer:
        return zh_target, en_target

    # 某一分支不足时将剩余容量转移给另一分支。
    remaining = capacity - zh_target - en_target
    if remaining:
        if zh_target < zh_ideal:
            addition = min(remaining, en_count - en_target)
            en_target += addition
            remaining -= addition
        if remaining and en_target < en_ideal:
            addition = min(remaining, zh_count - zh_target)
            zh_target += addition
            remaining -= addition
        if remaining:
            addition = min(remaining, zh_count - zh_target)
            zh_target += addition
            remaining -= addition
        if remaining:
            en_target += min(remaining, en_count - en_target)

    return zh_target, en_target


def build_language_coverage_contract(
    required_reference_count: int,
    zh_ratio: float,
    *,
    min_zh: int,
    min_en: int,
    eligible_zh: int,
    eligible_en: int,
    affinity: str = "balanced",
) -> Dict[str, Any]:
    """构造贯穿检索、写作与最终门禁的双语覆盖契约。

    ``calculate_branch_targets`` 仍负责候选池的软配额；这里保存用户最终正文
    所需的最低语言覆盖，且不会因当前某一分支候选不足而静默转移。
    """
    required = max(0, int(required_reference_count))
    ratio = min(1.0, max(0.0, float(zh_ratio)))
    desired_zh = round(required * ratio)
    desired_en = required - desired_zh
    if required >= max(0, int(min_zh)) + max(0, int(min_en)):
        minimum_zh = max(0, int(min_zh))
        minimum_en = max(0, int(min_en))
    elif required >= 2:
        minimum_zh = 1
        minimum_en = 1
    else:
        minimum_zh = required if ratio >= 0.5 else 0
        minimum_en = required - minimum_zh
    eligible_zh = max(0, int(eligible_zh))
    eligible_en = max(0, int(eligible_en))
    deficits = {
        "zh": max(0, minimum_zh - eligible_zh),
        "en": max(0, minimum_en - eligible_en),
    }
    return {
        "enabled": required > 0,
        "affinity": str(affinity or "balanced"),
        "required_total": required,
        "desired_zh": desired_zh,
        "desired_en": desired_en,
        "minimum_zh": minimum_zh,
        "minimum_en": minimum_en,
        "eligible_zh": eligible_zh,
        "eligible_en": eligible_en,
        "deficits": deficits,
        "satisfied_at_screening": not any(deficits.values()),
    }


# ============================================================
# 第二阶段去重：合并阶段（identity key + score 比较）
# ============================================================
def global_cross_language_deduplicate(
    zh_candidates: List[Dict[str, Any]],
    en_candidates: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """跨语言去重：DOI / identity key 相同的论文保留元数据更完整者。

    胜者判据是 ``_metadata_completeness``（字段完整度 + 绝对质量分），
    不是 ``_merge_score``：merge_score 的质量分量已做分支内归一化，
    不再反映单条记录本身的完整程度；同一论文的双语版本应保留字段
    更全的记录，与语言无关。

    此阶段允许标准化标题作为 identity key，因为在合并后执行，
    不会影响评分阶段的独立性。

    Returns:
        ``(clean_zh, clean_en)`` 去重后的两个分支列表。
    """
    seen_keys: Dict[str, str] = {}

    clean_zh: List[Dict[str, Any]] = []
    clean_en: List[Dict[str, Any]] = []

    combined: List[Tuple[Dict[str, Any], str]] = [
        *[(p, "zh") for p in zh_candidates],
        *[(p, "en") for p in en_candidates],
    ]
    # 按元数据完备度降序处理：同键重复时先到的必然是更完整的记录，
    # 后续重复项直接跳过即可，不存在“替换已保留记录”的分支。
    combined.sort(
        key=lambda item: _metadata_completeness(item[0]),
        reverse=True,
    )

    for paper, branch in combined:
        key = _identity_key_for_paper(paper)
        if key in seen_keys:
            continue

        seen_keys[key] = branch

        if branch == "zh":
            clean_zh.append(paper)
        else:
            clean_en.append(paper)

    logger.info(
        "Cross-language dedup: zh %d→%d, en %d→%d",
        len(zh_candidates), len(clean_zh),
        len(en_candidates), len(clean_en),
    )
    return clean_zh, clean_en


# ============================================================
# 主入口
# ============================================================
def merge_language_branches(
    zh_ranked: List[Dict[str, Any]],
    en_ranked: List[Dict[str, Any]],
    top_k: int,
    zh_ratio: float = 0.40,
    min_zh: int = 8,
    min_en: int = 12,
    branch_candidate_multiplier: float = 1.5,
    *,
    allow_quota_transfer: bool = True,
) -> List[Dict[str, Any]]:
    """中英文分支合并主入口。

    流程：百分位归一化（含小分支保护）→ 计算配额 →
    跨语言去重 → 按配额选取 → 高质量补位。
    英文配额 = 总量 - 中文配额，由 ``zh_ratio`` 推导。

    Returns:
        合并后的最终论文列表（已按 ``_merge_score`` 降序排列）。
    """
    # 1. 分支内归一化（小分支使用校准原始分）
    zh_ranked = normalize_scores_by_percentile(zh_ranked)
    en_ranked = normalize_scores_by_percentile(en_ranked)

    # 2. 计算 merge_score（含分支可靠性衰减）
    #    相关性与质量分都在分支内归一化：绝对质量分依赖元数据完备性，
    #    跨分支直接比较会把元数据稀疏一侧的论文系统性压低。
    for branch_papers in (zh_ranked, en_ranked):
        for paper in branch_papers:
            reliability = float(paper.get("_branch_reliability", 1.0))
            adjusted_pct = (
                float(paper.get("_branch_percentile", 0.5)) * reliability
                + _calibrated_raw_score(paper) * (1.0 - reliability)
            )
            adjusted_quality = (
                float(paper.get("_quality_percentile", 0.5)) * reliability
                + _calibrated_raw_score(paper, "_quality_score") * (1.0 - reliability)
            )
            paper["_merge_score"] = round(
                adjusted_pct * 0.80 + adjusted_quality * 0.20, 4,
            )

    # 3. 分支内排序
    zh_ranked.sort(key=lambda p: p["_merge_score"], reverse=True)
    en_ranked.sort(key=lambda p: p["_merge_score"], reverse=True)

    # 4. 计算配额与候选池
    zh_target, en_target = calculate_branch_targets(
        top_k, zh_ratio, len(zh_ranked), len(en_ranked),
        min_zh=min_zh, min_en=min_en,
        allow_quota_transfer=allow_quota_transfer,
    )
    zh_pool_size = min(len(zh_ranked), max(zh_target, int(zh_target * branch_candidate_multiplier)))
    en_pool_size = min(len(en_ranked), max(en_target, int(en_target * branch_candidate_multiplier)))

    zh_candidates = zh_ranked[:zh_pool_size]
    en_candidates = en_ranked[:en_pool_size]

    # 5. 跨语言去重
    zh_candidates, en_candidates = global_cross_language_deduplicate(
        zh_candidates, en_candidates,
    )

    # 6. 按软配额选取
    selected_zh = zh_candidates[:zh_target]
    selected_en = en_candidates[:en_target]
    selected = selected_zh + selected_en

    # 7. 高质量补位：配额不足时由另一分支高分论文补齐。异常低通过率保护
    # 启用时禁止补位，否则刚保留的语言缺口会在此被再次静默填平。
    if allow_quota_transfer and len(selected) < top_k:
        selected_ids = {_identity_key_for_paper(p) for p in selected}
        remaining = [
            p for p in zh_candidates + en_candidates
            if _identity_key_for_paper(p) not in selected_ids
        ]
        remaining.sort(key=lambda p: p["_merge_score"], reverse=True)
        selected.extend(remaining[:top_k - len(selected)])

    # 8. 最终排序
    selected.sort(key=lambda p: p["_merge_score"], reverse=True)
    selected = selected[:top_k]

    logger.info(
        "Branch merge: zh_target=%d selected=%d, en_target=%d selected=%d, final=%d",
        zh_target, len(selected_zh), en_target, len(selected_en), len(selected),
    )
    return selected
