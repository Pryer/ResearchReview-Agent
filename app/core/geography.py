"""研究地域证据判定；禁止以论文语言替代国内外分类。"""

from __future__ import annotations

from typing import Any, Iterable

_GEO_FIELDS = (
    "country", "region", "affiliation_country", "author_countries",
    "study_country", "research_region", "sample_country", "sample_region",
)
_CHINA_MARKERS = {
    "china", "pr china", "people's republic of china", "中国", "中国大陆",
    "mainland china", "hong kong", "香港", "macau", "澳门", "taiwan", "台湾",
}


def geographic_bucket(paper: dict[str, Any]) -> str | None:
    """返回 china/other；只有显式地域元数据才能形成分类。"""
    values = [
        token for field in _GEO_FIELDS for token in _flatten(paper.get(field)) if token
    ]
    if not values:
        return None
    lowered = {value.strip().lower() for value in values if value.strip()}
    if any(
        marker == value or marker in value
        for value in lowered
        for marker in _CHINA_MARKERS
    ):
        return "china"
    broad = {
        "asia", "asian", "europe", "european", "global", "international",
        "亚洲", "欧洲", "全球", "国际",
    }
    if lowered and not lowered.issubset(broad):
        return "other"
    return None


def has_reliable_geographic_comparison(papers: Iterable[dict[str, Any]]) -> bool:
    """仅当论文池同时具有可靠的中国与其他地域证据时返回 True。"""
    buckets = {bucket for paper in papers if (bucket := geographic_bucket(paper))}
    return {"china", "other"}.issubset(buckets)


def _flatten(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(item) for item in value.values() if item is not None]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]
