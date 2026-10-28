"""字段支持资格：相似度只召回候选，不能证明事实成立。"""
from __future__ import annotations

import hashlib
import re
from typing import Any

EVIDENCE_POLICY_VERSION = "independent-literal-evidence-v1"


def normalized(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def literal_support(claim: str, source: str, *, fragment: bool = False) -> bool:
    value, text = normalized(claim), normalized(source)
    if not value or not text:
        return False
    if value == text:
        return True
    if not fragment:
        return False
    # WHY: 数据集/指标短标签可逐字定位，但省去否定或条件时不能取得字段支持资格。
    qualifiers = r"\b(?:not|no|never|without|neither|unless|only|if|however|but)\b|n't|未|不|无|仅|只有|但是|然而"
    if re.findall(qualifiers, value) != re.findall(qualifiers, text):
        return False
    return re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text) is not None


def independent_spans(card: dict[str, Any]) -> list[dict[str, Any]]:
    """只接纳独立的来源记录；兼容带位置的旧记录，不从抽取字段补造原文。"""
    result = []
    for span in card.get("evidence_spans") or []:
        if (
            not isinstance(span, dict)
            or not span.get("evidence_id")
            or not str(span.get("text") or "").strip()
            or span.get("source_type") not in {"abstract", "full_text", "table"}
            or span.get("source_status") == "source_unverified"
        ):
            continue
        start, end = span.get("char_start"), span.get("char_end")
        if start is not None or end is not None:
            if (
                type(start) is not int or type(end) is not int
                or not 0 <= start < end or end - start != len(span["text"])
            ):
                continue
        elif not span.get("source_url") and not (type(span.get("page")) is int and span["page"] > 0):
            continue
        text_hash = hashlib.sha256(span["text"].encode("utf-8")).hexdigest()
        if span.get("source_text_hash") and span["source_text_hash"] != text_hash:
            continue
        result.append(dict(span))
    return result


def authorized_field_claim(card: dict[str, Any], claim: dict[str, Any]) -> bool:
    if not isinstance(claim, dict) or not claim.get("explicitly_reported"):
        return False
    evidence_id = claim.get("evidence_id")
    field = next((field for field, claims in (card.get("field_claims") or {}).items() if claim in claims), None)
    for span in independent_spans(card):
        if span.get("evidence_id") == evidence_id:
            source = span["text"]
            if claim.get("source_text") and normalized(claim["source_text"]) != normalized(source):
                return False
            # WHY: source_section 是原文章节，不是 PaperCard 字段名。
            return literal_support(claim.get("claim"), source, fragment=field in {"dataset", "metrics"})
    return False
