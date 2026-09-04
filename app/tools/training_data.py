"""从已核验声明或字段证据构造训练候选数据。

本模块只负责数据集派生，不参与在线声明核验。自动标签用于冷启动，正式训练前
仍需人工抽检或复标。
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_qlora_records(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """将验证报告转换为三分类训练候选记录。"""
    records: List[Dict[str, Any]] = []
    for claim in report.get("claims", []):
        if not claim.get("factual") or claim.get("support_status") == "not_applicable":
            continue
        evidence = "\n".join(
            item.get("text", "") for item in claim.get("evidence_snippets", [])
        )
        records.append({
            "instruction": "判断论文证据对生成主张的支持程度。",
            "claim": claim.get("sentence", ""),
            "evidence": evidence,
            "label": claim.get("support_status"),
            "issues": claim.get("issues", []),
        })
    return records


def build_controlled_training_records(
    paper_cards: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """从字段—证据映射构造可控的三分类训练候选样本。"""
    records: List[Dict[str, Any]] = []
    field_names = ("research_problem", "method", "results", "limitations")
    card_evidence: Dict[str, Dict[str, str]] = {}

    for card in paper_cards:
        paper_id = str(card.get("paper_id") or "")
        span_map = {
            str(span.get("evidence_id") or ""): str(span.get("text") or "")
            for span in (card.get("evidence_spans") or [])
            if span.get("evidence_id") and span.get("text")
        }
        field_map: Dict[str, str] = {}
        for field in field_names:
            value = card.get(field)
            claim = (
                "；".join(str(item) for item in value if item)
                if isinstance(value, list) else str(value or "")
            )
            evidence_ids = (card.get("field_evidence") or {}).get(field, [])
            evidence = "\n".join(span_map[eid] for eid in evidence_ids if eid in span_map)
            if not claim or not evidence:
                continue
            field_map[field] = evidence
            records.append({
                "instruction": "判断论文证据对生成主张的支持程度。",
                "paper_id": paper_id,
                "field": field,
                "claim": claim,
                "evidence": evidence,
                "label": "supported",
                "mutation_type": "none",
                "label_source": "controlled_generation_requires_review",
            })
            records.append({
                "instruction": "判断论文证据对生成主张的支持程度。",
                "paper_id": paper_id,
                "field": field,
                "claim": f"该论文首次且显著地表明：{claim}",
                "evidence": evidence,
                "label": "partially_supported",
                "mutation_type": "overstated_language",
                "label_source": "controlled_generation_requires_review",
            })
        card_evidence[paper_id] = field_map

    paper_ids = list(card_evidence)
    for index, card in enumerate(paper_cards):
        if len(paper_ids) < 2:
            break
        paper_id = str(card.get("paper_id") or "")
        negative_id = paper_ids[(index + 1) % len(paper_ids)]
        for field, wrong_evidence in card_evidence.get(negative_id, {}).items():
            value = card.get(field)
            claim = (
                "；".join(str(item) for item in value if item)
                if isinstance(value, list) else str(value or "")
            )
            if not claim:
                continue
            records.append({
                "instruction": "判断论文证据对生成主张的支持程度。",
                "paper_id": paper_id,
                "evidence_paper_id": negative_id,
                "field": field,
                "claim": claim,
                "evidence": wrong_evidence,
                "label": "unsupported",
                "mutation_type": "cross_paper_evidence_swap",
                "label_source": "controlled_generation_requires_review",
            })
            break
    return records
