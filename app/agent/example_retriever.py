"""人工审核回归案例的轻量检索器。

案例只用于帮助 LLM 理解关系结构，不能向当前请求注入案例实体。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_CASE_PATH = Path(__file__).with_name("regression_cases.jsonl")
_STOPWORDS = {"调研", "研究", "论文", "分析", "识别", "使用", "进行", "基于", "相关"}


def retrieve_semantic_examples(
    message: str,
    top_k: int = 5,
    minimum_score: float = 0.12,
    case_path: Path | None = None,
) -> list[dict[str, Any]]:
    """使用字符 n-gram 与英文词项相似度检索少量人工标注案例。"""
    query_tokens = _tokens(message)
    if not query_tokens:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for case in _load_cases(case_path or _CASE_PATH):
        case_tokens = _tokens(str(case.get("input") or ""))
        union = query_tokens | case_tokens
        score = len(query_tokens & case_tokens) / len(union) if union else 0.0
        if score >= minimum_score:
            scored.append((score, case))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "case_id": case.get("case_id"),
            "input": case.get("input"),
            "expected_frame": case.get("expected_frame"),
            "error_tags": case.get("error_tags") or [],
            "similarity": round(score, 3),
        }
        for score, case in scored[: max(0, top_k)]
    ]


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _tokens(text: str) -> set[str]:
    lowered = str(text or "").lower()
    tokens = {
        token for token in re.findall(r"[a-z][a-z0-9_-]{2,}", lowered)
        if token not in _STOPWORDS
    }
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
        tokens.update(
            sequence[index:index + 2]
            for index in range(len(sequence) - 1)
            if sequence[index:index + 2] not in _STOPWORDS
        )
    return tokens
