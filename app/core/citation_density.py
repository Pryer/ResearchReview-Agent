# -*- coding: utf-8 -*-
"""引用密度与堆砌检测工具模块。"""

from __future__ import annotations

import re
from typing import Any, Dict, List

MAX_CITATIONS_PER_GROUP = 3


def detect_citation_dumps(
    text: str,
    max_per_group: int = MAX_CITATIONS_PER_GROUP,
) -> List[Dict[str, Any]]:
    """检测单处或连续连排引用超过阈值（默认 5 篇）的堆砌位置。

    支持检测两种形式：
    1. 单方括号内逗号分隔的密集引用，例如：[1, 2, 3, 4, 5, 6] 或 [p1, p2, p3, p4, p5, p6]
    2. 紧密连续相连的多个方括号引用，例如：[1][2][3][4][5][6]
    """
    if not text:
        return []

    dumps: List[Dict[str, Any]] = []

    # 1. 检测单方括号内多篇引用: [id1, id2, id3, ...]
    bracket_pattern = re.compile(r"\[([^[\]]+)\]")
    for match in bracket_pattern.finditer(text):
        content = match.group(1).strip()
        parts = [p.strip() for p in re.split(r"[,，;；、\s]+", content) if p.strip()]
        if len(parts) > max_per_group:
            dumps.append({
                "type": "single_bracket_dump",
                "start": match.start(),
                "end": match.end(),
                "matched_text": match.group(0),
                "citation_count": len(parts),
                "citations": parts,
            })

    # 2. 检测连续连排方括号: [id1][id2][id3][id4][id5][id6]。
    #    两支及以上连续括号、合计引用数超过阈值即视为堆砌——
    #    防止把 20 篇拆成 [1,2,3][4,5,6]… 逐组卡在单括号阈值之下规避检测。
    consecutive_pattern = re.compile(r"(?:\[[^[\]]+\]\s*){2,}")
    for match in consecutive_pattern.finditer(text):
        matched_str = match.group(0)
        sub_brackets = bracket_pattern.findall(matched_str)
        all_ids = []
        for sb in sub_brackets:
            all_ids.extend([p.strip() for p in re.split(r"[,，;；、\s]+", sb) if p.strip()])
        if len(all_ids) > max_per_group:
            if not any(d["start"] == match.start() and d["end"] == match.end() for d in dumps):
                dumps.append({
                    "type": "consecutive_brackets_dump",
                    "start": match.start(),
                    "end": match.end(),
                    "matched_text": matched_str.strip(),
                    "citation_count": len(all_ids),
                    "citations": all_ids,
                })

    return dumps


def break_citation_dumps(
    text: str,
    max_per_group: int = MAX_CITATIONS_PER_GROUP,
) -> str:
    """确定性拆散引用堆砌：每处只保留前 ``max_per_group`` 个引用，其余删除。

    被删掉的引用如实反映为正文唯一引用数下降，由引用数量校验和
    最终质量门禁报告缺口；不再用"拆成连续小组"的方式规避检测。
    """
    dumps = detect_citation_dumps(text, max_per_group=max_per_group)
    # 连排区间与其内部的单括号堆砌会同时命中（如 [1,2,3,4][5][6][7]），
    # 二者区间重叠；若各自用原始偏移量做二次切割，第二次替换会在已缩短的
    # 字符串上错位、截掉后续正文。先按包含关系去重：保留覆盖范围更大的
    # 外层区间，一次替换即已把内部括号一并收敛到阈值之内。
    unique_dumps: List[Dict[str, Any]] = []
    for dump in sorted(dumps, key=lambda d: (d["start"], -d["end"])):
        if any(
            kept["start"] <= dump["start"] and dump["end"] <= kept["end"]
            for kept in unique_dumps
        ):
            continue
        unique_dumps.append(dump)
    if not unique_dumps:
        return text
    result = str(text or "")
    for dump in sorted(unique_dumps, key=lambda d: d["start"], reverse=True):
        kept = dump["citations"][:max_per_group]
        replacement = "[" + ", ".join(kept) + "]"
        result = result[:dump["start"]] + replacement + result[dump["end"]:]
    return result


def has_citation_dumps(
    text: str,
    max_per_group: int = MAX_CITATIONS_PER_GROUP,
) -> bool:
    """快速判定文本中是否存在引用倾倒/堆砌。"""
    return bool(detect_citation_dumps(text, max_per_group=max_per_group))
