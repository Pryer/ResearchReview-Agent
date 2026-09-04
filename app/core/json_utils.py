"""公共 JSON 解析工具。

提供 ``parse_json_object`` — 一个对 LLM 输出容错的 JSON 解析函数，
依次尝试三种策略（直接解析 → 平衡括号法 → 去注释重试），
均失败则返回空字典，不抛异常。

用法示例::

    from app.core.json_utils import parse_json_object

    result = parse_json_object(llm_response_text)
"""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_json_comments(text: str) -> str:
    """去除 JSON 文本中的 ``//`` 行注释与 ``/* */`` 块注释。

    逐字符扫描并跟踪字符串字面量状态，字符串内部的 ``//``
    （如 URL ``https://…``）不会被误当作注释截断。
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    escape_next = False
    while i < n:
        ch = text[i]
        if escape_next:
            out.append(ch)
            escape_next = False
            i += 1
            continue
        if in_string:
            out.append(ch)
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            # 行注释：跳过到行尾（保留换行符本身）
            j = text.find("\n", i + 2)
            i = n if j == -1 else j
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            # 块注释：跳过到闭合 */
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def parse_json_object(text: str) -> dict[str, Any]:
    """对 LLM 输出进行健壮 JSON 解析，依次尝试三种策略。

    策略顺序：
    1. 直接 ``json.loads``（最快，适合格式规整的输出）
    2. **平衡括号法**：找到第一个 ``{``，逐字符跟踪深度，提取完整 JSON
       对象（处理嵌套结构 / 尾部多余文本）
    3. 去除行注释（``//``）和块注释（``/* */``）后重试

    Args:
        text: LLM 返回的原始文本，可能包含 markdown 代码块、注释、
              或推理模型的 ``reasoning_content`` 前缀。

    Returns:
        解析成功时返回字典；三种策略均失败时返回 ``{}``，不抛异常。
    """
    if not text:
        return {}

    text = str(text).strip()
    # 去除 markdown 代码块包裹（```json ... ``` 或 ``` ... ```）
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    # ── 策略 1：直接解析 ──────────────────────────────────────────────
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        # 输入本身是合法 JSON，但根节点类型不符；不能从数组内部擅自抽取
        # 第一个对象并改变调用方看到的语义。
        return {}
    except (json.JSONDecodeError, ValueError):
        pass

    # ── 策略 2：平衡括号法提取完整 JSON 对象 ─────────────────────────
    try:
        start = text.index("{")
        depth = 0
        in_string = False
        escape_next = False
        end = -1
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            result = json.loads(text[start:end])
            if isinstance(result, dict):
                return result
    except (ValueError, json.JSONDecodeError):
        pass

    # ── 策略 3：去除注释后重试 ────────────────────────────────────────
    try:
        cleaned = _strip_json_comments(text)
        result = json.loads(cleaned.strip())
        return result if isinstance(result, dict) else {}
    except (ValueError, json.JSONDecodeError):
        return {}


def parse_json_list(text: str) -> list[Any]:
    """对 LLM 输出进行健壮 JSON 数组解析。

    策略与 ``parse_json_object`` 类似，但提取 ``[...]`` 而非 ``{...}``。

    Returns:
        解析成功时返回列表；失败时返回 ``[]``，不抛异常。
    """
    if not text:
        return []

    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    # 策略1：直接解析
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, ValueError):
        pass

    # 策略2：平衡括号法提取数组
    try:
        start = text.index("[")
        depth = 0
        in_string = False
        escape_next = False
        end = -1
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            result = json.loads(text[start:end])
            return result if isinstance(result, list) else []
    except (ValueError, json.JSONDecodeError):
        pass

    return []
