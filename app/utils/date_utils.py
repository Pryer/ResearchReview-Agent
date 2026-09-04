"""日期与年份工具。

MVP 阶段避免硬编码当前年份（因为工作流脚本禁止 Date.now），
所有需要当前年份的地方由配置或调用方传入。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Tuple
from zoneinfo import ZoneInfo


def now_info(timezone: str | None = None) -> Dict[str, Any]:
    """返回当前时间的结构化信息（含可选 IANA 时区）。

    时钟核心实现放在 utils 层；``app.tools.get_current_time`` 是面向
    Agent 节点的薄工具封装，依赖方向保持 utils ← tools。
    """
    if timezone:
        now = datetime.now(ZoneInfo(timezone))
    else:
        now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "timezone": str(now.tzinfo),
    }


def current_year() -> int:
    """返回运行时当前年份。"""
    return int(now_info()["year"])


def default_year_range(current_year: int, offset: int = 3) -> Tuple[int, int]:
    """计算默认年份范围。

    Args:
        current_year: 当前年份（由调用方提供）。
        offset: 向前回溯年数（含当前年）。

    Returns:
        (start_year, end_year) 元组。
    """
    start = max(current_year - offset + 1, 1900)
    return start, current_year


def parse_year(text: str) -> int | None:
    """从文本中提取 4 位年份。

    Args:
        text: 含年份的文本。

    Returns:
        找到的年份，或 None。
    """
    import re

    m = re.search(r"(19|20)\d{2}", text)
    return int(m.group()) if m else None


def format_datetime(dt: datetime) -> str:
    """格式化为 ISO 格式字符串。"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")
