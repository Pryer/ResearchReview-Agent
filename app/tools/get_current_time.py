"""当前时间工具。

用于解析“近五年”“今年”“2022 年以后”等依赖运行时日期的请求。
时钟核心实现在 ``app.utils.date_utils.now_info``，本模块是面向
Agent 节点的薄工具封装。
"""

from __future__ import annotations

from typing import Any, Dict

from app.utils.date_utils import now_info


def get_current_time(timezone: str | None = None) -> Dict[str, Any]:
    """返回当前时间的结构化信息。

    Args:
        timezone: IANA 时区名，如 Asia/Shanghai。为空时使用系统本地时区。

    Returns:
        包含 iso/date/year/timezone 的字典。
    """
    return now_info(timezone)
