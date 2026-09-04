"""文件与路径工具。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

from app.core.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在，不存在则创建。

    Args:
        path: 目录路径。

    Returns:
        Path 对象。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_filename(name: str, max_len: int = 128) -> str:
    """将任意字符串安全的文件名。

    移除或替换不可用于文件名的字符，截断过长部分。

    Args:
        name: 原始名称。
        max_len: 最大长度。

    Returns:
        安全的文件名（不含扩展名）。
    """
    if not name:
        return "untitled"
    # 替换 Windows/Linux 不允许的字符
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    # 移除控制字符
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    # 合并连续空格
    name = re.sub(r"\s+", " ", name).strip()
    # 截断
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name or "untitled"


def get_file_size_mb(path: Union[str, Path]) -> float:
    """获取文件大小（MB）。"""
    p = Path(path)
    if not p.exists():
        return 0.0
    return p.stat().st_size / (1024 * 1024)
