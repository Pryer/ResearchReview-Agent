"""日志模块。

提供统一的日志配置，支持控制台输出和文件滚动日志。
通过 ``get_logger(name)`` 在任意模块获取以模块名命名的 logger。
"""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict

from app.core.config import get_settings

# ---------- 全局配置 ----------
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3

# 已获取 logger 的缓存（仅作查找加速；handler 配置由锁保证只执行一次）
_configured_loggers: Dict[str, logging.Logger] = {}
_setup_lock = threading.Lock()
_root_initialized = False

# 日志根目录
_LOG_DIR = Path("logs")


def _setup_root_logger() -> None:
    """配置根日志器（仅调用一次）。"""
    settings = get_settings()
    log_level = logging.DEBUG if settings.app_debug else logging.INFO

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    # 根 logger
    root = logging.getLogger("research_review_agent")
    root.setLevel(log_level)
    root.addHandler(console_handler)

    # 文件 handler
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        _LOG_DIR / "app.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """获取以 ``name`` 命名的 logger。

    Args:
        name: 通常传入 ``__name__``，用于标识日志来源模块。

    Returns:
        配置好的 ``logging.Logger`` 实例。
    """
    global _root_initialized
    # 双重检查锁：多线程首次并发调用时根日志器也只配置一次，
    # 避免 handler 被重复添加导致日志重复输出。
    if not _root_initialized:
        with _setup_lock:
            if not _root_initialized:
                _setup_root_logger()
                _root_initialized = True

    logger = logging.getLogger(f"research_review_agent.{name}")
    _configured_loggers[name] = logger
    return logger
