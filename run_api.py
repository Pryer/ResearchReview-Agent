"""FastAPI 服务启动入口。

运行方式::

    python run_api.py

或在项目根目录::

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import uvicorn

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def main() -> None:
    """启动 FastAPI 服务。"""
    logger.info("Starting API server at %s:%s", settings.app_host, settings.app_port)
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level="info" if not settings.app_debug else "debug",
    )


if __name__ == "__main__":
    main()
