"""可选的 API Key 认证依赖。"""

from __future__ import annotations

import secrets
import ipaddress

from fastapi import Header, HTTPException

from app.core.config import get_settings


def validate_deployment_security() -> None:
    """阻止无认证服务监听非回环接口。"""
    settings = get_settings()
    host = settings.app_host.strip().lower()
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host == "localhost"
    if not is_loopback and not settings.app_api_key.strip():
        raise RuntimeError(
            "APP_HOST 指向非回环接口时必须配置 APP_API_KEY；"
            "本地开发请使用 127.0.0.1。"
        )


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """配置 APP_API_KEY 后保护业务与指标接口；本地未配置时保持兼容。"""
    expected = get_settings().app_api_key.strip()
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="无效或缺失的 X-API-Key")
