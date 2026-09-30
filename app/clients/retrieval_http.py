"""百炼原生模型接口的有界请求和响应校验。"""

from __future__ import annotations

import math
import time
from urllib.parse import urlparse

import requests


class RetrievalProviderError(RuntimeError):
    pass


def post_json(url, key, body, *, timeout, retries, session=None, should_cancel=None, kind="model"):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or not key:
        raise RetrievalProviderError("检索模型端点或密钥未正确配置")
    transport = session or requests
    for attempt in range(retries + 1):
        from app.agent.execution_budget import active_budget, check_execution, record_retrieval_request

        check_execution()
        if should_cancel and should_cancel():
            raise InterruptedError("检索模型调用已取消")
        active = active_budget()
        effective_timeout = min(timeout, max(0.001, active["deadline"] - time.monotonic())) if active else timeout
        record_retrieval_request(kind)
        try:
            response = transport.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body,
                timeout=effective_timeout,
            )
        except requests.RequestException as exc:
            if attempt == retries:
                raise RetrievalProviderError("检索模型网络请求失败") from exc
            delay = min(2 ** attempt, 8)
        else:
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise RetrievalProviderError("检索模型返回无效 JSON") from exc
                if not isinstance(payload, dict):
                    raise RetrievalProviderError("检索模型返回格式错误")
                return payload
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == retries:
                # WHY: 不记录响应体/请求头，避免 provider 回显私人论文或密钥。
                raise RetrievalProviderError(f"检索模型请求失败，HTTP {response.status_code}")
            raw_wait = response.headers.get("Retry-After", "")
            try:
                delay = min(max(float(raw_wait), 0), 15) if raw_wait else min(2 ** attempt, 8)
            except ValueError:
                delay = min(2 ** attempt, 8)
        if should_cancel and should_cancel():
            raise InterruptedError("检索模型调用已取消")
        time.sleep(delay)
    raise RetrievalProviderError("检索模型重试次数耗尽")


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RetrievalProviderError("检索模型返回非有限数值")
    return float(value)
