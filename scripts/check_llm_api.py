"""测试项目根目录 .env 中的 OpenAI 兼容 LLM API 配置是否可用。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_env(path: Path) -> dict[str, str]:
    """读取简单 KEY=VALUE 格式的环境配置，不输出任何密钥。"""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def chat_endpoint(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def test_target(
    name: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
) -> bool:
    print(f"\n[{name}]")
    print(f"Base URL: {base_url or '(未配置)'}")
    print(f"Model: {model or '(未配置)'}")

    missing = [
        field
        for field, value in (
            ("API_KEY", api_key),
            ("BASE_URL", base_url),
            ("MODEL", model),
        )
        if not value
    ]
    if missing:
        print(f"结果: 失败，缺少配置：{', '.join(missing)}")
        return False

    endpoint = chat_endpoint(base_url)
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "temperature": 0,
            "max_tokens": 8,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        elapsed = time.perf_counter() - started
        print(f"Endpoint: {endpoint}")
        print(f"HTTP: {exc.code}，耗时: {elapsed:.2f}s")
        print(f"结果: 失败，{error_message(body)}")
        return False
    except urllib.error.URLError as exc:
        elapsed = time.perf_counter() - started
        print(f"Endpoint: {endpoint}")
        print(f"耗时: {elapsed:.2f}s")
        print(f"结果: 连接失败，{exc.reason}")
        return False
    except TimeoutError:
        elapsed = time.perf_counter() - started
        print(f"Endpoint: {endpoint}")
        print(f"结果: 请求超时，耗时: {elapsed:.2f}s")
        return False

    elapsed = time.perf_counter() - started
    try:
        data = json.loads(body)
        content = data["choices"][0]["message"].get("content", "")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        print(f"Endpoint: {endpoint}")
        print(f"HTTP: {status}，耗时: {elapsed:.2f}s")
        print("结果: 接口可访问，但响应不是预期的 OpenAI Chat Completions 格式")
        print(f"响应摘要: {body[:300]}")
        return False

    print(f"Endpoint: {endpoint}")
    print(f"HTTP: {status}，耗时: {elapsed:.2f}s")
    print(f"回复: {content!r}")
    print("结果: 可用")
    return True


def error_message(body: str) -> str:
    try:
        data = json.loads(body)
        error = data.get("error") or {}
        if isinstance(error, dict):
            message = error.get("message") or error.get("type")
            if message:
                return str(message)
    except json.JSONDecodeError:
        pass
    return body[:300] or "服务端未返回错误详情"


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 .env 中的 LLM API 配置")
    parser.add_argument("--env", default=".env", help="环境文件路径，默认 .env")
    parser.add_argument(
        "--target",
        choices=("primary", "backup", "all"),
        default="primary",
        help="测试主服务、备用服务或两者，默认 primary",
    )
    parser.add_argument("--timeout", type=float, default=None, help="请求超时秒数")
    args = parser.parse_args()

    env_path = Path(args.env)
    if not env_path.is_file():
        print(f"找不到环境文件：{env_path.resolve()}")
        return 2

    env = load_env(env_path)
    timeout = args.timeout or float(env.get("LLM_REQUEST_TIMEOUT", "30"))
    results: list[bool] = []

    if args.target in {"primary", "all"}:
        results.append(
            test_target(
                "主服务",
                api_key=env.get("LLM_API_KEY", ""),
                base_url=env.get("LLM_BASE_URL", ""),
                model=env.get("LLM_MODEL", ""),
                timeout=timeout,
            )
        )

    if args.target in {"backup", "all"}:
        results.append(
            test_target(
                "备用服务",
                api_key=env.get("LLM_BACKUP_API_KEY", ""),
                base_url=env.get("LLM_BACKUP_BASE_URL", ""),
                model=env.get("LLM_BACKUP_MODEL", ""),
                timeout=timeout,
            )
        )

    return 0 if results and all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
