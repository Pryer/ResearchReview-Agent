"""LLM 服务。

封装 OpenAI 兼容 API 调用，支持 DeepSeek / OpenAI / Qwen / Ollama / LongCat。
这是 Agent 调用 LLM 的统一入口。

主用失败（超时/连接错误/额度不足/429/5xx）时自动切换到备用提供商再试一次；
主备都失败后抛出 ``LLMInvocationError``，由各调用方确定性兜底。
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError

from app.core.config import get_settings
from app.core.exceptions import LLMInvocationError
from app.core.logger import get_logger

logger = get_logger(__name__)


# 触发「主用 → 备用」切换的可降级 HTTP 状态码。部分兼容 OpenAI 的
# 提供商会用 402 表示当前账号 Token 额度不足；该错误不应阻止已配置的
# 备用模型继续服务。
_RETRYABLE_STATUS_CODES = {402, 429, 500, 502, 503, 504}

_WRITING_OPERATION_PREFIXES = (
    "write_section:", "write_deliverable:", "repair_deliverable:",
    "repair_english_fragments:", "polish_fallback:", "backfill_citations:",
)


class LLMService:
    """OpenAI 兼容 LLM 客户端。

    使用 ``openai`` SDK 连接任意兼容 OpenAI API 的服务。
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.llm_api_key
        self.base_url = settings.llm_base_url
        self.model = settings.llm_model
        self.provider = settings.llm_provider
        self.thinking_enabled = settings.llm_thinking_enabled
        self.thinking_effort = settings.llm_thinking_effort
        self.thinking_max_tokens = settings.llm_thinking_max_tokens
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.control_plane_max_tokens = settings.llm_control_plane_max_tokens
        self.request_timeout = settings.llm_request_timeout
        self.failover_total_timeout = settings.llm_failover_total_timeout
        self._client = None
        # 备用提供商：主用失败后自动切换。
        self.backup_api_key = settings.llm_backup_api_key
        self.backup_base_url = settings.llm_backup_base_url
        self.backup_model = settings.llm_backup_model
        self.backup_provider = settings.llm_backup_provider
        self.backup_enabled = settings.llm_backup_enabled
        self._backup_client = None

    @property
    def client(self):
        """懒加载主用 openai 客户端。"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key or "sk-placeholder",
                    base_url=self.base_url,
                    timeout=self.request_timeout,
                    max_retries=0,
                )
            except ImportError:
                raise LLMInvocationError("openai SDK 未安装，请运行 pip install openai")
        return self._client

    @property
    def backup_client(self):
        """懒加载备用 openai 客户端。仅在配置了备用提供商时可用。"""
        if self._backup_client is None:
            try:
                from openai import OpenAI
                self._backup_client = OpenAI(
                    api_key=self.backup_api_key or "sk-placeholder",
                    base_url=self.backup_base_url,
                    timeout=self.request_timeout,
                    max_retries=0,
                )
            except ImportError:
                raise LLMInvocationError("openai SDK 未安装，请运行 pip install openai")
        return self._backup_client

    def complete(
        self,
        prompt: str,
        response_format: str = "text",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        retry_empty: bool = True,
        operation: str = "completion",
        thinking_enabled: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """调用 LLM 获取单次回复。

        Args:
            prompt: 用户提示词。
            response_format: "text" 或 "json"；"json_object" 作为等价别名
                兼容历史调用方（与 OpenAI 线上格式值同名）。
            temperature: 可选，覆盖默认温度。
            max_tokens: 可选，覆盖默认 max_tokens。
            timeout: 可选，覆盖默认请求超时时间（秒）。
            retry_empty: 是否在返回空内容时按 reasoning 模型规则扩 token 重试。
            thinking_enabled: 可选，覆盖全局思考模式。仅对 DeepSeek 请求生效。
                最终正文写作可传 True，其余调用不传时沿用全局默认值。
            reasoning_effort: DeepSeek 思考强度，可选 low/high/max；未传时使用
                LLM_THINKING_EFFORT，且仅在思考模式开启时发送。

        Returns:
            LLM 响应文本。

        主用提供商失败（超时/连接错误/额度不足/429/5xx）时自动切换到备用提供商再试
        一次；主备都失败则抛出 ``LLMInvocationError``，由调用方确定性兜底。
        """
        if not self.api_key and not self.backup_enabled:
            logger.warning("LLM_API_KEY not set and no backup configured, returning empty response")
            return ""

        started = time.monotonic()
        effective_requested_thinking = (
            self.thinking_enabled
            if thinking_enabled is None
            else thinking_enabled
        )
        is_writing_operation = (
            operation == "completion"
            or operation.startswith(_WRITING_OPERATION_PREFIXES)
        )
        if max_tokens is not None:
            effective_max_tokens = int(max_tokens)
        elif effective_requested_thinking and is_writing_operation:
            effective_max_tokens = max(self.max_tokens, self.thinking_max_tokens)
        elif not is_writing_operation:
            effective_max_tokens = self.control_plane_max_tokens
        else:
            effective_max_tokens = self.max_tokens
        logical_timeout = max(0.001, float(self.failover_total_timeout))
        per_provider_timeout = max(
            0.001,
            float(timeout if timeout is not None else self.request_timeout),
        )
        deadline = started + logical_timeout
        logger.info(
            "LLM_CALL_START operation=%s timeout=%s total_deadline=%s max_tokens=%s retry_empty=%s thinking=%s",
            operation,
            per_provider_timeout,
            logical_timeout,
            effective_max_tokens,
            retry_empty,
            effective_requested_thinking,
        )

        # 构造一次调用的基础参数；每个 provider 各取一份副本，避免相互污染
        # （例如 response_format 降级会 pop 掉该字段）。
        base_kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": effective_max_tokens,
        }
        if response_format in ("json", "json_object"):
            base_kwargs["response_format"] = {"type": "json_object"}

        # 主用 → 备用 两段式，每个 provider 一次尝试，不做同提供商重试。
        providers: list[tuple[str, object, str, str]] = []
        if self.api_key:
            providers.append(("primary", self.client, self.model, self.provider))
        if self.backup_enabled:
            providers.append(
                ("backup", self.backup_client, self.backup_model, self.backup_provider)
            )

        try:
            last_exc: Exception | None = None
            for prov_name, client, model, provider in providers:
                kwargs = dict(base_kwargs)
                kwargs["model"] = model
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    last_exc = TimeoutError(
                        f"LLM failover deadline exhausted before {prov_name} attempt"
                    )
                    break
                kwargs["timeout"] = min(per_provider_timeout, remaining)
                # DeepSeek V4 默认是 thinking 模式。OpenAI SDK 需要通过
                # extra_body 传递厂商扩展字段；关闭后 token 会用于最终正文，
                # 不再大量消耗在 reasoning_content。其他兼容提供商不接收该字段。
                if provider.strip().lower() == "deepseek":
                    effective_thinking = (
                        self.thinking_enabled
                        if thinking_enabled is None
                        else thinking_enabled
                    )
                    kwargs["extra_body"] = {
                        "thinking": {
                            "type": "enabled" if effective_thinking else "disabled"
                        }
                    }
                    # DeepSeek 官方说明思考模式不支持 temperature；虽然传入时
                    # 不会报错，但参数不会生效。开启 thinking 时将其移除，避免
                    # 日志与调用方误以为最终写作仍受温度值控制。
                    if effective_thinking:
                        kwargs["extra_body"]["reasoning_effort"] = (
                            reasoning_effort or self.thinking_effort
                        )
                        kwargs.pop("temperature", None)
                try:
                    content = self._attempt_once(
                        kwargs,
                        response_format,
                        retry_empty,
                        client,
                        operation=operation,
                        provider=provider,
                    )
                except (APITimeoutError, APIConnectionError) as e:
                    last_exc = e
                    logger.warning(
                        "LLM provider=%s failed (%s)", prov_name, type(e).__name__,
                    )
                    if prov_name == "primary" and self.backup_enabled:
                        logger.info(
                            "LLM_BACKUP_SWITCH operation=%s reason=timeout_or_conn", operation,
                        )
                        continue  # 切到备用
                    break  # 备用也失败，跳出
                except APIStatusError as e:
                    if e.status_code in _RETRYABLE_STATUS_CODES:
                        last_exc = e
                        logger.warning(
                            "LLM provider=%s failed (status=%d)", prov_name, e.status_code,
                        )
                        if prov_name == "primary" and self.backup_enabled:
                            logger.info(
                                "LLM_BACKUP_SWITCH operation=%s reason=status_%d",
                                operation, e.status_code,
                            )
                            continue
                        break
                    raise  # 非可重试状态码（401/403/400 等），直接冒泡到外层 except
                # BadRequestError 不在此捕获——由 _call_with_possible_fallback 内部处理或冒泡

                # 空 content 也触发主用→备用切换（LongCat reasoning 模型偶发把 token 烧在 reasoning 里导致 content 为空）。
                # 备用 DeepSeek 非 reasoning 模型，基本不会空；备用也空则返回空让调用方兜底（行为同现状）。
                if not content.strip() and prov_name == "primary" and self.backup_enabled:
                    logger.warning(
                        "LLM provider=primary returned empty content; switching to backup (op=%s)",
                        operation,
                    )
                    logger.info(
                        "LLM_BACKUP_SWITCH operation=%s reason=empty_content", operation,
                    )
                    continue  # 切到备用

                # 该 provider 成功（非空 content，或主用空但无备用/已是备用）
                if prov_name == "backup":
                    # backup 成功但返回空内容：同样视为失败（不记录 BACKUP_OK）
                    if not content.strip():
                        logger.warning(
                            "LLM provider=backup also returned empty content (op=%s), treating as failure",
                            operation,
                        )
                        last_exc = last_exc or ValueError("backup returned empty content")
                        break
                    logger.info(
                        "LLM_BACKUP_OK operation=%s provider=backup duration_ms=%d",
                        operation, int((time.monotonic() - started) * 1000),
                    )
                logger.debug("LLM response (%d chars): %s...", len(content), content[:80])
                logger.info(
                    "LLM_CALL_END operation=%s duration_ms=%d content_length=%d",
                    operation,
                    int((time.monotonic() - started) * 1000),
                    len(content),
                )
                return content

            # 两个提供商都失败：让最后一次异常冒泡到外层 except，统一包成 LLMInvocationError。
            raise last_exc  # type: ignore[misc]

        except Exception as e:
            logger.error(
                "LLM_CALL_FAILED operation=%s duration_ms=%d error=%s",
                operation,
                int((time.monotonic() - started) * 1000),
                e,
            )
            raise LLMInvocationError(f"LLM 调用失败: {e}")

    def _attempt_once(
        self,
        kwargs: dict,
        response_format: str,
        retry_empty: bool,
        client,
        operation: str = "completion",
        provider: str = "",
    ) -> str:
        """对单个 client 执行一次完整尝试。

        包含 ``response_format`` 降级和 reasoning 模型空内容扩 token 重试。
        超时/连接/状态错误不在本层处理，直接向上传播由 ``complete()`` 决定切换。

        每个真实 API 响应的 usage 恰好记录一次，并附带真实的
        operation / provider 与耗时（此前恒为默认值/0，指标不可用）。
        """
        started = time.monotonic()

        def _record(resp) -> None:
            usage = getattr(resp, "usage", None)
            if not usage:
                return
            from app.core.metrics import get_metrics_collector
            get_metrics_collector().record_llm_call(
                model=str(kwargs.get("model") or self.model),
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                duration_ms=int((time.monotonic() - started) * 1000),
                operation=operation,
                provider=provider,
            )

        def _content_of(resp) -> str:
            # 部分兼容提供商在内容过滤/异常时会返回空 choices，
            # 直接取 choices[0] 会 IndexError；视为空内容交给上层
            # 的主用→备用切换逻辑处理。
            choices = getattr(resp, "choices", None) or []
            if not choices:
                return ""
            return choices[0].message.content or ""

        resp = self._call_with_possible_fallback(kwargs, response_format, client=client)
        if retry_empty:
            resp = self._retry_if_content_empty(kwargs, resp, client=client)
        _record(resp)
        content = _content_of(resp)

        if (
            retry_empty
            and response_format == "json"
            and not content.strip()
            and "response_format" in kwargs
        ):
            logger.warning("LLM returned empty JSON response, retrying without response_format")
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**kwargs)
            resp = self._retry_if_content_empty(kwargs, resp, client=client)
            _record(resp)
            content = _content_of(resp)
        return content

    def _call_with_possible_fallback(self, kwargs: dict, response_format: str, client=None):
        """发起单次 LLM 调用，在 ``response_format`` 不被模型支持时降级重试一次。

        超时（APITimeoutError）、连接错误（APIConnectionError）和 API 状态错误
        （APIStatusError）不在本层处理，直接向上传播，由 ``complete()`` 的
        provider 循环决定是否切换到备用提供商。
        """
        if client is None:
            client = self.client
        try:
            return client.chat.completions.create(**kwargs)
        except BadRequestError:
            # 模型不支持 response_format=json_object —— 移除参数重试一次。
            if "response_format" not in kwargs:
                raise
            logger.info("Model does not support response_format, retrying without it")
            kwargs.pop("response_format", None)
            return client.chat.completions.create(**kwargs)

    def _retry_if_content_empty(self, kwargs: dict, resp, client=None):
        """LongCat 等 reasoning 模型可能先消耗 token 到 reasoning_content。

        当正文 content 为空且 finish_reason=length 时，自动提高 max_tokens 重试一次，
        避免上层拿到空字符串后误判为 LLM 规划失败。
        """
        if client is None:
            client = self.client
        # 部分兼容提供商在内容过滤/网关异常时不返回标准响应对象，而是一个
        # 原始字符串（或其他无 choices 的对象）。此处早于 _content_of 执行，
        # 直接取 resp.choices 会抛 AttributeError('str' object has no attribute
        # 'choices')。该异常不属于 APITimeout/APIConnection/APIStatus，会绕过
        # 主用→备用切换直接冒泡成 LLMInvocationError，使备用提供商永远得不到
        # 机会。这里与 _content_of 一样做防御：无 choices 时原样返回，交由空
        # 内容分支触发备用切换。
        if not getattr(resp, "choices", None):
            return resp
        choice = resp.choices[0]
        message = choice.message
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None)
        if content.strip() or choice.finish_reason != "length" or not reasoning:
            return resp

        current_max = int(kwargs.get("max_tokens") or self.max_tokens or 0)
        retry_ceiling = max(
            self.max_tokens,
            self.thinking_max_tokens * 2,
            512,
        )
        retry_max = min(max(current_max * 2, 512), retry_ceiling)
        if retry_max <= current_max:
            logger.warning(
                "LLM returned empty content after exhausting reasoning budget; "
                "retry ceiling=%d reached",
                retry_ceiling,
            )
            return resp
        retry_kwargs = dict(kwargs)
        retry_kwargs["max_tokens"] = retry_max
        logger.warning(
            "LLM returned empty content after reasoning tokens; retrying with max_tokens=%d",
            retry_max,
        )
        return client.chat.completions.create(**retry_kwargs)

    def is_available(self) -> bool:
        """检查 LLM 是否配置可用（主用或备用任一可用即视为可用）。"""
        return bool(self.api_key) or self.backup_enabled
