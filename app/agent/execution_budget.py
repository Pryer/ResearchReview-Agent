"""跨主决策和专业任务共享的确定性执行预算。"""

from __future__ import annotations

import time
import json
import threading
from contextvars import ContextVar, copy_context
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable
from app.agent.execution import AgentCancelledError
from app.core.logger import get_logger

logger = get_logger(__name__)


class AgentBudgetExceeded(RuntimeError):
    pass


class AgentExecutionCancelled(AgentCancelledError):
    pass


class AgentExecutionStale(AgentExecutionCancelled):
    """持久化租约失效（owner 被接管或租约过期未续约）。

    继承 AgentExecutionCancelled 是为了复用所有已有的取消透传边界；但它
    语义上不是"用户取消"，主循环必须把结果状态判为 failed 而非 cancelled。
    一旦抛出，本执行对会话已无所有权，任何节点都不得吞掉它继续跑——
    否则会继续产生外部调用，最终保存仍会被 CAS 拒绝（只留下烧费空转）。
    """
    pass


def is_control_exception(exc: BaseException) -> bool:
    """判断是否为必须立即向执行边界传播的停止信号。

    WHY: 预算耗尽、用户取消、租约失效、账本 CAS 冲突都代表本执行已无权
    继续。节点若把它们当普通可恢复失败吞掉，会在已停止的会话里继续发起
    provider 调用、备用切换或覆盖补建（只烧费不产出），并把"系统停止"
    误报成"质量失败"或"N 条判定失败"。AgentExecutionCancelled/Stale 都是
    AgentCancelledError 子类，故一并覆盖。
    """
    if isinstance(exc, (AgentBudgetExceeded, AgentCancelledError)):
        return True
    from app.core.exceptions import LLMProviderUnavailableError
    if isinstance(exc, LLMProviderUnavailableError):
        return True
    from app.database.runtime_repository import RuntimeConflict

    return isinstance(exc, RuntimeConflict)


def _runtime_check(runtime):
    """把账本层 RuntimeConflict 转换为图内统一的"执行失效"终止异常。"""
    from app.database.runtime_repository import RuntimeConflict

    try:
        runtime.check()
    except RuntimeConflict as exc:
        raise AgentExecutionStale(str(exc)) from exc


def _runtime_persist_budget(runtime, ledger):
    """persist_budget 内部也会 check()；同样转换，避免裸 RuntimeConflict 被降级。"""
    from app.database.runtime_repository import RuntimeConflict

    try:
        runtime.persist_budget(ledger)
    except RuntimeConflict as exc:
        raise AgentExecutionStale(str(exc)) from exc


_ACTIVE = ContextVar("research_budget", default=None)
_ACTION_DEPTH = ContextVar("research_action_depth", default=0)
_LOCK = threading.RLock()


def active_budget():
    return _ACTIVE.get()


@contextmanager
def budget_scope(state, should_cancel=None):
    from app.core.config import get_settings
    existing = _ACTIVE.get()
    if existing is not None:
        state["agent_execution_budget"] = existing["ledger"]
        yield existing
        return
    settings = get_settings()
    ledger = state.setdefault("agent_execution_budget", {})
    ledger.setdefault("token_limit", settings.agent_main_token_budget)
    ledger.setdefault("retrieval_limit", settings.agent_retrieval_budget)
    ledger.setdefault("recovery_limit", settings.recovery_total_action_budget)
    ledger["recovery_used"] = max(int(ledger.get("recovery_used") or 0), int(state.get("recovery_action_count") or 0))
    # WHY: 跨请求保存累计用量；截止时间属于本次执行，不能保存 monotonic 进数据库。
    execution = {"ledger": ledger, "cancel": should_cancel,
                 "deadline": time.monotonic() + settings.agent_execution_deadline_seconds}
    token = _ACTIVE.set(execution)
    try:
        check_execution()
        yield execution
    finally:
        _ACTIVE.reset(token)


def check_execution():
    from app.services.durable_execution_service import active_runtime
    runtime = active_runtime()
    if runtime:
        _runtime_check(runtime)
    current = _ACTIVE.get()
    if current:
        if current.get("exhausted"):
            raise AgentBudgetExceeded(current["exhausted"])
        if (int(current["ledger"].get("llm_tokens") or 0)
                + int(current["ledger"].get("retrieval_model_tokens") or 0)
                > int(current["ledger"]["token_limit"])):
            raise AgentBudgetExceeded("actual token usage exceeded the reserved budget")
        if current["cancel"] and current["cancel"]():
            raise AgentExecutionCancelled("research execution cancelled")
        if time.monotonic() >= current["deadline"]:
            raise AgentBudgetExceeded("research execution deadline exceeded")


def consume(kind: str, amount: int = 1):
    check_execution()
    current = _ACTIVE.get()
    if current is None:
        return
    with _LOCK:
        ledger = current["ledger"]
        count = int(ledger.get(kind + "_used") or 0)
        if count + amount > int(ledger[kind + "_limit"]):
            current["exhausted"] = f"{kind} budget exhausted"
            raise AgentBudgetExceeded(f"{kind} budget exhausted")
        ledger[kind + "_used"] = count + amount
        from app.services.durable_execution_service import active_runtime
        runtime = active_runtime()
        if runtime:
            _runtime_persist_budget(runtime, ledger)


def record_retrieval_request(kind: str):
    """远程 embedding/rerank 每次 HTTP 尝试均计入同一执行账本。"""
    check_execution()
    current = _ACTIVE.get()
    if current is None:
        return
    from app.core.config import get_settings

    with _LOCK:
        ledger = current["ledger"]
        used = int(ledger.get("retrieval_model_requests") or 0)
        if used >= get_settings().retrieval_model_request_limit:
            current["exhausted"] = "retrieval model request budget exhausted"
            raise AgentBudgetExceeded(current["exhausted"])
        ledger["retrieval_model_requests"] = used + 1
        ledger[f"retrieval_{kind}_requests"] = int(ledger.get(f"retrieval_{kind}_requests") or 0) + 1
        from app.services.durable_execution_service import active_runtime

        runtime = active_runtime()
        if runtime:
            _runtime_persist_budget(runtime, ledger)


def record_retrieval_usage(usage):
    current = _ACTIVE.get()
    if current is None:
        return
    with _LOCK:
        ledger = current["ledger"]
        value = usage.get("total_tokens") if isinstance(usage, dict) else None
        if isinstance(value, (int, float)) and value >= 0:
            ledger["retrieval_model_tokens"] = int(ledger.get("retrieval_model_tokens") or 0) + int(value)
        else:
            ledger["retrieval_usage_unknown_count"] = int(ledger.get("retrieval_usage_unknown_count") or 0) + 1
        from app.services.durable_execution_service import active_runtime

        runtime = active_runtime()
        if runtime:
            _runtime_persist_budget(runtime, ledger)
    check_execution()


def submit_with_context(executor, fn, *args):
    def run():
        check_execution()
        return fn(*args)
    return executor.submit(copy_context().run, run)


def budgeted_create(client, *, on_response=None, operation="", **kwargs):
    """真实 provider 请求前预留，返回后结算；重试与备用分别记账。"""
    check_execution()
    current = _ACTIVE.get()
    if current is None:
        response = client.chat.completions.create(**kwargs)
        if on_response:
            on_response(response)
        return response
    ledger = current["ledger"]
    # WHY: 账本限制总 token，缓存命中不能抵扣；未知服务商 tokenizer 时保留字节上界。
    # 输出预留必须覆盖实际请求允许的完整上限，不能用平均输出比例冒充硬预算。
    prompt_bound = len(json.dumps({k: kwargs[k] for k in ("messages", "tools") if k in kwargs},
                                  ensure_ascii=False).encode("utf-8")) + 128
    reservation = prompt_bound + int(kwargs.get("max_tokens") or 4096)
    with _LOCK:
        used = int(ledger.get("llm_tokens") or 0) + int(ledger.get("retrieval_model_tokens") or 0)
        reserved = int(ledger.get("tokens_reserved") or 0)
        if used + reserved + reservation > int(ledger["token_limit"]):
            current["exhausted"] = "token reservation exceeds remaining budget"
            # WHY: 拒绝点记录可核对的结构化诊断（操作/已用/已预留/请求预留/
            # 剩余/上限/阶段），但绝不含 prompt 或凭据。此次请求尚未调用
            # provider，故不计入 llm_requests、不占用预留，账面无需回滚。
            limit = int(ledger["token_limit"])
            logger.warning(
                "BUDGET_RESERVATION_REJECTED stage=reservation operation=%s model=%s "
                "used=%d reserved=%d requested=%d remaining=%d limit=%d",
                operation or "unknown", str(kwargs.get("model") or ""),
                used, reserved, reservation, max(0, limit - used - reserved), limit,
            )
            raise AgentBudgetExceeded("token reservation exceeds remaining budget")
        ledger["tokens_reserved"] = reserved + reservation
        ledger["llm_requests"] = int(ledger.get("llm_requests") or 0) + 1
    kwargs["timeout"] = min(float(kwargs.get("timeout") or 60), max(.001, current["deadline"] - time.monotonic()))
    cost = reservation
    estimated = True
    from app.services.durable_execution_service import active_runtime, active_task_id
    runtime = active_runtime()
    attempt_id = None
    settled_usage = {"model": str(kwargs.get("model") or "")}
    requested = False
    try:
        if runtime:
            with _LOCK:
                # reserve_attempt 内部会 check() 租约，失效时统一转成
                # AgentExecutionStale，避免在发请求前把终止错误降级。
                from app.database.runtime_repository import RuntimeConflict

                try:
                    attempt_id = runtime.reserve_attempt(active_task_id(), reservation, ledger)
                except RuntimeConflict as exc:
                    raise AgentExecutionStale(str(exc)) from exc
        requested = True
        response = client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None and (getattr(usage, "total_tokens", None) or
                                  getattr(usage, "prompt_tokens", None) is not None):
            cost = int(getattr(usage, "total_tokens", 0) or
                       (int(getattr(usage, "prompt_tokens", 0) or 0) + int(getattr(usage, "completion_tokens", 0) or 0)))
            estimated = False
            for field in ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
                value = getattr(usage, field, None)
                if value is not None:
                    settled_usage[field] = int(value)
            # WHY: reasoning 是 completion 的子集，只记录明细，不重复加入总量；
            # 服务未报告时保持缺省，不能把未知思考用量当成零。
            reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
            if reasoning is not None:
                settled_usage["reasoning_tokens"] = int(reasoning)
        # WHY: 响应已产生费用，即使随后收到取消/截止，也要记录 usage。
        if on_response:
            on_response(response)
    finally:
        with _LOCK:
            if not requested:
                cost, estimated = 0, False
                ledger["llm_requests"] -= 1
            ledger["tokens_reserved"] -= reservation
            ledger["llm_tokens"] = int(ledger.get("llm_tokens") or 0) + cost
            if estimated:
                ledger["estimated_tokens"] = int(ledger.get("estimated_tokens") or 0) + cost
                ledger["usage_estimated"] = True
            if runtime and attempt_id:
                runtime.settle_attempt(attempt_id, {**settled_usage, "total_tokens": cost}, ledger, estimated=estimated)
    # WHY: 必须先结算本次真实用量，再检查超支/取消/截止，超限响应不可交给解析和研究提交。
    check_execution()
    return response


def research_budget(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        import inspect
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        key = "initial_state" if "initial_state" in bound.arguments else "research_state"
        state = dict(bound.arguments.get(key) or {})
        state["user_operation_sequence"] = int(state.get("user_operation_sequence") or 0) + 1
        bound.arguments[key] = state
        with budget_scope(state, bound.arguments.get("should_cancel")):
            return fn(*bound.args, **bound.kwargs)
    return wrapped


def single_action_charge(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        token = _ACTION_DEPTH.set(_ACTION_DEPTH.get() + 1)
        try:
            return fn(*args, **kwargs)
        finally:
            _ACTION_DEPTH.reset(token)
    return wrapped


def reserve_recovery(state):
    if active_budget():
        consume("recovery")
    else:
        # WHY: 服务层可在恢复图启动前登记动作，必须写入同一持久化账本。
        with budget_scope(state):
            consume("recovery")


def reserve_action(
    state: dict[str, Any],
    *,
    limit: int,
    should_cancel: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
) -> None:
    check_execution()
    if should_cancel and should_cancel():
        raise AgentExecutionCancelled("agent task cancelled before execution")
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise AgentBudgetExceeded("agent execution deadline exceeded")
    if _ACTION_DEPTH.get() > 1:
        return
    ledger = state.setdefault("agent_execution_budget", {})
    ledger.setdefault("action_limit", max(0, int(limit)))
    ledger.setdefault("actions_reserved", 0)
    ledger.setdefault("actions_committed", 0)
    configured = int(ledger.get("action_limit") or 0)
    used = int(ledger.get("actions_reserved") or 0)
    if used >= configured:
        raise AgentBudgetExceeded("agent action budget exhausted")
    ledger["actions_reserved"] = used + 1


def commit_action(state: dict[str, Any]) -> None:
    if _ACTION_DEPTH.get() > 1:
        return
    ledger = state.setdefault("agent_execution_budget", {})
    ledger["actions_committed"] = int(ledger.get("actions_committed") or 0) + 1
