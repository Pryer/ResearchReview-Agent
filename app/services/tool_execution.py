"""副作用边界：Handler继承受限执行身份，回执与费用结算分别持久化。"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
import inspect
import json
import logging
import time
from uuid import uuid4

from app.database.runtime_repository import RuntimeConflict


class ToolAuthorizationError(RuntimeConflict):
    pass


@dataclass(frozen=True)
class ToolContext:
    delegation_id: str
    invocation_id: str
    invocation_attempt_id: str
    task_id: str
    action: str
    grant_revision: int


_CONTEXT = ContextVar('research_tool_context', default=None)
_CALL = ContextVar('research_tool_call', default=None)
_ARTIFACT_STORAGE = ContextVar('research_artifact_storage', default=False)


def current_tool_context():
    return _CONTEXT.get()


def artifact_tool_context():
    return None if _ARTIFACT_STORAGE.get() else _CONTEXT.get()


@contextmanager
def artifact_storage_scope():
    # WHY: Repository内的纯资料存取不再递归创建工具回执；作用域不关闭其他工具校验。
    token = _ARTIFACT_STORAGE.set(True)
    try:
        yield
    finally:
        _ARTIFACT_STORAGE.reset(token)


def current_tool_call_id():
    return _CALL.get()


@contextmanager
def tool_execution_scope(invocation):
    from app.services.durable_execution_service import active_task_id
    from app.services.invocation_scope import current_invocation
    current = current_invocation()
    task_id = active_task_id()
    if not current or current[0] != invocation or not task_id:
        raise ToolAuthorizationError('tool scope requires an active action attempt')
    context = ToolContext(invocation.delegation_id, invocation.invocation_id, current[1],
                          task_id, invocation.action, invocation.grant_revision)
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def _argument_hash(arguments):
    def public(value):
        if isinstance(value, dict):
            return {str(k): public(v) for k, v in value.items()
                    if str(k).lower() not in {'key', 'api_key', 'access_token', 'refresh_token', 'token'}
                    and not any(s in str(k).lower() for s in ('secret', 'password', 'authorization'))
                    and not str(k).lower().endswith('_api_key')
                    and str(k) not in {'self', 'client', 'session', 'settings', 'on_response'}}
        if isinstance(value, (tuple, list)):
            return [public(v) for v in value]
        return value if value is None or isinstance(value, (str, int, float, bool)) else type(value).__name__
    return hashlib.sha256(json.dumps(public(arguments), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class ToolExecutor:
    @staticmethod
    def execute(name, call, *, arguments=None, boundary='internal'):
        context = _CONTEXT.get()
        if context is None:
            # WHY: 旧运行时和独立API保持原行为；v2只在确定性Handler入口签发此作用域。
            return call()
        from app.agent.execution_budget import check_execution
        from app.agent.registry.tool_registry import TOOLS, tools_for_action
        from app.services.durable_execution_service import active_runtime
        check_execution()
        if name not in TOOLS or name not in tools_for_action(context.action):
            raise ToolAuthorizationError('tool is outside the current action capability: ' + name)
        runtime = active_runtime()
        if runtime is None:
            raise ToolAuthorizationError('tool execution requires durable receipts')
        call_id = uuid4().hex
        payload = {**context.__dict__, 'call_id': call_id, 'tool': name,
                   'effect': TOOLS[name].effect, 'retry': TOOLS[name].retry,
            'arguments_hash': _argument_hash(arguments or {}), 'boundary': boundary}
        runtime.begin_tool_call(payload)
        token = _CALL.set(call_id)
        started = time.monotonic()
        failure = None
        try:
            # WHY: durable ack可能等待写锁；确认后再次检查取消，才能进入真实外部能力。
            check_execution()
            from app.services.paper_resources import RESOURCE_TOOLS, check_file
            scoped = name in RESOURCE_TOOLS or (name == 'paper_search' and context.action == 'fetch_metadata')
            guard = runtime.validate_tool_resources(context, name, arguments or {}, boundary) if scoped else {}
            check_file(guard)
            result = call()
            if scoped:
                runtime.record_tool_resources(context, name, arguments or {}, result, guard)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            _CALL.reset(token)
            receipt = {'status': 'failed' if failure else 'completed',
                       'error_type': type(failure).__name__ if failure else None,
                       'duration_ms': int((time.monotonic() - started) * 1000)}
            try:
                runtime.finish_tool_call(call_id, receipt)
            except Exception as receipt_error:
                if failure is None:
                    raise
                # WHY: 原始取消/失权或业务错误不能被审计故障覆盖；持久start保持未决供恢复判定。
                logging.getLogger(__name__).warning('Tool receipt unavailable (%s)', type(receipt_error).__name__)
        check_execution()
        return result


execute_tool = ToolExecutor.execute


def research_tool(name):
    """包装真实客户端边界；纯函数、预算/检查点基础设施不套工具层。"""
    def decorate(fn):
        signature = inspect.signature(fn)
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if _CONTEXT.get() is None:
                return fn(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            tool = name(bound.arguments) if callable(name) else name
            return execute_tool(tool, lambda: fn(*args, **kwargs), arguments=bound.arguments,
                                boundary=fn.__module__ + '.' + fn.__qualname__)
        return wrapped
    return decorate
