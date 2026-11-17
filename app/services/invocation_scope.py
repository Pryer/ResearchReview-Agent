"""委派不占据 Action task_scope；每次尝试有独立执行身份。"""
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

_INVOCATION = ContextVar("research_invocation", default=None)


def current_invocation():
    return _INVOCATION.get()


@contextmanager
def invocation_scope(invocation):
    token = _INVOCATION.set((invocation, uuid4().hex))
    try:
        yield
    finally:
        _INVOCATION.reset(token)
