"""会话服务拥有资料存取；执行期依赖不进入研究状态。"""
from contextvars import ContextVar
from functools import wraps

_STORE = ContextVar("research_artifact_store", default=None)


def artifact_store():
    return _STORE.get()


def with_artifact_store(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        from app.services.research_artifact_service import ResearchArtifactService
        from app.agent.execution_budget import budget_scope
        request = args[0] if args else kwargs.get("request")
        if request is None or not request.session_id:
            raise ValueError("多轮研究请求必须提供 session_id")
        saved = self.repo.get(str(request.session_id)) if request and request.session_id else None
        saved_state = (saved or {}).get("state") or {}
        budget_state = dict(saved_state.get("editable_research_state") or saved_state)
        token = _STORE.set(ResearchArtifactService(self.db))
        try:
            # WHY: 一个用户回合内的解析、图执行和自动恢复共用截止时间与账本。
            with budget_scope(budget_state, self.should_cancel):
                from app.services.durable_execution_service import durable_scope
                from app.services.research_memory_service import ResearchMemoryService
                with durable_scope(self.db, str(request.session_id), budget_state,
                                   budget_state["agent_execution_budget"]) as runtime:
                    memory = ResearchMemoryService(self.db)
                    importing = not memory.legacy_imported(str(request.session_id))
                    memory.import_legacy_history(str(request.session_id), saved_state)
                    if saved and importing:
                        # WHY: 归档标记与旧历史键一起落库，后续读取不会把同一条旧消息再当新消息。
                        self.repo.save(session_id=str(request.session_id), status=saved["status"],
                            original_query=saved["original_query"], state=saved_state,
                            clarification=saved.get("clarification"))
                    memory.append_event(session_id=str(request.session_id), event_type="user_turn",
                        payload={"content": getattr(request, "clarification_answer", None)
                            or getattr(request, "instruction", None) or getattr(request, "user_query", "")},
                        event_key="turn:" + runtime.owner)
                    self.db.commit()
                    result = fn(self, *args, **kwargs)
                    current = result.get("research_state") if isinstance(result, dict) else None
                    if current and result.get("status") != "cancelled" and not (self.should_cancel and self.should_cancel()):
                        runtime.snapshot(current, ledger=budget_state["agent_execution_budget"])
                    return result
        finally:
            _STORE.reset(token)
    return wrapped


def checkpoint_artifacts(state):
    store = artifact_store()
    session_id = state.get("session_id")
    if store is not None and session_id:
        stored = store.externalize_state(session_id, state)
        state["artifact_manifest"] = stored["artifact_manifest"]
        store.repo.db.commit()
