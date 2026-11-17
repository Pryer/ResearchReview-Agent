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
        # WHY: 配置修订以运行态CAS记录为准，可能比最后一轮会话结果更新，领取前先读取。
        import json
        from app.database.models import ResearchRuntimeModel
        row=self.db.get(ResearchRuntimeModel,str(request.session_id),populate_existing=True)
        runtime_state=json.loads(row.state_json or "{}") if row else {}
        if runtime_state.get("effective_config"):
            budget_state["effective_config"]=runtime_state["effective_config"]
            budget_state["delivery_mode"]=runtime_state.get("delivery_mode", "strict")
        if row:
            budget_state["agent_execution_budget"]=json.loads(row.budget_json)
        # WHY: 澄清/恢复/修订继承原任务；已结束会话中的新研究重新计工作轮次，费用与请求额度仍累计。
        new_research = bool(fn.__name__ == "handle" and saved
            and saved.get("status") in {"completed", "partial", "failed", "cancelled"}
            and not getattr(request, "resume_from_checkpoint", False)
            and not getattr(request, "clarification_answer", None))
        if new_research:
            # WHY: 外层先建立新截止时间；其余工作计数由acquire读取权威账本后只重置一次。
            budget_state.setdefault("agent_execution_budget", {})["elapsed_seconds"] = 0
            budget_state.pop("recovery_action_count", None)
        from app.agent.runtime_config import initialize_delivery
        requested=getattr(request,"delivery_mode",None)
        if requested:
            if budget_state.get("effective_config") and budget_state["effective_config"]["values"].get("review_delivery_mode")!=requested:
                raise ValueError("请先显式修订已保存任务的交付配置")
            budget_state["delivery_mode"]=requested
        initialize_delivery(budget_state,getattr(request,"user_query", ""),new_task=not bool(saved))
        from app.services.execution_report_capture import Capture,capture_scope
        from app.services.execution_report_service import freeze_source,finalize_capture
        from app.core.config import get_settings
        cap=Capture(str(request.session_id),str(getattr(request,'user_query',None) or getattr(request,'instruction',None) or (saved or {}).get('original_query','')),
            getattr(self,'execution_report_id',None)) if get_settings().execution_report_enabled else None
        result=None
        original_llm=self.llm
        token = _STORE.set(ResearchArtifactService(self.db))
        try:
            # WHY: 一个用户回合内的解析、图执行和自动恢复共用截止时间与账本。
            with budget_scope(budget_state, self.should_cancel):
                bind=getattr(original_llm,"with_task_settings",None)
                if callable(bind):self.llm=bind()
                from app.services.durable_execution_service import durable_scope
                from app.services.research_memory_service import ResearchMemoryService
                with durable_scope(self.db, str(request.session_id), budget_state,
                                   budget_state["agent_execution_budget"], new_research=new_research) as runtime:
                    if getattr(self, 'execution_job_id', None):
                        from app.services.execution_progress import bind_job
                        bind_job(runtime, self.execution_job_id)
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
                    from contextlib import nullcontext
                    if cap:
                        cap.start_version=runtime.version
                        cap.owner=runtime.owner
                    with capture_scope(cap) if cap else nullcontext():
                        try:
                            result = fn(self, *args, **kwargs)
                            current = result.get("research_state") if isinstance(result, dict) else None
                            if current and result.get("status") != "cancelled" and not (self.should_cancel and self.should_cancel()):
                                runtime.snapshot(current, ledger=budget_state["agent_execution_budget"])
                            return result
                        except BaseException as exc:
                            if cap:cap.error_type=type(exc).__name__
                            raise
                        finally:
                            if cap:
                                cap.end_version=runtime.version
                                try:cap.source=freeze_source(self.db,cap)
                                except Exception as exc:
                                    # 诊断导出不能覆盖原研究结果或停止异常。
                                    import logging
                                    logging.getLogger(__name__).warning('Report snapshot failed (%s)',type(exc).__name__)
                                    cap.capture_error_type=type(exc).__name__
        except BaseException as exc:
            if cap:cap.error_type=cap.error_type or type(exc).__name__
            raise
        finally:
            self.llm=original_llm
            _STORE.reset(token)
            if cap:
                if getattr(self,'defer_execution_report',False):
                    self.pending_execution_report=(cap,result)
                else:
                    status=None
                    if self.should_cancel:
                        try:status='cancelled' if self.should_cancel() else None
                        except Exception:pass  # 诊断读取失败不能替换业务结果。
                    descriptor=finalize_capture(self.db,cap,result,status)
                    if isinstance(result,dict):result['execution_report']=descriptor
    return wrapped


def checkpoint_artifacts(state):
    store = artifact_store()
    session_id = state.get("session_id")
    if store is not None and session_id:
        stored = store.externalize_state(session_id, state)
        state["artifact_manifest"] = stored["artifact_manifest"]
        store.repo.db.commit()
