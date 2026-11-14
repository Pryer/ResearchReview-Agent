"""工作流回归的子决策桩；真实子循环、契约和 Controller 仍正常执行。"""
import json


def delegated_response(messages, operation):
    if operation not in {"search_agent_decision", "analysis_agent_decision", "writing_agent_decision", "check_agent_decision"}:
        return None
    task = json.loads(messages[-1]["content"])["state"]["delegation"]
    if task["operation"] not in task["completed_operations"]:
        return {"name": task["operation"], "arguments": task["arguments"], "tool_call_id": "child-action"}
    return {"name": "report_to_main", "arguments": {"status": "completed", "message": "任务已执行"},
            "tool_call_id": "child-report"}


def checked_result(state):
    """桩只提交检查报告；待核验标记和版本由真实Controller结算。"""
    from app.agent.checking import build_check_report, check_input_fingerprint
    state.setdefault("claim_verification", {"valid": True, "supported": 1, "unsupported": 0,
                                          "unverified": 0, "verification_coverage": 1.0})
    build_check_report(state, check_input_fingerprint(state))


def accept_check_kwargs(fn):
    """旧组件桩只接收其声明的参数；生产代码仍传入完整检查契约。"""
    import inspect
    signature = inspect.signature(fn)
    variadic = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
    def call(state, **kwargs):
        return fn(state, **(kwargs if variadic else {k:v for k,v in kwargs.items() if k in signature.parameters}))
    return call


def completed_verifier(fn):
    """整个验证链被桩替换时，显式提供完成记录；不替换生产检查逻辑。"""
    call = accept_check_kwargs(fn)
    def verify(state, **kwargs):
        result = call(state, **kwargs)
        state.setdefault("claim_verification", {"valid":True,"supported":1,"unsupported":0,"unverified":0,"verification_coverage":1.0})
        state.setdefault("generation_quality", {"passed":True,"support_rate":1.0})
        state.setdefault("quality_gate", {"passed":True,"draft_released":True})
        return result
    return verify
