"""工作流回归的子决策桩；真实子循环、契约和 Controller 仍正常执行。"""
import json


def delegated_response(messages, operation):
    if operation not in {"search_agent_decision", "analysis_agent_decision", "writing_agent_decision"}:
        return None
    task = json.loads(messages[-1]["content"])["state"]["delegation"]
    if task["operation"] not in task["completed_operations"]:
        return {"name": task["operation"], "arguments": task["arguments"], "tool_call_id": "child-action"}
    return {"name": "report_to_main", "arguments": {"status": "completed", "message": "任务已执行"},
            "tool_call_id": "child-report"}
