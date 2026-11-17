"""专业角色的局部模型策略；实际权限仍由统一Policy执行端复核。"""
import json
from app.agent.action_registry import tool_schemas, validate_arguments
from app.agent.execution_budget import check_execution
from app.agent.runtime.errors import AgentDecisionError
from app.agent.runtime.strategies import PlanAndActStrategy
from app.core.config import get_settings
from app.core.exceptions import NativeToolsUnsupportedError
from app.core.json_utils import parse_json_object
from app.agent.react_context import ContextBudgetExceeded
from app.prompt.delegation import LOCAL_REACT_PROMPT, LOCAL_PLAN_PROMPT


class _LocalProtocol:
    def __init__(self, llm):
        self.llm = llm
        self.native_tools_disabled = False

    def request(self, messages, tools, operation, decode, validate):
        native = bool(getattr(self.llm, 'native_tools_enabled', False)) and not self.native_tools_disabled
        for _ in range(get_settings().agent_decision_repair_attempts):
            check_execution()
            try:
                if native:
                    try:
                        raw = decode(self.llm.complete_tool_call(messages, tools=tools, operation=operation))
                    except NativeToolsUnsupportedError:
                        # WHY: 能力降级不消耗一次格式纠错机会；控制异常始终原样传播。
                        native = False
                        self.native_tools_disabled = True
                if not native:
                    json_messages = [dict(message) for message in messages]
                    json_messages[0]['content'] += '\n动作契约：' + json.dumps(tools, ensure_ascii=False)
                    raw = parse_json_object(self.llm.complete_messages(json_messages, response_format='json',
                        operation=operation, retry_empty=False))
                return validate(raw)
            except ContextBudgetExceeded:
                raise
            except (ValueError, TypeError, KeyError) as exc:
                messages.append({'role': 'user', 'content': '上次输出未通过协议校验，请修正：' + str(exc)[:200]})
        raise AgentDecisionError('local decision repair exhausted')


class RoleReActStrategy:
    def __init__(self, llm):
        self.protocol = _LocalProtocol(llm)

    def __call__(self, grant, context, available):
        tools = tool_schemas(list(grant.allowed_actions)) + [{'type': 'function', 'function': {
            'name': 'report_to_main', 'description': '当前任务需全局帮助时交回Main',
            'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}]
        messages = [
            {'role': 'system', 'content': LOCAL_REACT_PROMPT.format(role=grant.role.value)},
            {'role': 'user', 'content': json.dumps({'context': context,
                'allowed_actions': [*available, 'report_to_main']}, ensure_ascii=False)},
        ]
        def validate(raw):
            if not isinstance(raw, dict) or set(raw) != {'action', 'arguments'}:
                raise ValueError('decision requires only action and arguments')
            name, arguments = raw.get('action'), raw.get('arguments')
            if name not in [*available, 'report_to_main'] or not isinstance(arguments, dict):
                raise ValueError('invalid local role decision')
            if name == 'report_to_main':
                if arguments:
                    raise ValueError('report_to_main does not accept action parameters')
            else:
                validate_arguments(name, arguments)
            return name, arguments
        from app.agent.task_context import planning_task_brief
        with planning_task_brief(grant, context):
            return self.protocol.request(messages, tools, grant.role.value + '_agent_decision',
                lambda raw: {'action': raw['name'], 'arguments': raw['arguments']}, validate)


class RolePlanAndActStrategy(PlanAndActStrategy):
    def __init__(self, llm):
        self.protocol = _LocalProtocol(llm)
        super().__init__(self._plan)

    def _plan(self, grant, context, available):
        if grant.profile != 'evidence_build':
            raise ValueError('model planning is limited to evidence_build')
        choices = [{'type': 'object', 'properties': {
            'action': {'type': 'string', 'enum': [tool['function']['name']]},
            'arguments': tool['function']['parameters']}, 'required': ['action', 'arguments'],
            'additionalProperties': False} for tool in tool_schemas(list(grant.allowed_actions))]
        tools = [{'type': 'function', 'function': {'name': 'submit_local_plan',
            'description': '提出当前委派的有界研究计划', 'parameters': {
                'type': 'object', 'properties': {'steps': {'type': 'array', 'minItems': 1,
                    'maxItems': grant.max_steps, 'items': {'anyOf': choices}}},
                'required': ['steps'], 'additionalProperties': False}}}]
        messages = [{'role': 'system', 'content': LOCAL_PLAN_PROMPT},
            {'role': 'user', 'content': json.dumps({'context': context, 'allowed_actions': available}, ensure_ascii=False)}]

        def decode(raw):
            if raw.get('name') != 'submit_local_plan':
                raise ValueError('expected submit_local_plan')
            return raw['arguments']

        def validate(raw):
            if not isinstance(raw, dict) or set(raw) != {'steps'}:
                raise ValueError('plan requires only steps')
            # WHY: 先验证整份计划，避免执行前几步后才发现后续参数不合法。
            plan = PlanAndActStrategy(lambda *args: raw['steps']).propose_plan(grant, context, available)
            if plan[0]['action'] not in available:
                raise ValueError('first plan step is currently unavailable')
            return plan

        from app.agent.task_context import planning_task_brief
        with planning_task_brief(grant, context):
            return self.protocol.request(messages, tools, 'analysis_agent_plan', decode, validate)


# 兼容已接入Search入口；专业策略实现只维护一份。
SearchReActStrategy = RoleReActStrategy
