"""唯一生产运行时、历史配置升级及直接Python入口的持久化边界。"""
import ast
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent import graph
from app.agent.runtime_config import task_settings_scope
from app.core.config import Settings, get_settings
from app.database.models import ResearchRuntimeModel, ResearchTaskModel, ResearchEventModel
from tests.test_v2_delegation_runtime import db


def test_production_has_one_runtime_and_no_retired_dependency():
    root = Path(__file__).resolve().parents[1]
    retired = {'app.agent.main_loop', 'app.agent.main_policy', 'app.agent.subagents.loop', 'app.agent.delegation_context'}
    assert all(not (root / (module.replace('.', '/') + '.py')).exists() for module in retired)
    for directory in ('app', 'scripts'):
        for source in (root / directory).rglob('*.py'):
            tree = ast.parse(source.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                names = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                         [n.name for n in node.names] if isinstance(node, ast.Import) else [])
                assert not any(name in retired or name.startswith('tests.') for name in names), str(source.relative_to(root))
                if isinstance(node, ast.ClassDef):
                    assert node.name not in {'MainAgentLoop', 'SubagentLoop', 'MainAgentPolicy'}
    assert get_settings().agent_architecture_version == 'v2'
    with pytest.raises(ValidationError):
        Settings(agent_architecture_version='v1')


@pytest.mark.parametrize('architecture', [None, 'v1'])
def test_old_config_upgrade_keeps_constraints_models_and_spend(architecture):
    current = {'start_year': 2021, 'end_year': 2023, 'required_reference_count': 40,
        'agent_execution_budget': {'llm_tokens': 1234}, 'delivery_mode': 'strict',
        'effective_config': {'version': 'old', 'revision': 4, 'values': {
            'llm_model': 'frozen-model', 'agent_main_max_rounds': 12, 'review_delivery_mode': 'strict'}}}
    if architecture:
        current['effective_config']['values']['agent_architecture_version'] = architecture
    before = copy.deepcopy(current)
    with task_settings_scope(current) as settings:
        assert settings.agent_architecture_version == 'v2' and settings.llm_model == 'frozen-model'
        assert settings.agent_main_max_rounds == 12 and settings.review_delivery_mode == 'strict'
    assert all(current[k] == before[k] for k in ('start_year', 'end_year', 'required_reference_count', 'agent_execution_budget'))
    assert current['effective_config']['revision'] == 4
    assert current['effective_config']['architecture_migration']['to'] == 'v2'
    upgraded = copy.deepcopy(current)
    with task_settings_scope(current):
        pass
    assert current == upgraded


def test_unknown_config_refused_without_partial_upgrade():
    current = {'effective_config': {'values': {'agent_architecture_version': 'v99'}}}
    before = copy.deepcopy(current)
    with pytest.raises(ValueError, match='unsupported persisted'):
        with task_settings_scope(current):
            pytest.fail('unsupported runtime admitted')
    assert current == before


def test_invalid_old_settings_do_not_partially_commit_architecture_upgrade():
    current = {'effective_config': {'values': {'agent_architecture_version': 'v1', 'agent_context_window_tokens': 0}}}
    before = copy.deepcopy(current)
    with pytest.raises(ValidationError):
        with task_settings_scope(current):
            pytest.fail('invalid context admitted')
    assert current == before


def test_default_python_entry_uses_real_durable_action_and_tool_receipts(db, monkeypatch):
    from tests.test_research_delegation_service import configure
    from tests.test_subagent_react import ScriptedLLM
    from app.services import durable_execution_service
    from app.services.tool_execution import execute_tool
    calls = []
    configure(monkeypatch, calls)
    monkeypatch.setattr(durable_execution_service, 'standalone_session', lambda: Session(db.get_bind()))
    llm = ScriptedLLM({'main_agent_decision': [('delegate_agent', {
        'agent': 'search', 'profile': 'initial_research', 'goal': 'research'}), ('request_finish', {})]})
    monkeypatch.setattr(graph, '_get_llm', lambda: llm)
    def plan(current, **kwargs):
        assert durable_execution_service.active_runtime() is not None
        current.update(intent='search_papers', topic='test', core_deliverables=[], max_papers=1,
            required_reference_count=1, research_plan={'task_graph': []})
    monkeypatch.setattr(graph, 'plan_node', plan)
    search = graph._search_rank_with_refinement
    monkeypatch.setattr(graph, '_search_rank_with_refinement', lambda *a, **kw:
        execute_tool('paper_search', lambda: search(*a, **kw)))
    output = graph.run_research_agent('帮我找几篇关于目标检测的论文', initial_state={'session_id': 'standalone-test'})
    assert output['status'] == 'success' and calls == ['search']
    assert output['research_state']['architecture_version'] == 2
    assert output['research_state']['task_main_control']
    assert output['research_state']['delegation_control']
    assert durable_execution_service.active_runtime() is None
    row = db.get(ResearchRuntimeModel, 'standalone-test', populate_existing=True)
    assert row.owner is None and row.version > 0
    saved = json.loads(row.state_json)
    assert saved['architecture_version'] == 2 and saved['task_main_control']
    tasks = list(db.scalars(select(ResearchTaskModel).where(ResearchTaskModel.session_id == 'standalone-test')))
    assert len(tasks) == 1 and tasks[0].status == 'completed'
    events = list(db.scalars(select(ResearchEventModel).where(ResearchEventModel.session_id == 'standalone-test')))
    assert sum(e.event_type == 'tool_started' for e in events) == 1
    assert sum(e.event_type == 'tool_finished' for e in events) == 1


def test_result_projection_preserves_v2_recovery_identity():
    state = {'architecture_version': 2, 'active_delegation_id': 'active',
        'delegation_control': {'active': {'rounds': 2}}, 'task_main_control': {'stalled': 1},
        'legacy_checkpoint_migration': {'version': 1}}
    saved = graph._build_output(state)['research_state']
    assert all(saved[key] == value for key, value in state.items())


def test_inventory_is_read_only_and_aggregates_without_private_state(db):
    from scripts.audit_runtime_migration import audit
    db.add(ResearchRuntimeModel(session_id='private-session', state_json=json.dumps({
        'topic': 'private-topic', 'active_subagent': 'unknown', 'subagent_sessions': {}}), budget_json='{}'))
    db.add(ResearchRuntimeModel(session_id='malformed', state_json='[]', budget_json='{}'))
    db.commit()
    before = [(row.session_id, row.state_json, row.budget_json) for row in db.scalars(select(ResearchRuntimeModel))]
    result = audit(db.get_bind())
    assert result['runtime_rows'] == 2 and result['classification']['historical_active_needs_review'] == 1
    assert result['classification']['malformed_snapshot'] == 1
    assert 'private-' not in json.dumps(result)
    assert [(row.session_id, row.state_json, row.budget_json) for row in db.scalars(select(ResearchRuntimeModel))] == before
