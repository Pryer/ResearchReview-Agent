"""真实规划、写后验证与 Controller 提交契约的组合回归；外部模型使用替身。"""
import json

import pytest

from app.agent import graph
from app.agent.action_contracts import validate_patch
from app.agent.action_registry import ACTION_REGISTRY
from app.agent.main_loop import MainAgentLoop
from app.agent.writing_plan import build_writing_plan


class OutlineLLM:
    def complete(self, prompt, **kwargs):
        assert kwargs['operation'] == 'induce_background_outline'
        return json.dumps({
            'paragraph_goals': [
                {'id': str(i), 'label': f'证据目标{i}', 'writing_goal': f'基于证据解释问题{i}'}
                for i in range(3)
            ],
            'comparison_dimensions': ['证据中的研究条件'],
        }, ensure_ascii=False)


def _state():
    claim = '该研究通过课堂行为识别分析学生参与度'
    return {
        'topic': '课堂行为分析', 'intent': 'generate_review',
        'required_reference_count': 40, 'start_year': 2024, 'end_year': 2026,
        'year_range_explicit': True, 'max_papers_explicit': True,
        'core_deliverables': ['research_background'],
        'claim_plans': [{'route_id': 'r1', 'claims': [{
            'claim_id': 'c1', 'claim_text': claim, 'claim_type': 'problem',
            'evidence_ids': ['p1:e001'], 'support_level': 'single',
        }]}],
        'paper_cards': [{
            'paper_id': 'p1', 'title': '测试证据', 'quality_status': 'valid',
            'evidence_source': 'abstract', 'relation_type': 'direct',
            'eligible_deliverables': ['research_background'],
            'field_claims': {'research_problem': [{
                'claim': claim, 'explicitly_reported': True, 'evidence_id': 'p1:e001',
            }]},
        }],
        'paper_details': [{'paper_id': 'p1', 'title': '测试证据', 'authors': ['测试作者'],
                           'year': 2024, 'venue': '测试期刊'}],
        'review': claim + '[1]。', 'citation_map': {'p1': 1},
        'steps': [], 'errors': [],
    }


def test_background_outline_is_local_and_cannot_be_reused_from_old_state():
    state = _state()
    plan = build_writing_plan('research_background', state, llm=OutlineLLM())
    assert plan.sections[0].claims_to_establish == [f'基于证据解释问题{i}' for i in range(3)]
    assert plan.sections[0].comparison_dimensions == ['证据中的研究条件']
    assert '_dynamic_background_outline' not in state
    state['_dynamic_background_outline'] = {'paragraph_goals': [
        {'writing_goal': '过期主题的目标'} for _ in range(3)
    ]}
    fresh = build_writing_plan('research_background', state)
    assert '过期主题的目标' not in fresh.sections[0].claims_to_establish


@pytest.mark.parametrize('operation', ['generate_deliverables', 'rewrite_sections', 'validate_result'])
def test_real_postwriting_checks_commit_without_releasing_insufficient_draft(monkeypatch, operation):
    state = _state()
    llm = OutlineLLM()
    draft = state.pop('review')
    monkeypatch.setattr(graph, '_get_llm', lambda: llm)

    def generate(current, **kwargs):
        # 不模拟整个验证链：真实规划器产生计划，后续对齐、引用检查、门禁均由生产代码执行。
        plan = build_writing_plan('research_background', current, llm=llm)
        current['writing_plans'] = [plan.model_dump(mode='json')]
        current['review'] = draft

    monkeypatch.setattr(graph, '_generate_deliverables_or_block', generate)
    # 此回归不调用外部语义模型；不替换确定性的对齐、引用授权或最终质量门禁。
    monkeypatch.setattr(graph, 'verify_claims_node', lambda current, **kwargs: current)
    if operation == 'validate_result':
        generate(state)

    def run(loop, current, *, handlers, finish_validator, **kwargs):
        result = loop.controller.execute(
            role=ACTION_REGISTRY[operation].role, operation=operation, objective='核验写作提交边界',
            state=current, handler=lambda working: handlers[operation](working, {}),
        )
        assert result.status.value == 'completed'
        assert current['claim_alignment']
        assert current['claim_citation_consistency']
        assert current['required_reference_count'] == 40
        assert (current['start_year'], current['end_year']) == (2024, 2026)
        assert not current['quality_gate']['passed']
        assert not current['quality_gate']['draft_released']
        assert not finish_validator(current)
        assert '_dynamic_background_outline' not in result.state_patch
        assert 'claim_alignment' in result.state_patch
        assert 'claim_citation_consistency' in result.state_patch
        return 'blocked'

    monkeypatch.setattr(MainAgentLoop, 'run', run)
    output = graph._run_autonomous_pipeline(state)
    assert output['status'] == 'blocked'


@pytest.mark.parametrize('operation', ['generate_deliverables', 'rewrite_sections', 'validate_result'])
def test_writing_output_remains_strict(operation):
    for patch in ({'claim_alignment': []}, {'claim_citation_consistency': 'invalid'},
                  {'_dynamic_background_outline': {}}, {'required_reference_count': 1},
                  {'unregistered_field': {}}):
        with pytest.raises(ValueError):
            validate_patch(operation, patch, [])
