"""润色只修改已发送主张的表述，不改变身份、证据授权或数字。"""
from copy import deepcopy
import json

import pytest

from app.agent.claim_plan import _llm_refine_claims, enforce_claim_evidence_gate
from app.agent.execution_budget import AgentBudgetExceeded, AgentExecutionStale


def claims(count=3):
    return [{"claim_id": f"c{i}", "claim_text": f"模型{i}在当前实验中达到94.2%的准确率。",
             "claim_type": "finding", "support_level": "single", "allowed_language": "一项研究报告",
             "evidence_ids": [f"p{i}:e1"]} for i in range(count)]


class Editor:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def complete(self, prompt, **kwargs):
        self.calls.append(prompt)
        sent, _ = json.JSONDecoder().raw_decode(prompt.split("当前主张：\n", 1)[1])
        refined = [{"claim_id": c["claim_id"], "claim_text": "研究报告：" + c["claim_text"]}
                   for c in reversed(sent)]
        return json.dumps({"refined_claims": self.response(refined) if self.response else refined}, ensure_ascii=False)


def test_refines_only_sent_twelve_and_maps_reordered_response_by_id():
    source = claims(15)
    before = deepcopy(source)
    def extra_fields(rows):
        for row in rows:
            row.update(evidence_ids=['wrong:evidence'], support_level='established', claim_type='trend')
        return rows
    editor = Editor(extra_fields)
    assert _llm_refine_claims(source, '路线', '问题', editor) is source
    sent, _ = json.JSONDecoder().raw_decode(editor.calls[0].split('当前主张：\n', 1)[1])
    assert len(editor.calls) == 1 and len(sent) == 12
    assert [c['claim_id'] for c in sent] == [c['claim_id'] for c in before[:12]]
    for i, current in enumerate(source[:12]):
        assert current == {**before[i], 'claim_text': '研究报告：' + before[i]['claim_text']}
    assert source[12:] == before[12:]


@pytest.mark.parametrize('corrupt', [
    lambda r: r[:-1],
    lambda r: r + [r[0]],
    lambda r: {'rows': r},
    lambda r: r[:-1] + [None],
    lambda r: r[:-1] + [{'claim_text': '缺少身份'}],
    lambda r: r[:-1] + [{**r[-1], 'claim_id': 'unknown'}],
    lambda r: r[:-1] + [{**r[-1], 'claim_id': r[0]['claim_id']}],
    lambda r: r[:-1] + [{**r[-1], 'claim_text': None}],
    lambda r: r[:-1] + [{**r[-1], 'claim_text': '  '}],
])
def test_bad_batch_is_rejected_before_any_write(corrupt, caplog):
    source = claims()
    before = deepcopy(source)
    _llm_refine_claims(source, '路线', '问题', Editor(corrupt))
    assert source == before
    assert 'LLM claim refinement skipped' in caplog.text


@pytest.mark.parametrize('original, changed', [
    ('准确率为94.2%。', '准确率为94.3%。'),
    ('准确率为94.2%。', '准确率为94.2。'),
    ('准确率为94.2%。', '准确率较高。'),
    ('基线为90.1%，新方法为94.2%。', '基线为94.2%，新方法为90.1%。'),
    ('两次结果均为94.2%，分别为94.2%。', '结果为94.2%。'),
    ('变化为-3，阈值为1e-3。', '变化为3，阈值为1e-3。'),
    ('阈值为1e-3。', '阈值为1e3。'),
    ('样本数为1,000。', '样本数为1000。'),
    ('阈值为.05。', '阈值为05。'),
    ('变化为−.5。', '变化为.5。'),
    ('只有定性结果。', '结果提升3%。'),
])
def test_numeric_change_keeps_original_but_other_valid_claims_can_refine(original, changed, caplog):
    source = claims(2)
    source[0]['claim_text'] = original
    before = deepcopy(source)
    def change(rows):
        next(row for row in rows if row['claim_id'] == 'c0')['claim_text'] = changed
        return rows
    _llm_refine_claims(source, '路线', '问题', Editor(change))
    assert source[0] == before[0]
    assert source[1] == {**before[1], 'claim_text': '研究报告：' + before[1]['claim_text']}
    assert 'numeric_tokens_changed' in caplog.text


def test_signed_scientific_and_percentage_tokens_can_be_retained_exactly():
    source = claims(1)
    source[0]['claim_text'] = '变化为−3.5%，阈值为1e-3，样本数为1,000。'
    original = source[0]['claim_text']
    _llm_refine_claims(source, '路线', '问题', Editor())
    assert source[0]['claim_text'] == '研究报告：' + original


@pytest.mark.parametrize('mutate', [
    lambda c: c[0].pop('claim_id'),
    lambda c: c[1].update(claim_id=c[0]['claim_id']),
])
def test_ambiguous_input_identity_skips_model_call(mutate):
    source = claims()
    mutate(source)
    before = deepcopy(source)
    editor = Editor()
    _llm_refine_claims(source, '路线', '问题', editor)
    assert source == before and not editor.calls


@pytest.mark.parametrize('error', [RuntimeError, AgentBudgetExceeded, AgentExecutionStale])
def test_optional_refinement_failure_preserves_text_but_control_errors_propagate(error):
    class Broken:
        def complete(self, *args, **kwargs):
            raise error('stopped')
    source = claims()
    before = deepcopy(source)
    if error is RuntimeError:
        _llm_refine_claims(source, '路线', '问题', Broken())
    else:
        with pytest.raises(error):
            _llm_refine_claims(source, '路线', '问题', Broken())
    assert source == before


def test_empty_input_does_not_call_model():
    editor = Editor()
    assert _llm_refine_claims([], '路线', '问题', editor) == []
    assert not editor.calls


def test_number_guard_does_not_replace_downstream_semantic_gate():
    source = claims(1)
    source[0]['claim_text'] = '模型在当前实验中未提升准确率。'
    def unsupported(rows):
        rows[0]['claim_text'] = '模型在当前实验中提升准确率。'
        return rows
    _llm_refine_claims(source, '路线', '问题', Editor(unsupported))
    class Verifier:
        def complete(self, *args, **kwargs):
            return '{"results":[{"claim_id":"c0","label":"insufficient","confidence":0.99}]}'
    cards = [{'paper_id': 'p0', 'evidence_spans': [{'evidence_id': 'p0:e1',
              'text': '模型在当前实验中未提升准确率。', 'source_type': 'abstract',
              'source_url': 'https://example.org/paper'}]}]
    repaired, audit = enforce_claim_evidence_gate([{'route_id': 'r', 'claims': source}], cards, llm=Verifier())
    assert repaired[0]['claims'] == []
    assert audit['entailment_failed_claims'] == 1
