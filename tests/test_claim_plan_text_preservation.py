"""Claim Planner 去重不得损坏发送给蕴含核验和写作的原始主张。"""
import pytest

from app.agent.claim_plan import build_background_claim_plan, build_claim_plans, enforce_claim_evidence_gate


@pytest.mark.parametrize('planner', ['background', 'route'])
def test_deduplication_preserves_complete_claim_and_its_evidence(planner):
    text = ('We do NOT claim that retrieval always improves factual accuracy; '
            'the reported improvement applies only to the evaluated datasets.')
    evidence_id = 'arxiv:example:e001'
    card = {
        'paper_id': 'arxiv:example', 'title': 'Retrieval evaluation',
        'evidence_state': {'access_level': 'full_text'},
        'evidence_spans': [{'evidence_id': evidence_id, 'text': text, 'source_type': 'full_text',
                            'char_start': 0, 'char_end': len(text)}],
        'field_claims': {'research_problem': [{'claim': text, 'source_text': text,
            'evidence_id': evidence_id, 'explicitly_reported': True}]},
    }
    if planner == 'background':
        plans = build_background_claim_plan({'paragraph_goals': [
            {'id': 'g', 'goal': 'retrieval evaluation'}]}, [card])
    else:
        plans = build_claim_plans([{'route_id': 'r', 'name': 'Retrieval evaluation',
                                   'core_paper_ids': [card['paper_id']]}], [card])
    assert plans[0]['claims'][0]['claim_text'] == text
    assert plans[0]['claims'][0]['evidence_ids'] == [evidence_id]

    class Verifier:
        def complete(self, prompt, **kwargs):
            # 端到端检查 Planner 输出进入真实蕴含协议，否定词及80字符后的限定条件完整保留。
            assert text in prompt
            return '{"results": [{"claim_id": "' + plans[0]['claims'][0]['claim_id'] + '\", "label": "entailed", "confidence": 0.99}]}'

    retained, report = enforce_claim_evidence_gate(plans, [card], llm=Verifier())
    assert report['entailment_checked_claims'] == 1
    assert retained[0]['claims'][0]['claim_text'] == text
