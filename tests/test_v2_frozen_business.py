"""冻结合成材料跑通任务级四角色；保留真实写作、引用、Check和最终门禁。"""
import copy
import hashlib
import json
from pathlib import Path
import pytest

from app.agent import graph
from app.agent.execution_budget import budget_scope
from app.agent.task_main import TaskMainLoop
from app.core.config import get_settings
from app.schemas.deliverable_schema import WritingPlan
from app.services.durable_execution_service import durable_scope
from tests.test_v2_delegation_runtime import db, state
from tests.test_subagent_react import ScriptedLLM

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / 'examples/architecture-v2/frozen-business-input.json').read_text(encoding='utf-8'))


@pytest.fixture(autouse=True)
def artifact_store(db):
    from app.services.research_artifact_service import ResearchArtifactService
    from app.services.research_execution_service import _STORE
    token = _STORE.set(ResearchArtifactService(db))
    try:
        yield
    finally:
        _STORE.reset(token)


def materials():
    cards, claims = [], []
    for paper, text in zip(FIXTURE['papers'], FIXTURE['sentences']):
        pid, eid = paper['paper_id'], paper['paper_id'] + ':e1'
        cards.append({**paper, 'quality_status': 'partial', 'relation_type': 'direct',
            'evidence_source': 'abstract', 'eligible_deliverables': ['research_background'],
            'evidence_spans': [{'evidence_id': eid, 'text': text, 'source_type': 'abstract',
                'source_url': paper['url'], 'char_start': 0, 'char_end': len(text),
                'source_text_hash': hashlib.sha256(text.encode()).hexdigest()}],
            'field_claims': {'research_problem': [{'claim': text, 'source_text': text, 'evidence_id': eid,
                'explicitly_reported': True, 'source_section': 'Abstract', 'evidence_level': 'abstract'}]}})
        claims.append({'claim_id': pid + ':claim', 'claim_text': text, 'claim_type': 'background_fact',
            'evidence_ids': [eid], 'support_level': 'single', 'evidence_access_limit': 'abstract_only'})
    return cards, [{'route_id': 'r', 'route_name': '研究背景', 'claims': claims}]


def writing_plan():
    return WritingPlan(deliverable_type='research_background', purpose='离线控制流验证', organizing_strategy='evidence_driven',
        sections=[{'id': 'background_body', 'title': '研究背景', 'purpose': '说明检索增强生成的研究背景',
            'supporting_paper_ids': [p['paper_id'] for p in FIXTURE['papers']], 'supporting_claim_ids': []}],
        citation_policy={'minimum_unique_references': 3})


class FixtureLLM(ScriptedLLM):
    def __init__(self, *, native=True, label='entailed'):
        self.label, self.semantic_calls = label, []
        steps = [{'action': name, 'arguments': {}} for name in ('extract_paper_cards', 'validate_routes', 'plan_claims')]
        super().__init__({'main_agent_decision': [
            ('delegate_agent', {'agent': 'search', 'profile': 'initial_research', 'goal': 'collect fixture sources'}),
            ('delegate_agent', {'agent': 'analysis', 'profile': 'evidence_build', 'goal': 'authorize fixture evidence'}),
            ('delegate_agent', {'agent': 'writing', 'profile': 'draft_generation', 'goal': 'write fixture review'}),
            ('delegate_agent', {'agent': 'check', 'profile': 'independent_check', 'goal': 'check current candidate'}),
            ('request_finish', {}), ('report_blocked', {'reason': '冻结材料未通过最终门禁'})],
            'search_agent_decision': [('search_and_rank', {}), ('fetch_metadata', {})],
            'analysis_agent_decision': [(step['action'], {}) for step in steps],
            'analysis_agent_plan': [('submit_local_plan', {'steps': steps})]}, native=native)

    def complete_messages(self, messages, *, operation, **kwargs):
        if operation == 'analysis_agent_plan':
            _, args = self._next(messages, operation)
            return json.dumps(args)
        return super().complete_messages(messages, operation=operation, **kwargs)

    def complete(self, prompt, *, operation='completion', **kwargs):
        if operation == 'verify_claim_entailment':
            from app.services.tool_execution import current_tool_context
            assert current_tool_context().action == 'validate_result'
            payload, _ = json.JSONDecoder().raw_decode(prompt.split('待验证项目：', 1)[1].lstrip())
            self.semantic_calls.extend(payload)
            return json.dumps({'results': [{'claim_id': item['claim_id'], 'label': self.label,
                'confidence': 0.99, 'reason': 'offline scripted verdict'} for item in payload]})
        assert operation.startswith(('write_', 'repair_', 'polish_', 'backfill_')), operation
        return FIXTURE['body']


def prepare(monkeypatch, *, native=True, local='deterministic', label='entailed'):
    current = state()
    current.update(user_query='写2022年检索增强生成研究背景，引用3篇', topic=FIXTURE['topic'],
        canonical_topic=FIXTURE['topic'], intent='generate_review', core_deliverables=['research_background'],
        required_reference_count=3, max_papers=3, max_papers_explicit=True, start_year=2022, end_year=2022,
        year_range_explicit=True, delivery_mode='strict', allow_evidence_expansion=True)
    settings = get_settings()
    monkeypatch.setattr(settings, 'agent_v2_local_strategy', 'react' if local == 'react' else 'deterministic')
    monkeypatch.setattr(settings, 'agent_v2_analysis_strategy', 'plan_and_act' if local == 'plan_and_act' else 'inherit')
    llm = FixtureLLM(native=native, label=label)
    cards, claims = materials()
    calls = []
    def search(w, args):
        from app.services.retrieval_ranking_service import admission_scope_fingerprint, _document
        calls.append('search')
        papers = copy.deepcopy(FIXTURE['papers'])
        for paper in papers:
            paper.update(_screening_decision='include', _topic_relation='direct',
                _eligible_deliverables=['research_background'], _semantic_conditions_confirmed=True,
                _semantic_contract_complete=True, _screening_confidence=0.99)
            paper['_screened_content_fingerprint'] = _document(paper).content_fingerprint
            paper['_screened_scope_fingerprint'] = admission_scope_fingerprint(w)
        w.update(candidate_papers=copy.deepcopy(papers), ranked_papers=papers)
    def fetch(w, args):
        calls.append('fetch')
        w['paper_details'] = copy.deepcopy(w['ranked_papers'])
    def extract(w, args):
        calls.append('extract')
        w['paper_cards'] = copy.deepcopy(cards)
    def routes(w, args):
        calls.append('routes')
        w['validated_routes'] = [{'route_id': 'r', 'route_name': '研究背景', 'paper_ids': [p['paper_id'] for p in FIXTURE['papers']]}]
    def authorize(w, args):
        calls.append('claims')
        w['claim_plans'] = copy.deepcopy(claims)
    def generate(w, *, llm, **kwargs):
        from app.tools.write_deliverable import write_deliverable
        from app.tools.validate_deliverable import validate_deliverable
        calls.append('write')
        plan = writing_plan()
        w['writing_plans'] = [plan.model_dump(mode='json')]
        w['writing_version'] = int(w.get('writing_version') or 0) + 1
        w['review'] = write_deliverable(plan, w, llm=llm)
        w['deliverable_validation'] = [validate_deliverable(w['review'], plan, w)]
    monkeypatch.setattr(graph, '_generate_deliverables_or_block', generate)
    handlers, finish = graph.build_research_actions(current, llm=llm)
    handlers.update(search_and_rank=search, fetch_metadata=fetch, extract_paper_cards=extract,
        validate_routes=routes, plan_claims=authorize)
    return current, llm, handlers, finish, calls


@pytest.mark.parametrize('native', [False, True], ids=['json', 'native'])
@pytest.mark.parametrize('local', ['deterministic', 'react', 'plan_and_act'])
def test_frozen_four_role_chain_honors_actual_publication_gates(db, monkeypatch, native, local):
    from app.agent.checking import check_is_current
    current, llm, handlers, finish, calls = prepare(monkeypatch, native=native, local=local)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        status = TaskMainLoop(llm).run(current, handlers=handlers, finish_validator=finish)
        if status != 'completed':
            pytest.fail(json.dumps({'status': status, 'calls': calls, 'gate': current.get('quality_gate'),
                'structure': current.get('deliverable_validation'),
                'claims': [{k: claim.get(k) for k in ('sentence', 'issues', 'support_status', 'verification_status')}
                    for claim in current.get('claim_verification', {}).get('claims', [])]}, ensure_ascii=True))
        assert check_is_current(current) and current['check_report']['verdict'] == 'passed'
        assert llm.semantic_calls
        assert calls == ['search', 'fetch', 'extract', 'routes', 'claims', 'write']
        assert current['unique_valid_cited_paper_count'] == 3
        output = graph._build_output(current)
        assert output['body'] and len(output['references']) == 3
        assert not any(token in output['body'] for token in (*[p['paper_id'] for p in FIXTURE['papers']], 'artifact://', 'state://', 'localhost'))


def test_scripted_contradiction_is_blocked_by_real_check_and_finish(db, monkeypatch):
    current, llm, handlers, finish, calls = prepare(monkeypatch, label='contradicted')
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        status = TaskMainLoop(llm).run(current, handlers=handlers, finish_validator=finish)
        assert status == 'blocked'
        assert current['check_report']['verdict'] != 'passed'
        assert not current['quality_gate']['passed']
        assert not graph._build_output(current)['body']
        assert current['required_reference_count'] == 3


def test_finish_before_independent_check_cannot_expose_candidate(db, monkeypatch):
    current, llm, handlers, finish, calls = prepare(monkeypatch)
    llm.scripts['main_agent_decision'] = iter([
        ('delegate_agent', {'agent': 'search', 'profile': 'initial_research', 'goal': 'collect'}),
        ('delegate_agent', {'agent': 'analysis', 'profile': 'evidence_build', 'goal': 'authorize'}),
        ('delegate_agent', {'agent': 'writing', 'profile': 'draft_generation', 'goal': 'write'}),
        ('request_finish', {}), ('report_blocked', {'reason': '候选必须独立核验'})])
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        assert TaskMainLoop(llm).run(current, handlers=handlers, finish_validator=finish) == 'blocked'
        assert current['pending_result_validation'] and not current.get('check_report')
        assert not llm.semantic_calls and not graph._build_output(current)['body']


def test_final_gate_enforces_reference_shortfall_without_reducing_goal(db, monkeypatch):
    current, llm, handlers, finish, calls = prepare(monkeypatch)
    current.update(required_reference_count=5, max_papers=5,
        user_query=current['user_query'].replace('引用3篇', '引用5篇'))
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        assert TaskMainLoop(llm).run(current, handlers=handlers, finish_validator=finish) == 'blocked'
        assert current['required_reference_count'] == 5 and current['start_year'] == current['end_year'] == 2022
        assert any(issue['code'] == 'minimum_cited_references_not_met' for issue in current['quality_gate']['blocking_issues'])
        assert not graph._build_output(current)['body']


def test_changed_body_cannot_reuse_a_previously_passed_check(db, monkeypatch):
    current, llm, handlers, finish, calls = prepare(monkeypatch)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        assert TaskMainLoop(llm).run(current, handlers=handlers, finish_validator=finish) == 'completed'
        current['review'] += '\n\n未经本轮核验的新结论。'
        assert not finish(current)
        assert not graph._build_output(current)['body']


def test_section_revision_hands_off_to_check_again_before_finish(db, monkeypatch):
    from app.agent.checking import check_is_current
    from tests.test_tool_execution import events
    current, llm, handlers, finish, calls = prepare(monkeypatch)
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        assert TaskMainLoop(llm).run(current, handlers=handlers, finish_validator=finish) == 'completed'
        old_version = current['writing_version']
        llm.scripts['main_agent_decision'] = iter([
            ('delegate_agent', {'agent': 'writing', 'profile': 'section_revision', 'goal': 'revise approved section',
                'scope': {'section_ids': ['background_body']}}),
            ('delegate_agent', {'agent': 'check', 'profile': 'independent_check', 'goal': 'check revision'}),
            ('request_finish', {})])
        status = TaskMainLoop(llm).run(current, handlers=handlers, finish_validator=finish)
        assert status == 'completed', (current.get('errors'), current.get('task_main_control'),
            [(item['action'], item.get('status'), item.get('error')) for item in current['main_agent_trajectory']['memory']])
        assert current['writing_version'] == old_version + 1
        assert check_is_current(current) and not current['pending_result_validation']
        assert [event['operation'] for event in events(db, 'task_started')].count('validate_result') == 2


def test_current_failed_check_dispatches_scoped_revision_with_fresh_recheck(db, monkeypatch):
    from app.agent.checking import check_is_current
    from tests.test_tool_execution import events

    current, llm, handlers, finish, _ = prepare(monkeypatch)
    generate = handlers['generate_deliverables']
    rewrite = handlers['rewrite_sections']

    def first_draft(working, arguments):
        generate(working, arguments)
        working['deliverable_validation'][0].update(valid=False, errors=['初稿缺少要求的结构'])

    def revise(working, arguments):
        assert arguments == {'section_ids': ['background_body']}
        return rewrite(working, arguments)

    handlers['generate_deliverables'] = first_draft
    handlers['rewrite_sections'] = revise
    llm.scripts['main_agent_decision'] = iter([
        ('delegate_agent', {'agent': 'search', 'profile': 'initial_research', 'goal': 'collect'}),
        ('delegate_agent', {'agent': 'analysis', 'profile': 'evidence_build', 'goal': 'authorize'}),
        ('delegate_agent', {'agent': 'writing', 'profile': 'draft_generation', 'goal': 'write'}),
        ('delegate_agent', {'agent': 'check', 'profile': 'independent_check', 'goal': 'check first draft'}),
        ('request_finish', {}),
    ])
    with budget_scope(current), durable_scope(db, 'v2', current, current['agent_execution_budget']):
        assert TaskMainLoop(llm).run(current, handlers=handlers, finish_validator=finish) == 'completed'
        operations = [event['operation'] for event in events(db, 'task_started')]
        assert operations[-3:] == ['validate_result', 'rewrite_sections', 'validate_result']
        assert check_is_current(current) and current['check_report']['verdict'] == 'passed'
        assert len(graph._build_output(current)['references']) == 3
