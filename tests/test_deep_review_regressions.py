from collections import Counter
from datetime import datetime, timedelta
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.claim_plan import _count_independent_sources, _determine_support_level
from app.agent.execution_budget import budget_scope, budgeted_create
from app.database.models import Base, ResearchSessionModel, ResearchJobModel
from app.services.research_job_service import ResearchJobService
from app.tools.hybrid_retrieval import LexicalIndex, tokenize


@pytest.mark.parametrize('doi', [False, True])
def test_same_author_missing_dataset_cannot_become_strong_with_distinct_dois(doi):
    ids = [f'p{i}' for i in range(4)]
    cards = {pid: {'authors': ['  Same Author  ' if i % 2 else 'same author'],
                   'doi': f'10.1234/{pid}' if doi else None, 'dataset': None} for i, pid in enumerate(ids)}
    assert _count_independent_sources(ids, cards) == 1
    assert _determine_support_level(4, 'finding', ids, cards) == 'single'


def test_doi_duplicates_and_author_dataset_dependencies_combine_transitively():
    cards = {
        'a': {'doi': 'https://doi.org/10.1234/A', 'authors': ['author A'], 'dataset': 'known'},
        'b': {'doi': '10.1234/a', 'authors': ['author B'], 'dataset': 'known'},
        'c': {'doi': '10.1234/c', 'authors': ['AUTHOR B'], 'dataset': 'known'},
    }
    assert _count_independent_sources(['a', 'b', 'c', 'a'], cards) == 1
    cards['d'] = {'doi': '10.1234/d', 'authors': ['author C'], 'dataset': 'known'}
    assert _count_independent_sources(list(cards), cards) == 2


def test_unknown_authors_do_not_inflate_independence_and_known_datasets_remain_distinct():
    ids = ['a', 'b', 'c', 'd']
    assert _count_independent_sources(ids, {}) == 1
    assert _determine_support_level(4, 'finding', ids, {}) == 'single'
    cards = {'a': {'authors': ['author'], 'dataset': 'A'},
             'b': {'authors': ['author'], 'dataset': 'B'}}
    assert _count_independent_sources(ids, cards) == 2
    assert _count_independent_sources([], {}) == 0


def test_legacy_strong_plan_is_weakened_by_analysis_and_blocked_by_read_only_check():
    from copy import deepcopy
    from app.agent.claim_plan import enforce_claim_evidence_gate
    from app.agent.checking import build_check_report, check_input_fingerprint
    cards = [{'paper_id': f'p{i}', 'authors': ['same author'], 'dataset': None,
              'evidence_spans': [{'evidence_id': f'p{i}:e1', 'text': '研究报告一个结果。',
                                  'source_type': 'abstract', 'source_url': 'https://example.org/paper'}]}
             for i in range(4)]
    plans = [{'route_id': 'r', 'claims': [{'claim_id': 'c', 'claim_type': 'finding',
              'claim_text': '研究报告一个结果。', 'support_level': 'strong',
              'evidence_ids': [f'p{i}:e1' for i in range(4)]}]}]
    before = deepcopy(plans)
    state = {'review': '研究报告一个结果[1]。', 'paper_cards': cards, 'claim_plans': plans,
             'quality_gate': {'passed': True, 'draft_released': True},
             'claim_verification': {'factual_claims': 1, 'verification_coverage': 1}}
    report = build_check_report(state, check_input_fingerprint(state))
    assert report.verdict == 'needs_revision'
    assert any(i.code == 'claim_source_independence_overstated' for i in report.issues)
    assert not state['quality_gate']['passed'] and not state['quality_gate']['draft_released']
    assert plans == before
    repaired, audit = enforce_claim_evidence_gate(plans, cards)
    assert repaired[0]['claims'][0]['support_level'] == 'single'
    assert repaired[0]['claims'][0]['independent_source_count'] == 1
    assert audit['weakened_claims'] == 1


def test_claim_source_policy_change_invalidates_prior_check_and_draft(monkeypatch):
    from app.agent.checking import check_input_fingerprint, check_is_current
    from app.agent.graph import _draft_fingerprint
    state = {'review': '当前研究综述。'}
    state['check_report'] = {'input_fingerprint': check_input_fingerprint(state)}
    old_draft = _draft_fingerprint(state)
    assert check_is_current(state)
    monkeypatch.setattr('app.agent.claim_plan.CLAIM_SUPPORT_POLICY_VERSION', 'future-policy')
    assert not check_is_current(state)
    assert _draft_fingerprint(state) != old_draft


def test_bm25_index_reuses_corpus_and_preserves_reference_scores():
    import app.tools.hybrid_retrieval as module
    from rank_bm25 import BM25Okapi
    texts = ['课堂行为识别 teacher interaction', 'teacher observation', '', 'unrelated battery material']
    queries = ['课堂行为', 'teacher', 'absent', '', 'battery material']
    corpus = [tokenize(t) for t in texts]
    reference = BM25Okapi(corpus)
    with patch.object(module, 'tokenize', wraps=tokenize) as called:
        index = LexicalIndex(texts)
        actual = [index.rank(q, 10) for q in queries]
    counts = Counter(c.args[0] for c in called.call_args_list)
    assert counts[texts[0]] == counts[texts[1]] == counts[texts[3]] == 1
    for query, ranking in zip(queries, actual):
        terms = tokenize(query)
        scores = reference.get_scores(terms)
        expected = [(i, float(scores[i])) for i, words in enumerate(corpus) if set(terms).intersection(words)]
        expected.sort(key=lambda item: (-item[1], item[0]))
        assert ranking == expected
    assert LexicalIndex([]).rank('anything', 3) == []
    assert LexicalIndex(['', '']).rank('anything', 3) == []


@pytest.mark.parametrize('job_status,terminal', [('running', 'failed'), ('cancel_requested', 'cancelled')])
def test_restart_updates_session_but_does_not_overwrite_newer_jobs_or_completed_session(monkeypatch, job_status, terminal):
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine)
    old = datetime(2026, 1, 1)
    with maker() as db:
        for sid in ('interrupted', 'newer', 'finished'):
            db.add(ResearchSessionModel(session_id=sid, status='completed' if sid == 'finished' else 'running',
                                        state_json=json.dumps({'keep': 'original evidence'})))
            db.add(ResearchJobModel(job_id=sid, session_id=sid, status=job_status, created_at=old))
        db.add(ResearchJobModel(job_id='new-completed', session_id='newer', status='completed', created_at=old+timedelta(days=1)))
        db.commit()
    monkeypatch.setattr('app.services.research_job_service.SessionLocal', maker)
    monkeypatch.setattr('app.services.research_job_service._finish_job_report', lambda *args: None)
    ResearchJobService.recover_after_restart()
    with maker() as db:
        assert db.get(ResearchSessionModel, 'interrupted').status == terminal
        assert db.get(ResearchSessionModel, 'newer').status == 'running'
        assert db.get(ResearchSessionModel, 'finished').status == 'completed'
        assert json.loads(db.get(ResearchSessionModel, 'interrupted').state_json)['keep'] == 'original evidence'
        assert db.get(ResearchJobModel, 'interrupted').status == terminal
    engine.dispose()


def test_compact_reservation_keeps_text_and_full_output_limit():
    prompt = [{'role': 'user', 'content': '保留  中文空格\n证据 "引文"'}]
    payload = {'messages': prompt}
    expected = len(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()) + 128
    legacy = len(json.dumps(payload, ensure_ascii=False).encode()) + 128
    client = NS(chat=NS(completions=NS(create=lambda **kwargs: NS(usage=NS(prompt_tokens=12, completion_tokens=3)))))
    state = {}
    with budget_scope(state):
        budgeted_create(client, messages=prompt, max_tokens=1024)
    detail = state['agent_execution_budget']['last_token_reservation']
    assert detail['input_upper_bound'] == expected <= legacy
    assert detail['output_limit'] == 1024
    assert detail['reported_input_tokens'] == 12
    assert prompt[0]['content'] == '保留  中文空格\n证据 "引文"'


def test_late_response_does_not_overwrite_newer_reservation_diagnostic():
    inner = NS(chat=NS(completions=NS(create=lambda **kw: NS(usage=NS(prompt_tokens=5, completion_tokens=1)))))
    def outer_create(**kwargs):
        budgeted_create(inner, messages=[{'role': 'user', 'content': 'inner'}], max_tokens=10)
        return NS(usage=NS(prompt_tokens=30, completion_tokens=2))
    outer = NS(chat=NS(completions=NS(create=outer_create)))
    state = {}
    with budget_scope(state):
        budgeted_create(outer, messages=[{'role': 'user', 'content': 'outer'}], max_tokens=20)
    detail = state['agent_execution_budget']['last_token_reservation']
    assert detail['output_limit'] == 10 and detail['reported_input_tokens'] == 5


def test_restart_session_write_failure_rolls_back_job_transition(monkeypatch):
    from sqlalchemy import event
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine)
    with maker() as db:
        db.add(ResearchSessionModel(session_id='s', status='running'))
        db.add(ResearchJobModel(job_id='j', session_id='s', status='running'))
        db.commit()
    def fail_session_write(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith('UPDATE research_sessions'):
            raise RuntimeError('injected session write failure')
    event.listen(engine, 'before_cursor_execute', fail_session_write)
    monkeypatch.setattr('app.services.research_job_service.SessionLocal', maker)
    with pytest.raises(RuntimeError, match='injected'):
        ResearchJobService.recover_after_restart()
    event.remove(engine, 'before_cursor_execute', fail_session_write)
    with maker() as db:
        assert db.get(ResearchJobModel, 'j').status == 'running'
        assert db.get(ResearchSessionModel, 's').status == 'running'
    engine.dispose()
