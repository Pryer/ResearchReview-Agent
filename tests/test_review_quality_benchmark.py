import copy
import json
import socket
from pathlib import Path

import pytest

from scripts.benchmark_review_quality import (
    check_cases, confusion, digest, make_cases, offline, reassess, revision_replay,
)


def test_unfinished_verification_is_not_a_miss_or_true_negative():
    metrics = confusion([{'expected_error': True, 'detected': None},
                         {'expected_error': False, 'detected': False}])
    assert metrics['unverified'] == 1 and metrics['fn'] == 0
    assert metrics['coverage'] == .5
    assert metrics['error_detection_rate'] is None


def test_replay_blocks_socket_connections():
    with offline(), socket.socket() as client, pytest.raises(RuntimeError, match='network forbidden'):
        client.connect(('example.com', 443))


def test_saved_rankings_require_exact_inputs_and_unique_identities(tmp_path):
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps({'tasks': [{'task_id': 't', 'candidate_papers': [{'paper_id': 'p'}]}]}))
    qrels = tmp_path / 'qrels.jsonl'
    qrels.write_text(json.dumps({'task_id': 't', 'paper_id': 'p', 'relevance': 2}))
    report = tmp_path / 'ranking.json'
    value = {'input_sha256': {'snapshot': digest(snapshot), 'qrels': digest(qrels)},
             'tasks': [{'task_id': 't', 'raw_rankings': {m: ['p'] for m in ('bm25', 'dense', 'rrf', 'rrf_ce')}}]}
    report.write_text(json.dumps(value))
    result = reassess(report, snapshot, qrels)
    assert result['macro']['dense']['10']['precision'] == .1
    value['tasks'][0]['raw_rankings']['dense'] = ['p', 'p']
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='identity'):
        reassess(report, snapshot, qrels)
    qrels.write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        reassess(report, snapshot, qrels)


@pytest.fixture
def frozen():
    return json.loads((Path(__file__).parent / 'fixtures/writer_t06_20261110.json').read_text(encoding='utf-8'))


def test_controlled_cases_are_single_faults_and_use_production_verifier(frozen):
    cases = make_cases(frozen)
    assert len(cases) == 12
    for base, unknown, number in zip(cases[::3], cases[1::3], cases[2::3]):
        assert unknown['source_quote'] == number['source_quote'] == base['source_quote']
        assert unknown['card'] == number['card'] == base['card']
        assert '987.654%' in number['sentence']
        assert '987.654%' not in base['sentence']
    with offline():
        result = check_cases(cases)
    assert result['case_count'] == 12
    assert result['overall']['unverified'] == 0
    assert result['citation_precision'] is None


def test_extraction_replay_preserves_input_and_removed_paper_stays_removed(frozen):
    original = copy.deepcopy(frozen)
    with offline():
        result = revision_replay(frozen)
    assert frozen == original
    assert result['same_retained_evidence'] is True
    assert [p['papers_sent_for_extraction'] for p in result['pairs']] == [4, 0]
    assert result['latency_saving'] is None
