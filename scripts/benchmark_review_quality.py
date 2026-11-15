"""Offline ranking reassessment and bounded evidence-check benchmark.

Replay never instantiates a provider; live uses the production verifier and budget.
Raw artifacts remain under data/. Reports distinguish synthetic faults from gold.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import socket
import statistics
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ('bm25', 'dense', 'rrf', 'rrf_ce')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


@contextmanager
def offline():
    def denied(*args, **kwargs):
        raise RuntimeError('network forbidden in replay mode')
    with patch.object(socket.socket, 'connect', denied), patch.object(socket.socket, 'connect_ex', denied):
        yield


def reassess(report_path, snapshot_path, qrels_path):
    from scripts.evaluate_hybrid_retrieval import _metrics
    report, snapshot = read(report_path), read(snapshot_path)
    for key, path in [('snapshot', snapshot_path), ('qrels', qrels_path)]:
        if digest(path) != report['input_sha256'][key]:
            raise ValueError(f'{key} hash differs from saved ranking')
    labels = {}
    for line in Path(qrels_path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        group = labels.setdefault(row['task_id'], {})
        if row['paper_id'] in group or row['relevance'] not in range(4):
            raise ValueError('duplicate or invalid qrel')
        group[row['paper_id']] = row['relevance']
    pools = {t['task_id']: {p['paper_id'] for p in t['candidate_papers']} for t in snapshot['tasks']}
    rows = []
    for task in report['tasks']:
        task_id = task['task_id']
        if set(labels[task_id]) != pools[task_id] or task.get('degraded'):
            raise ValueError('incomplete labels or degraded historical ranking')
        for mode in MODES:
            ranking = task['raw_rankings'][mode]
            if len(ranking) != len(set(ranking)) or not set(ranking) <= pools[task_id]:
                raise ValueError('invalid ranking identity')
            for k in (10, 72, 120):
                rows.append({'task_id': task_id, 'mode': mode, 'k': k,
                             **_metrics(ranking, labels[task_id], k, 2)})
    means = {mode: {str(k): {metric: statistics.mean(
        row[metric] for row in rows if row['mode'] == mode and row['k'] == k and row[metric] is not None
    ) for metric in ('recall', 'precision', 'ndcg')} for k in (10, 72, 120)} for mode in MODES}
    return {'status': 'historical_recomputed', 'label_source': 'mixed_partial_human_model_rule',
            'task_count': len(report['tasks']), 'candidate_count': sum(map(len, pools.values())),
            'relevance_threshold': 2, 'rows': rows, 'macro': means,
            'evaluation_version': report.get('evaluation_version'), 'current_model_run': False}


def make_cases(fixture):
    from app.tools.verify_claims import split_review_sentences
    cases = []
    for card in fixture['paper_cards']:
        for span in card.get('evidence_spans', []):
            source_text = str(span.get('text') or '')
            candidates = split_review_sentences(' '.join(source_text.split()))
            text = next((s for s in candidates if re.search(r'\b\d+(?:\.\d+)?\s*%', s)
                         and any(t in s.lower() for t in ('accuracy', 'improv', 'outperform', 'result', 'achiev'))), '')
            number = re.search(r'\b\d+(?:\.\d+)?\s*%', text)
            if span.get('source_type') != 'full_text' or not number or len(text) > 700:
                continue
            # WHY: 仅在真实摘录中替换一个百分数，避免把任意年份误当作实验结果。
            if not any(term in text.lower() for term in ('accuracy', 'improv', 'outperform', 'result', 'achiev')):
                continue
            base = len(cases) // 3
            pid = card['paper_id']
            evidence = {**copy.deepcopy(card), 'evidence_spans': [copy.deepcopy(span)]}
            altered = text[:number.start()] + '987.654%' + text[number.end():]
            for kind, claim, cite, error in (
                ('control', text, pid, False),
                ('unknown_citation', text, 'doi:10.0000/benchmark-missing', True),
                ('numeric_corruption', altered, pid, True),
            ):
                cases.append({'case_id': f'{base}-{kind}', 'kind': kind, 'expected_error': error,
                              'sentence': claim.strip() + f' [{cite}].', 'card': evidence,
                              'source_quote': source_text, 'source_evidence_id': span['evidence_id']})
            if len(cases) >= 12:
                break
        if len(cases) >= 12:
            break
    return cases


def confusion(rows):
    counts = dict(tp=0, fp=0, tn=0, fn=0, unverified=0)
    for row in rows:
        if row['detected'] is None:
            counts['unverified'] += 1
        else:
            counts[('t' if row['expected_error'] == row['detected'] else 'f') +
                   ('p' if row['detected'] else 'n')] += 1
    ratio = lambda a, b: a / b if b else None
    return {**counts, 'error_detection_rate': ratio(counts['tp'], counts['tp'] + counts['fn']),
            'false_positive_rate': ratio(counts['fp'], counts['fp'] + counts['tn']),
            'coverage': ratio(len(rows) - counts['unverified'], len(rows))}


def check_cases(cases, llm=None, on_result=None):
    from app.tools.verify_claims import verify_review_claims
    if not cases:
        raise ValueError('no traceable percentage statements available')
    rows = []
    for case in cases:
        result = verify_review_claims(case['sentence'], [case['card']], llm=llm)
        claims = [c for c in result.get('claims', []) if c.get('factual')]
        rejected = any(c.get('support_status') == 'unsupported' and c.get('verification_status') == 'verified' for c in claims)
        complete = bool(claims) and all(c.get('verification_status') == 'verified' for c in claims)
        # WHY: 未完成语义判断不能作为支持、漏检或误报计数；确定性拒绝仍可计检出。
        detected = True if rejected else False if complete else None
        rows.append({k: case[k] for k in ('case_id', 'kind', 'expected_error')} |
                    {'detected': detected, 'report': result})
        if on_result is not None:
            on_result(rows)
    return {'status': 'controlled_faults', 'mode': 'live_semantic' if llm else 'deterministic_only',
            'case_count': len(rows), 'by_kind': {kind: confusion([r for r in rows if r['kind'] == kind])
                for kind in sorted({r['kind'] for r in rows})}, 'overall': confusion(rows), 'cases': rows,
            'citation_precision': None, 'faithfulness': None,
            'scope': 'production claim verifier component; not full Check orchestration or natural-error accuracy'}


def revision_replay(fixture):
    from app.agent.nodes.extraction import extract_card_node
    from app.agent.extraction_reuse import extraction_fingerprint
    from app.core.config import get_settings
    cards = fixture['paper_cards']
    papers = [{k: c.get(k) for k in ('paper_id', 'title', 'authors', 'year', 'doi', 'source', 'url')} |
              {'_topic_relation': c.get('relation_type'), '_eligible_deliverables': c.get('eligible_deliverables', [])}
              for c in cards]
    original = {'topic': fixture.get('canonical_topic', ''), 'delivery_mode': 'draft_first',
                'paper_details': papers, 'steps': [], 'errors': [], 'paper_cards': copy.deepcopy(cards)}
    cache = {extraction_fingerprint(p, None, original['topic'], get_settings()): copy.deepcopy(c)
             for p, c in zip(papers, cards)}
    outputs = []
    for mode in ('recompute', 'reuse'):
        state = copy.deepcopy(original)
        state['paper_details'].pop(1)
        state['extraction_cache'] = copy.deepcopy(cache) if mode == 'reuse' else {}
        work = []
        def recorded_batch(papers, *args, **kwargs):
            work.extend(p['paper_id'] for p in papers)
            return [copy.deepcopy(next(c for c in cards if c['paper_id'] == p['paper_id'])) for p in papers]
        with patch('app.tools.extract_paper_card.batch_extract_paper_cards', recorded_batch):
            extract_card_node(state)
        if state.get('errors'):
            raise ValueError('extraction replay failed')
        if len(state['paper_cards']) != len(papers) - 1:
            raise ValueError('removed paper retained or remaining paper missing')
        outputs.append({'condition': mode, 'papers_sent_for_extraction': len(work),
                        'cards_retained': len(state['paper_cards']),
                        'evidence': sorted((c['paper_id'], s['evidence_id'], s['text'])
                            for c in state['paper_cards'] for s in c.get('evidence_spans', []))})
    same = outputs[0]['evidence'] == outputs[1]['evidence']
    for row in outputs:
        row['evidence_hash'] = hashlib.sha256(json.dumps(row.pop('evidence'), ensure_ascii=False).encode()).hexdigest()
    return {'status': 'component_replay', 'pairs': outputs, 'same_retained_evidence': same,
            'paid_calls': 0, 'token_saving': None, 'latency_saving': None,
            'scope': 'source-bound PaperCard reuse, recorded extraction results, no writing or PDF reparse'}


def summary(output, retrieval, check, revision):
    lines = ['# Review quality benchmark', '', '## Retrieval', '',
             'Historical frozen pool, mixed-source labels; relevance >= 2. No new ranking calls.', '',
             '| Method | Recall@120 | Precision@72 | nDCG@72 |', '| --- | ---: | ---: | ---: |']
    if retrieval.get('macro'):
        for mode, value in retrieval['macro'].items():
            lines.append(f"| {mode} | {value['120']['recall']:.4f} | {value['72']['precision']:.4f} | {value['72']['ndcg']:.4f} |")
    else:
        lines.append('Unavailable: ' + retrieval.get('reason', 'missing input'))
    lines += ['', '## Controlled checks', '', f"Mode: {check.get('mode')}; cases: {check.get('case_count', 0)}.",
              '', '| Kind | TP | FP | TN | FN | Unverified |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for kind, c in check.get('by_kind', {}).items():
        lines.append(f"| {kind} | {c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} | {c['unverified']} |")
    lines += ['', '## Revision replay', '', 'Recorded extraction responses; no real LLM timing or cost estimate.', '']
    for row in revision.get('pairs', []):
        lines.append(f"- {row['condition']}: {row['papers_sent_for_extraction']} papers sent for extraction; {row['cards_retained']} retained.")
    lines += ['', 'Citation Precision, Faithfulness and full revision E2E savings remain unmeasured.', '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['replay', 'live'], default='replay')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--assets', type=Path, default=ROOT / 'data/provider_retrieval_eval/20261030_continue_10m')
    parser.add_argument('--fixture', type=Path, default=ROOT / 'tests/fixtures/writer_t06_20261110.json')
    parser.add_argument('--request-limit', type=int, default=24)
    parser.add_argument('--token-budget', type=int, default=300000)
    parser.add_argument('--deadline-seconds', type=int, default=1200)
    args = parser.parse_args()
    if args.mode == 'live' and not args.execute:
        parser.error('live requires --execute')
    if not (1 <= args.request_limit <= 60 and 1 <= args.token_budget <= 300000 and 1 <= args.deadline_seconds <= 1200):
        parser.error('limits exceed plan bounds or are not positive')
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / 'data'):
        parser.error('output must be under repository data/')
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'mode': args.mode, 'stages': {}, 'inputs': {}, 'budget': {},
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'tracked_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)).hexdigest(),
                'runner_sha256': digest(__file__), 'label_source': 'mixed; controlled faults are synthetic'}
    retrieval = check = revision = {}
    def stage(name, run):
        try:
            value = run()
            manifest['stages'][name] = value.get('status', 'completed')
        except Exception as exc:
            value = {'status': 'unavailable', 'reason': type(exc).__name__}
            manifest['stages'][name] = value
        write(output / f'{name}.json', value)
        write(output / 'manifest.json', manifest)
        return value
    try:
        with offline():
            template_path = ROOT / 'data/evidence_repairs/20261101/review_v2/review_template.json'
            if template_path.is_file():
                template = read(template_path)
                write(output / 'field_review_inventory.json', {
                    'template_sha256': digest(template_path),
                    'paper_count': len(template.get('papers', [])),
                    'field_task_count': sum(len(p.get('fields', [])) for p in template.get('papers', [])),
                    'human_gold': False, 'status': 'existing_review_package_located_pending_independent_review',
                    'note': 'Inventory does not validate old parser compatibility or human signatures.'})
            paths = [args.assets / 'enriched_retrieval.json', args.assets / 'inputs/real_merged_snapshot_enriched.json',
                     args.assets / 'inputs/human_all_qrels.jsonl', args.fixture]
            manifest['inputs'] = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:
                                  digest(p) if p.is_file() else None for p in paths}
            retrieval = stage('retrieval_metrics', lambda: reassess(*paths[:3]))
            fixture = read(args.fixture)
            cases = make_cases(fixture)
            write(output / 'check_cases.json', cases)
            check = stage('check_replay', lambda: check_cases(cases))
            revision = stage('revision_pairs', lambda: revision_replay(fixture))
        if args.mode == 'live':
            from app.core.config import get_settings
            from app.agent.execution_budget import budget_scope
            from app.services.llm_service import LLMService
            cfg = get_settings()
            cfg.agent_main_token_budget = args.token_budget
            cfg.agent_model_request_limit = min(args.request_limit, 24)
            cfg.agent_execution_deadline_seconds = args.deadline_seconds
            cfg.llm_request_timeout = min(cfg.llm_request_timeout, 45)
            cfg.llm_failover_total_timeout = min(cfg.llm_failover_total_timeout, 60)
            cfg.llm_control_plane_max_tokens = 2048
            state = {}
            try:
                with budget_scope(state):
                    llm = LLMService()
                    manifest['live_model'] = {'provider': llm.provider, 'model': llm.model}
                    check = stage('check_live', lambda: check_cases(cases, llm,
                        on_result=lambda rows: write(output / 'check_live_partial.json', rows)))
            finally:
                # WHY: 中断或预算异常也保存累计消耗，不能因未到正常返回而丢失账本。
                manifest['budget'] = state.get('agent_execution_budget', {})
    finally:
        manifest['scope'] = 'ranking reassessment + verifier component + recorded extraction reuse'
        manifest['remaining'] = ['independent human field review', 'natural citation quality', 'paired full revision cost']
        write(output / 'manifest.json', manifest)
        summary(output, retrieval, check, revision)
    print(json.dumps({'stages': manifest['stages'], 'budget': manifest['budget']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
