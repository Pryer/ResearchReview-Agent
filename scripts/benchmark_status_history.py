"""Offline synthetic prompt-size experiment; does not enable a production mode.

No provider, database, or real-paper claims. UTF-8 upper bounds are not billed tokens.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def experiment(rounds=10):
    from app.agent.context_builder import build_main_agent_context
    from app.agent.react_context import ReActContext, estimate_tokens
    from app.core.config import Settings, _TASK_SETTINGS
    from app.schemas.agent_decision_schema import AgentDecision
    from app.schemas.context_schema import KeyEvidence
    from app.services.execution_report_capture import Capture, capture_scope, record_main_context
    cfg = Settings(_env_file=None, agent_context_window_tokens=2000000,
                   agent_context_memory_items=64, agent_context_memory_chars=2000000)
    token = _TASK_SETTINGS.set(cfg)
    results = {}
    try:
        for mode in ('append', 'transient_experiment'):
            state = {'user_query': 'Synthetic request', 'topic': 'Synthetic topic'}
            trajectory = ReActContext(state)
            capture = Capture('synthetic', 'synthetic')
            sizes, full_prefixes = [], []
            previous = []
            with capture_scope(capture):
                for i in range(rounds):
                    context = build_main_agent_context(state)
                    context.key_evidence = [KeyEvidence(evidence_id=f'synthetic-{j}',
                        finding='Synthetic evidence text. ' * 20) for j in range(5)]
                    context.state.budget_remaining = {'actions': rounds - i}
                    messages = trajectory.request(context, system='fixed synthetic rules', tools=[], allowed_actions=['search_and_rank'])
                    sizes.append(estimate_tokens(messages, []))
                    full_prefixes.append(bool(previous) and messages[:len(previous)] == previous)
                    if mode == 'transient_experiment':
                        # WHY: 仅实验移除本轮尾部状态；生产协议保持追加式，不能从体积结果推断质量/账单收益。
                        trajectory.data['messages'].pop()
                    decision = AgentDecision(action='search_and_rank', transport='native_tool', tool_call_id=f'call-{i}')
                    record_main_context(state, context, decision, model_input=True)
                    trajectory.proposed(decision)
                    trajectory.observe({'status': 'completed', 'findings': [f'Synthetic result {i}']}, archive=False)
                    previous = messages
            results[mode] = {'input_upper_bounds': sizes, 'sum_upper_bound': sum(sizes),
                'last_input_upper_bound': sizes[-1], 'full_prior_request_prefix_matches': sum(full_prefixes),
                'captured_contexts': len(capture.rounds)}
    finally:
        _TASK_SETTINGS.reset(token)
    return {'kind': 'synthetic_offline_prompt_size', 'rounds': rounds, 'results': results,
            'provider_requests': 0, 'provider_cache_hit_rate': None, 'quality_effect': None,
            'production_default': 'append',
            'decision': 'Transient reduces history but requires persistent request-level reporting and real quality/cost evaluation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.output.resolve()
    if not path.is_relative_to(ROOT / 'data'):
        parser.error('output must be under repository data/')
    path.parent.mkdir(parents=True, exist_ok=True)
    data = experiment()
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps(data))
