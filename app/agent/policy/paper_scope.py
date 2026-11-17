"""论文子集只用于局部抽卡；范围外产物由统一合并器保留。"""
import copy
from app.database.runtime_repository import RuntimeConflict

LISTS = ('candidate_papers', 'ranked_papers', 'paper_details', 'paper_cards')
MAPS = ('pdf_paths', 'parsed_papers')


def project(state, paper_ids):
    selected = set(paper_ids)
    result = dict(state)
    for field in LISTS:
        if field in result:
            result[field] = [item for item in result[field] if item.get('paper_id') in selected]
    for field in MAPS:
        if field in result:
            result[field] = {key: value for key, value in result[field].items() if key in selected}
    if 'extraction_cache' in result:
        result['extraction_cache'] = {key: value for key, value in result['extraction_cache'].items()
                                      if isinstance(value, dict) and value.get('paper_id') in selected}
    if 'pending_card_rebuild_ids' in result:
        result['pending_card_rebuild_ids'] = [pid for pid in result['pending_card_rebuild_ids'] if pid in selected]
    result.pop('artifact_manifest', None)
    return result


def merge(state, task, result):
    if task.operation != 'extract_paper_cards':
        raise RuntimeConflict('paper subset is limited to evidence extraction')
    selected = set(task.paper_scope)
    fields = set(result.state_patch) | set(result.removed_state_fields)
    if fields - {'paper_cards', 'extraction_cache', 'pending_card_rebuild_ids', 'steps', 'errors', 'step_metrics', 'contract_violations', 'state_invariant_check'}:
        raise RuntimeConflict('scoped extraction changed a global product')
    patch = copy.deepcopy(result.state_patch)
    removed = set(result.removed_state_fields)
    if 'paper_cards' in fields:
        cards = patch.get('paper_cards') or []
        ids = [card.get('paper_id') for card in cards]
        if not set(ids) <= selected or len(ids) != len(set(ids)):
            raise RuntimeConflict('scoped extraction produced an unauthorized paper')
        replacements = {card['paper_id']: card for card in cards}
        merged = []
        for old in state.get('paper_cards') or []:
            pid = old.get('paper_id')
            if pid not in selected:
                merged.append(copy.deepcopy(old))
            elif pid in replacements:
                merged.append(replacements.pop(pid))
        merged.extend(replacements.values())
        patch['paper_cards'] = merged
        removed.discard('paper_cards')
    if 'extraction_cache' in fields:
        cache = copy.deepcopy(state.get('extraction_cache') or {})
        updates = patch.get('extraction_cache') or {}
        if any(not isinstance(value, dict) or value.get('paper_id') not in selected or
               (key in cache and (not isinstance(cache[key], dict) or cache[key].get('paper_id') not in selected)) for key, value in updates.items()):
            raise RuntimeConflict('scoped extraction cache exceeds its paper set')
        # WHY: 局部删除只清理本范围旧缓存；其他论文的命中记录不随子集替换丢失。
        cache = {key: value for key, value in cache.items() if not isinstance(value, dict) or value.get('paper_id') not in selected}
        cache.update(updates)
        patch['extraction_cache'] = cache
        removed.discard('extraction_cache')
    if 'pending_card_rebuild_ids' in fields:
        pending = patch.get('pending_card_rebuild_ids') or []
        if not set(pending) <= selected:
            raise RuntimeConflict('scoped rebuild list exceeds its paper set')
        patch['pending_card_rebuild_ids'] = [pid for pid in state.get('pending_card_rebuild_ids') or [] if pid not in selected] + pending
        removed.discard('pending_card_rebuild_ids')
    return patch, removed
