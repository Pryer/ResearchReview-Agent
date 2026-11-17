"""direct Profile 参数只从受控任务范围构造，不增加模型调用。"""
import copy
from app.agent.action_registry import validate_arguments


def direct_arguments(grant):
    names = {
        'gap_diagnosis': 'diagnose_evidence_gaps',
        'scope_revision': 'revise_scope',
        'draft_generation': 'generate_deliverables',
        'section_revision': 'rewrite_sections',
        'independent_check': 'validate_result',
    }
    if grant.profile in {'single_action', 'legacy_single_v1'}:
        if len(grant.allowed_actions) != 1 or grant.immutable_arguments is None:
            raise ValueError('invalid single action grant')
        return grant.allowed_actions[0], copy.deepcopy(grant.immutable_arguments)
    if grant.profile not in names:
        raise ValueError('profile requires a local strategy')
    action = names[grant.profile]
    arguments = ({'section_ids': copy.deepcopy(grant.scope.get('section_ids') or [])}
                 if grant.profile == 'section_revision' else {})
    validate_arguments(action, arguments)
    return action, arguments
