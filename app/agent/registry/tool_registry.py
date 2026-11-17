"""受控副作用目录；权限由Action声明派生，不向模型授予底层客户端。"""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ToolSpec:
    name: str
    effect: str
    retry: str


TOOLS = MappingProxyType({spec.name: spec for spec in (
    ToolSpec('paper_search', 'remote_read', 'bounded_read'),
    ToolSpec('metadata_read', 'remote_read', 'bounded_read'),
    ToolSpec('pdf_download', 'content_cache', 'verify_content'),
    ToolSpec('pdf_parse', 'local_read', 'verify_source'),
    ToolSpec('llm_completion', 'paid_model', 'new_paid_attempt'),
    ToolSpec('embedding_request', 'paid_model', 'new_paid_attempt'),
    ToolSpec('rerank_request', 'paid_model', 'new_paid_attempt'),
    ToolSpec('vector_read', 'snapshot_read', 'verify_version'),
    ToolSpec('vector_write', 'snapshot_write', 'fenced_build'),
    ToolSpec('vector_admin', 'schema_or_delete', 'explicit_admin'),
    ToolSpec('artifact_read', 'snapshot_read', 'verify_scope'),
    ToolSpec('artifact_write', 'immutable_artifact', 'verify_lineage'),
    ToolSpec('candidate_read', 'candidate_read', 'verify_work'),
    ToolSpec('candidate_write', 'candidate_audit', 'new_candidate'),
    ToolSpec('section_cache_read', 'calculation_cache', 'verify_dependencies'),
    ToolSpec('section_cache_write', 'calculation_cache', 'immutable_result'),
)})


def tools_for_action(action):
    from app.agent.action_registry import action_spec
    return frozenset(action_spec(action).tool_dependencies)


def validate_tool_registry():
    from app.agent.action_registry import ACTION_REGISTRY
    for action in ACTION_REGISTRY.values():
        if set(action.tool_dependencies) - TOOLS.keys():
            raise ValueError('unregistered tool dependency: ' + action.name)
        if len(action.tool_dependencies) != len(set(action.tool_dependencies)):
            raise ValueError('duplicate tool dependency: ' + action.name)
