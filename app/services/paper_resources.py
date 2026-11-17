"""从动作输入绑定论文/文件身份；同Attempt的可信工具回执可扩展资源。"""
import hashlib
import json
from pathlib import Path
from sqlalchemy import select

from app.agent.action_contracts import CONTRACTS
from app.database.models import ResearchEventModel as Event
from app.database.runtime_repository import RuntimeConflict

RESOURCE_TOOLS = frozenset({'metadata_read', 'pdf_download', 'pdf_parse', 'vector_read', 'vector_write'})


def canonical(value):
    value = str(value or '').strip().casefold()
    for prefix in ('https://doi.org/', 'http://doi.org/', 'doi:', 'https://openalex.org/', 'openalex:', 'arxiv:', 's2:'):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def paper_record(paper, previous=None):
    previous = previous or {}
    aliases = set(previous.get('aliases') or [])
    aliases.update(canonical(paper.get(key)) for key in ('paper_id', 'doi', 'arxiv_id'))
    record = {**previous, 'paper_id': str(paper['paper_id']), 'aliases': sorted(aliases - {''}),
        'pdf_url_hash': hashlib.sha256(str(paper.get('pdf_url') or '').encode()).hexdigest(),
        'source': str(paper.get('source') or ''), 'is_open_access': paper.get('is_open_access')}
    signature = hashlib.sha256(json.dumps([record[k] for k in ('pdf_url_hash', 'source', 'is_open_access')]).encode()).hexdigest()
    record['transports'] = sorted(set(previous.get('transports') or []) | {signature})
    return record


def bind(state, action):
    from app.core.config import get_settings
    from app.services.document_ingestion import source_fingerprint
    from app.services.research_artifact_service import ResearchArtifactService
    inputs, papers, files = CONTRACTS[action].inputs, {}, {}
    for field in ('candidate_papers', 'ranked_papers', 'paper_cards', 'paper_details'):
        if field in inputs:
            for paper in state.get(field) or []:
                if paper.get('paper_id'):
                    pid = str(paper['paper_id'])
                    papers[pid] = paper_record(paper, papers.get(pid))
    if 'parsed_papers' in inputs:
        for pid, parsed in (state.get('parsed_papers') or {}).items():
            if pid in papers:
                papers[pid]['source_hash'] = source_fingerprint(parsed)
                papers[pid]['source_pdf_hash'] = parsed.get('source_pdf_hash')
    if 'pdf_paths' in inputs:
        for pid, path in (state.get('pdf_paths') or {}).items():
            if pid in papers and path:
                files[pid] = ResearchArtifactService.controlled_file_ref(path)
                if papers[pid].get('source_pdf_hash') and papers[pid]['source_pdf_hash'] != files[pid]['sha256']:
                    papers[pid].pop('source_hash', None)
    return {'papers': papers, 'files': files, 'vector_tenant': get_settings().card_vector_tenant}


def current(runtime, context, packet):
    resources = {**packet.get('paper_resources', {}),
                 'papers': dict(packet.get('paper_resources', {}).get('papers') or {}),
                 'files': dict(packet.get('paper_resources', {}).get('files') or {})}
    for row in runtime.db.scalars(select(Event).where(Event.session_id == runtime.session_id,
            Event.event_key.like('resource-update:' + context.task_id + ':%')).order_by(Event.event_id)):
        update = json.loads(row.payload_json)
        resources['papers'].update(update.get('papers') or {})
        resources['files'].update(update.get('files') or {})
    return resources


def _paper(resources, pid):
    paper = resources['papers'].get(str(pid or ''))
    if paper is None:
        raise RuntimeConflict('paper is outside the current action resources')
    return paper


def _same_record(paper, approved):
    proposed = paper_record(paper)
    if (not set(proposed['aliases']) <= set(approved['aliases'])
            or not set(proposed['transports']) <= set(approved['transports'])):
        raise RuntimeConflict('paper transport metadata differs from the authorized source')


def validate(runtime, context, tool, arguments, boundary):
    from app.services.action_resources import _scope
    with runtime.transaction():
        _, packet = _scope(runtime, context, tool)
        resources = current(runtime, context, packet)
        if tool == 'metadata_read' and 'papers' in arguments:
            for paper in arguments['papers']:
                _same_record(paper, _paper(resources, paper.get('paper_id')))
            return {'papers': resources['papers']}
        if tool == 'metadata_read' or (tool == 'paper_search' and context.action == 'fetch_metadata'):
            identifier = next((arguments[k] for k in ('doi', 'identifier', 'work_id', 'query') if k in arguments), '')
            if not canonical(identifier) or not any(canonical(identifier) in p['aliases'] for p in resources['papers'].values()):
                raise RuntimeConflict('metadata identifier is outside the current paper scope')
        elif tool == 'pdf_download':
            paper = arguments.get('paper') or {}
            approved = _paper(resources, paper.get('paper_id'))
            _same_record(paper, approved)
            if arguments.get('save_dir'):
                from app.core.config import get_settings
                if not Path(arguments['save_dir']).resolve().is_relative_to(Path(get_settings().pdf_save_dir).resolve()):
                    raise RuntimeConflict('download directory is outside controlled research storage')
            return {'paper': approved}
        elif tool == 'pdf_parse':
            pid = str(arguments.get('paper_id') or '')
            approved = _paper(resources, pid)
            file = resources['files'].get(pid)
            if not file or Path(arguments['pdf_path']).resolve() != Path(file['path']):
                raise RuntimeConflict('PDF file is outside the current paper scope')
            return {'paper': approved, 'file': file}
        elif tool in {'vector_read', 'vector_write'}:
            spec = arguments.get('spec')
            pid = spec.paper_id if spec else arguments.get('paper_id')
            paper = _paper(resources, pid)
            tenant = spec.tenant if spec else arguments.get('tenant')
            source = spec.source_hash if spec else arguments.get('source_hash')
            if tenant != resources.get('vector_tenant') or not paper.get('source_hash') or source != paper['source_hash']:
                raise RuntimeConflict('vector tenant or source is outside the current paper scope')
        return {}


def check_file(guard):
    if guard.get('file'):
        from app.services.research_artifact_service import ResearchArtifactService
        if ResearchArtifactService.controlled_file_ref(guard['file']['path']) != guard['file']:
            raise RuntimeConflict('authorized PDF content changed')


def record_result(runtime, context, tool, arguments, result, guard):
    from app.services.action_resources import _scope
    from app.services.tool_execution import current_tool_call_id
    from app.services.research_artifact_service import ResearchArtifactService
    from app.services.document_ingestion import source_fingerprint
    update = {}
    if tool == 'metadata_read' and 'papers' in arguments:
        ids = {str(p.get('paper_id')) for p in arguments['papers']}
        if not isinstance(result, list) or len(result) != len(ids) or {str(p.get('paper_id')) for p in result} != ids:
            raise RuntimeConflict('metadata enrichment changed the authorized paper set')
        update['papers'] = {str(p['paper_id']): paper_record(p, guard['papers'].get(str(p['paper_id']))) for p in result}
    elif tool == 'pdf_download' and result:
        file = ResearchArtifactService.controlled_file_ref(result)
        update['files'] = {guard['paper']['paper_id']: file}
        if file['sha256'] != guard['paper'].get('source_pdf_hash'):
            paper = dict(guard['paper'])
            # WHY: 新PDF尚未解析，不能继续用旧正文摘要授权向量读写。
            paper.pop('source_hash', None)
            update['papers'] = {paper['paper_id']: paper}
    elif tool == 'pdf_parse':
        check_file(guard)
        if result.get('paper_id') != guard['paper']['paper_id'] or result.get('source_pdf_hash') != guard['file']['sha256']:
            raise RuntimeConflict('parsed PDF identity differs from its authorized file')
        update['papers'] = {guard['paper']['paper_id']: {**guard['paper'], 'source_hash': source_fingerprint(result),
                                                      'source_pdf_hash': result['source_pdf_hash']}}
    if update:
        with runtime.transaction():
            _scope(runtime, context, tool)
            runtime.event('resource-update:' + context.task_id + ':' + current_tool_call_id(),
                          'action_resources_extended', {'task_id': context.task_id, **update})
