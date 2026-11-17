"""将后台终态与服务端已发布版本投影为聊天消息，不处理候选稿授权。"""
import copy


def result_metadata(result):
    metadata = {key: copy.deepcopy(result.get(key)) for key in (
        'status', 'execution_report', 'clarification', 'intent', 'topic', 'session_id',
        'execution_status', 'delivery_status', 'verification_status', 'requirements_satisfied',
        'draft_released', 'published_draft_version')}
    metadata['body'] = result.get('body') or ''
    for key in ('paper_cards', 'steps', 'references', 'quality_recovery_history'):
        metadata[key] = copy.deepcopy(result.get(key) or [])
    for key in ('claim_verification', 'quality_gate', 'quality_recovery_decision', 'reference_coverage_stats'):
        metadata[key] = copy.deepcopy(result.get(key) or {})
    metadata['source_diagnostics'] = copy.deepcopy((result.get('search_report') or {}).get(
        'source_diagnostics') or result.get('source_diagnostics') or [])
    return metadata


def terminal_job_message(job):
    status = job.get('status')
    result = copy.deepcopy(job.get('result') or {})
    if status in {'cancelled', 'failed'}:
        # WHY: 异常返回中的候选正文不具备发布授权；停止后的正文只来自服务端发布投影。
        result = {}
    publication = job.get('available_delivery') or {}
    if publication.get('draft_released') is True and publication.get('body'):
        for key in ('body', 'references', 'delivery_status', 'verification_status',
                    'requirements_satisfied', 'draft_released', 'published_draft_version', 'quality_gate',
                    'claim_verification', 'paper_cards', 'citation_validation'):
            result[key] = copy.deepcopy(publication.get(key))
    # WHY: 交付等级与后台执行终态分别保留，已有草稿不能把取消、失败或澄清改成成功。
    result.update(status=status, session_id=job.get('session_id') or result.get('session_id'))
    if status in {'cancelled', 'failed'}:
        result['execution_status'] = status
    if job.get('execution_report'):
        result['execution_report'] = copy.deepcopy(job['execution_report'])
    defaults = {'completed': '✅ 任务已完成。', 'partial': '已生成可用草稿，请查看限制说明。',
                'blocked': '当前要求尚未满足，请查看质量门禁。',
                'needs_clarification': '请补充研究信息后继续。'}
    if status == 'cancelled':
        answer = '⏹ 任务已取消，后续步骤未再执行。'
    elif status == 'failed':
        from app.agent.public_errors import public_result_failure_reason
        answer = '❌ 任务失败：' + (job.get('error') or public_result_failure_reason(job.get('result')))
    else:
        answer = result.get('answer') or defaults.get(status, '任务已停止。')
    if status == 'needs_clarification':
        question = (result.get('clarification') or {}).get('question')
        if question and question not in answer:
            answer += '\n\n' + question
    published_answer = publication.get('answer') or publication.get('body')
    if result.get('draft_released') is True and published_answer and published_answer not in answer:
        answer += '\n\n' + published_answer
    return answer, result_metadata(result)


def can_download_draft(metadata):
    gate = metadata.get('quality_gate') or {}
    delivery = metadata.get('delivery_status')
    return bool((delivery == 'draft' or (not delivery and metadata.get('status') == 'partial'))
                and (metadata.get('draft_released') is True or gate.get('draft_released') is True)
                and str(metadata.get('body') or '').strip())
