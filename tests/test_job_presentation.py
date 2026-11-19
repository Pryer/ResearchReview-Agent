"""聊天终态保留已发布正文，并保留执行状态和独立核验状态。"""
import copy

import pytest

from app.frontend.job_presentation import (
    can_download_draft,
    draft_markdown,
    draft_notice,
    terminal_job_message,
)


def publication():
    return {'answer': '已有草稿及限制说明', 'body': '已发布正文',
            'references': [{'title': '已核准参考文献'}], 'status': 'partial',
            'delivery_status': 'draft', 'verification_status': 'pending',
            'requirements_satisfied': False, 'draft_released': True,
            'published_draft_version': 2, 'quality_gate': {'passed': False, 'draft_released': True}}


@pytest.mark.parametrize('status', ['cancelled', 'failed', 'blocked', 'needs_clarification'])
def test_stop_preserves_publication_without_promoting_execution_or_candidate(status):
    job = {'status': status, 'session_id': 's', 'error': '执行已停止',
           'result': {'body': '未核验的新候选', 'references': [{'title': '候选参考文献'}],
                      'clarification': {'needed': True, 'kind': 'main_agent', 'question': '补充主题？'}},
           'available_delivery': publication(), 'execution_report': {'state': 'pending', 'report_id': 'report'}}
    before = copy.deepcopy(job)
    answer, metadata = terminal_job_message(job)
    assert metadata['status'] == status and metadata['session_id'] == 's'
    assert metadata['body'] == '已发布正文' and metadata['references'] == publication()['references']
    assert metadata['verification_status'] == 'pending' and not metadata['requirements_satisfied']
    assert metadata['published_draft_version'] == 2 and can_download_draft(metadata)
    assert metadata['execution_report']['report_id'] == 'report'
    assert '未核验的新候选' not in answer
    assert job == before
    if status == 'cancelled':
        assert '取消' in answer
    elif status == 'failed':
        assert '失败' in answer
    elif status == 'needs_clarification':
        assert '补充主题？' in answer and metadata['clarification']['needed']


@pytest.mark.parametrize('status', ['cancelled', 'failed'])
def test_unpublished_candidate_never_becomes_stopped_job_delivery(status):
    answer, metadata = terminal_job_message({'status': status, 'session_id': 's',
        'result': {'answer': 'PRIVATE CANDIDATE', 'body': 'PRIVATE CANDIDATE',
                   'references': [{'title': 'PRIVATE'}], 'draft_released': False}})
    assert 'PRIVATE' not in answer and not metadata['body'] and not metadata['references']
    assert not can_download_draft(metadata)


@pytest.mark.parametrize('status', ['completed', 'partial', 'blocked', 'needs_clarification'])
def test_missing_answer_uses_actual_terminal_state(status):
    answer, metadata = terminal_job_message({'status': status, 'session_id': 's', 'result': {}})
    assert ('已完成' in answer) == (status == 'completed')
    assert metadata['status'] == status


def test_download_requires_release_flag_and_body():
    current = publication()
    assert can_download_draft(current)
    assert not can_download_draft({**current, 'body': ''})
    assert not can_download_draft({**current, 'draft_released': False, 'quality_gate': {'passed': False}})
    assert not can_download_draft({**current, 'delivery_status': 'none'})


@pytest.mark.parametrize(('status', 'label'), [
    ('complete', '当前版本的语义核验已通过'),
    ('partial', '语义核验尚未全部通过'),
    ('not_started', '语义核验尚未开始'),
    (None, '语义核验状态未知'),
])
def test_download_keeps_visible_draft_grade_and_verification_after_export(status, label):
    current = {**publication(), 'verification_status': status}
    notice = draft_notice(current)
    assert '正式交付门禁待通过' in notice and label in notice
    assert draft_markdown(current) == f"> **{notice}**\n\n{current['body']}"
    assert draft_markdown({**current, 'delivery_status': 'final'}) == ''


def test_completed_result_keeps_publication_metadata_and_question_is_not_invented():
    answer, metadata = terminal_job_message({'status': 'partial', 'session_id': 's', 'result': publication()})
    assert answer == publication()['answer'] and can_download_draft(metadata)
    assert metadata['verification_status'] == 'pending'
    assert metadata['clarification'] is None
