import copy
import pytest
from app.services.field_review_validation import validate_field_review


@pytest.fixture
def review():
    template={'papers':[{'paper_id':'p','source_pdf_sha256':'pdf','fields':['method'],
        'chunks':[{'chunk_id':0,'text':'We propose an exact source method.'}]}]}
    value={'schema_version':'field-evidence-review.v2','template_sha256':'template','reviewer':'Reader',
        'reviewer_type':'human','human_gold':False,'human_signature':{'reviewer':'Reader',
        'signed_at':'2026-11-01T22:00:00+08:00','independent':True,'source_basis':'original_pdf','task_keys':['p|method']},
        'tasks':[{'paper_id':'p','field':'method','source_pdf_sha256':'pdf','status':'complete',
            'rows':[{'chunk_id':0,'relevance':'relevant','evidence_quote':'We propose an exact source method.',
                     'human_confirmed':True,'quote_exact':True}]}]}
    return template,value


def test_independent_signature_is_validated_instead_of_frontend_flag(review):
    t,r=review
    result=validate_field_review(t,r,'template')
    assert result['human_gold'] and result['human_reviewed_tasks']==1


def test_ai_complete_never_counts_as_human(review):
    t,r=review;r.update(reviewer_type='ai',human_gold=True,human_signature=None)
    result=validate_field_review(t,r,'template')
    assert result['valid'] and result['annotation_complete_tasks']==1
    assert result['human_reviewed_tasks']==0 and not result['human_gold']


@pytest.mark.parametrize('change',[ 'quote','pdf','template','duplicate','confirmation','ai_signature','missing_signature'])
def test_untrusted_flags_do_not_bypass_review_validation(review,change):
    t,r=review;r=copy.deepcopy(r);r['human_gold']=True
    if change=='quote':r['tasks'][0]['rows'][0]['evidence_quote']='Fabricated quote.'
    if change=='pdf':r['tasks'][0]['source_pdf_sha256']='other'
    if change=='template':r['template_sha256']='other'
    if change=='duplicate':r['tasks'][0]['rows']*=2
    if change=='confirmation':r['tasks'][0]['rows'][0]['human_confirmed']=False
    if change=='ai_signature':r['reviewer_type']='ai'
    if change=='missing_signature':r['human_signature']=None
    result=validate_field_review(t,r,'template')
    assert not result['human_gold'] and result['human_reviewed_tasks']==0


def test_unresolved_human_review_does_not_become_resolved_gold(review):
    t,r=review;r['tasks'][0]['rows'][0]['relevance']='uncertain'
    result=validate_field_review(t,r,'template')
    assert result['human_reviewed_tasks']==1 and not result['human_gold']


def test_malformed_external_json_fails_closed_without_acceptance(review):
    t,r=review
    assert not validate_field_review(t,[],'template')['human_gold']
    r['tasks'][0]['rows'][0]['relevance']={'unexpected':'object'}
    result=validate_field_review(t,r,'template')
    assert not result['valid'] and not result['human_gold']
