import pytest
from scripts.reanchor_ai_field_review import reanchor_ai_review


def test_ai_rows_follow_source_positions_after_chunk_renumbering():
    c={'chunk_id':8,'page':2,'char_start':10,'char_end':21,'offset_basis':'page','text':'Exact quote'}
    old={'papers':[{'paper_id':'p','source_pdf_sha256':'pdf','fields':['method'],'chunks':[c]}]}
    new={'papers':[{**old['papers'][0],'chunks':[{**c,'chunk_id':3}]}]}
    review={'template_sha256':'old','reviewer_type':'ai','human_gold':False,'tasks':[
        {'paper_id':'p','source_pdf_sha256':'pdf','field':'method','rows':[
            {'chunk_id':8,'relevance':'relevant','evidence_quote':'Exact quote','human_confirmed':True}]}]}
    result,audit=reanchor_ai_review(old,new,review,'old','new')
    assert result['tasks'][0]['rows'][0]['chunk_id']==3
    assert result['tasks'][0]['rows'][0]['human_confirmed'] is False
    assert result['human_signature'] is None and not result['human_gold']
    assert audit['mapped_rows']==1 and audit['validation']['human_reviewed_tasks']==0
    with pytest.raises(ValueError,match='PDF source version'):
        reanchor_ai_review(old,{'papers':[{**new['papers'][0],'source_pdf_sha256':'different'}]},review,'old','new')
