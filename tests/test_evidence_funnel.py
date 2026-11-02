from scripts.diagnose_evidence_funnel import field_funnel


def test_coverage_selection_outside_topk_is_not_misreported_as_unretrieved_loss():
    chunks=[{'chunk_id':i} for i in range(3)]
    report={'indexed_chunk_ids':[0,1,2],'selected_chunk_ids':[2],
            'field_hits':{'method':[{'chunk_id':0}]}}
    card={'quality_status':'invalid','extraction_coverage':{'chunks':[{'chunk_id':2,'status':'processed'}]},
          'field_claims':{},'quality_issues':[]}
    rows=field_funnel(chunks,report,card,[{'field':'method','chunk_id':2,'label':'relevant'}])
    row=next(r for r in rows if r['field']=='method')
    assert row['relevant_selected_outside_topk']==[2]
    assert row['relevant_processed']==[2]
    assert row['quality_removed_claims'] is None
    assert row['earliest_observed_gap']=='post_extraction_stage_unknown'


def test_gate_removal_requires_observed_before_and_after_snapshots():
    chunks=[{'chunk_id':0}]
    report={'indexed_chunk_ids':[0],'selected_chunk_ids':[0],'field_hits':{'method':[{'chunk_id':0}]}}
    card={'quality_status':'invalid','extraction_coverage':{'chunks':[{'chunk_id':0,'status':'processed'}]},
          'field_claims':{},'quality_issues':['method:malformed_evidence_fragment']}
    rows=field_funnel(chunks,report,card,[{'field':'method','chunk_id':0,'label':'relevant'}],
        raw_card={'method':'An extracted statement'},attached_card={'field_claims':{'method':[{}]}})
    row=next(r for r in rows if r['field']=='method')
    assert row['quality_removed_claims']==1
    assert row['earliest_observed_gap']=='quality_gate_removed_field'
