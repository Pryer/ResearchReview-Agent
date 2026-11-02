"""离线字段审查验收：标注覆盖、AI预审和独立人工签署分别计算。"""
from __future__ import annotations
from datetime import datetime


def validate_field_review(template: dict, review: dict, template_sha256: str) -> dict:
    if not isinstance(review,dict):
        return {'valid':False,'errors':['invalid_review_document'],'human_gold':False,
                'human_reviewed_tasks':0,'acceptance_status':'pending_human_review'}
    errors=[]
    def error(code):
        errors.append(code)
    if review.get('template_sha256')!=template_sha256:
        error('template_version_mismatch')
    papers={p['paper_id']:p for p in template['papers']}
    expected={(p['paper_id'],f) for p in template['papers'] for f in p['fields']}
    kind=review.get('reviewer_type')
    if kind is not None and (not isinstance(kind,str) or kind not in {'human','ai'}):
        error('invalid_reviewer_type')
    signature=review.get('human_signature')
    signed_keys=set()
    signature_valid=False
    if signature is not None:
        if kind!='human':
            error('ai_cannot_sign_human_review')
        elif not isinstance(signature,dict):
            error('invalid_signature')
        else:
            keys=signature.get('task_keys',[])
            if not isinstance(keys,list) or any(not isinstance(k,str) for k in keys) or len(keys)!=len(set(keys)):
                error('invalid_signed_task_keys')
            else:
                signed_keys=set(keys)
            try:
                stamp=datetime.fromisoformat(signature.get('signed_at',''))
                timestamp_ok=stamp.tzinfo is not None
            except (TypeError,ValueError):
                timestamp_ok=False
            signature_valid=bool(
                review.get('schema_version')=='field-evidence-review.v2'
                and isinstance(review.get('reviewer'),str) and review['reviewer'].strip()
                and signature.get('reviewer')==review['reviewer']
                and signature.get('independent') is True
                and signature.get('source_basis')=='original_pdf'
                and timestamp_ok and signed_keys)
            if not signature_valid:
                error('incomplete_human_signature')
    tasks=review.get('tasks',[])
    if not isinstance(tasks,list):
        error('invalid_tasks');tasks=[]
    seen=set();results=[]
    for task in tasks:
        if not isinstance(task,dict):
            error('invalid_task');continue
        key=(task.get('paper_id'),task.get('field'))
        if any(not isinstance(v,str) for v in key) or key not in expected or key in seen:
            error('unknown_or_duplicate_task');continue
        seen.add(key)
        paper=papers[key[0]]
        if task.get('source_pdf_sha256')!=paper['source_pdf_sha256']:
            error('pdf_version_mismatch')
        chunks={c['chunk_id']:c for c in paper['chunks']}
        rows=task.get('rows',[])
        if not isinstance(rows,list):
            error('invalid_rows');rows=[]
        row_ids=set();judged=0;uncertain=0;all_confirmed=True
        for row in rows:
            if not isinstance(row,dict):
                error('invalid_row');continue
            cid=row.get('chunk_id')
            if type(cid) is not int or cid not in chunks or cid in row_ids:
                error('unknown_or_duplicate_chunk');continue
            row_ids.add(cid)
            label=row.get('relevance')
            if not isinstance(label,str) or label not in {'relevant','irrelevant','uncertain','not_reviewed'}:
                error('invalid_relevance');continue
            quote=row.get('evidence_quote','')
            if not isinstance(quote,str):
                error('invalid_quote');quote=''
            # WHY: quote_exact、status、human_gold均由外部输入声明，不能作为验收依据。
            if label=='relevant' and (not quote.strip() or quote not in chunks[cid]['text']):
                error('quote_not_in_original_chunk')
            judged+=label!='not_reviewed'
            uncertain+=label=='uncertain'
            all_confirmed=all_confirmed and row.get('human_confirmed') is True
        annotation_complete=row_ids==set(chunks) and judged==len(chunks)
        task_key=f'{key[0]}|{key[1]}'
        human_reviewed=bool(kind=='human' and signature_valid and task_key in signed_keys
                            and annotation_complete and all_confirmed)
        if task_key in signed_keys and not human_reviewed:
            error('signed_task_not_independently_confirmed')
        results.append({'paper_id':key[0],'field':key[1],
            'annotation_status':'complete' if annotation_complete else 'partial' if judged else 'not_reviewed',
            'judged_chunks':judged,'total_chunks':len(chunks),'uncertain_chunks':uncertain,
            'human_reviewed':human_reviewed,'human_resolved':human_reviewed and uncertain==0})
    if signed_keys-{f'{p}|{f}' for p,f in seen}:
        error('signature_covers_unknown_task')
    if errors:
        # WHY: 版本或来源被修改时整份验收失效，不能通过删除失败行继续累计人审覆盖。
        for task in results:
            task['human_reviewed']=task['human_resolved']=False
    reviewed=sum(t['human_reviewed'] for t in results)
    resolved=sum(t['human_resolved'] for t in results)
    gold=bool(not errors and len(expected)>0 and resolved==len(expected))
    return {'valid':not errors,'errors':errors,'reviewer_type':kind,
        'annotation_complete_tasks':sum(t['annotation_status']=='complete' for t in results),
        'human_reviewed_tasks':reviewed,'human_resolved_tasks':resolved,'total_tasks':len(expected),
        'human_review_coverage':reviewed/len(expected) if expected else None,
        'human_gold':gold,'acceptance_status':'complete_human_reference' if gold else
            'partial_human_review' if reviewed else 'ai_preliminary_only' if kind=='ai' else 'pending_human_review',
        'tasks':results,'signature_scope':'local_source_and_self_declared_signature_validation_not_identity_authentication'}
