"""按PDF版本、页码和精确原文区间迁移AI预审；不按旧chunk_id回填，不迁移人工签名。"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.services.field_review_validation import validate_field_review


def reanchor_ai_review(old_template,new_template,review,old_hash,new_hash):
    if review.get('reviewer_type')!='ai' or not validate_field_review(old_template,review,old_hash)['valid']:
        raise ValueError('only valid AI preliminary reviews can be reanchored')
    old_papers={p['paper_id']:p for p in old_template['papers']}
    new_papers={p['paper_id']:p for p in new_template['papers']}
    def anchor(c):
        return c['page'],c['char_start'],c['char_end'],c['offset_basis'],c['text']
    tasks=[];mapping=[];unmapped=[]
    for task in review['tasks']:
        pid=task['paper_id'];old=old_papers[pid];new=new_papers[pid]
        if old['source_pdf_sha256']!=new['source_pdf_sha256']:
            raise ValueError('PDF source version changed')
        original={c['chunk_id']:c for c in old['chunks']}
        by_anchor={}
        for c in new['chunks']:by_anchor.setdefault(anchor(c),[]).append(c)
        rows={c['chunk_id']:{'chunk_id':c['chunk_id'],'relevance':'not_reviewed',
            'evidence_quote':'','notes':'','human_confirmed':False} for c in new['chunks']}
        for row in task['rows']:
            matches=by_anchor.get(anchor(original[row['chunk_id']]),[])
            if len(matches)!=1:
                unmapped.append({'paper_id':pid,'field':task['field'],'old_row':row,
                    'source_anchor':original[row['chunk_id']], 'reason':'no_unique_exact_anchor_in_new_review_scope'})
                continue
            c=matches[0];quote=row.get('evidence_quote','')
            rows[c['chunk_id']]={**row,'chunk_id':c['chunk_id'],'human_confirmed':False,
                'quote_exact':quote in c['text'] if quote else None}
            mapping.append({'paper_id':pid,'field':task['field'],'old_chunk_id':row['chunk_id'],
                'new_chunk_id':c['chunk_id'],'matching_basis':'same_pdf_page_offsets_and_exact_text'})
        tasks.append({'paper_id':pid,'field':task['field'],'source_pdf_sha256':new['source_pdf_sha256'],
            'status':'partial','reviewed_at':None,'rows':list(rows.values())})
    value={'kind':'field_evidence_review','schema_version':'field-evidence-review.v2',
        'template_sha256':new_hash,'reviewer':review.get('reviewer'),'reviewer_type':'ai',
        'human_gold':False,'human_signature':None,'tasks':tasks,
        'provenance':{'kind':'AI_preliminary_exact_anchor_migration','old_template_sha256':old_hash}}
    validation=validate_field_review(new_template,value,new_hash)
    if not validation['valid']:raise ValueError('reanchored review failed validation')
    return value,{'mapped_rows':len(mapping),'archived_unmapped_rows':len(unmapped),
                  'mapping':mapping,'unmapped':unmapped,'validation':validation}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-template',required=True,type=Path)
    p.add_argument('--new-template',required=True,type=Path)
    p.add_argument('--review',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    args=p.parse_args();read=lambda f:json.loads(f.read_text(encoding='utf-8-sig'))
    value,audit=reanchor_ai_review(read(args.old_template),read(args.new_template),read(args.review),
        sha256(args.old_template.read_bytes()).hexdigest(),sha256(args.new_template.read_bytes()).hexdigest())
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'original_review.json').write_bytes(args.review.read_bytes())
    for name,data in [('ai_review_reanchored.json',value),('migration_audit.json',audit)]:
        (args.output_dir/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'mapped_rows':audit['mapped_rows'],'archived_unmapped_rows':audit['archived_unmapped_rows'],
        'human_reviewed_tasks':audit['validation']['human_reviewed_tasks']}))
