"""离线证据漏斗和完整原文审查包；不调用Provider，不代填人工结论。"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import html
import json
from pathlib import Path

FIELDS = ('research_problem', 'method', 'dataset', 'metrics', 'results', 'contributions', 'limitations')


def field_funnel(chunks, retrieval, card, references, *, raw_card=None, attached_card=None):
    """章节补选与Top-K是两条支路；历史缺少中间态时显式记录未知。"""
    by_id = {c['chunk_id']: c for c in chunks}
    indexed = set(retrieval['indexed_chunk_ids'])
    selected = set(retrieval['selected_chunk_ids'])
    coverage = (card.get('extraction_coverage') or {}).get('chunks') or []
    processed = {r['chunk_id'] for r in coverage if r['status'] == 'processed'}
    rows = []
    for field in FIELDS:
        relevant = {r['chunk_id'] for r in references if r['field'] == field and r['label'] == 'relevant'}
        uncertain = {r['chunk_id'] for r in references if r['field'] == field and r['label'] == 'uncertain'}
        top = {h['chunk_id'] for h in retrieval['field_hits'].get(field, [])}
        claims = (card.get('field_claims') or {}).get(field, [])
        prior = (attached_card.get('field_claims') or {}).get(field, []) if attached_card is not None else None
        row = {'field': field, 'reference_basis': 'model_weak_reference_not_human',
            'reference_relevant_chunks': sorted(relevant), 'reference_uncertain_chunks': sorted(uncertain),
            'relevant_indexed': sorted(relevant & indexed), 'relevant_topk': sorted(relevant & top),
            'relevant_selected': sorted(relevant & selected),
            'relevant_selected_outside_topk': sorted((relevant & selected) - top),
            'relevant_processed': sorted(relevant & processed),
            'not_indexed': sorted(relevant - indexed),
            'indexed_not_topk': sorted((relevant & indexed) - top),
            'topk_not_selected': sorted((relevant & top) - selected),
            'selected_not_processed': sorted((relevant & selected) - processed),
            'final_field_claims': len(claims),
            'pre_quality_field_claims': len(prior) if prior is not None else None,
            'quality_removed_claims': max(0, len(prior)-len(claims)) if prior is not None else None,
            'raw_field_present': bool(raw_card.get(field)) if raw_card is not None else None,
            'missing_raw_diagnostics': raw_card is None or attached_card is None,
            'card_quality': card['quality_status'],
            'field_issues': [v for v in card.get('quality_issues', []) if v.startswith(field + ':')],
            'topic_relation': card.get('relation_type')}
        # WHY: 不能从最终字段为空反推“LLM未抽取”或“门禁拒绝”，两者需要独立中间快照。
        row['earliest_observed_gap'] = (
            'no_positive_reference' if not relevant else
            'not_indexed' if not (relevant & indexed) else
            'not_selected_for_extraction' if not (relevant & selected) else
            'not_processed' if not (relevant & processed) else
            'retained_field_claims' if claims else
            'quality_gate_removed_field' if prior else
            'raw_field_empty' if raw_card is not None and not raw_card.get(field) else
            'post_extraction_stage_unknown')
        assert all(i in by_id for i in relevant | uncertain)
        rows.append(row)
    return rows


def build_package(source_dir, evaluation_dir, output, *, review_only=False):
    output.mkdir(parents=True, exist_ok=False)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    sources = {d['parsed']['paper_id']: d for d in [read(p) for p in (source_dir/'chunk_sources').glob('*.json')]}
    retrievals = {} if review_only else {r['paper_id']: r['report'] for r in read(source_dir/'chunk_retrieval.json')['papers']}
    cards = read(evaluation_dir/'rate_limited_resume/paper_cards.json')
    references = [] if review_only else read(evaluation_dir/'reference_v2/model_reference.json')
    package, funnels, diagnoses = [], [], []
    for card in cards:
        pid=card['paper_id']; data=sources[pid]; chunks=[c for c in data['chunks'] if not c['excluded']]
        if not review_only:
            rows=field_funnel(data['chunks'],retrievals[pid],card,[r for r in references if r['paper_id']==pid])
            funnels.append({'paper_id':pid,'title':card['title'],'fields':rows})
        statuses=Counter(r['status'] for r in card['extraction_coverage']['chunks'])
        if not review_only:
            diagnoses.append({'paper_id':pid,'title':card['title'],'quality':card['quality_status'],
            'processed':statuses['processed'],'eligible':len(chunks),'issues':card['quality_issues'],
            'meaningful_fields':{f:bool(card.get(f)) for f in ['research_problem','method','results']},
            'quality_rule':'invalid iff research_problem, method and results are all empty after the quality gate',
            'topic_admission_separate':True,'raw_failure_cause':'unknown_without_extraction_snapshots'})
        package.append({'paper_id':pid,'title':card['title'],'url':card['url'],
            'pdf_url':f"https://arxiv.org/pdf/{pid.split(':',1)[1]}",
            'source_pdf_sha256':data['parsed']['source_pdf_hash'],'fields':list(FIELDS),
            'parser_profile':data['parsed'].get('parser_profile'),
            'chunks':[dict(c) for c in chunks]})
    payload={'kind':'human_review_template','schema_version':'field-evidence-review.v2','human_gold':False,'papers':package,
        'reviews':[],'reviewer':None,'reviewed_at':None,'reviewer_type':None,
        'field_definitions':{'research_problem':'待解决的问题或研究动机；不把实验覆盖范围、研究发现混入。',
        'method':'本文实际采用的方法和研究设计。','dataset':'实际使用的数据、样本或材料；完整性单独评价。',
        'metrics':'明示的评价指标或测量标准。','results':'本文报告的发现，含基准分析，不限于性能提升。',
        'contributions':'作者明示的贡献。','limitations':'本文方法自身的局限及适用条件；前人局限需区分。'}}
    def save(name,data):
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    save('review_template.json',payload)
    if not review_only:
        save('evidence_funnel.json',funnels);save('invalid_card_diagnosis.json',diagnoses)
    # 浏览器端仅渲染原文和未填写表单；隐藏检索排名及模型标签，减少锚定效应。
    embedded=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>21项字段证据审查</title>
<style>body{font:16px/1.7 sans-serif;max-width:1100px;margin:25px auto;padding:15px}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}select,input,button,textarea{font:inherit;margin:6px;padding:5px}article{border-bottom:1px solid #ccc;padding-bottom:20px}textarea{width:95%}.toolbar{position:sticky;top:0;background:white;border-bottom:1px solid #ccc;padding:8px}</style>
<h1>21项字段证据审查</h1><p>全文展示，无1200字符截断。complete只表示标注覆盖完成，人工验收由独立校验器重新计算。可分批导出并导入继续填写；AI预审不会自动转为人工结论。</p>
<div class="toolbar"><select id="task"></select><input id="reviewer" placeholder="审查者姓名"><select id="reviewerType"><option value="">审查者类型</option><option value="human">人类审查者</option><option value="ai">AI辅助审查</option></select><select id="sourceBasis"><option value="parsed_text">仅解析文本</option><option value="original_pdf">已核对原始PDF</option></select><label><input id="attest" type="checkbox">我本人独立核对原PDF并逐条确认</label><button id="save">导出JSON</button><input id="load" type="file" accept=".json"></div><main id="body"></main>
<script>const template=__DATA__;const answers={};let current='';
const task=document.getElementById('task');template.papers.forEach((p,pi)=>p.fields.forEach(f=>{let o=document.createElement('option');o.value=pi+':'+f;o.textContent=p.title+' / '+f;task.append(o)}));
function persist(){if(!current)return;answers[current]=Array.from(document.querySelectorAll('article')).map(e=>({chunk_id:Number(e.dataset.id),relevance:e.querySelector('select').value,evidence_quote:e.querySelector('textarea').value,notes:e.querySelector('input[type=text]').value,human_confirmed:document.getElementById('reviewerType').value==='human'&&e.querySelector('input[type=checkbox]').checked}));}
function render(){persist();current=task.value;let [pi,f]=current.split(':');let p=template.papers[Number(pi)];let root=document.getElementById('body');root.replaceChildren();let h=document.createElement('h2');h.textContent=f+'：'+template.field_definitions[f];root.append(h);let a=document.createElement('a');a.href=p.pdf_url;a.textContent='打开原始PDF（页码从1开始）';a.target='_blank';root.append(a);
p.chunks.forEach(c=>{let e=document.createElement('article');e.dataset.id=c.chunk_id;let h=document.createElement('h3');h.textContent='PDF第'+c.page+'页 · 原文块'+(c.chunk_id+1)+' · '+c.text.length+'字符（完整）';let pre=document.createElement('pre');pre.textContent=c.text;let select=document.createElement('select');[['not_reviewed','未审查'],['relevant','相关证据'],['irrelevant','不相关'],['uncertain','不确定']].forEach(v=>{let o=document.createElement('option');o.value=v[0];o.textContent=v[1];select.append(o)});let quote=document.createElement('textarea');quote.placeholder='相关时粘贴连续原文证据；不要翻译改写';let note=document.createElement('input');note.type='text';note.placeholder='边界、字段分类或完整性说明';let confirm=document.createElement('input');confirm.type='checkbox';confirm.disabled=document.getElementById('reviewerType').value!=='human';let label=document.createElement('label');label.append(confirm,document.createTextNode('我已独立对照原PDF确认此条'));let old=(answers[current]||[]).find(x=>x.chunk_id===c.chunk_id);if(old){select.value=old.relevance;quote.value=old.evidence_quote;note.value=old.notes;confirm.checked=old.human_confirmed===true;}e.append(h,select,pre,quote,note,label);root.append(e)});}
task.onchange=render;render();document.getElementById('reviewerType').onchange=()=>{persist();current='';render();};document.getElementById('save').onclick=()=>{persist();let reviewer=document.getElementById('reviewer').value.trim(),kind=document.getElementById('reviewerType').value;let result={kind:'field_evidence_review',schema_version:'field-evidence-review.v2',template_sha256:'__HASH__',reviewer:reviewer||null,reviewer_type:kind||null,exported_at:new Date().toISOString(),human_gold:false,human_signature:null,tasks:[]};let signed=[];
Object.entries(answers).forEach(([key,rows])=>{let [pi,f]=key.split(':');let p=template.papers[Number(pi)];rows.forEach(r=>{let c=p.chunks.find(c=>c.chunk_id===r.chunk_id);r.quote_exact=r.evidence_quote?c.text.includes(r.evidence_quote):null;});let complete=rows.length===p.chunks.length&&rows.every(r=>r.relevance!=='not_reviewed'&&(r.relevance!=='relevant'||r.quote_exact===true));let independentlyConfirmed=complete&&rows.every(r=>r.human_confirmed===true);if(independentlyConfirmed)signed.push(p.paper_id+'|'+f);result.tasks.push({paper_id:p.paper_id,source_pdf_sha256:p.source_pdf_sha256,field:f,status:complete?'complete':rows.some(r=>r.relevance!=='not_reviewed')?'partial':'not_reviewed',annotated_at:complete?new Date().toISOString():null,reviewed_at:null,rows});});if(kind==='human'&&reviewer&&signed.length&&document.getElementById('attest').checked&&document.getElementById('sourceBasis').value==='original_pdf'){result.human_signature={reviewer,signed_at:new Date().toISOString(),independent:true,source_basis:'original_pdf',task_keys:signed};}let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));a.download='field-evidence-review.json';a.click();URL.revokeObjectURL(a.href);};
document.getElementById('load').onchange=async(e)=>{let d=JSON.parse(await e.target.files[0].text());if(d.template_sha256!=='__HASH__'){alert('审查模板版本不一致，未导入；请先按原文位置重新对齐');return;}d.tasks.forEach(t=>{let pi=template.papers.findIndex(p=>p.paper_id===t.paper_id);if(pi>=0)answers[pi+':'+t.field]=t.rows.map(r=>({...r,human_confirmed:d.reviewer_type==='human'&&d.human_signature?r.human_confirmed===true:false}));});document.getElementById('reviewer').value=d.reviewer||'';document.getElementById('reviewerType').value=d.reviewer_type||'';document.getElementById('attest').checked=false;document.getElementById('sourceBasis').value='parsed_text';current='';render();};</script></html>'''
    template_hash=sha256((output/'review_template.json').read_bytes()).hexdigest()
    (output/'REVIEW_21_FIELDS.html').write_text(page.replace('__DATA__',embedded).replace('__HASH__',template_hash),encoding='utf-8')
    if review_only:
        return {'tasks':sum(len(p['fields']) for p in package),'full_chunks':sum(len(p['chunks']) for p in package),
                'human_completed_tasks':0,'requires_new_retrieval_evaluation':True}
    prior=read(evaluation_dir/'manual_review_priority10.json')
    lines=['# 九条样本的完整审查上下文','','所有模型判断仍为待人工复核；下列原文不截断。','']
    for n,r in enumerate(prior,1):
        lines += [f"## {n}. {r['paper_title']}",f"字段：{r.get('field') or r.get('id')}"]
        if r['kind']=='chunk_reference':
            c=next(c for c in sources[r['paper_id']]['chunks'] if c['chunk_id']==r['chunk_id'])
            lines += [f"PDF第{c['page']}页，完整{len(c['text'])}字符。",'```text',c['text'],'```']
        else:
            lines += ['抽取主张：',r['claim']]
        lines += ['人工判定：待填写','']
    (output/'PRIORITY9_FULL_CONTEXT.md').write_text('\n'.join(lines),encoding='utf-8')
    report=['# 证据漏斗与卡片诊断','','下列相关性依据为上轮模型参考，不是人工金标准。Top-K与章节补选为支路，不强行视作单调漏斗。','',
        '| 论文 | 字段 | 参考相关块 | Top-K命中 | 选中 | 已处理 | 最终主张 | 最早可定位状态 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for paper in funnels:
        for r in paper['fields']:
            report.append(f"| {paper['title']} | {r['field']} | {len(r['reference_relevant_chunks'])} | {len(r['relevant_topk'])} | {len(r['relevant_selected'])} | {len(r['relevant_processed'])} | {r['final_field_claims']} | {r['earliest_observed_gap']} |")
    report += ['','历史运行未保留逐块原始响应和质量门禁前快照，不能把最终空字段自动归因为LLM漏抽或门禁误拒。',
        '两张invalid卡片的直接质量条件为research_problem/method/results均空；覆盖不足、摘录拒绝、字段清空为并存观察，不凭最终状态断言单一根因。',
        '主题indirect另由语义准入/词法兜底决定，不能当作invalid质量判定的原因。']
    (output/'FUNNEL_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return {'tasks':sum(len(p['fields']) for p in funnels),'full_chunks':sum(len(p['chunks']) for p in package),
            'unknown_intermediate_tasks':sum(r['missing_raw_diagnostics'] for p in funnels for r in p['fields'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',required=True,type=Path)
    parser.add_argument('--evaluation-dir',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--review-only',action='store_true',help='新解析版本只生成审查包，不重用历史块排名或参考标签')
    args=parser.parse_args()
    print(json.dumps(build_package(args.source_dir,args.evaluation_dir,args.output_dir,review_only=args.review_only)))
