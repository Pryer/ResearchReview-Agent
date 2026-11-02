"""真实Provider的8/12/16块抽取预算对照；默认仅预检。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CURRENT=ContextVar('card_ablation_case',default=None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',required=True,type=Path)
    p.add_argument('--previous-evaluation',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--token-budget',type=int,default=10_000_000)
    p.add_argument('--deadline-seconds',type=int,default=900)
    p.add_argument('--request-limit',type=int,default=128)
    args=p.parse_args()
    if not (0<args.token_budget<=10_000_000 and 0<args.deadline_seconds<=900 and 0<args.request_limit<=128):
        p.error('exceeds authorized limits')
    out=args.output_dir.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()):p.error('output must be under data')
    out.mkdir(parents=True,exist_ok=False)
    os.environ.update(APP_LOG_DIR=str(out/'logs'),AGENT_MAIN_TOKEN_BUDGET=str(args.token_budget),
        AGENT_EXECUTION_DEADLINE_SECONDS=str(args.deadline_seconds))
    from app.core.config import get_settings
    get_settings.cache_clear();settings=get_settings()
    from app.agent.execution_budget import budget_scope,submit_with_context,check_execution,AgentBudgetExceeded
    from app.schemas.chunk_schema import ChunkRetrievalReport
    from app.services.document_ingestion import validate_chunk_sources
    from app.services.llm_service import LLMService
    import app.services.llm_service as lm
    import app.tools.extract_paper_card as extraction
    from app.tools.chunk_retriever import select_extraction_chunks
    from app.services.retrieval_ranking_service import screen_candidates
    from scripts.e2e_reporting import source_identity,write_json
    from scripts.diagnose_evidence_funnel import field_funnel
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    lock=threading.RLock();rate_lock=threading.Lock();cache_locks={};last_request=0.0;stopped=threading.Event()
    save=lambda name,data:write_json(out/name,data)
    sources={d['parsed']['paper_id']:d for d in [read(path) for path in (args.source_dir/'chunk_sources').glob('*.json')]}
    retrievals={r['paper_id']:r['report'] for r in read(args.source_dir/'chunk_retrieval.json')['papers']}
    old_cards=read(args.previous_evaluation/'rate_limited_resume/paper_cards.json')
    papers=[]
    for c in old_cards:
        pid=c['paper_id'];data=sources[pid]
        assert sha256(Path(data['source_pdf']).read_bytes()).hexdigest()==data['parsed']['source_pdf_hash']
        validate_chunk_sources(pid,data['chunks'],data['parsed'])
        papers.append({key:c.get(key) for key in ['paper_id','title','authors','year','venue','doi','url','source','arxiv_id','publication_status']})
        papers[-1].update(abstract=data['parsed'].get('abstract'),pdf_url=f"https://arxiv.org/pdf/{pid.split(':')[1]}",is_open_access=True)
    reference=read(args.previous_evaluation/'reference_v2/model_reference.json')
    plan={'budgets':[8,12,16],'papers':papers,'limits':vars(args),'topic':'retrieval augmented generation for knowledge-intensive tasks and open-domain question answering',
        'retrieval':'same_frozen_rankings_and_production_selection_algorithm',
        'llm_response_reuse':'exact_prompt_single_flight_within_this_round','human_gold':False}
    plan['limits']={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    save('plan.json',plan)
    if not args.execute:
        print(json.dumps({'precheck':'passed','papers':len(papers),'budgets':plan['budgets']}));return 0
    manifest={'status':'running','started_at':datetime.now().astimezone().isoformat(),
        'limits':plan['limits'],'provider_attempts':0,'responses':0,'cache_hits':0,'completed_cases':[],'errors':[]}
    manifest.update(source_identity(ROOT));save('run.json',manifest)
    state={'agent_execution_budget':{'token_limit':args.token_budget}}
    original_create=lm.budgeted_create
    original_settings=get_settings
    def case_settings():
        current=CURRENT.get()
        return settings.model_copy(update={'card_extraction_max_chunks':current['budget']}) if current else settings
    def observed(client,**kwargs):
        nonlocal last_request
        check_execution()
        with lock:
            if stopped.is_set() or manifest['provider_attempts']>=args.request_limit:
                raise AgentBudgetExceeded('evaluation stopped/request limit')
            manifest['provider_attempts']+=1
        with rate_lock:
            delay=1.3-(time.monotonic()-last_request)
            if delay>0:time.sleep(delay)
            last_request=time.monotonic()
        previous=kwargs.get('on_response')
        def received(response):
            with lock:
                manifest['responses']+=1
                usage=getattr(response,'usage',None)
                with (out/'usage.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'operation':kwargs.get('operation'),'model':getattr(response,'model',None),
                        'total_tokens':getattr(usage,'total_tokens',None),'at':datetime.now().astimezone().isoformat()})+'\n')
            if previous:previous(response)
        kwargs['on_response']=received
        try:return original_create(client,**kwargs)
        except BaseException:
            stopped.set();raise
    class LiveLLM:
        def __init__(self,cache=False):
            self.service=LLMService();self.service.backup_enabled=False;self.service.thinking_enabled=False
            self.cache=cache
        def complete(self,prompt,**kwargs):
            check_execution()
            if stopped.is_set():raise AgentBudgetExceeded('first provider failure')
            kwargs.update(retry_empty=False,temperature=0,timeout=90)
            kwargs.setdefault('max_tokens',4096)
            kwargs.setdefault('operation','card_budget_extraction' if self.cache else 'card_budget_admission')
            if not self.cache:return self.service.complete(prompt,**kwargs)
            key=sha256(json.dumps([settings.llm_model,prompt,kwargs],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            with lock:key_lock=cache_locks.setdefault(key,threading.Lock())
            path=out/'response_cache'/f'{key}.json'
            with key_lock:
                if path.exists():
                    with lock:manifest['cache_hits']+=1
                    value=read(path);assert value['key']==key
                    return value['response']
                response=self.service.complete(prompt,**kwargs)
                path.parent.mkdir(exist_ok=True)
                write_json(path,{'key':key,'response':response,'created_at':datetime.now().astimezone().isoformat()})
                return response
    def replay(pid,chunks,**kwargs):
        assert chunks==sources[pid]['chunks']
        report=ChunkRetrievalReport.model_validate(retrievals[pid])
        report=select_extraction_chunks(chunks,report,CURRENT.get()['budget'])
        CURRENT.get()['retrieval']=report.model_dump(mode='json')
        return report
    def observe_stage(name,fn):
        def wrapped(*a,**kw):
            result=fn(*a,**kw)
            current=CURRENT.get()
            if current is not None:current[name]=result.model_dump(mode='json')
            return result
        return wrapped
    def run_case(paper,budget):
        pid=paper['paper_id'];case={'paper_id':pid,'budget':budget};token=CURRENT.set(case)
        started=time.monotonic()
        try:
            card=extraction.extract_paper_card(paper,sources[pid]['parsed'],LiveLLM(cache=True),plan['topic'])
            value=card.model_dump(mode='json');case.update(card=value,elapsed_seconds=time.monotonic()-started)
            case['funnel']=field_funnel(sources[pid]['chunks'],case['retrieval'],value,
                [r for r in reference if r['paper_id']==pid],raw_card=case.get('raw'),attached_card=case.get('attached'))
            filename=f"case_{budget}_{pid.replace(':','_')}.json";save(filename,case)
            with lock:
                manifest['completed_cases'].append(filename)
                manifest['budget_ledger']=copy.deepcopy(state['agent_execution_budget']);save('run.json',manifest)
            print(json.dumps({'stage':'case','budget':budget,'quality':value['quality_status'],
                'processed':sum(r['status']=='processed' for r in value['extraction_coverage']['chunks'])}),flush=True)
            return case
        finally:CURRENT.reset(token)
    cases=[]
    try:
        with budget_scope(state),patch.object(lm,'budgeted_create',observed),patch('app.core.config.get_settings',case_settings):
            admitted,diagnostics=screen_candidates({'topic':plan['topic'],'required_reference_count':3},
                copy.deepcopy(papers),LiveLLM(),target=len(papers),strict_provider_errors=True)
            save('semantic_admission.json',{'admitted':admitted,'diagnostics':diagnostics})
            if len(admitted)!=len(papers):raise ValueError('not_all_frozen_papers_admitted')
            with patch('app.tools.chunk_retriever.retrieve_chunks',replay),\
                 patch.object(extraction,'extract_from_full_text',observe_stage('raw',extraction.extract_from_full_text)),\
                 patch.object(extraction,'_deduplicate_card_fields',observe_stage('deduplicated',extraction._deduplicate_card_fields)),\
                 patch.object(extraction,'_attach_evidence',observe_stage('attached',extraction._attach_evidence)),\
                 ThreadPoolExecutor(max_workers=3) as pool:
                futures=[submit_with_context(pool,run_case,paper,b) for b in [8,12,16] for paper in admitted]
                for future in as_completed(futures):cases.append(future.result())
            save('cases.json',cases)
            manifest['status']='completed_ablation_pending_human'
    except BaseException as exc:
        stopped.set();manifest['status']='stopped';manifest['errors'].append({'type':type(exc).__name__})
    finally:
        manifest['finished_at']=datetime.now().astimezone().isoformat()
        manifest['budget_ledger']=state['agent_execution_budget']
        manifest['source_unchanged']=source_identity(ROOT)['source_tree_sha256']==manifest['source_tree_sha256']
        save('run.json',manifest)
        print(json.dumps({k:manifest.get(k) for k in ['status','provider_attempts','responses','cache_hits','budget_ledger','errors']}),flush=True)
    return 0 if manifest['status'].startswith('completed') else 1


if __name__=='__main__':raise SystemExit(main())
