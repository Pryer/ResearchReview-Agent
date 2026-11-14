"""Raw evidence dependencies are separate from semantic admission and display ranks."""
import copy
import hashlib
import json
from pathlib import Path

def extraction_fingerprint(paper, parsed, topic, settings):
    from app.services.evidence_support import EVIDENCE_POLICY_VERSION
    from app.services.rag_runtime import RAG_PIPELINE_VERSION
    # WHY: 排名和准入资格变化不改变原文；原文、身份、Prompt或模型变化才重抽。
    original={k:paper.get(k) for k in ("paper_id","title","abstract","authors","year","doi","arxiv_id","url","pdf_url","source")}
    prompt=Path(__file__).parents[1]/"prompt/paper_card.py"
    payload={"version":"extraction-reuse.v1","paper":original,"parsed":parsed,"topic":topic,
        "policy":EVIDENCE_POLICY_VERSION,"pipeline":RAG_PIPELINE_VERSION,
        "prompt":hashlib.sha256(prompt.read_bytes()).hexdigest(),
        "endpoint_fingerprints":[hashlib.sha256(str(v).encode()).hexdigest() for v in (settings.llm_base_url,settings.llm_backup_base_url,settings.retrieval_embedding_url,settings.retrieval_rerank_url)],
        "models":[settings.llm_provider,settings.llm_model,settings.llm_backup_model,
            settings.retrieval_embedding_model,settings.retrieval_rerank_model, settings.retrieval_cache_version],
        "settings":{k:v for k,v in settings.model_dump().items() if k.startswith("card_") and not any(x in k for x in ("password","url","dsn"))}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()

def refresh_card_eligibility(card,paper):
    value=copy.deepcopy(card)
    value.update(relation_type=str(paper.get("_topic_relation") or value.get("relation_type") or "indirect"),
        eligible_deliverables=list(paper.get("_eligible_deliverables") or []),
        anchor_low_confidence=bool(paper.get("_anchor_low_confidence")))
    return value
