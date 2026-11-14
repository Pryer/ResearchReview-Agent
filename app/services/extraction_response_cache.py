"""Atomic source-bound chunk responses; failed attempts are recorded, never treated as evidence."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import time

class ExtractionResponseCache:
    def __init__(self,settings,paper,parsed_text,prompt):
        from app.services.rag_runtime import RAG_PIPELINE_VERSION
        self.root=Path(settings.retrieval_cache_dir)/"chunk_extraction"
        self.source=["chunk-extraction.v1",RAG_PIPELINE_VERSION,paper.get("paper_id"),paper.get("title"),
            hashlib.sha256(json.dumps(parsed_text,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest(),
            prompt,settings.llm_provider,settings.llm_model,settings.llm_backup_model,
            hashlib.sha256((settings.llm_base_url+settings.llm_backup_base_url).encode()).hexdigest()]
    def key(self,chunk):
        return hashlib.sha256(json.dumps([self.source,chunk],sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
    def get(self,key):
        try:
            value=json.loads((self.root/f"{key}.json").read_text(encoding="utf-8"))
            return value.get("data") if value.get("status")=="success" else None
        except (OSError,ValueError,TypeError):return None
    def put(self,key,data=None,*,error=""):
        self.root.mkdir(parents=True,exist_ok=True)
        name=None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root,suffix=".tmp",mode="w",encoding="utf-8",delete=False) as f:
                name=f.name;json.dump({"status":"failed" if error else "success","data":data,
                    "error_type":error,"at":time.time()},f,ensure_ascii=False)
                f.flush();os.fsync(f.fileno())
            os.replace(name,self.root/f"{key}.json")
        finally:
            if name and os.path.exists(name):os.unlink(name)
