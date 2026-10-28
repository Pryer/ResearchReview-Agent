"""论文内部检索诊断；分数和命中不授予引用权限。"""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from app.schemas.vector_schema import IndexCoverage


class ChunkHit(BaseModel):
    chunk_id: int
    bm25: Optional[float] = None
    cosine: Optional[float] = None
    rrf: float = 0.0
    rerank: Optional[float] = None


class ChunkRetrievalReport(BaseModel):
    mode: Literal["sections", "hybrid"]
    backend: str = "numpy_exact"
    indexed_chunk_ids: List[int] = Field(default_factory=list)
    selected_chunk_ids: List[int] = Field(default_factory=list)
    field_hits: Dict[str, List[ChunkHit]] = Field(default_factory=dict)
    degraded: List[str] = Field(default_factory=list)
    index_truncated: bool = False
    index_coverage: Optional[IndexCoverage] = None
