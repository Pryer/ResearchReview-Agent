"""向量快照的来源、所有权与覆盖状态。"""
from dataclasses import dataclass
from typing import Literal, Optional, Protocol

from pydantic import BaseModel


class VectorStoreUnavailable(RuntimeError):
    """连接/依赖不可用，必须停止持久化检索，不能包装为章节模式成功。"""


class VectorIntegrityError(RuntimeError):
    """来源、维度、权限或所有权不匹配，禁止当作普通来源失败吞掉。"""


class IndexCoverage(BaseModel):
    backend: str = "pgvector"
    snapshot_id: Optional[str] = None
    source_revision: Optional[int] = None
    status: Literal["complete", "partial", "failed"]
    total_chunks: int = 0
    indexed_chunks: int = 0
    reused: bool = False
    index_truncated: bool = False


@dataclass(frozen=True)
class BuildTicket:
    snapshot_id: str
    builder_id: str
    fencing_token: int
    source_revision: int


@dataclass(frozen=True)
class IndexSpec:
    tenant: str
    paper_id: str
    source_hash: str
    processing_profile: str
    embedding_profile: str
    dimension: int
    expected_count: int
    total_count: int


class ChunkVectorIndex(Protocol):
    """Dense 后端共同契约：返回当前候选集合的位置及余弦相似度。"""
    def search(self, query, limit: int) -> list[tuple[int, float]]: ...
