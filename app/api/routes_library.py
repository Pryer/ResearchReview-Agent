"""本地论文库相关 API 路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.logger import get_logger
from app.database.db import get_db
from app.schemas.common_schema import APIResponse
from app.services.library_service import LibraryService

logger = get_logger(__name__)
router = APIRouter()


@router.post("/import_pdf", response_model=APIResponse, summary="导入本地 PDF")
def import_pdf_api(
    file_path: str,
    db: Session = Depends(get_db),
):
    """导入本地 PDF 文件到论文库。"""
    service = LibraryService(db)
    result = service.import_pdf(file_path)
    if not result:
        raise HTTPException(status_code=400, detail="PDF 导入失败")
    return APIResponse.ok(result)


@router.post("/search", response_model=APIResponse, summary="搜索本地论文库")
def search_library_api(
    query: str = Query(..., description="检索关键词"),
    top_k: int = Query(default=5, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """在本地论文库中语义检索相关论文。"""
    service = LibraryService(db)
    cards = service.search_local_library(query, top_k=top_k)
    return APIResponse.ok([c.model_dump() for c in cards])


@router.post("/rebuild_index", response_model=APIResponse, summary="重建向量索引")
def rebuild_index_api(
    db: Session = Depends(get_db),
):
    """重建本地论文库的向量索引。"""
    service = LibraryService(db)
    count = service.rebuild_index()
    return APIResponse.ok({"indexed_count": count})


@router.get("/papers", response_model=APIResponse, summary="列出本地论文库")
def list_library_papers_api(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """列出本地论文库中的论文卡片。"""
    service = LibraryService(db)
    cards = service.list_papers(limit=limit)
    return APIResponse.ok([c.model_dump() for c in cards])
