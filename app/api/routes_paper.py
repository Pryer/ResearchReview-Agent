"""论文相关 API 路由。"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.logger import get_logger
from app.database.db import get_db
from app.schemas.common_schema import APIResponse
from app.schemas.paper_schema import PaperMetadata, PaperSearchRequest
from app.services.paper_service import PaperService

logger = get_logger(__name__)
router = APIRouter()


@router.post("/search", response_model=APIResponse, summary="检索论文")
def search_papers_api(
    request: PaperSearchRequest,
    db: Session = Depends(get_db),
):
    """根据主题和年份范围检索论文并返回排序结果。"""
    service = PaperService(db)
    papers = service.search(request)
    return APIResponse.ok([p.model_dump() for p in papers])


@router.post("/search/core", response_model=APIResponse, summary="检索论文统一字段")
def search_papers_core_api(
    request: PaperSearchRequest,
    db: Session = Depends(get_db),
):
    """检索论文并只返回统一爬取字段：标题、作者、年份、摘要、期刊/会议、DOI、URL、PDF URL。"""
    service = PaperService(db)
    papers = service.search(request)
    return APIResponse.ok([p.to_crawl_metadata().model_dump() for p in papers])


@router.get("/{paper_id}", response_model=APIResponse, summary="获取论文详情")
def get_paper_detail_api(
    paper_id: str,
    db: Session = Depends(get_db),
):
    """根据论文 ID 获取元数据详情。"""
    service = PaperService(db)
    paper = service.get_by_id(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")
    return APIResponse.ok(paper.model_dump())


@router.get("/", response_model=APIResponse, summary="列出本地论文库")
def list_papers_api(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """列出本地已入库的论文列表。"""
    service = PaperService(db)
    papers = service.list_papers(limit=limit)
    return APIResponse.ok([p.model_dump() for p in papers])


@router.post("/{paper_id}/card", response_model=APIResponse, summary="生成单篇论文卡片")
def generate_paper_card_api(
    paper_id: str,
    db: Session = Depends(get_db),
):
    """对指定论文生成结构化 PaperCard。"""
    service = PaperService(db)
    card = service.build_paper_card(paper_id)
    if not card:
        raise HTTPException(status_code=404, detail="论文不存在或无法生成卡片")
    return APIResponse.ok(card.model_dump())
