"""文献综述相关 API 路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.logger import get_logger
from app.database.db import get_db
from app.schemas.agent_schema import AgentRequest, ResearchRevisionRequest
from app.schemas.common_schema import APIResponse
from app.services.review_service import ReviewService

logger = get_logger(__name__)
router = APIRouter()

# 历史同步入口 POST /generate 与 POST /agent 已彻底移除（原为 410 占位）：
# 长任务统一走 POST /jobs 异步提交，避免超时、支持取消与进度轮询。


@router.post("/jobs", response_model=APIResponse, summary="提交后台研究任务")
def create_research_job_api(
    request: AgentRequest,
    db: Session = Depends(get_db),
):
    """立即返回 job_id，前端通过状态接口轮询，不再阻塞长连接。"""
    from app.services.research_job_service import (
        ResearchJobCapacityError,
        ResearchJobService,
        ResearchSessionBusyError,
    )

    try:
        return APIResponse.ok(ResearchJobService(db).submit(request))
    except ResearchJobCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ResearchSessionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/jobs/revise", response_model=APIResponse, summary="提交增量修订任务")
def create_research_revision_job_api(
    request: ResearchRevisionRequest,
    db: Session = Depends(get_db),
):
    """排除指定论文，只重新执行聚类、生成与引用验证。"""
    from app.services.research_job_service import (
        ResearchJobCapacityError,
        ResearchJobService,
        ResearchSessionBusyError,
    )

    try:
        return APIResponse.ok(ResearchJobService(db).submit_revision(request))
    except ResearchJobCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ResearchSessionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=APIResponse, summary="查询研究任务")
def get_research_job_api(job_id: str, db: Session = Depends(get_db)):
    from app.services.research_job_service import ResearchJobService

    job = ResearchJobService(db).get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="研究任务不存在")
    return APIResponse.ok(job)


@router.post("/jobs/{job_id}/cancel", response_model=APIResponse, summary="取消研究任务")
def cancel_research_job_api(job_id: str, db: Session = Depends(get_db)):
    from app.services.research_job_service import ResearchJobService

    job = ResearchJobService(db).cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="研究任务不存在")
    return APIResponse.ok(job)


from pydantic import BaseModel, Field
from typing import Any

class ConfigurationRevision(BaseModel):
    expected_revision: int = Field(ge=0)
    values: dict[str, Any]

@router.patch("/sessions/{session_id}/configuration", response_model=APIResponse, summary="显式修订空闲任务配置")
def revise_configuration_api(session_id: str, request: ConfigurationRevision, db: Session = Depends(get_db)):
    from app.database.runtime_repository import ResearchRuntimeRepository, RuntimeConflict
    try:
        return APIResponse.ok(ResearchRuntimeRepository(db,session_id).revise_configuration(
            expected_revision=request.expected_revision,values=request.values))
    except RuntimeConflict as exc:
        raise HTTPException(status_code=409,detail=str(exc)) from exc
    except ValueError as exc:
        # Settings errors can include invalid secret-bearing input; do not echo it.
        raise HTTPException(status_code=400,detail="配置无效、低于已消耗额度或包含禁止字段") from exc


@router.get("/sessions/{session_id}/reports", response_model=APIResponse, summary="查看任务运行报告列表")
def list_execution_reports_api(session_id: str, db: Session = Depends(get_db)):
    from app.services.execution_report_service import list_reports
    return APIResponse.ok(list_reports(db,session_id))


@router.get("/sessions/{session_id}/reports/{report_id}/files/{name}", summary="读取运行报告及导出文件")
def execution_report_file_api(session_id: str, report_id: str, name: str, db: Session = Depends(get_db)):
    from fastapi.responses import FileResponse
    from app.services.execution_report_service import report_file
    path=report_file(db,session_id,report_id,name)
    if path is None:raise HTTPException(status_code=404,detail="运行报告不存在或尚未完成")
    media='text/html' if name=='report.html' else 'application/zip' if name=='bundle.zip' else 'application/gzip' if name.endswith('.gz') else 'application/json'
    return FileResponse(path,media_type=media,headers={
        'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff',
        'Content-Security-Policy':"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"})


@router.get("/sessions/{session_id}", response_model=APIResponse, summary="查询研究会话记忆")
def get_research_session_api(session_id: str, db: Session = Depends(get_db)):
    """返回可展示的会话历史和当前结果，不暴露内部大体积执行状态。"""
    from app.database.repositories import ResearchSessionRepository

    session = ResearchSessionRepository(db).get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="研究会话不存在")
    state = session.get("state") or {}
    from app.services.research_artifact_service import ResearchArtifactService
    public=ResearchArtifactService(db).read_published_delivery(session_id,status=session.get("status") or "running")
    return APIResponse.ok(
        {
            "session_id": session_id,
            "status": session.get("status"),
            "original_query": session.get("original_query"),
            "conversation_history": [
                {key: value for key, value in item.items() if not key.startswith("_")}
                for item in state.get("conversation_history") or [] if isinstance(item, dict)
            ],
            "revision_history": state.get("revision_history") or [],
            "revision_number": state.get("revision_number") or 0,
            "selected_paper_ids": state.get("selected_paper_ids") or [],
            "excluded_paper_ids": state.get("excluded_paper_ids") or [],
            "result": {**(state.get("result_snapshot") or {}),**(public or {})} if public else state.get("result_snapshot"),
        }
    )


@router.post("/from_papers", response_model=APIResponse, summary="基于指定论文生成综述")
def generate_review_from_papers_api(
    paper_ids: list[str],
    db: Session = Depends(get_db),
):
    """旧同步写作入口已移除；指定论文应作为后台任务的会话上下文处理。"""
    raise HTTPException(
        status_code=410,
        detail={
            "error": "endpoint_deprecated",
            "message": "该同步写作入口已移除，请使用后台研究任务接口",
        },
    )


@router.get("/{review_id}", response_model=APIResponse, summary="查看历史综述")
def get_review_api(
    review_id: int,
    db: Session = Depends(get_db),
):
    """根据 ID 获取已保存的综述。"""
    service = ReviewService(db)
    review = service.get_by_id(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="综述不存在")
    return APIResponse.ok(review)


@router.get("/", response_model=APIResponse, summary="列出历史综述")
def list_reviews_api(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """列出已生成的历史综述。"""
    service = ReviewService(db)
    reviews = service.list_reviews(limit=limit)
    return APIResponse.ok(reviews)
