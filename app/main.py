"""FastAPI 应用主入口。

创建 FastAPI 实例，注册路由、中间件、异常处理。
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_library import router as library_router
from app.api.routes_paper import router as paper_router
from app.api.routes_review import router as review_router
from app.core.config import get_settings
from app.core.exceptions import AgentBaseError
from app.core.logger import get_logger
from app.core.security import require_api_key, validate_deployment_security
from app.database.db import init_db
from app.schemas.common_schema import APIResponse

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时建表、关闭时清理。"""
    validate_deployment_security()
    logger.info("Starting %s ...", settings.app_name)
    init_db()
    from app.services.research_job_service import ResearchJobService

    recovery = ResearchJobService.recover_after_restart()
    if recovery["requeued"] or recovery["interrupted"] or recovery["cancelled"]:
        logger.warning("Recovered research jobs after restart: %s", recovery)
    logger.info("Application started successfully.")
    yield
    # 排空后台任务执行器：排队中的任务取消（状态留在库里，由下次启动的
    # 恢复流程统一修复），不再等待运行中的分钟级 LLM 任务自然结束——
    # 否则 SIGTERM 后进程会挂起很久。
    ResearchJobService.shutdown_executor()
    logger.info("Application shutting down.")


_INTERNAL_DETAIL_RE = re.compile(
    r"[a-zA-Z]+://\S+|SELECT\s.+?\sFROM\s|INSERT\s+INTO|sqlite3\.",
    re.IGNORECASE | re.DOTALL,
)


def _sanitize_error_message(message: str) -> str:
    """对外错误消息脱敏。

    异常消息可能内嵌底层细节（内部 URL、SQL 片段、驱动名）；日志保留
    全量，检测到内部细节时对前端整体替换为通用提示。
    """
    text = str(message or "")
    if _INTERNAL_DETAIL_RE.search(text):
        return "服务处理请求时出现问题，请稍后重试（详情见服务端日志）"
    return text


def create_app() -> FastAPI:
    """FastAPI 应用工厂。

    Returns:
        配置好的 FastAPI 实例。
    """
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="面向多学科研究主题的检索、证据结构化与可验证综述智能体",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ---------- CORS ----------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- 异常处理 ----------
    @app.exception_handler(AgentBaseError)
    async def agent_base_error_handler(request, exc: AgentBaseError):
        from fastapi.responses import JSONResponse

        logger.error("[%s] %s", exc.__class__.__name__, exc.message)
        status_code = int(getattr(exc, "status_code", 400) or 400)
        return JSONResponse(
            status_code=status_code,
            content=APIResponse.error(
                code=status_code,
                message=_sanitize_error_message(exc.message),
            ).model_dump(),
        )

    # ---------- 健康检查 ----------
    @app.get("/health", tags=["系统"])
    async def health_check():
        return APIResponse.ok({"status": "healthy", "app": settings.app_name})

    # ---------- 性能指标（P2-3）----------
    # 注意：与 /health 一致，当前未加鉴权。若部署到公网环境，建议加上
    # 基础认证或内网限制，避免节点耗时/成功率等运行细节被外部探测。
    @app.get("/metrics", tags=["系统"], dependencies=[Depends(require_api_key)])
    async def metrics_report():
        from app.core.metrics import get_metrics_collector

        return APIResponse.ok(get_metrics_collector().get_report())

    # ---------- 路由 ----------
    protected = [Depends(require_api_key)]
    app.include_router(paper_router, prefix="/api/papers", tags=["论文"], dependencies=protected)
    app.include_router(review_router, prefix="/api/reviews", tags=["综述"], dependencies=protected)
    app.include_router(library_router, prefix="/api/library", tags=["论文库"], dependencies=protected)

    return app


app = create_app()
