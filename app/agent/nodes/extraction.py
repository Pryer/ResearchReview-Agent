"""PDF 下载、解析与 PaperCard 提取节点。"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from app.agent.decorators import node, optional, provides, requires
from app.agent.nodes.base import (
    _compact_debug_value,
    _latest_step,
    _needs_current_time_tool,
    _paper_debug_item,
    _paper_identity_key,
    _preview_text,
    _select_branch_diverse_keywords,
    _select_search_keywords,
    _summarize_papers,
    append_step,
)
from app.core.config import get_settings
from app.core.logger import get_logger
from app.schemas.paper_schema import SourceDiagnostic

if TYPE_CHECKING:
    from app.agent.state import ResearchAgentState

logger = get_logger(__name__)


@node(name="download_pdf", category="retrieval", description="下载开放获取 PDF；失败则降级")
@requires("paper_details")
@provides("pdf_paths")
def download_pdf_node(
    state: "ResearchAgentState",
    should_cancel: Optional[Callable[[], bool]] = None,
) -> "ResearchAgentState":
    """下载开放获取 PDF；失败则降级。"""
    t0 = time.time()
    try:
        from app.tools.download_pdf import (
            allows_pdf_download,
            batch_download_pdfs,
            is_open_access,
        )

        papers = state.get("paper_details") or []
        pdf_paths = batch_download_pdfs(
            papers, state.get("pdf_paths") or {}, should_cancel=should_cancel,
        )
        state["pdf_paths"] = pdf_paths
        policy_skipped_ids = {
            str(paper.get("paper_id") or "")
            for paper in papers
            if paper.get("paper_id") and not allows_pdf_download(paper)
        }
        eligible_ids = {
            str(paper.get("paper_id") or "")
            for paper in papers
            if paper.get("paper_id") and is_open_access(paper)
        }
        downloaded_ids = {
            paper_id
            for paper_id in eligible_ids
            if pdf_paths.get(paper_id)
        }
        append_step(
            state, "download_pdf", "success",
            tool_name="download_pdf",
            input_data={
                "papers": len(papers),
                "paper_sample": _summarize_papers(papers, limit=10),
            },
            output_data={
                "eligible": len(eligible_ids),
                "skipped_by_policy": len(policy_skipped_ids),
                "policy_skipped_ids": sorted(policy_skipped_ids)[:20],
                "downloaded": len(downloaded_ids),
                "failed": len(eligible_ids - downloaded_ids),
                "unavailable": max(
                    0,
                    len(papers) - len(eligible_ids) - len(policy_skipped_ids),
                ),
                "failed_ids": sorted(eligible_ids - downloaded_ids)[:20],
                "pdf_paths": pdf_paths,
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except InterruptedError as exc:
        from app.agent.graph import AgentCancelledError

        raise AgentCancelledError(str(exc)) from exc
    except Exception as e:
        from app.agent.exceptions import DegradableAgentError

        error = DegradableAgentError(
            str(e), step="download_pdf", degraded_mode="abstract_only", original_error=e,
        )
        logger.error("download_pdf_node failed: %s", error.message)
        state.setdefault("errors", []).append(f"download_pdf: {e}")
        append_step(state, "download_pdf", "failed", error=str(e))
    return state


# ============================================================
# Parse PDF 节点
# ============================================================
@node(name="parse_pdf", category="extraction", description="解析已下载 PDF 的全文与章节结构")
@requires("pdf_paths")
@provides("parsed_papers")
def parse_pdf_node(
    state: "ResearchAgentState",
    should_cancel: Optional[Callable[[], bool]] = None,
) -> "ResearchAgentState":
    """解析 PDF，抽取结构化文本。"""
    t0 = time.time()
    try:
        from app.tools.parse_pdf import batch_parse_pdfs

        existing_parsed = dict(state.get("parsed_papers") or {})
        incremental_ids = {
            str(item) for item in (state.get("incremental_new_paper_ids") or [])
            if str(item)
        }
        pending_paths = {
            paper_id: path
            for paper_id, path in (state.get("pdf_paths") or {}).items()
            if (
                paper_id not in existing_parsed
                and (not state.get("incremental_retrieval") or paper_id in incremental_ids)
            )
        }
        parsed_papers = {
            **existing_parsed,
            **batch_parse_pdfs(pending_paths, should_cancel=should_cancel),
        }
        state["parsed_papers"] = parsed_papers
        append_step(
            state, "parse_pdf", "success",
            tool_name="parse_pdf",
            input_data={"pdf_paths": state.get("pdf_paths") or {}},
            output_data={
                "parsed": len(parsed_papers),
                "parsed_ids": list(parsed_papers.keys())[:20],
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except InterruptedError as exc:
        from app.agent.graph import AgentCancelledError

        raise AgentCancelledError(str(exc)) from exc
    except Exception as e:
        logger.error("parse_pdf_node failed: %s", e)
        state.setdefault("errors", []).append(f"parse_pdf: {e}")
        append_step(state, "parse_pdf", "failed", error=str(e))
    return state


# ============================================================
# Extract Card 节点
# ============================================================
@node(name="extract_paper_cards", category="generation", description="从论文全文或摘要生成 PaperCard")
@requires("paper_details", "topic")
@provides("paper_cards")
@optional("parsed_papers")
def extract_card_node(
    state: "ResearchAgentState",
    llm=None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> "ResearchAgentState":
    """从论文全文或摘要生成 PaperCard。"""
    t0 = time.time()
    try:
        from app.tools.extract_paper_card import batch_extract_paper_cards

        existing_cards = (
            list(state.get("paper_cards") or [])
            if state.get("incremental_retrieval") else []
        )
        existing_keys = {
            _paper_identity_key(card) for card in existing_cards
            if _paper_identity_key(card)
        }
        papers_to_extract = [
            paper for paper in (state.get("paper_details") or [])
            if _paper_identity_key(paper) not in existing_keys
        ]
        cards = batch_extract_paper_cards(
            papers_to_extract,
            state.get("parsed_papers") or {},
            llm=llm,
            topic=state.get("topic", ""),
            should_cancel=should_cancel,
        )
        new_cards = [
            c if isinstance(c, dict) else c.model_dump(mode="json") for c in cards
        ]
        state["paper_cards"] = [*existing_cards, *new_cards]
        append_step(
            state, "extract_card", "success",
            tool_name="extract_paper_card",
            input_data={
                "papers": len(state.get("paper_details") or []),
                "reused_cards": len(existing_cards),
                "new_papers": len(papers_to_extract),
                "parsed": len(state.get("parsed_papers") or {}),
                "topic": state.get("topic", ""),
            },
            output_data={
                "cards": len(state["paper_cards"]),
                "new_cards": len(new_cards),
                "card_sample": state["paper_cards"][:5],
            },
            duration_ms=int((time.time() - t0) * 1000),
        )
    except InterruptedError as exc:
        from app.agent.graph import AgentCancelledError

        raise AgentCancelledError(str(exc)) from exc
    except Exception as e:
        from app.agent.exceptions import LLMGenerationError

        error = LLMGenerationError(str(e), step="extract_card", original_error=e)
        logger.error("extract_card_node failed: %s", error.message)
        state.setdefault("errors", []).append(f"extract_card: {e}")
        append_step(state, "extract_card", "failed", error=str(e))
    return state

