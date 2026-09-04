"""多轮研究会话编排服务。"""

from __future__ import annotations

import re
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.agent.topic_disambiguation import (
    analyze_topic_ambiguity,
    build_scoped_query,
    resolve_scope_conversational,
)
from app.database.repositories import ResearchSessionRepository
from app.schemas.agent_schema import AgentRequest, ResearchRevisionRequest

# 质量决策问句的选项词表：问句措辞与解析判据同源。
# WHY: 问句曾提供"基于现有证据保守重写"，但解析器里 force_generate 的宽口径
# 正则 基于(?:当前|现有).{0,12}(?:…|写) 排在前面先命中，系统等于提供了一个自己
# 解析不了的选项——保守重写实际跑了完整 LLM 重写，支持率反而从 47.1% 掉到
# 36.1%。词表的遍历顺序即解析优先级，保守重写必须排在 force_generate 之前。
_QUALITY_DECISION_OPTIONS: dict[str, tuple[str, str, str]] = {
    # option_id: (action, 问句中的选项原文, 命中该选项的判据)
    "stop": ("stop", "结束本次任务", r"结束本次任务|结束任务"),
    "conservative_rewrite": (
        "regenerate_existing",
        "基于现有证据保守重写",
        r"保守重写|重新保守生成|保守生成",
    ),
    "accept_available": ("accept_available", "接受当前篇数", r"接受当前篇数"),
    "expand_time_range": ("expand_time_range", "扩大时间范围", r"扩大时间范围"),
    "include_more_types": (
        "include_more_types",
        "纳入更多文献类型",
        r"纳入更多文献类型",
    ),
    "broaden_scope": ("broaden_scope", "放宽主题范围", r"放宽主题范围"),
    "retry_search": ("retry_search", "补充检索", r"补充检索"),
    "keep_searching": ("retry_search", "保持条件继续检索", r"保持条件继续检索"),
    "repair_taxonomy": ("repair_taxonomy", "自动重新分类", r"自动重新分类"),
    "best_effort_draft": (
        "force_generate",
        "直接基于现有证据生成最佳可用草稿",
        r"直接基于现有证据生成最佳可用草稿",
    ),
}

_QUALITY_DECISION_ORDER: tuple[str, ...] = (
    "stop",
    "conservative_rewrite",
    "accept_available",
    "expand_time_range",
    "include_more_types",
    "repair_taxonomy",
    "broaden_scope",
    "retry_search",
    "keep_searching",
    "best_effort_draft",
)


def _quality_option_label(option_id: str) -> str:
    """取问句中使用的选项原文，使问句与解析器不可能各说各话。"""
    return _QUALITY_DECISION_OPTIONS[option_id][1]


_CHINESE_YEAR_DIGITS = {
    "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _extract_years(text: str) -> int:
    """从答复中抽取期望的时间跨度；未给出时沿用既有默认 5 年。"""
    number = re.search(r"近\s*(\d{1,2})\s*年", text)
    if number:
        return int(number.group(1))
    char = re.search(r"近\s*([四五六七八九十])\s*年", text)
    if char:
        return _CHINESE_YEAR_DIGITS.get(char.group(1), 5)
    return 5


def _extract_paper_count(text: str) -> int | None:
    """从答复中抽取用户明确写出的篇数。"""
    number = re.search(r"(\d{1,3})\s*篇", text)
    return int(number.group(1)) if number else None


class ResearchConversationService:
    """在耗时检索前处理主题澄清，并持久化可恢复状态。"""

    def __init__(
        self,
        db: Session,
        llm=None,
        agent_runner: Callable[..., dict] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.db = db
        self.repo = ResearchSessionRepository(db)
        if llm is None:
            from app.services.llm_service import LLMService

            llm = LLMService()
        self.llm = llm
        if agent_runner is None:
            from app.agent.graph import run_research_agent

            agent_runner = run_research_agent
        self.agent_runner = agent_runner
        self.should_cancel = should_cancel
        self.progress_callback = progress_callback

    def handle(self, request: AgentRequest) -> dict[str, Any]:
        if self.should_cancel and self.should_cancel():
            from app.agent.graph import AgentCancelledError

            raise AgentCancelledError("任务在会话处理前已取消")
        if not request.session_id:
            raise ValueError("多轮研究请求必须提供 session_id")
        if request.clarification_answer:
            return self._resume_after_clarification(request)
        # 服务端兜底：同一 session 若正等待澄清，下一条自然语言消息
        # 默认就是该澄清回答。这样前端即使遗漏字段，也不会把回答
        # 误当成新的研究会话或新的主题请求。
        existing_session = self.repo.get(str(request.session_id))
        if existing_session and existing_session.get("status") == "needs_clarification":
            request.clarification_answer = request.user_query
            return self._resume_after_clarification(request)
        return self._start_turn(request)

    def _start_turn(
        self,
        request: AgentRequest,
        trusted_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session_id = str(request.session_id)
        # trusted_state 只由服务内部的会话恢复路径提供；外部请求仍经过
        # AgentRequest 的公开字段白名单校验。
        initial_state = dict(trusted_state if trusted_state is not None else (request.state or {}))
        # 顶层意图只在新研究请求上重新识别。澄清恢复由专用路径显式标记为
        # ``working_query``，避免把同一 session 携带的旧状态误当成当前请求。
        initial_state["intent_context_role"] = "request"
        # 一个对话只对应一个 session。新研究请求仍可在同一对话中开始，
        # 但必须保留历史上下文；研究证据等工作状态由本轮重新解析，
        # 避免把上一主题的论文池混入新主题。
        existing_session = self.repo.get(session_id)
        existing_state = (existing_session or {}).get("state") or {}
        history = list(
            initial_state.get("conversation_history")
            or existing_state.get("conversation_history")
            or []
        )
        current_query = str(request.user_query or "").strip()
        already_recorded = any(
            str(item.get("type") or "") == "research_request"
            and str(item.get("content") or "").strip() == current_query
            for item in history
            if isinstance(item, dict)
        )
        if current_query and not already_recorded:
            history.append({
                "role": "user",
                "content": current_query,
                "type": "research_request",
            })
        if history:
            initial_state["conversation_history"] = history[-50:]
        if existing_state.get("revision_history") and "revision_history" not in initial_state:
            initial_state["revision_history"] = list(existing_state["revision_history"])

        # 能力检查必须早于主题消歧和论文检索，避免为不支持的任务消耗模型与数据源调用。
        from app.agent.unsupported_task_guard import check_unsupported_task

        guard = check_unsupported_task(request.user_query)
        initial_state["unsupported_task_guard"] = guard.model_dump(mode="json")
        if not guard.allowed:
            initial_state["conversation_history"] = [
                {"role": "user", "content": request.user_query, "type": "research_request"},
                {"role": "assistant", "content": guard.message, "type": "unsupported_task"},
            ]
            result = {
                "session_id": session_id,
                "status": "blocked",
                "answer": guard.message,
                "intent": None,
                "topic": None,
                "unsupported_task_guard": initial_state["unsupported_task_guard"],
                "generation_blocked": True,
                "core_deliverables": [item.value for item in guard.supported_deliverables],
                "steps": [
                    {
                        "step_name": "unsupported_task_guard",
                        "status": "blocked",
                        "tool_name": None,
                        "input_data": {"user_query": request.user_query},
                        "output_data": initial_state["unsupported_task_guard"],
                        "error": None,
                        "duration_ms": 0,
                    }
                ],
                "references": [],
                "paper_cards": [],
                "clusters": [],
                "errors": [],
            }
            initial_state["result_snapshot"] = dict(result)
            self.repo.save(
                session_id=session_id,
                status="blocked",
                original_query=request.user_query,
                state=initial_state,
            )
            self.db.commit()
            return result

        if self.progress_callback:
            self.progress_callback("preflight:semantic_parsing_and_scope", 0, 14)
        analysis = analyze_topic_ambiguity(request.user_query, llm=self.llm)
        if self.should_cancel and self.should_cancel():
            from app.agent.graph import AgentCancelledError

            raise AgentCancelledError("任务在主题消歧后已取消")
        initial_state["research_request"] = analysis["research_request"]
        initial_state["canonical_topic"] = analysis["research_request"].get("topic")
        initial_state["research_semantic_frame"] = (
            analysis["research_request"].get("semantic_frame") or {}
        )
        initial_state["topic_interpretations"] = analysis["ambiguity"].get("scopes") or []

        # 相关工作依赖用户自己的研究问题与方法方向，应在耗时检索前澄清。
        from app.agent.deliverable_router import (
            check_deliverable_readiness,
            extract_user_paper_profile,
            resolve_core_deliverables,
        )
        from app.schemas.deliverable_schema import CoreDeliverableType

        research_request = analysis["research_request"]
        core_deliverables = resolve_core_deliverables(
            str(research_request.get("task_type") or ""),
            research_request.get("requested_sections") or [],
        )
        initial_state["core_deliverables"] = [item.value for item in core_deliverables]
        profile = extract_user_paper_profile(request.user_query, initial_state)
        initial_state["user_paper_profile"] = profile.model_dump(mode="json")
        if CoreDeliverableType.RELATED_WORK in core_deliverables:
            readiness = check_deliverable_readiness(
                CoreDeliverableType.RELATED_WORK,
                initial_state,
                phase="pre_retrieval",
            )
            if not readiness.ready:
                question = readiness.clarification_question or "你的论文主要解决什么问题，并计划采用什么方法或研究路线？"
                clarification = {
                    "kind": "user_paper_profile",
                    "slot": "user_paper_profile",
                    "question": question,
                    "missing_inputs": readiness.missing_inputs,
                }
                initial_state["conversation_history"] = [
                    {"role": "user", "content": request.user_query, "type": "research_request"},
                    {"role": "assistant", "content": question, "type": "clarification"},
                ]
                self.repo.save(
                    session_id=session_id,
                    status="needs_clarification",
                    original_query=request.user_query,
                    state=initial_state,
                    clarification=clarification,
                )
                self.db.commit()
                return {
                    "session_id": session_id,
                    "status": "needs_clarification",
                    "answer": question,
                    "topic": research_request.get("topic"),
                    "research_request": research_request,
                    "clarification": clarification,
                    "steps": [], "references": [], "paper_cards": [], "clusters": [], "errors": [],
                }

        # 调用方已经显式指定范围时，完成交付物前置检查后不再重复消歧。
        if initial_state.get("selected_scope"):
            return self._run_and_persist(session_id, request.user_query, initial_state)

        if analysis["needs_clarification"]:
            if self.progress_callback:
                self.progress_callback("preflight:clarification_ready", 0, 14)
            clarification = analysis["ambiguity"]
            initial_state["conversation_history"] = [
                {"role": "user", "content": request.user_query, "type": "research_request"},
                {
                    "role": "assistant",
                    "content": clarification.get("question") or "请确认研究范围。",
                    "type": "clarification",
                },
            ]
            self.repo.save(
                session_id=session_id,
                status="needs_clarification",
                original_query=request.user_query,
                state=initial_state,
                clarification=clarification,
            )
            self.db.commit()
            return {
                "session_id": session_id,
                "status": "needs_clarification",
                "answer": clarification.get("question") or "请确认研究范围。",
                "topic": analysis["research_request"].get("topic"),
                "research_request": analysis["research_request"],
                "clarification": clarification,
                "steps": [],
                "references": [],
                "paper_cards": [],
                "clusters": [],
                "errors": [],
            }

        ambiguity = analysis["ambiguity"]
        if ambiguity.get("recommended_strategy") == "multi_branch":
            branches = ambiguity.get("scopes") or []
            combined_scope = {
                "scope_id": "multi_branch",
                "label": "跨分支综合",
                "description": "按识别出的多个研究范围进行综合检索",
                "include_terms": [
                    term for branch in branches for term in (branch.get("include_terms") or [])
                ],
                "exclude_terms": [],
                "seed_queries": [
                    query for branch in branches for query in (branch.get("seed_queries") or [])
                ],
                "branches": branches,
            }
            initial_state["selected_scope"] = combined_scope
            initial_state["intent_context_role"] = "working_query"
            scoped_query = build_scoped_query(request.user_query, combined_scope)
            from app.agent.research_semantic_parser import parse_research_semantics

            initial_state["research_semantic_frame"] = parse_research_semantics(
                scoped_query,
                str(initial_state.get("canonical_topic") or request.user_query),
                deliverables=initial_state.get("core_deliverables") or [],
                llm=self.llm,
            ).model_dump(mode="json")
            return self._run_and_persist(
                session_id,
                scoped_query,
                initial_state,
                original_query=request.user_query,
            )
        return self._run_and_persist(session_id, request.user_query, initial_state)

    def _resume_after_clarification(self, request: AgentRequest) -> dict[str, Any]:
        session_id = str(request.session_id)
        session = self.repo.get(session_id)
        if not session:
            raise ValueError("研究会话不存在或已过期，请重新提交原始研究请求")
        if session.get("status") != "needs_clarification":
            raise ValueError("当前研究会话没有待回答的澄清问题")

        clarification = session.get("clarification") or {}
        state = dict(session.get("state") or {})
        state.update(request.state or {})
        answer = request.clarification_answer or ""
        if clarification.get("kind") == "quality_decision":
            return self._resume_after_quality_decision(
                session_id=session_id,
                session=session,
                state=state,
                clarification=clarification,
                answer=answer,
            )
        if clarification.get("kind") == "user_paper_profile":
            from app.agent.deliverable_router import (
                check_deliverable_readiness,
                extract_user_paper_profile,
            )
            from app.schemas.deliverable_schema import CoreDeliverableType

            profile = extract_user_paper_profile(
                answer,
                {"our_work": state.get("user_paper_profile") or state.get("our_work") or {}},
            )
            state["user_paper_profile"] = profile.model_dump(mode="json")
            state["our_work"] = {
                "research_problem": profile.research_problem,
                "method_name": profile.proposed_method or profile.research_direction or "",
                "method_summary": profile.proposed_method or profile.research_direction or "",
                "innovations": profile.main_contributions,
            }
            readiness = check_deliverable_readiness(
                CoreDeliverableType.RELATED_WORK, state, phase="pre_retrieval"
            )
            history = state.setdefault("conversation_history", [])
            history.append({"role": "user", "content": answer, "type": "clarification_answer"})
            if not readiness.ready:
                question = readiness.clarification_question or "请同时说明你的研究问题和计划采用的方法路线。"
                history.append({"role": "assistant", "content": question, "type": "clarification"})
                updated = {**clarification, "question": question, "missing_inputs": readiness.missing_inputs}
                self.repo.save(
                    session_id=session_id,
                    status="needs_clarification",
                    original_query=session["original_query"],
                    state=state,
                    clarification=updated,
                )
                self.db.commit()
                return {
                    "session_id": session_id, "status": "needs_clarification", "answer": question,
                    "topic": (state.get("research_request") or {}).get("topic"),
                    "research_request": state.get("research_request") or {}, "clarification": updated,
                    "steps": [], "references": [], "paper_cards": [], "clusters": [], "errors": [],
                }
            return self._start_turn(
                AgentRequest(
                    session_id=session_id,
                    user_query=session["original_query"],
                ),
                trusted_state=state,
            )
        if self.progress_callback:
            self.progress_callback("preflight:resolving_clarification", 0, 14)
        resolution = resolve_scope_conversational(clarification, answer, llm=self.llm)
        selected = resolution.get("selected_scope")
        history = state.setdefault("conversation_history", [])
        history.append(
            {
                "role": "user",
                "content": answer,
                "type": "clarification_answer",
            }
        )
        if not selected:
            question = resolution.get("question") or "你希望这次研究主要侧重哪个方向？"
            history.append(
                {
                    "role": "assistant",
                    "content": question,
                    "type": "clarification",
                }
            )
            clarification = {
                **clarification,
                "question": question,
                "turn_count": int(clarification.get("turn_count") or 1) + 1,
            }
            self.repo.save(
                session_id=session_id,
                status="needs_clarification",
                original_query=session["original_query"],
                state=state,
                clarification=clarification,
            )
            self.db.commit()
            return {
                "session_id": session_id,
                "status": "needs_clarification",
                "answer": question,
                "topic": (state.get("research_request") or {}).get("topic"),
                "research_request": state.get("research_request") or {},
                "clarification": clarification,
                "steps": [],
                "references": [],
                "paper_cards": [],
                "clusters": [],
                "errors": [],
            }

        state["selected_scope"] = selected
        state["intent_context_role"] = "working_query"
        state["topic_interpretations"] = clarification.get("scopes") or []
        scoped_query = build_scoped_query(
            session["original_query"],
            selected,
            clarification_answer=answer,
        )
        from app.agent.research_semantic_parser import parse_research_semantics

        semantic_query = (
            session["original_query"].rstrip()
            + "\n用户澄清原文（明确方法、对象、先后关系和分析目标均须保留）："
            + answer.strip()
        )
        state["research_semantic_frame"] = parse_research_semantics(
            semantic_query,
            str((state.get("research_request") or {}).get("topic") or session["original_query"]),
            deliverables=state.get("core_deliverables") or [],
            llm=self.llm,
        ).model_dump(mode="json")
        return self._run_and_persist(
            session_id,
            scoped_query,
            state,
            original_query=session["original_query"],
        )

    def _resume_after_quality_decision(
        self,
        session_id: str,
        session: dict[str, Any],
        state: dict[str, Any],
        clarification: dict[str, Any],
        answer: str,
    ) -> dict[str, Any]:
        """理解用户对质量门禁的自由回答，并从最小必要阶段恢复。"""
        decision = self._parse_quality_decision(answer, clarification)
        history = list(state.get("conversation_history") or [])
        history.append({
            "role": "user",
            "content": answer,
            "type": "quality_decision_answer",
        })
        if decision is None:
            question = (
                "我还不能确定你希望如何调整。请直接说明：直接生成当前最佳草稿、"
                "接受当前篇数、扩大到近几年、纳入更多文献类型、放宽主题范围，"
                "还是保持条件继续检索？"
            )
            updated = {
                **clarification,
                "question": question,
                "turn_count": int(clarification.get("turn_count") or 1) + 1,
            }
            history.append({"role": "assistant", "content": question, "type": "clarification"})
            state["conversation_history"] = history[-50:]
            self.repo.save(
                session_id=session_id,
                status="needs_clarification",
                original_query=session["original_query"],
                state=state,
                clarification=updated,
            )
            self.db.commit()
            return self._clarification_result(session_id, state, updated, question)

        if decision["action"] == "stop":
            snapshot = dict(state.get("result_snapshot") or {})
            message = "已按你的决定结束本次任务；未通过质量门禁的正文不会作为正式结果交付。"
            snapshot.update({
                "session_id": session_id,
                "status": "completed",
                "answer": message,
            })
            history.append({"role": "assistant", "content": message, "type": "quality_decision"})
            state["conversation_history"] = history[-50:]
            state["result_snapshot"] = snapshot
            self.repo.save(
                session_id=session_id,
                status="completed",
                original_query=session["original_query"],
                state=state,
            )
            self.db.commit()
            return snapshot

        editable = dict(state.get("editable_research_state") or {})
        recovery_attempts = int(
            editable.get("quality_recovery_attempts")
            or state.get("quality_recovery_attempts")
            or clarification.get("recovery_attempts")
            or 0
        )
        if (
            recovery_attempts >= 3
            and decision["action"] not in {"force_generate", "accept_available", "stop"}
        ):
            question = (
                "已经连续尝试3次调整，但同一写作门禁仍未通过。为避免无限循环，"
                "系统不再自动重复检索或放宽范围。你可以回答“直接生成”以输出当前"
                "最佳可用草稿，或回答“结束任务”。"
            )
            updated = {
                **clarification,
                "question": question,
                "recovery_attempts": recovery_attempts,
                "recovery_options": ["直接生成当前最佳可用草稿", "结束任务"],
            }
            history.append({"role": "assistant", "content": question, "type": "clarification"})
            state["conversation_history"] = history[-50:]
            self.repo.save(
                session_id=session_id,
                status="needs_clarification",
                original_query=session["original_query"],
                state=state,
                clarification=updated,
            )
            self.db.commit()
            return self._clarification_result(session_id, state, updated, question)
        from app.agent.topic_disambiguation import reconcile_selected_scope_from_history

        reconciled_scope = reconcile_selected_scope_from_history(
            editable.get("selected_scope") or state.get("selected_scope"),
            state.get("topic_interpretations") or [],
            history,
        )
        if reconciled_scope:
            state["selected_scope"] = reconciled_scope
            editable["selected_scope"] = reconciled_scope
        research_request = dict(
            editable.get("research_request")
            or state.get("research_request")
            or {}
        )
        refreshed_frame = self._refresh_open_alternative_semantics(
            session["original_query"],
            history,
            editable.get("research_semantic_frame") or state.get("research_semantic_frame") or {},
            str(
                editable.get("canonical_topic")
                or state.get("canonical_topic")
                or research_request.get("topic")
                or session["original_query"]
            ),
            editable.get("core_deliverables") or state.get("core_deliverables") or [],
            llm=self.llm,
        )
        if refreshed_frame:
            editable["research_semantic_frame"] = refreshed_frame
            state["research_semantic_frame"] = refreshed_frame
            research_request["semantic_frame"] = refreshed_frame
        if decision["action"] == "force_generate":
            if not editable or not (editable.get("paper_cards") or []):
                raise ValueError("当前会话没有可用于生成的论文证据，请重新提交研究请求")
            # WHY: force_generate 是"照原要求尽力生成"，不是"把要求降到当前水平"。
            # 曾在此用阻断消息里的 actual（生成后即已引用篇数）覆盖
            # required_reference_count/max_papers，使 synthesis 的 cited < requested
            # 恒假、用户显式的 40 篇要求凭空消失并被判为达标。降低达标线只能由
            # accept_available 分支执行（用户明确说"接受当前 N 篇"，属授权路径）。
            editable.update({
                "best_effort_generation": True,
                "allow_unvalidated_taxonomy": True,
                "quality_recovery_attempts": recovery_attempts,
                "research_request": research_request,
                "conversation_history": history[-50:],
            })
            return self._regenerate_and_persist(
                session_id, session["original_query"], state, editable
            )
        if decision["action"] in {"accept_available", "regenerate_existing", "repair_taxonomy"}:
            available = int(
                decision.get("available")
                or clarification.get("available")
                or len(editable.get("paper_cards") or [])
            )
            if available < 1 or not editable:
                raise ValueError("当前会话没有可恢复的论文证据，请重新提交研究请求")
            if decision["action"] == "accept_available":
                if "original_required_reference_count" not in editable:
                    editable["original_required_reference_count"] = editable.get("required_reference_count")
                if "original_required_reference_count" not in research_request:
                    research_request["original_required_reference_count"] = research_request.get("required_reference_count")
                editable.update({
                    "required_reference_count": available,
                    "max_papers": available,
                    "max_papers_explicit": True,
                    "generation_limit": max(available, int(editable.get("generation_limit") or 0)),
                })
                research_request.update({
                    "required_reference_count": available,
                    "max_papers": available,
                    "max_papers_explicit": True,
                })
            else:
                editable["conservative_regeneration"] = True
                
            if decision["action"] == "repair_taxonomy":
                from app.schemas.taxonomy_schema import DynamicTaxonomy
                from app.tools.cluster_papers import taxonomy_fingerprint
                
                taxonomy_data = editable.get("dynamic_taxonomy") or state.get("dynamic_taxonomy") or {}
                if taxonomy_data and taxonomy_data.get("themes"):
                    current_fp = taxonomy_fingerprint(DynamicTaxonomy.model_validate(taxonomy_data))
                    last_fp = editable.get("last_taxonomy_fingerprint")
                    if last_fp and current_fp == last_fp:
                        raise ValueError("分类结果已稳定无法继续细分。请尝试'扩大年份'、'加入预印本'或'接受现有论文'。")
                    editable["last_taxonomy_fingerprint"] = current_fp
                
                editable["force_taxonomy_remediation"] = True
                
            editable["research_request"] = research_request
            editable["conversation_history"] = history[-50:]
            editable["quality_recovery_attempts"] = recovery_attempts + 1
            return self._regenerate_and_persist(
                session_id, session["original_query"], state, editable
            )

        # 生成后“引用篇数不足”不一定意味着证据池不足。若现有卡片已经达到
        # 用户要求，优先做确定性保守重写，避免用完全相同的查询重复检索。
        if decision["action"] == "retry_search" and clarification.get("phase") == "post_generation":
            requested = int(clarification.get("requested") or 0)
            evidence_pool = len(editable.get("paper_cards") or [])
            if requested and evidence_pool >= requested and editable:
                editable["conservative_regeneration"] = True
                editable["conversation_history"] = history[-50:]
                editable["quality_recovery_attempts"] = recovery_attempts + 1
                return self._regenerate_and_persist(
                    session_id, session["original_query"], state, editable
                )

        # 改变检索边界时以 editable_research_state 为工作底稿，复用前轮候选、
        # 详情和证据卡片；只有范围本身被重新定义时才需要完整重新规划。
        query_suffix = ""
        editable["quality_recovery_attempts"] = recovery_attempts + 1
        state["quality_recovery_attempts"] = recovery_attempts + 1
        if decision["action"] == "expand_time_range":
            years = int(decision.get("years") or 5)
            end_year = int(research_request.get("end_year") or editable.get("end_year") or 0)
            if not end_year:
                from app.utils.date_utils import current_year

                end_year = current_year()
            previous_start = int(
                research_request.get("start_year") or editable.get("start_year") or end_year
            )
            expanded_start = end_year - years + 1
            research_request.update({
                "start_year": expanded_start,
                "end_year": end_year,
                "year_range_explicit": True,
            })
            editable.update({
                "start_year": expanded_start,
                "end_year": end_year,
                "year_range_explicit": True,
                "incremental_retrieval": True,
                "incremental_search_window": {
                    "start_year": expanded_start,
                    "end_year": previous_start - 1,
                },
            })
            query_suffix = f"用户在质量决策中将时间范围扩大为近{years}年。"
        elif decision["action"] == "include_more_types":
            research_request["include_preprints"] = True
            editable["incremental_retrieval"] = True
            query_suffix = "用户明确允许纳入高相关会议论文和预印本。"
        elif decision["action"] == "broaden_scope":
            research_request["scope_adjustment"] = answer
            relaxed_scope = dict(editable.get("selected_scope") or state.get("selected_scope") or {})
            relaxed_scope["exclude_terms"] = []
            relaxed_scope["label"] = str(relaxed_scope.get("label") or "研究范围") + "（已放宽）"
            editable["selected_scope"] = relaxed_scope
            state["selected_scope"] = relaxed_scope
            editable["incremental_retrieval"] = True
            query_suffix = f"用户在质量决策中放宽研究范围：{answer}"
        else:
            editable["incremental_retrieval"] = True
            query_suffix = "用户要求保持当前约束并继续补充检索。"

        state["research_request"] = research_request
        state["intent_context_role"] = "working_query"
        state["conversation_history"] = history[-50:]
        editable["research_request"] = research_request
        editable["conversation_history"] = history[-50:]
        resumed_query = f"{session['original_query']}\n\n{query_suffix}"
        if decision["action"] in {
            "expand_time_range", "retry_search", "broaden_scope", "include_more_types"
        } and editable:
            return self._continue_retrieval_and_persist(
                session_id,
                session["original_query"],
                state,
                editable,
            )
        return self._run_and_persist(
            session_id,
            resumed_query,
            state,
            original_query=session["original_query"],
        )

    def _continue_retrieval_and_persist(
        self,
        session_id: str,
        original_query: str,
        state: dict[str, Any],
        editable: dict[str, Any],
    ) -> dict[str, Any]:
        """从持久化证据池继续检索，不重新执行语义解析与研究规划。"""
        self.repo.save(
            session_id=session_id,
            status="running",
            original_query=original_query,
            state=state,
        )
        self.db.commit()
        try:
            from app.agent.graph import continue_research_agent

            result = continue_research_agent(
                editable,
                should_cancel=self.should_cancel,
                progress_callback=self.progress_callback,
            )
        except Exception as exc:
            from app.agent.graph import AgentCancelledError

            self.repo.save(
                session_id=session_id,
                status="cancelled" if isinstance(exc, AgentCancelledError) else "failed",
                original_query=original_query,
                state=state,
            )
            self.db.commit()
            raise
        return self._persist_or_pause_result(
            session_id, original_query, state, result
        )

    @staticmethod
    def _refresh_open_alternative_semantics(
        original_query: str,
        history: list[dict[str, Any]],
        semantic_frame: dict[str, Any],
        canonical_topic: str,
        deliverables: list[str],
        llm=None,
    ) -> dict[str, Any] | None:
        """使用 LLM 重新解析旧会话语义；不再用连接词正则猜测方法关系。"""
        if llm is None:
            return None
        semantic_answers = [
            str(item.get("content") or "").strip()
            for item in history
            if item.get("type") == "clarification_answer"
            and str(item.get("content") or "").strip()
        ]
        semantic_text = "\n".join([str(original_query or "").strip(), *semantic_answers])

        from app.agent.research_semantic_parser import parse_research_semantics

        refreshed = parse_research_semantics(
            semantic_text,
            canonical_topic,
            deliverables=deliverables,
            llm=llm,
        ).model_dump(mode="json")
        return refreshed if refreshed.get("evidence_requirements") else None

    @staticmethod
    def _parse_quality_decision(
        answer: str,
        clarification: dict[str, Any],
    ) -> dict[str, Any] | None:
        text = str(answer or "").strip().lower()
        # 先剔除否定式表达（“不要取消”“别停止”），避免被误判为停止指令；
        # “不想做”“不做了”等独立停止短语不受影响。
        text = re.sub(
            r"(?:不要|不想|先别|别|不能|不可|无需|不用)(?:取消任务|取消|停止|结束|终止)",
            "",
            text,
        )
        post_generation = clarification.get("phase") == "post_generation"
        has_count_issue = any(
            issue.get("code") in {
                "minimum_references_not_met", "minimum_cited_references_not_met"
            }
            for issue in clarification.get("blocking_issues") or []
        )
        # 问句原样选项优先于自由文本模式：系统提供的每个选项都必须能被解析，
        # 否则会反复追问，或落到语义相反的分支（保守重写 → 强制重新生成）。
        for option_id in _QUALITY_DECISION_ORDER:
            action, _label, pattern = _QUALITY_DECISION_OPTIONS[option_id]
            if not re.search(pattern, text):
                continue
            if option_id == "stop":
                return {"action": "stop"}
            if option_id == "conservative_rewrite":
                if not post_generation:
                    continue
                return {
                    "action": action,
                    "available": clarification.get("available"),
                }
            if option_id == "accept_available":
                # 无篇数类阻断时"接受当前篇数"就是"照现状生成"，沿用既有语义。
                if not has_count_issue:
                    return {"action": "force_generate"}
                return {
                    "action": action,
                    "available": (
                        _extract_paper_count(text) or clarification.get("available")
                    ),
                }
            if option_id == "expand_time_range":
                return {"action": action, "years": _extract_years(text)}
            return {"action": action}
        if re.search(r"结束|停止|不做了|取消任务|取消|不想做", text):
            return {"action": "stop"}
        # 保守重写必须排在 force_generate 之前：后者的宽口径正则
        # 基于(?:当前|现有).{0,12}(?:…|写) 会吞掉"基于现有证据保守重写"。
        if post_generation and re.search(
            r"保守重写|重新保守生成|保守生成|重新生成|重新写|重写", text
        ):
            return {
                "action": "regenerate_existing",
                "available": clarification.get("available"),
            }
        if re.search(
            r"直接生成|立即生成|就此生成|按(?:当前|现有).{0,8}生成|生成(?:当前|现有).{0,8}(?:正文|草稿|结果)"
            # 宽口径的"直接…"句式：用户常答"直接基于现有证据生成最佳可用草稿"，
            # "直接"与"生成/草稿"之间隔了其他词，逐字模式匹配不上导致反复追问。
            r"|直接.{0,12}(?:生成|草稿|输出|写)"
            r"|基于(?:当前|现有).{0,12}(?:生成|草稿|输出|写)"
            r"|(?:当前|现有).{0,6}最佳.{0,6}(?:草稿|版本|结果)"
            r"|最佳(?:可用)?草稿",
            text,
        ):
            return {"action": "force_generate"}
        # 特异性高的意图优先匹配，避免 "接受/继续生成" 等宽泛模式覆盖具体意图
        if re.search(r"近\s*[四五六七八九十\d]+\s*年|扩大.*年|扩大.*时间|放宽.*时间|更长.*时间", text):
            return {"action": "expand_time_range", "years": _extract_years(text)}
        if re.search(r"预印本|会议论文|更多文献类型|更多类型", text):
            return {"action": "include_more_types"}
        if re.search(r"重新分类|修复分类|排除.*低相关|删除.*兜底|处理.*其他", text):
            return {"action": "repair_taxonomy"}
        if re.search(r"放宽.*(?:主题|范围)|扩大.*(?:主题|范围)|相邻领域|交叉领域", text):
            return {"action": "broaden_scope"}
        if re.search(r"补充检索|补充搜索|继续检索|继续搜索|保持.*条件|不要改变.*范围|再找", text):
            return {"action": "retry_search"}
        if re.search(r"接受|就按|按当前|当前篇数|少于|降低.*篇|继续生成", text):
            if not has_count_issue:
                return {"action": "force_generate"}
            return {
                "action": "accept_available",
                "available": (
                    _extract_paper_count(text) or clarification.get("available")
                ),
            }
        return None

    @staticmethod
    def _quality_clarification(result: dict[str, Any]) -> dict[str, Any] | None:
        gate = result.get("quality_gate") or {}
        if gate.get("passed") is not False:
            return None
        # 只有明确获准发布的 best-effort 草稿才进入 partial 终态；普通失败
        # 草稿仍处于隔离区，必须进入质量决策流程，不能被当作可展示结果。
        if gate.get("draft_released") is True:
            return None
        issues = gate.get("blocking_issues") or []
        phase = str(gate.get("phase") or "post_generation")
        count_issue = next(
            (
                issue for issue in issues
                if issue.get("code") in {
                    "minimum_references_not_met",
                    "minimum_cited_references_not_met",
                }
            ),
            None,
        )
        requested = int((count_issue or {}).get("requested") or 0)
        available = int(
            (count_issue or {}).get("available")
            or (count_issue or {}).get("actual")
            or (result.get("generation_readiness") or {}).get("usable_reference_count")
            or len(result.get("paper_cards") or [])
        )
        evidence_pool_available = len(result.get("paper_cards") or [])
        issue_codes = {str(issue.get("code") or "") for issue in issues}
        if count_issue and phase == "post_generation" and evidence_pool_available >= requested:
            question = (
                f"现有证据池包含 {evidence_pool_available} 篇论文，但正文当前只有效引用了 {available} 篇，"
                f"未达到至少 {requested} 篇的要求。你希望{_quality_option_label('conservative_rewrite')}、"
                f"{_quality_option_label('retry_search')}，还是{_quality_option_label('stop')}？"
            )
        elif count_issue:
            question = (
                f"当前只核验到 {available} 篇可用论文，未达到至少 {requested} 篇的要求。"
                f"你希望{_quality_option_label('accept_available')}、"
                f"{_quality_option_label('expand_time_range')}、"
                f"{_quality_option_label('include_more_types')}、"
                f"{_quality_option_label('broaden_scope')}，"
                f"还是{_quality_option_label('keep_searching')}？"
            )
        elif issue_codes & {"fallback_theme_present", "taxonomy_not_ready"}:
            question = (
                "当前论文数量已经足够，但动态分类仍包含兜底或单篇碎片主题。"
                f"你可以{_quality_option_label('best_effort_draft')}，也可以"
                f"{_quality_option_label('repair_taxonomy')}、"
                f"{_quality_option_label('retry_search')}、"
                f"{_quality_option_label('broaden_scope')}或{_quality_option_label('stop')}。"
            )
        elif phase == "pre_generation":
            question = (
                "当前动态分类或证据结构未达到写作条件。"
                f"你可以{_quality_option_label('best_effort_draft')}，也可以"
                f"{_quality_option_label('broaden_scope')}、"
                f"{_quality_option_label('retry_search')}或{_quality_option_label('stop')}。"
            )
        else:
            question = (
                "生成草稿未通过引用或主张证据检查。"
                f"你希望{_quality_option_label('conservative_rewrite')}、"
                f"{_quality_option_label('retry_search')}，还是{_quality_option_label('stop')}？"
            )
        return {
            "kind": "quality_decision",
            "slot": "quality_decision",
            "phase": phase,
            "question": question,
            "requested": requested,
            "available": available,
            "evidence_pool_available": evidence_pool_available,
            "blocking_issues": issues,
            "recovery_options": gate.get("recovery_options") or [],
            "turn_count": 1,
            "recovery_attempts": 0,
            "resume_from": "generation_readiness" if phase == "pre_generation" else "post_generation_repair",
        }

    @staticmethod
    def _clarification_result(
        session_id: str,
        state: dict[str, Any],
        clarification: dict[str, Any],
        question: str,
    ) -> dict[str, Any]:
        snapshot = state.get("result_snapshot") or {}
        return {
            **snapshot,
            "session_id": session_id,
            "status": "needs_clarification",
            "answer": question,
            "clarification": clarification,
            "quality_failure_answer": snapshot.get("answer"),
            "steps": snapshot.get("steps") or [],
            "references": snapshot.get("references") or [],
            "paper_cards": snapshot.get("paper_cards") or [],
            "clusters": snapshot.get("clusters") or [],
            "errors": snapshot.get("errors") or [],
        }

    def _regenerate_and_persist(
        self,
        session_id: str,
        original_query: str,
        state: dict[str, Any],
        editable: dict[str, Any],
    ) -> dict[str, Any]:
        self.repo.save(
            session_id=session_id,
            status="running",
            original_query=original_query,
            state=state,
        )
        self.db.commit()
        try:
            from app.agent.graph import regenerate_research_agent

            # P0 修复已移除 regenerate_research_agent 的 db 参数（Agent 不应
            # 持有数据库会话），但两处调用点当时未同步更新，导致 TypeError。
            result = regenerate_research_agent(
                editable,
                should_cancel=self.should_cancel,
                progress_callback=self.progress_callback,
            )
        except Exception as exc:
            from app.agent.graph import AgentCancelledError

            self.repo.save(
                session_id=session_id,
                status="cancelled" if isinstance(exc, AgentCancelledError) else "failed",
                original_query=original_query,
                state=state,
            )
            self.db.commit()
            raise
        return self._persist_or_pause_result(
            session_id, original_query, state, result
        )

    def _persist_or_pause_result(
        self,
        session_id: str,
        original_query: str,
        state: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        result["session_id"] = session_id
        history = list(state.get("conversation_history") or [])
        if not history:
            history.append({"role": "user", "content": original_query, "type": "research_request"})
        snapshot = {
            key: value for key, value in result.items()
            if key not in {"research_state", "session_id", "status", "clarification"}
        }
        persisted_state = {
            **state,
            "topic": result.get("topic"),
            "intent": result.get("intent"),
            "selected_paper_ids": [
                card.get("paper_id") for card in (result.get("paper_cards") or [])
            ],
            "editable_research_state": result.get("research_state") or state.get("editable_research_state") or {},
            "result_snapshot": snapshot,
            "revision_history": list(state.get("revision_history") or []),
            "revision_number": int(state.get("revision_number") or 0),
        }
        clarification = self._quality_clarification(result)
        if clarification:
            recovery_attempts = int(
                (result.get("research_state") or {}).get("quality_recovery_attempts")
                or persisted_state.get("quality_recovery_attempts")
                or 0
            )
            clarification["recovery_attempts"] = recovery_attempts
            if recovery_attempts >= 3:
                clarification.update({
                    "question": (
                        "已经连续尝试3次调整，但同一写作门禁仍未通过。为避免无限循环，"
                        "系统不再自动重复检索或放宽范围。你可以回答“直接生成”以输出"
                        "当前最佳可用草稿，或回答“结束任务”。"
                    ),
                    "recovery_options": ["直接生成当前最佳可用草稿", "结束任务"],
                })
            question = clarification["question"]
            history.append({"role": "assistant", "content": question, "type": "quality_decision"})
            persisted_state["conversation_history"] = history[-50:]
            self.repo.save(
                session_id=session_id,
                status="needs_clarification",
                original_query=original_query,
                state=persisted_state,
                clarification=clarification,
            )
            self.db.commit()
            return self._clarification_result(
                session_id, persisted_state, clarification, question
            )

        from app.agent.graph import derive_result_status

        raw_status = str(result.get("status") or derive_result_status(result))
        result_status = "completed" if raw_status == "success" else raw_status
        result["status"] = result_status
        history.append({
            "role": "assistant",
            "content": (
                "研究任务已完成" if result_status == "completed"
                else "研究任务部分完成" if result_status == "partial"
                else "研究任务未能完成"
            ),
            "type": "generated_result",
            "paper_count": len(result.get("paper_cards") or []),
        })
        persisted_state["conversation_history"] = history[-50:]
        persisted_state["result_snapshot"] = {
            key: value for key, value in result.items() if key != "research_state"
        }
        self.repo.save(
            session_id=session_id,
            status=result_status,
            original_query=original_query,
            state=persisted_state,
        )
        self.db.commit()
        return result

    def _run_and_persist(
        self,
        session_id: str,
        query: str,
        state: dict[str, Any],
        original_query: str | None = None,
    ) -> dict[str, Any]:
        self.repo.save(
            session_id=session_id,
            status="running",
            original_query=original_query or query,
            state=state,
        )
        self.db.commit()
        try:
            # Agent 只消费可序列化状态，不持有长生命周期数据库会话。
            # 持久化由本服务在节点运行前后负责。
            runner_kwargs = {"initial_state": state}
            if self.should_cancel is not None:
                runner_kwargs["should_cancel"] = self.should_cancel
            if self.progress_callback is not None:
                runner_kwargs["progress_callback"] = self.progress_callback
            result = self.agent_runner(query, **runner_kwargs)
        except Exception as exc:
            from app.agent.graph import AgentCancelledError

            cancelled = isinstance(exc, AgentCancelledError)
            self.repo.save(
                session_id=session_id,
                status="cancelled" if cancelled else "failed",
                original_query=original_query or query,
                state=state,
            )
            self.db.commit()
            raise

        return self._persist_or_pause_result(
            session_id,
            original_query or query,
            state,
            result,
        )

    def revise(self, request: ResearchRevisionRequest) -> dict[str, Any]:
        """排除指定论文并仅重跑聚类、生成和验证阶段。"""
        session = self.repo.get(request.session_id)
        if not session:
            raise ValueError("研究会话不存在或已过期")
        if session.get("status") not in {"completed", "partial"}:
            raise ValueError("只有已完成的研究会话可以修订")

        state = dict(session.get("state") or {})
        editable = dict(state.get("editable_research_state") or {})
        if not editable:
            raise ValueError("该会话由旧版本生成，没有可编辑研究状态，请重新生成一次")

        cards = list(editable.get("paper_cards") or [])
        known_ids = {
            str(card.get("paper_id") or "")
            for card in cards
            if card.get("paper_id")
        }
        excluded = {str(value) for value in request.excluded_paper_ids if str(value) in known_ids}
        instruction_text = request.instruction or ""
        requested_deliverable = None
        deliverable_patterns = (
            (r"研究背景|research_background", "research_background"),
            (r"研究现状|research_status", "research_status"),
            (r"相关工作|related_work", "related_work"),
            (r"叙述性综述|完整综述|narrative_review", "narrative_review"),
        )
        for pattern, value in deliverable_patterns:
            if re.search(pattern, instruction_text, re.IGNORECASE):
                requested_deliverable = value
                break
        if instruction_text:
            # 支持稳定 ID、完整标题以及“删除第 2、5 篇”等自然语言引用。
            excluded.update(paper_id for paper_id in known_ids if paper_id in instruction_text)
            for card in cards:
                title = str(card.get("title") or "").strip()
                if len(title) >= 4 and title in instruction_text:
                    excluded.add(str(card.get("paper_id")))
            if re.search(r"删除|去除|去掉|移除|排除|不符合", instruction_text):
                ordinal_numbers = {
                    int(value)
                    for value in re.findall(r"(?<!\d)(\d{1,3})(?!\d)", instruction_text)
                }
                chinese_digits = {
                    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
                }
                ordinal_numbers.update(
                    chinese_digits[value]
                    for value in re.findall(r"第([一二三四五六七八九十])篇", instruction_text)
                    if value in chinese_digits
                )
                for number in ordinal_numbers:
                    if 1 <= number <= len(cards):
                        paper_id = str(cards[number - 1].get("paper_id") or "")
                        if paper_id:
                            excluded.add(paper_id)
        if not excluded and not requested_deliverable:
            raise ValueError("没有识别到论文排除或交付物修改，请说明论文序号、标题或希望生成的四类任务")

        def keep(item: dict) -> bool:
            return str(item.get("paper_id") or "") not in excluded

        if excluded:
            editable["paper_details"] = [
                item for item in (editable.get("paper_details") or []) if keep(item)
            ]
            editable["paper_cards"] = [
                item for item in (editable.get("paper_cards") or []) if keep(item)
            ]
        if requested_deliverable:
            editable["core_deliverables"] = [requested_deliverable]
            editable["requested_sections"] = [requested_deliverable]
        # 仅修改交付物类型（未排除论文）时该键可能不存在，不能直接下标访问。
        if not (editable.get("paper_details") or []):
            raise ValueError("不能删除全部论文；请至少保留一篇")

        revision_number = int(state.get("revision_number") or 0) + 1
        instruction = request.instruction or f"排除 {len(excluded)} 篇不符合要求的论文"
        history = list(state.get("conversation_history") or [])
        history.append({"role": "user", "content": instruction, "type": "revision"})
        self.repo.save(
            session_id=request.session_id,
            status="running",
            original_query=session["original_query"],
            state={**state, "conversation_history": history[-50:]},
        )
        self.db.commit()

        try:
            from app.agent.graph import regenerate_research_agent

            result = regenerate_research_agent(
                editable,
                should_cancel=self.should_cancel,
                progress_callback=self.progress_callback,
            )
        except Exception as exc:
            from app.agent.graph import AgentCancelledError

            self.repo.save(
                session_id=request.session_id,
                status="cancelled" if isinstance(exc, AgentCancelledError) else "failed",
                original_query=session["original_query"],
                state=state,
            )
            self.db.commit()
            raise

        # 与主流程 _persist_or_pause_result 对齐：门禁未过且无部分成功草稿时，
        # 进入澄清流程，而不是把未达标结果标记为 completed。
        clarification = self._quality_clarification(result)
        if clarification:
            revision_status = "needs_clarification"
        else:
            revision_gate = result.get("quality_gate") or {}
            revision_status = (
                "partial"
                if revision_gate.get("passed") is False and revision_gate.get("partial_success")
                else "completed"
            )
        result.update(
            {
                "session_id": request.session_id,
                "status": revision_status,
                "revision_number": revision_number,
                "excluded_paper_ids": sorted(excluded),
                "incremental_regeneration": True,
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": (
                    str(clarification.get("question"))
                    if clarification
                    else f"已排除 {len(excluded)} 篇论文并完成增量重生成"
                ),
                "type": "quality_decision" if clarification else "revised_result",
                "revision_number": revision_number,
            }
        )
        revisions = list(state.get("revision_history") or [])
        revisions.append(
            {
                "revision_number": revision_number,
                "instruction": instruction,
                "excluded_paper_ids": sorted(excluded),
                "remaining_paper_count": len(result.get("paper_cards") or []),
            }
        )
        snapshot = {key: value for key, value in result.items() if key != "research_state"}
        persisted = {
            **state,
            "editable_research_state": result.get("research_state") or editable,
            "result_snapshot": snapshot,
            "selected_paper_ids": [
                card.get("paper_id") for card in (result.get("paper_cards") or [])
            ],
            "excluded_paper_ids": sorted(
                set(state.get("excluded_paper_ids") or []) | excluded
            ),
            "conversation_history": history[-50:],
            "revision_history": revisions,
            "revision_number": revision_number,
        }
        self.repo.save(
            session_id=request.session_id,
            status=revision_status,
            original_query=session["original_query"],
            state=persisted,
            clarification=clarification,
        )
        self.db.commit()
        if clarification:
            return self._clarification_result(
                request.session_id, persisted, clarification, clarification["question"]
            )
        return result
