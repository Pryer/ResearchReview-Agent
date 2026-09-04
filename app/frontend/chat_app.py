"""ChatGPT 风格的多轮对话前端。

支持多会话、连续问答、研究范围澄清、后台进度和任务取消。

运行方式：
    streamlit run app/frontend/chat_app.py
"""

from __future__ import annotations

import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.frontend.query_utils import build_agent_request_payload

st.set_page_config(
    page_title="ResearchReview-Agent Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = get_settings().frontend_api_base_url.rstrip("/")

SUPPORTED_TASKS = (
    ("研究背景", "梳理研究问题的现实需求、学术价值、技术发展和主要挑战"),
    ("研究现状", "按动态主题分类综合近年研究方法、证据、进展与不足"),
    ("论文相关工作", "围绕你的论文问题与方法，比较前置、同类及竞争工作"),
    ("叙述性综述初稿", "形成包含分类体系、研究脉络、比较和研究空白的综述草稿"),
)

EXAMPLE_QUERIES = (
    "调研近三年课堂行为分析论文，并生成研究背景和研究现状，引用论文不少于40篇",
    "调研近五年检索增强生成中减少幻觉的方法，并生成研究现状",
    "我的论文研究少样本时序动作定位，采用原型对齐与时序建模方法，请生成论文相关工作",
    "围绕多模态学习分析生成一份叙述性综述初稿，重点比较数据模态、方法和评价指标",
)


def _api_headers() -> dict[str, str]:
    key = get_settings().app_api_key.strip()
    return {"X-API-Key": key} if key else {}


def init_session_state() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    # API 地址是部署配置，不接受浏览器会话输入。
    st.session_state.api_url = API_BASE
    if "active_job" not in st.session_state:
        st.session_state.active_job = None


def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> None:
    if session_id not in st.session_state.conversations:
        st.session_state.conversations[session_id] = {
            "messages": [],
            "created_at": datetime.now(),
            "title": content[:50] + ("..." if len(content) > 50 else ""),
        }
    st.session_state.conversations[session_id]["messages"].append({
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "timestamp": datetime.now(),
    })


def get_messages(session_id: str) -> list[dict]:
    conversation = st.session_state.conversations.get(session_id) or {}
    return conversation.get("messages") or []


def create_new_session() -> str:
    session_id = uuid.uuid4().hex
    st.session_state.conversations[session_id] = {
        "messages": [],
        "created_at": datetime.now(),
        "title": "新对话",
    }
    st.session_state.current_session_id = session_id
    return session_id


def _job_request(
    method: str,
    path: str,
    payload: dict | None = None,
) -> dict | None:
    try:
        response = requests.request(
            method,
            f"{API_BASE}/reviews{path}",
            json=payload,
            headers=_api_headers(),
            timeout=15,
        )
        if response.status_code == 200:
            return response.json().get("data") or {}
        st.error(f"API 返回错误：{response.status_code} — {response.text}")
    except requests.ConnectionError:
        st.error(f"无法连接 API：{API_BASE}")
    except Exception as exc:
        st.error(f"请求失败：{exc}")
    return None


def submit_agent_job(
    user_query: str,
    session_id: str,
    clarification_answer: str | None = None,
) -> dict | None:
    payload = build_agent_request_payload(
        user_query,
        session_id,
        clarification_answer=clarification_answer,
    )
    return _job_request("POST", "/jobs", payload)


def _is_clarification_answer(messages: list[dict]) -> bool:
    if not messages:
        return False
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        metadata = message.get("metadata") or {}
        return metadata.get("status") == "needs_clarification"
    return False


def _result_metadata(result: dict) -> dict:
    return {
        "status": result.get("status"),
        "clarification": result.get("clarification"),
        "paper_cards": result.get("paper_cards") or [],
        "claim_verification": result.get("claim_verification") or {},
        "steps": result.get("steps") or [],
        "references": result.get("references") or [],
        "intent": result.get("intent"),
        "topic": result.get("topic"),
        "quality_gate": result.get("quality_gate") or {},
        "source_diagnostics": (result.get("search_report") or {}).get("source_diagnostics") or result.get("source_diagnostics") or [],
        "session_id": result.get("session_id"),
    }


def _finish_job(job: dict) -> None:
    active = st.session_state.active_job or {}
    session_id = active.get("session_id")
    if not session_id:
        st.session_state.active_job = None
        return

    status = job.get("status")
    if status in {"completed", "partial", "blocked", "needs_clarification"}:
        result = job.get("result") or {}
        answer = result.get("answer") or "✅ 任务已完成。"
        add_message(session_id, "assistant", answer, _result_metadata(result))
    elif status == "cancelled":
        add_message(session_id, "assistant", "⏹ 任务已取消，后续步骤未再执行。")
    else:
        add_message(
            session_id,
            "assistant",
            f"❌ 任务失败：{job.get('error') or '未知错误'}",
        )
    st.session_state.active_job = None


@st.fragment(run_every="2s")
def render_active_job() -> None:
    active = st.session_state.active_job
    if not active:
        return
    job = _job_request("GET", f"/jobs/{active['job_id']}")
    if not job:
        return

    status = job.get("status", "queued")
    current = int(job.get("progress_current") or 0)
    total = max(int(job.get("progress_total") or 14), 1)
    step = job.get("current_step") or "等待执行"
    st.info(f"后台任务：{status} · {step}")
    st.progress(min(current / total, 1.0), text=f"进度 {current}/{total}")

    if status in {"queued", "running", "cancel_requested"}:
        if st.button(
            "⏹ 取消任务" if status != "cancel_requested" else "正在取消…",
            disabled=status == "cancel_requested",
            key=f"cancel_{active['job_id']}",
        ):
            _job_request("POST", f"/jobs/{active['job_id']}/cancel")
            st.rerun()
        return

    _finish_job(job)
    st.rerun()


def render_user_message(content: str) -> None:
    with st.chat_message("user", avatar="👤"):
        st.markdown(content)


def render_assistant_message(content: str, metadata: dict | None = None) -> None:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(content)
        if metadata:
            render_message_metadata(metadata)


_SOURCE_OUTCOME_LABELS = {
    "success_with_results": "有结果",
    "success_empty": "零结果",
    "query_not_adapted": "查询不适配",
    "rate_limited": "限流",
    "timeout": "超时",
    "authentication_failed": "认证失败",
    "api_failed": "来源/API失败",
    "human_action_required": "需要人工处理",
    "skipped": "已跳过",
}


def _safe_source_message(item: dict) -> str:
    """只展示来源、分类和服务端错误摘要，不泄漏内部路径、ID或提示词。"""
    message = str(item.get("message") or "").replace("\n", " ").strip()
    message = re.sub(r"https?://\S+|[A-Za-z]:\\[^ ]+|/[^ ]+", "", message)
    return message[:180]


def render_source_diagnostics(diagnostics: list) -> None:
    if not diagnostics:
        return
    with st.expander("来源诊断", expanded=False):
        language_gap = []
        for raw in diagnostics[:30]:
            item = raw if isinstance(raw, dict) else raw.model_dump(mode="json")
            source = str(item.get("source") or "未知来源")
            outcome = str(item.get("outcome") or item.get("status") or "unknown")
            label = _SOURCE_OUTCOME_LABELS.get(outcome, "来源状态")
            count = int(item.get("returned_count") or 0)
            st.caption(f"{source}：{label}（{count} 篇）")
            if outcome == "query_not_adapted":
                language_gap.append(source)
            detail = _safe_source_message(item)
            if detail:
                st.caption(detail)
        if language_gap:
            st.info(
                "语言覆盖缺口：" + "、".join(sorted(set(language_gap)))
                + " 不支持当前检索语言；建议提供对应语言关键词，或启用匹配语言的数据源。"
            )


def render_message_metadata(metadata: dict) -> None:
    cards = metadata.get("paper_cards") or []
    verification = metadata.get("claim_verification") or {}
    steps = metadata.get("steps") or []
    references = metadata.get("references") or []
    quality_gate = metadata.get("quality_gate") or {}

    if metadata.get("status") == "needs_clarification":
        st.caption("请直接在下方输入框中用自然语言回答这个问题。")
    render_source_diagnostics(metadata.get("source_diagnostics") or [])
    if quality_gate and not quality_gate.get("passed", True):
        with st.expander("⚠️ 质量门禁"):
            for issue in quality_gate.get("blocking_issues") or []:
                st.markdown(f"- {issue.get('message') or issue}")
    if cards:
        with st.expander(f"📄 论文与证据卡片（{len(cards)}）"):
            render_papers_compact(cards)
    if verification:
        with st.expander("🔎 引用验证"):
            render_verification_compact(verification)
    if steps:
        with st.expander(f"⚙️ 执行步骤（{len(steps)}）"):
            render_steps_compact(steps)
    if references:
        with st.expander(f"📚 参考文献（{len(references)}）"):
            render_references_compact(references)


def render_papers_compact(papers: list) -> None:
    for index, paper in enumerate(papers[:40], 1):
        title = paper.get("title") or "未命名论文"
        year = paper.get("year") or "—"
        source = paper.get("source") or "—"
        st.markdown(f"**{index}. {title}** · {year} · {source}")
        relation = paper.get("relation_to_topic") or paper.get("relevance_reason")
        if relation:
            st.caption(str(relation))


def render_verification_compact(verification: dict) -> None:
    supported = verification.get("supported", 0)
    partial = verification.get("partially_supported", 0)
    unsupported = verification.get("unsupported", 0)
    col1, col2, col3 = st.columns(3)
    col1.metric("完全支持", supported)
    col2.metric("部分支持", partial)
    col3.metric("不支持", unsupported)
    support_rate = float(verification.get("support_rate") or 0.0)
    st.progress(min(max(support_rate, 0.0), 1.0), text=f"支持率：{support_rate:.1%}")


def render_steps_compact(steps: list) -> None:
    for step in steps:
        status = step.get("status") or "—"
        icon = "✅" if status == "success" else "⏭️" if status == "skipped" else "⚠️"
        st.caption(f"{icon} {step.get('step_name') or 'unknown'} · {status}")


def render_references_compact(references: list) -> None:
    for index, reference in enumerate(references, 1):
        st.caption(f"[{index}] {reference}")


def render_capability_guide(*, show_examples: bool = True) -> None:
    """展示当前产品真实支持的交付物边界。"""
    st.markdown("#### 当前可生成的任务")
    for name, description in SUPPORTED_TASKS:
        st.markdown(f"- **{name}**：{description}")
    st.caption(
        "系统只生成以上四类学术写作交付物。实验设计、代码实现、完整论文、"
        "数据分析等请求会在检索前说明暂不支持。"
    )
    if show_examples:
        st.markdown("#### 使用示例")
        st.caption("可复制任一示例到下方输入框，再按你的主题和约束修改。")
        for query in EXAMPLE_QUERIES:
            st.code(query, language=None)


def render_sidebar() -> None:
    with st.sidebar:
        st.title("💬 对话")
        if st.button("＋ 新建对话", use_container_width=True, type="primary"):
            create_new_session()
            st.rerun()

        conversations = sorted(
            st.session_state.conversations.items(),
            key=lambda item: item[1].get("created_at", datetime.min),
            reverse=True,
        )
        for session_id, conversation in conversations:
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(
                    conversation.get("title") or "新对话",
                    key=f"select_{session_id}",
                    use_container_width=True,
                ):
                    st.session_state.current_session_id = session_id
                    st.rerun()
            with col2:
                if st.button("×", key=f"delete_{session_id}"):
                    del st.session_state.conversations[session_id]
                    if st.session_state.current_session_id == session_id:
                        st.session_state.current_session_id = None
                    st.rerun()

        st.markdown("---")
        with st.expander("📌 可生成的任务范围", expanded=True):
            render_capability_guide(show_examples=False)

        st.markdown("---")
        st.subheader("⚙️ 设置")
        st.caption(f"API：{API_BASE}")
        st.caption("ResearchReview-Agent")
        st.caption("可像 ChatGPT 一样连续对话")


def render_chat_interface() -> None:
    if not st.session_state.current_session_id:
        create_new_session()
    session_id = st.session_state.current_session_id

    st.title("📚 ResearchReview-Agent Chat")
    st.caption("多轮研究对话 · 可澄清范围 · 可取消后台任务")

    messages = get_messages(session_id)
    if not messages:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("👋 你好！请告诉我研究主题、时间范围、期望篇数和需要生成的内容。")
            render_capability_guide(show_examples=True)
    else:
        for message in messages:
            if message.get("role") == "user":
                render_user_message(message.get("content") or "")
            else:
                render_assistant_message(
                    message.get("content") or "",
                    message.get("metadata") or {},
                )

    active = st.session_state.active_job
    if active and active.get("session_id") == session_id:
        render_active_job()

    user_input = st.chat_input(
        "输入你的问题或回答 Agent 的澄清问题…",
        disabled=bool(st.session_state.active_job),
    )
    if not user_input:
        return

    clarification_answer = user_input if _is_clarification_answer(messages) else None
    add_message(session_id, "user", user_input)
    if len(messages) == 0:
        st.session_state.conversations[session_id]["title"] = (
            user_input[:50] + ("..." if len(user_input) > 50 else "")
        )
    job = submit_agent_job(user_input, session_id, clarification_answer)
    if job:
        st.session_state.active_job = {
            "job_id": job.get("job_id"),
            "session_id": session_id,
        }
    else:
        add_message(session_id, "assistant", "❌ 提交任务失败，请检查 API 服务。")
    st.rerun()


def main() -> None:
    init_session_state()
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
