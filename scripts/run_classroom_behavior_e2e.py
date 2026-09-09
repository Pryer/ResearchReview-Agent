"""真实课堂行为研究端到端验收；默认走会话服务及其自动恢复控制器。"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _force_utf8(stream):
    recon = getattr(stream, "reconfigure", None)
    if callable(recon):
        recon(encoding="utf-8")
        return stream
    return io.TextIOWrapper(stream.buffer, encoding="utf-8")


sys.stdout = _force_utf8(sys.stdout)
sys.stderr = _force_utf8(sys.stderr)

from app.core.config import get_settings  # noqa: E402

settings = get_settings()

QUERY = (
    "调研近三年课堂行为分析论文，侧重教育技术视角下的行为编码与教学互动分析，"
    "并生成研究背景和研究现状，引用论文不少于40篇"
)


def _progress(step: str, current: int, total: int) -> None:
    print(f"[progress] {current}/{total} {step}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=QUERY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="独立输出目录；默认按时间写入 data/e2e_runs/，不会覆盖旧快照",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="只运行 Agent 图；默认运行带持久化和自动恢复的会话服务",
    )
    return parser.parse_args()


def _run_conversation(query: str, output_dir: Path) -> dict:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database.models import Base
    from app.schemas.agent_schema import AgentRequest
    from app.services.research_conversation_service import ResearchConversationService

    database_path = (output_dir / "session.db").resolve()
    engine = create_engine(
        f"sqlite:///{database_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with session_factory() as db:
        service = ResearchConversationService(db, progress_callback=_progress)
        return service.handle(AgentRequest(
            session_id=f"classroom-e2e-{uuid4().hex}",
            user_query=query,
        ))


def main() -> None:
    args = _parse_args()
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or Path("data/e2e_runs") / run_stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    out_txt = output_dir / "result.txt"
    out_json = output_dir / "result.json"

    print(f"query: {args.query}")
    print(f"configured sources: {settings.search_sources_list}")
    print(f"llm provider={settings.llm_provider} model={settings.llm_model}")
    print(f"mode={'graph' if args.graph_only else 'conversation'}")

    t0 = time.perf_counter()
    if args.graph_only:
        from app.agent.graph import run_research_agent

        result = run_research_agent(args.query, progress_callback=_progress)
    else:
        result = _run_conversation(args.query, output_dir)
    dt = time.perf_counter() - t0

    intent = result.get("intent")
    core_deliverables = result.get("core_deliverables") or []
    references = result.get("references") or []
    paper_cards = result.get("paper_cards") or []
    errors = result.get("errors") or []
    steps = result.get("steps") or []
    citation_validation = result.get("citation_validation") or {}
    claim_verification = result.get("claim_verification") or {}
    generation_blocked = result.get("generation_blocked")
    answer = result.get("answer") or ""

    failed_steps = [s for s in steps if s.get("status") not in ("success", "skipped")]

    lines = []
    lines.append("=" * 70)
    lines.append(f"query: {args.query}")
    lines.append(f"mode: {'graph' if args.graph_only else 'conversation'}")
    lines.append(f"耗时: {dt:.1f}s")
    lines.append(f"intent: {intent}")
    lines.append(f"core_deliverables: {core_deliverables}")
    lines.append(f"generation_blocked: {generation_blocked}")
    required = int(
        result.get("required_reference_count")
        or (result.get("research_state") or {}).get("required_reference_count")
        or 0
    )
    lines.append(f"参考文献数: {len(references)}  (要求 >= {required})")
    lines.append(f"paper_cards 数: {len(paper_cards)}")
    lines.append(f"errors 数: {len(errors)}")
    lines.append(f"失败/异常步骤数: {len(failed_steps)}")
    lines.append(f"citation_validation: {json.dumps(citation_validation, ensure_ascii=False)[:500]}")
    lines.append(f"claim_verification 摘要: {json.dumps(claim_verification, ensure_ascii=False)[:500]}")
    lines.append("")
    lines.append("-- steps 明细 --")
    for s in steps:
        lines.append(
            f"  [{s.get('status')}] {s.get('step_name')} "
            f"dur={s.get('duration_ms')}ms err={s.get('error')}"
        )
    lines.append("")
    if errors:
        lines.append("-- errors --")
        for e in errors:
            lines.append(f"  {str(e)[:300]}")
        lines.append("")
    lines.append("-- answer 全文 --")
    lines.append(answer)
    lines.append("")
    lines.append("-- references 列表 --")
    for i, ref in enumerate(references, start=1):
        lines.append(f"  [{i}] {ref}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"结果已写入 {out_txt} 和 {out_json}")
    print(f"intent={intent} core_deliverables={core_deliverables}")
    print(f"参考文献数={len(references)} (要求>={required})  errors={len(errors)}  耗时={dt:.1f}s")
    print(f"generation_blocked={generation_blocked}")


if __name__ == "__main__":
    main()
