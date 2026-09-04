"""端到端测试：调研近三年课堂行为分析论文，生成研究背景+研究现状，引用不少于40篇。

真实调用 LLM（.env 配置的 longcat）和真实检索 arxiv/semantic_scholar/openalex/
crossref/cnki（cnki 会启动 Selenium 浏览器，较慢）。运行结果写入
data/classroom_behavior_e2e_result.txt 和 .json（含完整 answer/references/steps 摘要）。
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

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

from app.agent.graph import run_research_agent  # noqa: E402
from app.core.config import get_settings  # noqa: E402

settings = get_settings()

QUERY = "调研近三年课堂行为分析论文，并生成研究背景和研究现状，不少于40篇引用论文"

OUT_TXT = Path("data/classroom_behavior_e2e_result.txt")
OUT_JSON = Path("data/classroom_behavior_e2e_result.json")


def _progress(step: str, current: int, total: int) -> None:
    print(f"[progress] {current}/{total} {step}", flush=True)


def main() -> None:
    print(f"query: {QUERY}")
    print(f"configured sources: {settings.search_sources_list}")
    print(f"llm provider={settings.llm_provider} model={settings.llm_model}")

    t0 = time.perf_counter()
    result = run_research_agent(QUERY, progress_callback=_progress)
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
    lines.append(f"query: {QUERY}")
    lines.append(f"耗时: {dt:.1f}s")
    lines.append(f"intent: {intent}")
    lines.append(f"core_deliverables: {core_deliverables}")
    lines.append(f"generation_blocked: {generation_blocked}")
    lines.append(f"参考文献数: {len(references)}  (要求 >= 40)")
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

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"结果已写入 {OUT_TXT} 和 {OUT_JSON}")
    print(f"intent={intent} core_deliverables={core_deliverables}")
    print(f"参考文献数={len(references)} (要求>=40)  errors={len(errors)}  耗时={dt:.1f}s")
    print(f"generation_blocked={generation_blocked}")


if __name__ == "__main__":
    main()
