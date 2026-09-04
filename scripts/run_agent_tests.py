"""批量运行 Agent 样例测试。

跑 10 个覆盖不同意图的 query，调用 run_research_agent，把每个样例的
intent / 候选论文数 / 是否触发 cnki / 耗时 / 错误 汇总到
data/agent_test_results.txt（UTF-8）。

注意：会真实调用 LLM（.env 配置的 longcat）和真实抓 CNKI（中文样例启动浏览器），
较慢且消耗额度。
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

# 把项目根加入 sys.path，使 `python scripts/run_agent_tests.py` 也能 import app。
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Force UTF-8 stdout so Chinese prints correctly on Windows.
def _force_utf8(stream):
    recon = getattr(stream, "reconfigure", None)
    if callable(recon):
        recon(encoding="utf-8")
        return stream
    return io.TextIOWrapper(stream.buffer, encoding="utf-8")


sys.stdout = _force_utf8(sys.stdout)
sys.stderr = _force_utf8(sys.stderr)

from app.agent.graph import run_research_agent
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
OUT = Path("data/agent_test_results.txt")

# 真实完整需求样例：带具体领域、时间范围、引用数量、产出要求。
# 这种重型需求会触发完整 agent 全流程（规划→检索→排序→修正→详情→综述→引用校验）。
SAMPLES = [
    "帮我调研近五年少样本动作识别相关论文，引用不少于50篇，并生成相关工作",
    "请帮我检索近三年大语言模型推理加速方向的论文，至少30篇，并写一份中文综述",
    "Survey retrieval-augmented generation over the last 5 years, cite at least 40 papers and generate a related-work section in English",
]


def _has_cnki(result) -> bool:
    """结果里是否出现了 cnki 来源的论文（看 references / paper_cards / steps）。"""
    for key in ("references", "paper_cards"):
        for item in result.get(key) or []:
            src = item.get("source") if isinstance(item, dict) else None
            if src == "cnki":
                return True
    # steps 里的 search_node debug 也可能记录 sources 含 cnki
    for step in result.get("steps") or []:
        out = step.get("output_data") or {}
        if isinstance(out, dict):
            blob = str(out)
            if "cnki" in blob:
                return True
    return False


def main() -> None:
    lines: list[str] = []
    lines.append(f"Agent 批量测试  共 {len(SAMPLES)} 个样例")
    lines.append(f"configured sources: {settings.search_sources_list}")
    lines.append("=" * 70)

    summary = []
    for i, query in enumerate(SAMPLES, start=1):
        lines.append(f"\n[{i}/{len(SAMPLES)}] query: {query}")
        t0 = time.perf_counter()
        try:
            result = run_research_agent(query)
            dt = time.perf_counter() - t0
            intent = result.get("intent")
            refs = result.get("references") or []
            cards = result.get("paper_cards") or []
            errors = result.get("errors") or []
            cnki = _has_cnki(result)
            answer = (result.get("answer") or "").strip()
            ok = "成功" if not errors else f"有{len(errors)}个错误"
            line = (f"  intent={intent} | 论文数={len(refs)} | 卡片={len(cards)} | "
                    f"cnki={cnki} | 耗时={dt:.1f}s | {ok}")
            lines.append(line)
            if errors:
                lines.append(f"  errors: {[str(e)[:120] for e in errors]}")
            lines.append(f"  answer前80字: {answer[:80]}")
            summary.append({
                "i": i, "query": query, "intent": intent, "papers": len(refs),
                "cnki": cnki, "ok": not errors, "dt": dt,
            })
        except Exception as exc:
            dt = time.perf_counter() - t0
            lines.append(f"  异常: {type(exc).__name__}: {str(exc)[:200]}  耗时={dt:.1f}s")
            summary.append({"i": i, "query": query, "ok": False, "dt": dt,
                            "error": f"{type(exc).__name__}: {exc}"})

    # 汇总
    lines.append("\n" + "=" * 70)
    lines.append("汇总:")
    succ = sum(1 for s in summary if s.get("ok"))
    cnki_cnt = sum(1 for s in summary if s.get("cnki"))
    lines.append(f"  成功: {succ}/{len(SAMPLES)}  触发cnki: {cnki_cnt}/{len(SAMPLES)}")
    for s in summary:
        lines.append(f"  [{s['i']}] {'OK ' if s.get('ok') else 'FAIL'} "
                     f"intent={s.get('intent')} papers={s.get('papers','-')} "
                     f"cnki={s.get('cnki')} {s.get('dt',0):.0f}s  {s['query'][:30]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果已写入 {OUT}")
    print(f"成功 {succ}/{len(SAMPLES)}，触发 cnki {cnki_cnt}/{len(SAMPLES)}")


if __name__ == "__main__":
    main()
