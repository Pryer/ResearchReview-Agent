"""持续监控指定 Job ID，每 5 秒打印一次进度。"""
import sys
import requests
import time
from datetime import datetime

JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "ad349cfaed8645a4bb6f63be6a570d33"
API_BASE = "http://localhost:8000"

STEP_NAMES = {
    "parse_intent":          "🎯 意图识别",
    "extract_slots":         "📌 槽位提取",
    "topic_disambiguation":  "🔍 主题消歧",
    "search_papers":         "🔎 检索论文",
    "rank_papers":           "📊 排名/Rerank",
    "refine_search":         "🔄 检索优化",
    "fetch_detail":          "📄 获取详情",
    "cluster":               "🗂️  聚类分析",
    "extract_card":          "🃏 证据卡片",
    "generation_readiness":  "✔️  生成准备",
    "generate_deliverables": "✍️  生成正文",
    "verify_claims":         "🔬 主张验证",
    "citation_check":        "📎 引用验证",
    "final_answer":          "📝 组装回复",
    "quality_gate":          "🚦 质量门禁",
}

last_step = None
last_status = None
seen_errors = set()
start = time.time()

print(f"▶  监控 Job {JOB_ID}")
print("=" * 70)

while True:
    try:
        r = requests.get(f"{API_BASE}/api/reviews/jobs/{JOB_ID}", timeout=10)
        d = r.json().get("data", {})
        status = d.get("status", "?")
        elapsed = int(time.time() - start)
        ts = datetime.now().strftime("%H:%M:%S")

        if status != last_status:
            icons = {"pending": "⏳", "queued": "⏳", "running": "🏃",
                     "completed": "✅", "failed": "❌", "needs_clarification": "❓"}
            print(f"\n[{ts}] [{elapsed:3d}s] {icons.get(status, '?')} 状态: {last_status or 'new'} → {status}")
            last_status = status

        steps = d.get("steps") or []
        if steps:
            cur = steps[-1]
            sname = cur.get("step_name", "")
            if sname != last_step:
                label = STEP_NAMES.get(sname, f"⚙️  {sname}")
                print(f"[{ts}] [{elapsed:3d}s] {label} [{cur.get('status','')}]")
                last_step = sname
                out = cur.get("output_data") or {}
                for key, label2 in [
                    ("total_results", "  └─ 检索总数"),
                    ("new_results",   "  └─ 新增"),
                    ("ranked",        "  └─ 排名保留"),
                    ("cards",         "  └─ 证据卡片"),
                    ("clusters",      "  └─ 聚类数"),
                    ("writing_plans", "  └─ 写作计划"),
                    ("unique_cited_paper_count", "  └─ 有效引用"),
                ]:
                    if key in out:
                        extra = ""
                        if key == "unique_cited_paper_count":
                            req = out.get("required_reference_count", 40)
                            pct = int(out[key] / req * 100) if req else 0
                            extra = f"/{req} ({pct}%)"
                        print(f"{label2}: {out[key]}{extra}")
                preview = out.get("answer_preview") or out.get("preview") or ""
                if preview:
                    print(f"  └─ 预览: {str(preview)[:100]}…")
                dur = cur.get("duration_ms")
                if dur:
                    print(f"  └─ 耗时: {dur/1000:.1f}s")

        for e in (d.get("errors") or []):
            if e not in seen_errors:
                print(f"[{ts}] ⚠️  {str(e)[:120]}")
                seen_errors.add(e)

        if status in ("completed", "failed", "needs_clarification", "cancelled"):
            print("\n" + "=" * 70)
            print(f"⏱️  总耗时: {elapsed//60}m{elapsed%60}s")
            if status == "completed":
                res = d.get("result") or {}
                answer = res.get("answer") or ""
                cited  = res.get("unique_cited_paper_count")
                refs   = res.get("references") or []
                print(f"  正文长度  : {len(answer):,} 字符")
                print(f"  有效引用  : {cited} 篇")
                print(f"  参考文献  : {len(refs)} 条")
                qg = res.get("quality_gate") or {}
                if qg.get("partial_success"):
                    print("  质量状态  : ⚠️  部分满足（带警告展示）")
                elif qg.get("passed"):
                    print("  质量状态  : ✅ 通过")
                # 保存
                ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
                import json
                with open(f"result_{JOB_ID[:8]}_{ts2}.json", "w", encoding="utf-8") as f2:
                    json.dump(d, f2, ensure_ascii=False, indent=2)
                print(f"  JSON 已保存: result_{JOB_ID[:8]}_{ts2}.json")
                if answer:
                    with open(f"result_{JOB_ID[:8]}_{ts2}.md", "w", encoding="utf-8") as f2:
                        f2.write(answer)
                    print(f"  正文已保存: result_{JOB_ID[:8]}_{ts2}.md")
            elif status == "failed":
                print(f"  错误: {d.get('error')}")
            elif status == "needs_clarification":
                clar = (d.get("result") or {}).get("clarification") or {}
                print(f"  需要澄清: {clar.get('question', '请看 result.clarification')}")
            break

        time.sleep(5)

    except KeyboardInterrupt:
        print("\n⏸  监控中断，任务仍在运行")
        print(f"  继续查询: GET {API_BASE}/api/reviews/jobs/{JOB_ID}")
        break
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 查询异常: {e}")
        time.sleep(5)
