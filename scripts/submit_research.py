"""提交并监控研究任务"""
import requests
import json
import time
from datetime import datetime

API_BASE = "http://localhost:8000"
API_KEY = ""  # 如果有 API Key 从 .env 读取

def get_api_key():
    try:
        with open(".env", encoding="utf-8") as f:
            for line in f:
                if line.startswith("APP_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except:
        pass
    return ""

def submit_task():
    """提交研究任务 - 使用 POST /api/reviews/jobs"""
    api_key = get_api_key()
    headers = {"X-API-Key": api_key} if api_key else {}

    # AgentRequest 格式：
    #   user_query: 用户自然语言请求（首次提交不带 clarification_answer）
    #   state: 可选初始状态（写作所需的本文工作信息等）
    # 注意：clarification_answer 只用于已有 session_id 的第二轮交互，
    #       首次提交时将其写入 user_query 末尾或放在 state 中。
    payload = {
        "user_query": (
            "调研近三年课堂行为分析论文，并生成研究背景和研究现状，引用论文不少于40篇。"
            "研究方向：先基于人工智能技术进行老师或学生行为的自动识别和自动行为编码，"
            "然后基于教育学来进行分析。"
        ),
        "state": {
            "our_work_description": (
                "先基于人工智能技术进行老师或学生行为的自动识别和自动行为编码，"
                "然后基于教育学来进行分析"
            ),
            "max_papers": 50,
            "required_reference_count": 40,
            "language": "zh",
        }
    }

    print("=" * 80)
    print("📤 提交研究任务")
    print("=" * 80)
    print(f"查询: {payload['user_query'][:60]}...")
    print(f"目标引用: 40 篇")
    print("=" * 80)

    try:
        response = requests.post(
            f"{API_BASE}/api/reviews/jobs",
            json=payload,
            headers=headers,
            timeout=10
        )
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

    if response.status_code == 200:
        result = response.json()
        if result.get("code") == 0:
            job_id = result["data"].get("job_id") or result["data"].get("id")
            print(f"\n✅ 任务提交成功")
            print(f"📋 Job ID: {job_id}")
            print(f"⏰ 提交时间: {datetime.now().strftime('%H:%M:%S')}")
            return job_id
        else:
            print(f"❌ 提交失败: {result.get('message')}")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return None
    else:
        print(f"❌ HTTP {response.status_code}")
        print(response.text[:500])
        return None


def get_job_status(job_id, headers):
    try:
        response = requests.get(f"{API_BASE}/api/reviews/jobs/{job_id}", headers=headers, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                return result["data"]
    except Exception as e:
        print(f"  查询异常: {e}")
    return None


# 步骤名称中文映射
STEP_NAMES = {
    "parse_intent":         "🎯 意图识别",
    "extract_slots":        "📌 槽位提取",
    "topic_disambiguation": "🔍 主题消歧",
    "search_papers":        "🔎 检索论文",
    "rank_papers":          "📊 论文排名",
    "refine_search":        "🔄 检索优化",
    "search_refined_queries":"🔄 补充检索",
    "fetch_paper_details":  "📄 获取详情",
    "cluster_papers":       "🗂️  论文聚类",
    "extract_paper_cards":  "🃏 生成证据卡片",
    "generation_readiness": "✔️  生成准备检查",
    "generate_deliverables":"✍️  生成正文",
    "generate_review":      "✍️  生成综述",
    "quality_gate":         "🚦 质量门禁",
    "rank_papers_llm":      "🤖 LLM 重排序",
    "citation_validation":  "📎 引用验证",
    "claim_verification":   "🔬 主张验证",
    "assemble_answer":      "📝 组装回复",
}


def fmt_step(name):
    return STEP_NAMES.get(name, f"⚙️  {name}")


def monitor_job(job_id, check_interval=4):
    api_key = get_api_key()
    headers = {"X-API-Key": api_key} if api_key else {}

    print("\n" + "=" * 80)
    print("🔍 开始监控任务执行（Ctrl+C 可停止监控，任务仍在后台运行）")
    print("=" * 80)

    last_step_name = None
    last_status = None
    start_time = time.time()
    seen_errors = set()
    last_paper_count = 0

    while True:
        try:
            job_data = get_job_status(job_id, headers)
            elapsed = int(time.time() - start_time)
            ts = datetime.now().strftime("%H:%M:%S")

            if not job_data:
                print(f"[{ts}] [{elapsed:3d}s] ⚠️  无法获取状态，等待...")
                time.sleep(check_interval)
                continue

            status = job_data.get("status", "unknown")

            # 状态变化
            if status != last_status:
                status_icon = {
                    "pending": "⏳", "running": "🏃", "completed": "✅",
                    "failed": "❌", "cancelled": "🚫"
                }.get(status, "❓")
                print(f"\n[{ts}] [{elapsed:3d}s] {status_icon} 状态变化: {last_status or 'new'} → {status}")
                last_status = status

            # ── 步骤信息 ────────────────────────────────────────────────
            steps = job_data.get("steps", [])
            if steps:
                cur = steps[-1]
                step_name = cur.get("step_name", "")
                step_status = cur.get("status", "")

                if step_name != last_step_name:
                    print(f"[{ts}] [{elapsed:3d}s] {fmt_step(step_name)} [{step_status}]")
                    last_step_name = step_name

                    out = cur.get("output_data") or {}

                    # 检索数量
                    total = out.get("total_results") or out.get("total")
                    new_res = out.get("new_results")
                    if total is not None:
                        print(f"    └─ 检索结果: {total} 篇{f' (新增 {new_res})' if new_res else ''}")

                    # 来源分布
                    sources = out.get("sources") or out.get("source_counts")
                    if isinstance(sources, dict) and sources:
                        parts = ", ".join(f"{k}:{v}" for k, v in sources.items())
                        print(f"    └─ 来源分布: {parts}")

                    # 排名
                    ranked = out.get("ranked_papers") or out.get("ranked_count")
                    if ranked is not None:
                        print(f"    └─ 排名论文: {ranked} 篇")

                    # 聚类
                    clusters = out.get("clusters")
                    if clusters is not None:
                        print(f"    └─ 聚类数量: {clusters} 个")

                    # 证据卡片
                    cards = out.get("paper_cards") or out.get("cards")
                    if cards is not None:
                        print(f"    └─ 证据卡片: {cards} 个")

                    # 引用数
                    cited = out.get("unique_cited_paper_count")
                    req = out.get("required_reference_count") or 40
                    if cited is not None:
                        bar = "█" * min(20, int(cited / req * 20))
                        print(f"    └─ 引用情况: {cited}/{req} 篇 [{bar:<20}]")

                    # 质量门禁
                    if step_name == "quality_gate":
                        qg = out.get("quality_gate") or {}
                        passed = qg.get("passed")
                        partial = qg.get("partial_success")
                        if passed is True:
                            print(f"    └─ 质量门禁: ✅ 通过")
                        elif partial:
                            print(f"    └─ 质量门禁: ⚠️  部分满足（草稿已生成，带警告展示）")
                        elif passed is False:
                            print(f"    └─ 质量门禁: ❌ 未通过")
                            for issue in (qg.get("blocking_issues") or []):
                                print(f"       - {issue.get('message', '')}")

                    # 正文预览
                    preview = out.get("answer_preview") or out.get("preview")
                    if preview:
                        print(f"    └─ 内容预览: {str(preview)[:120]}...")

                    # 耗时
                    dur_ms = cur.get("duration_ms")
                    if dur_ms:
                        print(f"    └─ 耗时: {dur_ms / 1000:.1f}s")

            # ── 进度快照 ────────────────────────────────────────────────
            # 每 20s 输出一次简要进度
            if elapsed % 20 == 0 and elapsed > 0 and status == "running":
                result_snap = job_data.get("result") or {}
                papers_now = len(result_snap.get("ranked_papers") or [])
                if papers_now != last_paper_count:
                    print(f"[{ts}] [{elapsed:3d}s] 📈 进度快照: 已排名 {papers_now} 篇")
                    last_paper_count = papers_now

            # ── 错误信息 ────────────────────────────────────────────────
            for err in job_data.get("errors", []):
                if err not in seen_errors:
                    print(f"[{ts}] ⚠️  {str(err)[:120]}")
                    seen_errors.add(err)

            # ── 终止 ─────────────────────────────────────────────────────
            if status in ("completed", "failed", "cancelled"):
                print(f"\n{'=' * 80}")
                elapsed_min = elapsed // 60
                elapsed_sec = elapsed % 60
                print(f"⏱️  总耗时: {elapsed_min}分{elapsed_sec}秒")

                if status == "completed":
                    result = job_data.get("result") or {}
                    answer = result.get("answer") or ""
                    papers = result.get("ranked_papers") or []
                    cards = result.get("paper_cards") or []
                    cited = result.get("unique_cited_paper_count")
                    refs = result.get("references") or []
                    qg = result.get("quality_gate") or {}

                    print(f"\n📊 最终结果摘要:")
                    print(f"  正文长度  : {len(answer):,} 字符")
                    print(f"  检索论文  : {len(papers)} 篇")
                    print(f"  证据卡片  : {len(cards)} 个")
                    print(f"  有效引用  : {cited if cited is not None else '未知'} 篇")
                    print(f"  参考文献  : {len(refs)} 条")
                    if qg.get("passed") is False and qg.get("partial_success"):
                        print(f"  质量状态  : ⚠️  部分满足（草稿已展示）")
                    elif qg.get("passed"):
                        print(f"  质量状态  : ✅ 通过")

                    save_result(job_id, job_data, elapsed)

                elif status == "failed":
                    print(f"\n❌ 任务失败")
                    errors = job_data.get("errors") or []
                    print(f"\n最后错误（共 {len(errors)} 条）:")
                    for err in errors[-5:]:
                        print(f"  - {err}")
                break

            time.sleep(check_interval)

        except KeyboardInterrupt:
            print(f"\n\n⏸️  监控中断，任务仍在后台运行")
            print(f"  Job ID: {job_id}")
            print(f"  继续查询: GET {API_BASE}/api/reviews/jobs/{job_id}")
            break
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 监控异常: {e}")
            time.sleep(check_interval)


def save_result(job_id, job_data, elapsed_sec):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"result_{job_id[:8]}_{ts}"

    # 保存完整 JSON
    json_file = f"{prefix}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(job_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 完整 JSON: {json_file}")

    # 保存正文 markdown
    result = job_data.get("result") or {}
    answer = result.get("answer") or ""
    if answer:
        md_file = f"{prefix}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# 课堂行为分析 - 研究背景与现状\n\n")
            f.write(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"> 耗时: {elapsed_sec // 60}分{elapsed_sec % 60}秒  \n")
            f.write(f"> Job ID: {job_id}\n\n")
            f.write("---\n\n")
            f.write(answer)
            refs = result.get("references") or []
            if refs:
                f.write("\n\n## 参考文献\n\n")
                for i, ref in enumerate(refs, 1):
                    f.write(f"{i}. {ref}\n")
        print(f"💾 正文 Markdown: {md_file}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         ResearchReview-Agent  全程监控                           ║
║         主题：课堂行为分析 · 研究背景与研究现状 · 引用≥40篇     ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    job_id = submit_task()
    if job_id:
        monitor_job(job_id, check_interval=4)
    else:
        print("\n任务提交失败，请检查后端服务是否正常运行")
        print(f"健康检查: GET {API_BASE}/health")
