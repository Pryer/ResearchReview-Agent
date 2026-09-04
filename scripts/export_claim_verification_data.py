"""将 Agent 输出中的验证报告导出为 QLoRA 三分类 JSONL 候选数据。

示例：
    python scripts/export_claim_verification_data.py agent_output.json data/claim_verification.jsonl

导出的标签来自当前规则基线，正式训练前应人工抽检或复标，并按论文/主题划分数据集。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.tools.training_data import build_qlora_records


def _reports(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            yield from _reports(item)
        return
    if not isinstance(payload, dict):
        return
    if "claim_verification" in payload and isinstance(payload["claim_verification"], dict):
        yield payload["claim_verification"]
    if "data" in payload:
        yield from _reports(payload["data"])


def export_records(input_path: Path, output_path: Path) -> int:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = []
    for report in _reports(payload):
        records.extend(build_qlora_records(report))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            record["label_source"] = "rule_baseline_requires_review"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Claim-Evidence QLoRA 候选数据")
    parser.add_argument("input", type=Path, help="Agent API 输出 JSON")
    parser.add_argument("output", type=Path, help="输出 JSONL")
    args = parser.parse_args()
    count = export_records(args.input, args.output)
    print(f"exported={count} output={args.output}")


if __name__ == "__main__":
    main()
