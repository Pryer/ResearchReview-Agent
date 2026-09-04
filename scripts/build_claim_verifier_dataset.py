"""从 Agent 输出的 Evidence Card 构建并按论文切分 QLoRA 候选数据集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.tools.training_data import build_controlled_training_records


def _cards(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            yield from _cards(item)
        return
    if not isinstance(payload, dict):
        return
    if isinstance(payload.get("paper_cards"), list):
        yield from payload["paper_cards"]
    if "data" in payload:
        yield from _cards(payload["data"])


def _split_for_paper(paper_id: str) -> str:
    bucket = int(hashlib.sha1(paper_id.encode("utf-8")).hexdigest(), 16) % 10
    if bucket == 8:
        return "dev"
    if bucket == 9:
        return "test"
    return "train"


def build_dataset(input_path: Path, output_dir: Path) -> Dict[str, int]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    unique_cards: Dict[str, Dict[str, Any]] = {}
    for card in _cards(payload):
        paper_id = str(card.get("paper_id") or "")
        if paper_id:
            unique_cards[paper_id] = card
    cards_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for paper_id, card in unique_cards.items():
        cards_by_split[_split_for_paper(paper_id)].append(card)
    # 先按论文切分，再在各 split 内构造跨论文负样本，避免 evidence_paper 泄漏。
    grouped = {
        split: build_controlled_training_records(split_cards)
        for split, split_cards in cards_by_split.items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_records in grouped.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in split_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {split: len(records) for split, records in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 Claim–Evidence QLoRA 三分类候选数据")
    parser.add_argument("input", type=Path, help="含 paper_cards 的 Agent API 输出 JSON")
    parser.add_argument("output_dir", type=Path, help="train/dev/test JSONL 输出目录")
    args = parser.parse_args()
    counts = build_dataset(args.input, args.output_dir)
    print(" ".join(f"{split}={count}" for split, count in counts.items()))


if __name__ == "__main__":
    main()
