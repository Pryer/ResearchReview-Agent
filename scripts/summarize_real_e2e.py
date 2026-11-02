"""根据人工填写的 CSV 更新指标，不修改原始运行记录或自动签署验收。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.e2e_reporting import refresh_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    output = Path(args.run_dir).resolve()
    if not output.is_relative_to((ROOT / "data").resolve()):
        parser.error("run directory must be under repository data")
    summary = refresh_summary(output)
    print("Summary updated; manual review errors:", len(summary["manual_review_errors"]))
    return 1 if summary["manual_review_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
