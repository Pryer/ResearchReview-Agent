"""检查真实分层 RAG 的配置与数据库；不消耗 Provider Token。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.services.rag_runtime import check_rag_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    report = check_rag_runtime(get_settings(), check_database=not args.config_only)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
