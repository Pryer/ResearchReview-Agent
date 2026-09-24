"""显式修改空闲研究会话 token 上限，不重置消耗或自动执行研究。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--expected-limit", type=int, required=True)
    parser.add_argument("--token-limit", type=int, required=True)
    args = parser.parse_args()
    from app.database.db import SessionLocal
    from app.database.runtime_repository import ResearchRuntimeRepository, RuntimeConflict
    try:
        with SessionLocal() as db:
            ledger = ResearchRuntimeRepository(db, args.session_id).update_token_limit(
                expected_limit=args.expected_limit, new_limit=args.token_limit,
            )
    except (ValueError, RuntimeConflict) as exc:
        parser.exit(1, f"Budget update rejected: {exc}\n")
    print(f"Token limit: {ledger['token_limit']}; consumed: {ledger.get('llm_tokens', 0)}; research not started")


if __name__ == "__main__":
    main()
