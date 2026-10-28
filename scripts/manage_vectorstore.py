"""显式管理可选向量库；连接信息仅从配置读取，不输出口令。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.storage.pgvector_store import PgVectorStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["migrate", "health", "register-source", "rollback"])
    parser.add_argument("--paper-id")
    parser.add_argument("--pdf")
    parser.add_argument("--expected-revision", type=int)
    parser.add_argument("--confirm-vector-data-loss", action="store_true")
    args = parser.parse_args()
    if args.action == "rollback" and not args.confirm_vector_data_loss:
        parser.error("rollback requires --confirm-vector-data-loss")
    settings = get_settings()
    store = PgVectorStore(settings)
    if args.action == "migrate":
        store.migrate()
        print("Vector schema migration applied")
    elif args.action == "health":
        print(json.dumps(store.health(), ensure_ascii=False))
    elif args.action == "rollback":
        with store._db(adapt=False) as cursor:
            cursor.execute((Path(__file__).resolve().parents[1] / "migrations/vectorstore/001_rollback.sql").read_text(encoding="utf-8"))
        print("Optional vector tables removed; business database untouched")
    else:
        if not args.pdf or not args.paper_id:
            parser.error("register-source requires --pdf and --paper-id")
        from app.tools.parse_pdf import parse_pdf
        from app.services.document_ingestion import source_fingerprint
        parsed = parse_pdf(args.pdf, paper_id=args.paper_id)
        revision = store.register_source(settings.card_vector_tenant, args.paper_id, source_fingerprint(parsed),
                                         expected_revision=args.expected_revision)
        print(json.dumps({"source_revision": revision}))


if __name__ == "__main__":
    main()
