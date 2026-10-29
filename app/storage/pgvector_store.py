"""独立向量数据库；显式迁移、带来源版本和构建 fencing 的快照。"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np

from app.agent.execution_budget import check_execution
from app.schemas.retrieval_schema import fingerprint
from app.schemas.vector_schema import BuildTicket, IndexSpec, VectorIntegrityError, VectorStoreUnavailable


@lru_cache(maxsize=8)
def _engine(url, pool_size, connect_timeout, statement_timeout, prepared_statements):
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url
    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql":
        raise VectorIntegrityError("向量库连接必须是独立 PostgreSQL")
    return create_engine(parsed.set(drivername="postgresql+psycopg"), pool_size=pool_size,
                         max_overflow=0, pool_timeout=connect_timeout, pool_pre_ping=True,
                         connect_args={"connect_timeout": connect_timeout,
                                       "options": f"-c statement_timeout={statement_timeout}",
                                       # WHY: 协议代理/事务连接池可能不保留会话级预备语句；显式关闭不改变参数绑定。
                                       "prepare_threshold": 5 if prepared_statements else None})


def normalized_vectors(vectors, dimension):
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != dimension or not np.isfinite(values).all():
        raise VectorIntegrityError("向量维度或数值无效")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any() or not np.isfinite(norms).all():
        raise VectorIntegrityError("向量范数无效")
    return values / norms


class PgVectorStore:
    def __init__(self, settings):
        self.settings = settings
        if not settings.card_vector_database_url:
            raise VectorStoreUnavailable("未配置独立向量库连接")

    @contextmanager
    def _db(self, *, adapt=True):
        check_execution()
        try:
            import psycopg
            from psycopg.rows import dict_row
            from pgvector.psycopg import register_vector
            from sqlalchemy.exc import SQLAlchemyError
        except ImportError as exc:
            raise VectorStoreUnavailable("请安装 requirements-vectorstore.txt") from exc
        try:
            s = self.settings
            engine = _engine(s.card_vector_database_url, s.card_vector_pool_size,
                             s.card_vector_connect_timeout, s.card_vector_statement_timeout_ms,
                             s.card_vector_prepared_statements)
            with engine.begin() as connection:
                raw = connection.connection.driver_connection
                if adapt:
                    register_vector(raw)
                with raw.cursor(row_factory=dict_row) as cursor:
                    yield cursor
                    check_execution()
        except (psycopg.Error, SQLAlchemyError) as exc:
            code = getattr(exc, "sqlstate", None) or getattr(getattr(exc, "orig", None), "sqlstate", None)
            if code == "42501" or (code and code.startswith("23")):
                raise VectorIntegrityError("向量库权限或约束校验失败") from exc
            raise VectorStoreUnavailable("向量库操作失败") from exc

    def migrate(self):
        dimension = self.settings.retrieval_embedding_dimension
        sql_path = Path(__file__).resolve().parents[2] / "migrations/vectorstore/001_initial.sql"
        with self._db(adapt=False) as cur:
            cur.execute("SELECT pg_advisory_xact_lock(174215013)")
            cur.execute(sql_path.read_text(encoding="utf-8").replace("__DIMENSION__", str(int(dimension))))
            cur.execute("SELECT dimension FROM rra_vector_schema WHERE singleton")
            if cur.fetchone()["dimension"] != dimension:
                raise VectorIntegrityError("已安装向量库维度不同，需独立迁移")

    def register_source(self, tenant, paper_id, source_hash, *, expected_revision=None):
        with self._db() as cur:
            cur.execute("INSERT INTO rra_vector_sources VALUES (%s,%s,1,%s) ON CONFLICT DO NOTHING",
                        (tenant, paper_id, source_hash))
            cur.execute("SELECT * FROM rra_vector_sources WHERE tenant=%s AND paper_id=%s FOR UPDATE", (tenant, paper_id))
            source = cur.fetchone()
            if source["source_hash"] == source_hash:
                return source["revision"]
            # WHY: 旧任务不能凭旧 PDF 自动递增来源版本；更新必须携带调用方观察到的 revision。
            if expected_revision != source["revision"]:
                raise VectorIntegrityError("来源变更需要匹配的 expected_revision")
            revision = source["revision"] + 1
            cur.execute("UPDATE rra_vector_sources SET revision=%s,source_hash=%s WHERE tenant=%s AND paper_id=%s",
                        (revision, source_hash, tenant, paper_id))
            return revision

    def _source(self, cur, spec, revision):
        cur.execute("SELECT * FROM rra_vector_sources WHERE tenant=%s AND paper_id=%s FOR UPDATE", (spec.tenant, spec.paper_id))
        row = cur.fetchone()
        if not row or row["revision"] != revision or row["source_hash"] != spec.source_hash:
            raise VectorIntegrityError("来源版本已失效或被删除")

    def claim(self, spec: IndexSpec, revision):
        with self._db() as cur:
            self._source(cur, spec, revision)
            cur.execute("SELECT *,(lease_expires_at > clock_timestamp()) AS live FROM rra_vector_snapshots "
                        "WHERE tenant=%s AND paper_id=%s AND source_revision=%s "
                        "AND processing_profile=%s AND embedding_profile=%s FOR UPDATE",
                        (spec.tenant, spec.paper_id, revision, spec.processing_profile, spec.embedding_profile))
            row = cur.fetchone()
            if row and (row["dimension"],row["expected_count"],row["total_count"]) != (
                    spec.dimension,spec.expected_count,spec.total_count):
                raise VectorIntegrityError("快照计划与维度不匹配")
            if row and row["status"] == "active":
                return str(row["id"]), None
            if row and row["status"] == "building" and row["live"]:
                raise VectorStoreUnavailable("索引正由其他构建者处理")
            snapshot = str(row["id"]) if row else str(uuid4())
            builder = str(uuid4())
            token = row["fencing_token"] + 1 if row else 1
            if row:
                cur.execute("DELETE FROM rra_vector_chunks WHERE snapshot_id=%s", (snapshot,))
                cur.execute("UPDATE rra_vector_snapshots SET status='building',builder_id=%s,fencing_token=%s,"
                            "lease_expires_at=clock_timestamp()+(%s * interval '1 second'),error_code=NULL WHERE id=%s",
                            (builder, token, self.settings.card_vector_build_lease_seconds, snapshot))
            else:
                cur.execute("INSERT INTO rra_vector_snapshots "
                            "(id,tenant,paper_id,source_revision,source_hash,processing_profile,embedding_profile,dimension,"
                            "expected_count,total_count,status,builder_id,fencing_token,lease_expires_at) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'building',%s,%s,"
                            "clock_timestamp()+(%s * interval '1 second'))",
                            (snapshot,spec.tenant,spec.paper_id,revision,spec.source_hash,spec.processing_profile,
                             spec.embedding_profile,spec.dimension,spec.expected_count,spec.total_count,
                             builder,token,self.settings.card_vector_build_lease_seconds))
            return snapshot, BuildTicket(snapshot,builder,token,revision)

    def _ticket(self, cur, spec, ticket):
        self._source(cur, spec, ticket.source_revision)
        cur.execute("SELECT * FROM rra_vector_snapshots WHERE id=%s AND tenant=%s AND paper_id=%s "
                    "AND status='building' AND builder_id=%s AND fencing_token=%s "
                    "AND lease_expires_at > clock_timestamp() FOR UPDATE",
                    (ticket.snapshot_id,spec.tenant,spec.paper_id,ticket.builder_id,ticket.fencing_token))
        row = cur.fetchone()
        if (not row or row["embedding_profile"] != spec.embedding_profile or row["processing_profile"] != spec.processing_profile
                or (row["dimension"],row["expected_count"],row["total_count"]) != (
                    spec.dimension,spec.expected_count,spec.total_count)):
            raise VectorIntegrityError("索引构建所有权或租约失效")
        return row

    def renew(self, spec, ticket):
        with self._db() as cur:
            self._ticket(cur, spec, ticket)
            cur.execute("UPDATE rra_vector_snapshots SET lease_expires_at=clock_timestamp()+"
                        "(%s * interval '1 second') WHERE id=%s", (self.settings.card_vector_build_lease_seconds,ticket.snapshot_id))

    def write(self, spec, ticket, chunks, vectors, *, start_order):
        matrix = normalized_vectors(vectors, spec.dimension)
        if len(matrix) != len(chunks):
            raise VectorIntegrityError("块与向量数量不同")
        with self._db() as cur:
            self._ticket(cur, spec, ticket)
            for order, (chunk, vector) in enumerate(zip(chunks,matrix), start_order):
                if chunk.get("paper_id") != spec.paper_id:
                    raise VectorIntegrityError("禁止跨论文写入")
                cur.execute("INSERT INTO rra_vector_chunks "
                            "(snapshot_id,tenant,paper_id,chunk_id,chunk_order,section,page,offset_basis,"
                            "start_offset,end_offset,text,content_hash,embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                            (ticket.snapshot_id,spec.tenant,spec.paper_id,chunk["chunk_id"],order,chunk["section"],
                             chunk["page"],chunk["offset_basis"],chunk["char_start"],chunk["char_end"],chunk["text"],
                             fingerprint(chunk["text"]),vector))

    def activate(self, spec, ticket):
        with self._db() as cur:
            self._ticket(cur, spec, ticket)
            cur.execute("SELECT count(*) AS n,min(chunk_order) AS first,max(chunk_order) AS last "
                        "FROM rra_vector_chunks WHERE snapshot_id=%s", (ticket.snapshot_id,))
            count = cur.fetchone()
            if count["n"] != spec.expected_count or count["first"] != 0 or count["last"] != spec.expected_count - 1:
                raise VectorIntegrityError("快照计划未完整写入")
            cur.execute("UPDATE rra_vector_snapshots SET status='superseded' WHERE tenant=%s AND paper_id=%s "
                        "AND embedding_profile=%s AND status='active'", (spec.tenant,spec.paper_id,spec.embedding_profile))
            cur.execute("UPDATE rra_vector_snapshots SET status='active',activated_at=clock_timestamp() WHERE id=%s", (ticket.snapshot_id,))

    def fail(self, spec, ticket, *, cancelled=False):
        with self._db() as cur:
            self._ticket(cur, spec, ticket)
            cur.execute("UPDATE rra_vector_snapshots SET status=%s,error_code=%s WHERE id=%s",
                        ("cancelled" if cancelled else "failed", "build_interrupted", ticket.snapshot_id))

    def _active(self, cur, spec, snapshot, revision):
        self._source(cur, spec, revision)
        cur.execute("SELECT * FROM rra_vector_snapshots WHERE id=%s AND tenant=%s AND paper_id=%s "
                    "AND source_revision=%s AND status='active'", (snapshot,spec.tenant,spec.paper_id,revision))
        row = cur.fetchone()
        if (not row or row["source_hash"] != spec.source_hash or row["processing_profile"] != spec.processing_profile
                or row["embedding_profile"] != spec.embedding_profile or row["dimension"] != spec.dimension
                or row["expected_count"] != spec.expected_count or row["total_count"] != spec.total_count):
            raise VectorIntegrityError("快照不属于当前论文、来源或模型")
        return row

    def load(self, spec, snapshot, revision):
        with self._db() as cur:
            self._active(cur,spec,snapshot,revision)
            cur.execute("SELECT chunk_id,chunk_order,text,content_hash,page,section,offset_basis,start_offset,end_offset "
                        "FROM rra_vector_chunks WHERE tenant=%s AND paper_id=%s AND snapshot_id=%s ORDER BY chunk_order",
                        (spec.tenant,spec.paper_id,snapshot))
            rows = cur.fetchall()
            if len(rows) != spec.expected_count:
                raise VectorIntegrityError("活动快照数量不完整")
            return rows

    def search(self, spec, snapshot, revision, query, limit):
        if type(limit) is not int or not 1 <= limit <= 512:
            raise VectorIntegrityError("无效向量查询上限")
        vector = normalized_vectors([query], spec.dimension)[0]
        with self._db() as cur:
            self._active(cur,spec,snapshot,revision)
            cur.execute("SELECT chunk_id,1-(embedding <=> %s) AS score FROM rra_vector_chunks "
                        "WHERE tenant=%s AND paper_id=%s AND snapshot_id=%s "
                        "ORDER BY embedding <=> %s,chunk_order,chunk_id LIMIT %s",
                        (vector,spec.tenant,spec.paper_id,snapshot,vector,limit))
            return [(row["chunk_id"],float(row["score"])) for row in cur.fetchall()]

    def delete(self, tenant, paper_id):
        with self._db() as cur:
            cur.execute("DELETE FROM rra_vector_sources WHERE tenant=%s AND paper_id=%s", (tenant,paper_id))

    def health(self):
        with self._db() as cur:
            cur.execute("SELECT version() AS server, extversion AS vector FROM pg_extension WHERE extname='vector'")
            return cur.fetchone()

    def ready(self):
        """只读检查扩展、迁移版本、维度及运行所需表；不得在业务入口隐式迁移。"""
        with self._db() as cur:
            cur.execute("SELECT version, dimension FROM rra_vector_schema WHERE singleton")
            schema = cur.fetchone()
            if not schema or (schema["version"], schema["dimension"]) != (1, self.settings.retrieval_embedding_dimension):
                raise VectorIntegrityError("向量库迁移版本或维度不匹配")
            for table in ("rra_vector_sources", "rra_vector_snapshots", "rra_vector_chunks"):
                cur.execute("SELECT to_regclass(%s) AS relation", (table,))
                if not cur.fetchone()["relation"]:
                    raise VectorIntegrityError("向量库缺少已迁移的运行表")
        return True
