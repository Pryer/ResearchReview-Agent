CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS rra_vector_schema (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    version integer NOT NULL, dimension integer NOT NULL
);
INSERT INTO rra_vector_schema(singleton, version, dimension)
VALUES (true, 1, __DIMENSION__) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS rra_vector_sources (
    tenant text NOT NULL, paper_id text NOT NULL, revision bigint NOT NULL CHECK (revision > 0),
    source_hash text NOT NULL, PRIMARY KEY(tenant, paper_id)
);
CREATE TABLE IF NOT EXISTS rra_vector_snapshots (
    id uuid PRIMARY KEY, tenant text NOT NULL, paper_id text NOT NULL,
    source_revision bigint NOT NULL, source_hash text NOT NULL,
    processing_profile text NOT NULL, embedding_profile text NOT NULL,
    dimension integer NOT NULL CHECK (dimension = __DIMENSION__),
    expected_count integer NOT NULL CHECK (expected_count > 0),
    total_count integer NOT NULL CHECK (total_count >= expected_count),
    status text NOT NULL CHECK (status IN ('building','active','superseded','failed','cancelled')),
    builder_id uuid NOT NULL, fencing_token bigint NOT NULL CHECK (fencing_token > 0),
    lease_expires_at timestamptz NOT NULL, created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    activated_at timestamptz, error_code text,
    FOREIGN KEY(tenant, paper_id) REFERENCES rra_vector_sources(tenant, paper_id) ON DELETE CASCADE,
    UNIQUE(tenant, paper_id, source_revision, processing_profile, embedding_profile),
    UNIQUE(id, tenant, paper_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS rra_vector_one_active
ON rra_vector_snapshots(tenant, paper_id, embedding_profile) WHERE status='active';
CREATE TABLE IF NOT EXISTS rra_vector_chunks (
    snapshot_id uuid NOT NULL, tenant text NOT NULL, paper_id text NOT NULL,
    chunk_id integer NOT NULL, chunk_order integer NOT NULL,
    section text NOT NULL, page integer, offset_basis text NOT NULL CHECK (offset_basis IN ('page','full_text')),
    start_offset integer NOT NULL CHECK(start_offset >= 0),
    end_offset integer NOT NULL CHECK(end_offset > start_offset),
    text text NOT NULL CHECK(length(text) > 0), content_hash text NOT NULL,
    embedding vector(__DIMENSION__) NOT NULL CHECK(vector_norm(embedding) > 0),
    PRIMARY KEY(snapshot_id, chunk_id), UNIQUE(snapshot_id, chunk_order),
    FOREIGN KEY(snapshot_id, tenant, paper_id)
      REFERENCES rra_vector_snapshots(id, tenant, paper_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS rra_vector_chunks_scope ON rra_vector_chunks(tenant, paper_id, snapshot_id);
