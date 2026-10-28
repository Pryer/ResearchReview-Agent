-- Explicit rollback: removes ONLY the optional RRA vector store and its data.
-- Keeps the shared vector extension and the business database untouched.
DROP TABLE IF EXISTS rra_vector_chunks;
DROP TABLE IF EXISTS rra_vector_snapshots;
DROP TABLE IF EXISTS rra_vector_sources;
DROP TABLE IF EXISTS rra_vector_schema;
