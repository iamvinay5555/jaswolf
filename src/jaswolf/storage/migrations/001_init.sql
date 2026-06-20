-- JasWolf initial schema (PostgreSQL + pgvector)
-- __EMBED_DIM__ is replaced with the configured embedding dimension at apply time.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT NOT NULL,
    agent_id TEXT,
    session_id TEXT,
    namespace TEXT NOT NULL DEFAULT 'default',
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding vector(__EMBED_DIM__),
    memory_type TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'active',
    importance REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.8,
    access_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_accessed TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_memories_scope
    ON memories (tenant_id, user_id, namespace, state, memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_hash
    ON memories (tenant_id, user_id, namespace, content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_expiry ON memories (expires_at);
CREATE INDEX IF NOT EXISTS idx_memories_activity
    ON memories (state, (COALESCE(last_accessed, updated_at)));

-- ANN index: HNSW, cosine. Build concurrently on large existing tables.
CREATE INDEX IF NOT EXISTS idx_memories_embedding
    ON memories USING hnsw (embedding vector_cosine_ops);

-- Full-text index for hybrid search.
CREATE INDEX IF NOT EXISTS idx_memories_fts
    ON memories USING gin (to_tsvector('english', content));

CREATE TABLE IF NOT EXISTS memory_versions (
    id BIGSERIAL PRIMARY KEY,
    memory_id UUID NOT NULL,
    content TEXT NOT NULL,
    reason TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_versions_memory ON memory_versions (memory_id);

CREATE TABLE IF NOT EXISTS memory_relationships (
    id BIGSERIAL PRIMARY KEY,
    from_id UUID NOT NULL,
    to_id UUID NOT NULL,
    relation TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rel_from ON memory_relationships (from_id);

CREATE TABLE IF NOT EXISTS memory_access_logs (
    id BIGSERIAL PRIMARY KEY,
    memory_id UUID NOT NULL,
    tenant_id TEXT NOT NULL,
    action TEXT NOT NULL,
    query TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_access_memory ON memory_access_logs (memory_id);
CREATE INDEX IF NOT EXISTS idx_access_time ON memory_access_logs (created_at);
