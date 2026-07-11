-- L0 conversation archive (semantic pyramid, v0.3.0).
-- Raw conversation turns — the evidence layer under extracted memories.
-- Opt-in via JASWOLF_CONVERSATION_CAPTURE; pruned by retention, never rewritten.

CREATE TABLE IF NOT EXISTS conversation_messages (
    id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT NOT NULL,
    agent_id TEXT,
    session_id TEXT,
    namespace TEXT NOT NULL DEFAULT 'default',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conv_scope
    ON conversation_messages (tenant_id, user_id, namespace, created_at);
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_messages (session_id);
CREATE INDEX IF NOT EXISTS idx_conv_fts
    ON conversation_messages USING gin (to_tsvector('english', content));
