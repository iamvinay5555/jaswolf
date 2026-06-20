-- DB-level metadata, e.g. the embedding fingerprint stamped at first open so
-- a model/provider switch on an existing DB is detected instead of silently
-- mixing incompatible vector spaces.
CREATE TABLE IF NOT EXISTS jas0_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
