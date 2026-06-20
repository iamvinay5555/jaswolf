"""Configuration via environment variables (prefix JASWOLF_) or constructor kwargs."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class JaswolfSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JASWOLF_", env_file=".env", extra="ignore")

    # --- storage ---
    # sqlite:///path/to/jaswolf.db (dev) or postgresql://user:pass@host:5432/jaswolf (prod)
    database_url: str = "sqlite:///./jaswolf.db"

    # --- cache ---
    redis_url: str | None = None        # e.g. redis://localhost:6379/0; None -> in-process LRU
    embed_cache_size: int = 10_000      # entries in the embedding cache
    embed_cache_ttl: int = 7 * 24 * 3600

    # --- embeddings ---
    # "auto": sentence-transformers if installed, else OpenAI-compatible if key set, else hash
    embedding_provider: str = "auto"    # auto | local | openai | hash
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    # load the model at service startup instead of on the first query — a CPU
    # box pays seconds-to-tens-of-seconds for the first sentence-transformers
    # embed; with prewarm that cost lands at boot, never on a live turn
    embedding_prewarm: bool = False
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"

    # --- scoring ---
    weight_importance: float = 0.4
    weight_relevance: float = 0.3
    weight_recency: float = 0.2
    weight_frequency: float = 0.1
    recency_half_life_days: float = 7.0
    frequency_saturation: int = 50      # access count at which frequency score ~= 1.0
    # query-driven search drops candidates below this cosine similarity, so an
    # important-but-irrelevant memory can never outrank actually relevant ones
    min_relevance: float = 0.1

    # --- write-path behavior ---
    dedup_threshold: float = 0.95       # cosine sim above which a new memory reinforces an existing one
    consolidation_threshold: float = 0.88
    consolidation_max_batch: int = 2000
    # corrections ("actually I prefer X now") archive the contradicted memory
    supersession_enabled: bool = True
    supersession_threshold: float = 0.5  # min similarity between correction and the fact it replaces
    # read-time current-state resolution: when retrieved memories fill the same
    # singleton slot ("User's office is X") with different values, inject only
    # the freshest. Complements write-time supersession for UNMARKED
    # contradictions; never collapses multi-valued relations. (temporal.py)
    temporal_resolution: bool = True

    # --- lifecycle (days without activity before transition) ---
    working_ttl_hours: float = 24.0
    active_to_warm_days: float = 14.0
    warm_to_cold_days: float = 60.0
    cold_to_archived_days: float = 180.0
    sweep_interval_seconds: float = 300.0

    # --- context builder ---
    context_token_budget: int = 1500
    context_candidate_pool: int = 48
    context_dedup_threshold: float = 0.92
    # identity-grade pinning: only preferences/goals clearing BOTH gates are
    # injected into context regardless of query relevance. Single-shot
    # extracted facts (confidence 0.75) don't qualify until reinforced.
    pin_min_importance: float = 0.7
    pin_min_confidence: float = 0.8
    # Only preferences/goals at/above this importance are force-pinned into
    # EVERY context regardless of query (identity/safety tier, e.g. "don't call
    # him Mr Smith"). Lower-importance preferences are included only when
    # query-relevant — stops a mediocre or stray preference from polluting every
    # turn (2026-06-15 live-pilot incident: a staging pref dominated all context).
    context_always_pin_importance: float = 0.9
    context_max_pins: int = 6  # cap on force-pinned identity items
    # strict mode (Jasmine, 2026-06-15): when true, ONLY memories explicitly
    # marked metadata.always_pin force-pin — high importance alone does not.
    # Prevents bulk-imported high-importance rows from injecting unconditionally.
    context_pin_requires_always_pin: bool = False
    # drop staging/test-marked memories from assembled context entirely
    exclude_test_memories: bool = True
    # context-boundary gate: a non-pinned vector candidate enters the prompt
    # only if its raw cosine clears a per-query threshold (calibration.py).
    # With a big-enough corpus the threshold is mean + noise_z*std of the
    # query's similarity to a background sample of the user's own memories —
    # anisotropy-invariant. noise_z>0 enables it; <=0 disables the gate.
    context_noise_z: float = 3.5
    context_background_sample: int = 256   # corpus embeddings sampled for the floor
    context_min_background: int = 24       # below this, fall back to fixed anchors
    # fallback only (tiny corpus): anchor-median floor + this margin
    context_similarity_margin: float = 0.08
    # keyword evidence: a query token exempts a memory from the context gate
    # only if it is discriminative — not a stopword, at least this long, and
    # present in at most this fraction of the corpus (IDF-style cut)
    keyword_min_token_len: int = 3
    keyword_max_df_ratio: float = 0.10
    # share of budget per section, normalized at build time
    context_share_preference: float = 0.20
    context_share_goal: float = 0.10
    context_share_semantic: float = 0.30
    context_share_procedural: float = 0.15
    context_share_episodic: float = 0.15
    context_share_relationship: float = 0.10

    # --- extraction ---
    extraction_strategy: str = "rules"  # rules | llm | hybrid
    llm_base_url: str | None = None     # OpenAI-compatible endpoint for extraction/merging
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0

    # --- API / security ---
    # "key1:tenantA,key2:tenantB" or "key1,key2" (tenant defaults to "default").
    api_keys: str = ""
    # Running the HTTP API without API keys requires this explicit opt-in;
    # otherwise startup fails. Embedded mode is unaffected.
    dev_open_mode: bool = False
    rate_limit_per_minute: int = 600
    cors_origins: str = ""  # empty = no cross-origin access (safe default)

    # --- MCP memory server (jaswolf mcp) ---
    # identity the server operates as; Hermes is single-user (Alice)
    mcp_user_id: str = "default"
    mcp_agent_id: str = "hermes"
    mcp_namespace: str = "default"
    mcp_host: str = "127.0.0.1"      # streamable-http transport bind
    mcp_port: int = 8765

    # --- misc ---
    log_level: str = "INFO"
    service_name: str = "jaswolf"

    def api_key_map(self) -> dict[str, str]:
        """Parse api_keys into {key: tenant_id}."""
        result: dict[str, str] = {}
        for part in self.api_keys.split(","):
            part = part.strip()
            if not part:
                continue
            key, _, tenant = part.partition(":")
            result[key.strip()] = tenant.strip() or "default"
        return result

    def context_shares(self) -> dict[str, float]:
        return {
            "preference": self.context_share_preference,
            "goal": self.context_share_goal,
            "semantic": self.context_share_semantic,
            "procedural": self.context_share_procedural,
            "episodic": self.context_share_episodic,
            "relationship": self.context_share_relationship,
        }
