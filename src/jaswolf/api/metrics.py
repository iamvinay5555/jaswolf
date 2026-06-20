"""Prometheus metrics with graceful no-op fallback when prometheus-client
isn't installed. Metric objects are module-level singletons so repeated app
construction (tests) never double-registers."""

from __future__ import annotations


class _Noop:
    def labels(self, *args, **kwargs):
        return self

    def observe(self, *args, **kwargs):
        return None

    def inc(self, *args, **kwargs):
        return None

    def set(self, *args, **kwargs):
        return None


try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    AVAILABLE = True
    REQUESTS = Counter("jaswolf_requests_total", "API requests", ["method", "route", "status"])
    REQUEST_LATENCY = Histogram(
        "jaswolf_request_latency_seconds", "API request latency", ["route"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )
    SEARCH_LATENCY = Histogram(
        "jaswolf_search_latency_seconds", "Memory search latency",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
    )
    CONTEXT_LATENCY = Histogram(
        "jaswolf_context_latency_seconds", "Context build latency",
        buckets=(0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0),
    )
    MEMORIES_CREATED = Counter("jaswolf_memories_created_total", "Memories created", ["memory_type"])
    MEMORIES_REINFORCED = Counter("jaswolf_memories_reinforced_total", "Write-path dedup hits")
    EMBED_CACHE = Counter("jaswolf_embed_cache_total", "Embedding cache lookups", ["result"])
    MEMORY_COUNT = Gauge("jaswolf_memories", "Total memories", ["tenant"])

    def render_latest() -> tuple[bytes, str]:
        return generate_latest(), CONTENT_TYPE_LATEST

except ImportError:  # pragma: no cover
    AVAILABLE = False
    REQUESTS = REQUEST_LATENCY = SEARCH_LATENCY = CONTEXT_LATENCY = _Noop()
    MEMORIES_CREATED = MEMORIES_REINFORCED = EMBED_CACHE = MEMORY_COUNT = _Noop()

    def render_latest() -> tuple[bytes, str]:
        return b"# prometheus-client not installed\n", "text/plain"
