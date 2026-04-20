"""Prometheus metrics for the recognize paths.

We keep the surface small on purpose — three metrics cover the questions
ops will actually ask in the first 90 days of operation:

- Are requests failing? (``recognize_requests_total{result="error"}``)
- Is p95 latency within the 500 ms budget? (``recognize_duration_seconds``)
- Are we leaking WebSocket turns? (``ws_active_turns``)

Label cardinality is bounded: ``path`` ∈ {text, file, stream}, ``result`` ∈
{ok, error, no_speech}, ``scene`` is a small finite set per deployment. We
deliberately do **not** label by intent_id or session_id to keep the cardinality
manageable under high traffic.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# One app-local registry so tests can snapshot/reset it without touching the
# default global, and so /metrics doesn't leak default Python process metrics
# we never asked for (gc, platform info). Keep it boring.
registry = CollectorRegistry()

recognize_requests_total = Counter(
    "recognize_requests_total",
    "Total recognition requests.",
    labelnames=("path", "scene", "result"),
    registry=registry,
)

recognize_duration_seconds = Histogram(
    "recognize_duration_seconds",
    "End-to-end recognition duration in seconds.",
    labelnames=("path",),
    # Buckets align with our 500 ms latency budget + tail observation.
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0),
    registry=registry,
)

ws_active_turns = Gauge(
    "ws_active_turns",
    "In-flight WebSocket recognition turns.",
    registry=registry,
)
