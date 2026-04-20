"""Phase 5f — /metrics, /readyz, JSON log format tests."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient

from sandroid.api.app import app
from sandroid.api.deps import (
    DEV_DEFAULT_API_KEY,
    get_asr,
    get_matcher,
    get_vad,
    reset_dependency_caches,
)
from sandroid.api.logging_config import LOG_FORMAT_ENV, configure_logging
from sandroid.core.matcher import StubMatcher
from sandroid.models.asr.stub import StubASR
from sandroid.vad.mock import MockVAD

AUTH_HEADERS = {"x-api-key": DEV_DEFAULT_API_KEY}


@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    reset_dependency_caches()
    yield
    app.dependency_overrides.clear()
    reset_dependency_caches()


# ---------- /healthz ----------


def test_healthz_is_cheap_and_always_ok() -> None:
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


# ---------- /readyz ----------


def test_readyz_ok_when_all_deps_resolve() -> None:
    # Overriding with always-working fakes so this test doesn't depend on
    # whether real ONNX artifacts are on disk.
    app.dependency_overrides[get_matcher] = StubMatcher
    app.dependency_overrides[get_asr] = StubASR
    app.dependency_overrides[get_vad] = MockVAD

    client = TestClient(app)
    resp = client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["checks"]) == {"registry", "session_store", "matcher", "vad", "asr"}
    assert all(v == "ok" for v in body["checks"].values())


def test_readyz_returns_503_when_probe_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Point SCENES_DIR at a non-existent directory so get_registry raises.
    monkeypatch.setenv("SANDROID_SCENES_DIR", "/definitely/does/not/exist")
    client = TestClient(app)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "error"
    assert body["checks"]["registry"].startswith("error:")


# ---------- /metrics ----------


def test_metrics_exposes_recognize_counters() -> None:
    app.dependency_overrides[get_matcher] = StubMatcher

    client = TestClient(app)
    # Drive at least one recognize/text so counters get populated.
    resp = client.post(
        "/api/v1/recognize/text",
        json={"scene_id": "example_bank", "text": "你好"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "text/plain" in metrics.headers["content-type"]
    body = metrics.text
    assert "recognize_requests_total" in body
    assert 'path="text"' in body
    assert "recognize_duration_seconds" in body
    assert "ws_active_turns" in body


def test_metrics_endpoint_unauthenticated() -> None:
    # Unlike /api/v1/* which requires the API key, /metrics must be
    # scrapeable by Prometheus without auth (restrict at network layer).
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200


# ---------- Structured logging ----------


def test_json_log_format(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(LOG_FORMAT_ENV, "json")
    configure_logging()
    logging.getLogger("sandroid.test").info(
        "hello", extra={"session_id": "s-1", "duration_ms": 12.3}
    )
    out = capsys.readouterr().out.strip().splitlines()
    assert out, "expected at least one log line"
    parsed = json.loads(out[-1])
    assert parsed["message"] == "hello"
    assert parsed["session_id"] == "s-1"
    assert parsed["duration_ms"] == 12.3
    # Restore default format so other tests aren't polluted.
    monkeypatch.delenv(LOG_FORMAT_ENV, raising=False)
    configure_logging()


def test_plain_log_format_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(LOG_FORMAT_ENV, raising=False)
    configure_logging()
    logging.getLogger("sandroid.test").info("plain hello")
    out = capsys.readouterr().out
    assert "plain hello" in out
    # Should NOT be JSON — a plain message has no JSON braces wrapping it.
    assert not out.strip().startswith("{")
