from __future__ import annotations

from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient, WebSocketTestSession

from sandroid.api.app import app
from sandroid.api.deps import (
    DEV_DEFAULT_API_KEY,
    get_asr,
    reset_dependency_caches,
)
from sandroid.models.asr.stub import StubASR

AUTH_HEADERS = {"x-api-key": DEV_DEFAULT_API_KEY}


@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    reset_dependency_caches()
    yield
    app.dependency_overrides.clear()
    reset_dependency_caches()


def _drain_until(ws: WebSocketTestSession, kinds: set[str]) -> list[dict]:
    """Consume frames until one of ``kinds`` is the frame type."""

    seen: list[dict] = []
    while True:
        msg = ws.receive_json()
        seen.append(msg)
        if msg.get("type") in kinds:
            return seen


def test_stream_happy_path() -> None:
    app.dependency_overrides[get_asr] = lambda: StubASR(scripted_transcript="你好")
    client = TestClient(app)
    with client.websocket_connect("/api/v1/recognize/stream", headers=AUTH_HEADERS) as ws:
        ws.send_json(
            {
                "type": "start",
                "session_id": "sess-1",
                "turn_id": 1,
                "scene_id": "example_bank",
            }
        )
        ws.send_bytes(b"\x00\x00" * 1600)
        ws.send_bytes(b"\x00\x00" * 1600)
        ws.send_json({"type": "stop", "session_id": "sess-1", "turn_id": 1})

        messages = _drain_until(ws, {"result", "error"})

    kinds = [m["type"] for m in messages]
    assert "partial" in kinds
    assert "final" in kinds
    assert kinds[-1] == "result"
    result = messages[-1]
    assert result["top"]["intent_id"] == "greeting"
    assert result["current_category_id"] == "root"
    # Id correlation on every frame.
    for m in messages:
        assert m["session_id"] == "sess-1"
        assert m["turn_id"] == 1


def test_stream_rejects_missing_api_key() -> None:
    client = TestClient(app)
    with (
        pytest.raises(Exception),  # noqa: B017 — starlette raises WebSocketDisconnect
        client.websocket_connect("/api/v1/recognize/stream"),
    ):
        pass


def test_stream_start_without_scene_closes() -> None:
    client = TestClient(app)
    with client.websocket_connect("/api/v1/recognize/stream", headers=AUTH_HEADERS) as ws:
        ws.send_json({"type": "start", "session_id": "sess-x", "turn_id": 1})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "BAD_START"
        assert msg["fatal"] is True


def test_stream_stale_turn_is_non_fatal() -> None:
    app.dependency_overrides[get_asr] = lambda: StubASR(scripted_transcript="你好")
    client = TestClient(app)
    with client.websocket_connect("/api/v1/recognize/stream", headers=AUTH_HEADERS) as ws:
        ws.send_json(
            {
                "type": "start",
                "session_id": "sess-2",
                "turn_id": 5,
                "scene_id": "example_bank",
            }
        )
        ws.send_bytes(b"\x00\x00" * 1600)
        ws.send_json({"type": "stop", "session_id": "sess-2", "turn_id": 5})
        _drain_until(ws, {"result"})

        # Stale turn_id — must produce a non-fatal error, socket stays usable.
        ws.send_json(
            {
                "type": "start",
                "session_id": "sess-2",
                "turn_id": 3,  # ≤ 5
                "scene_id": "example_bank",
            }
        )
        stale = ws.receive_json()
        assert stale["type"] == "error"
        assert stale["code"] == "STALE_TURN"
        assert stale["fatal"] is False

        # Next valid turn still works.
        ws.send_json(
            {
                "type": "start",
                "session_id": "sess-2",
                "turn_id": 6,
                "scene_id": "example_bank",
            }
        )
        ws.send_bytes(b"\x00\x00" * 1600)
        ws.send_json({"type": "stop", "session_id": "sess-2", "turn_id": 6})
        final_msgs = _drain_until(ws, {"result"})
        assert final_msgs[-1]["turn_id"] == 6


def test_stream_barge_in_cancels_prior_turn() -> None:
    app.dependency_overrides[get_asr] = lambda: StubASR(scripted_transcript="你好")
    client = TestClient(app)
    with client.websocket_connect("/api/v1/recognize/stream", headers=AUTH_HEADERS) as ws:
        ws.send_json(
            {
                "type": "start",
                "session_id": "sess-3",
                "turn_id": 1,
                "scene_id": "example_bank",
            }
        )
        ws.send_bytes(b"\x00\x00" * 1600)
        # No stop — interrupt with a new turn instead.
        ws.send_json(
            {
                "type": "start",
                "session_id": "sess-3",
                "turn_id": 2,
                "scene_id": "example_bank",
            }
        )
        ws.send_bytes(b"\x00\x00" * 1600)
        ws.send_json({"type": "stop", "session_id": "sess-3", "turn_id": 2})

        messages = _drain_until(ws, {"result"})

    # The cancelled turn must not produce a `result` frame.
    results = [m for m in messages if m["type"] == "result"]
    assert len(results) == 1
    assert results[0]["turn_id"] == 2
    # Turn 1 may have emitted a partial before cancellation; any such partial
    # must be id-tagged as turn 1 so the client can discard it.
    for m in messages:
        assert m["session_id"] == "sess-3"
        assert m["turn_id"] in {1, 2}


def test_stream_unknown_scene_is_non_fatal() -> None:
    app.dependency_overrides[get_asr] = lambda: StubASR(scripted_transcript="你好")
    client = TestClient(app)
    with client.websocket_connect("/api/v1/recognize/stream", headers=AUTH_HEADERS) as ws:
        ws.send_json(
            {
                "type": "start",
                "session_id": "sess-4",
                "turn_id": 1,
                "scene_id": "ghost_scene",
            }
        )
        ws.send_bytes(b"\x00\x00" * 1600)
        ws.send_json({"type": "stop", "session_id": "sess-4", "turn_id": 1})
        messages = _drain_until(ws, {"error"})
        err = messages[-1]
        assert err["code"] == "UNKNOWN_SCENE"
        assert err["fatal"] is False
        assert err["turn_id"] == 1

        # Socket still works — run a valid next turn.
        ws.send_json(
            {
                "type": "start",
                "session_id": "sess-4",
                "turn_id": 2,
                "scene_id": "example_bank",
            }
        )
        ws.send_bytes(b"\x00\x00" * 1600)
        ws.send_json({"type": "stop", "session_id": "sess-4", "turn_id": 2})
        final_msgs = _drain_until(ws, {"result"})
        assert final_msgs[-1]["type"] == "result"
        assert final_msgs[-1]["turn_id"] == 2
