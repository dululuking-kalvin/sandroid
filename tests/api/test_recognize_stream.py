from __future__ import annotations

from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient

from sandroid.api.app import app
from sandroid.api.deps import (
    DEV_DEFAULT_API_KEY,
    get_asr,
    reset_dependency_caches,
)
from sandroid.models.asr.stub import StubASR


@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    reset_dependency_caches()
    yield
    app.dependency_overrides.clear()
    reset_dependency_caches()


def test_stream_happy_path() -> None:
    app.dependency_overrides[get_asr] = lambda: StubASR(scripted_transcript="你好")
    client = TestClient(app)
    with client.websocket_connect(
        "/api/v1/recognize/stream",
        headers={"x-api-key": DEV_DEFAULT_API_KEY},
    ) as ws:
        ws.send_json({"scene_id": "example_bank"})
        ws.send_bytes(b"\x00\x00" * 1600)  # 100 ms of silence
        ws.send_bytes(b"\x00\x00" * 1600)
        ws.send_json({"type": "end"})

        messages: list[dict] = []
        while True:
            msg = ws.receive_json()
            messages.append(msg)
            if msg["type"] in {"result", "error"}:
                break

    kinds = [m["type"] for m in messages]
    assert "partial" in kinds
    assert "final" in kinds
    assert kinds[-1] == "result"
    result = messages[-1]
    assert result["top"]["intent_id"] == "greeting"
    assert result["current_category_id"] == "root"


def test_stream_rejects_missing_api_key() -> None:
    client = TestClient(app)
    with (
        pytest.raises(Exception),  # noqa: B017 — starlette raises WebSocketDisconnect
        client.websocket_connect("/api/v1/recognize/stream"),
    ):
        pass


def test_stream_requires_scene_or_path() -> None:
    client = TestClient(app)
    with client.websocket_connect(
        "/api/v1/recognize/stream",
        headers={"x-api-key": DEV_DEFAULT_API_KEY},
    ) as ws:
        ws.send_json({})
        msg = ws.receive_json()
        assert msg["type"] == "error"
