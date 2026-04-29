"""Phase 5f-d: WS /recognize/stream passes accumulated PCM16 to SLU.

The WS path streams raw PCM frames from client→server. We assert the SLU
adapter sees the concatenation of every frame the client sent — proving
``_TurnContext.audio_accum`` mirrors correctly through ``handle_pcm`` and
``_finalize_turn``. A second test exercises the overflow path: when the
buffer exceeds SANDROID_MAX_AUDIO_BYTES, Path B is dropped silently while
Path A still emits a ``result`` frame.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient, WebSocketTestSession

from sandroid.api.app import app
from sandroid.api.deps import (
    DEV_DEFAULT_API_KEY,
    MAX_AUDIO_BYTES_ENV,
    get_asr,
    get_matcher,
    get_orchestrator,
    get_registry,
    get_session_store,
    get_vad,
    reset_dependency_caches,
)
from sandroid.core.orchestrator import Orchestrator
from sandroid.models.asr.stub import StubASR
from sandroid.vad.mock import MockVAD
from tests._helpers.recording_slu import RecordingSLU

AUTH_HEADERS = {"x-api-key": DEV_DEFAULT_API_KEY}


def _drain_until(ws: WebSocketTestSession, kinds: set[str]) -> list[dict]:
    seen: list[dict] = []
    while True:
        msg = ws.receive_json()
        seen.append(msg)
        if msg.get("type") in kinds:
            return seen


@pytest.fixture
def recording_slu() -> Iterator[RecordingSLU]:
    reset_dependency_caches()
    rec = RecordingSLU()

    def _build_orch() -> Orchestrator:
        return Orchestrator(
            registry=get_registry(),
            sessions=get_session_store(),
            matcher=get_matcher(),
            slu=rec,
        )

    app.dependency_overrides[get_orchestrator] = _build_orch
    app.dependency_overrides[get_vad] = MockVAD
    app.dependency_overrides[get_asr] = lambda: StubASR(scripted_transcript="你好")
    yield rec
    app.dependency_overrides.clear()
    reset_dependency_caches()


def test_stream_accumulates_pcm_and_passes_to_slu(
    recording_slu: RecordingSLU,
) -> None:
    chunks = [b"\x01\x00\x02\x00" * 800, b"\x03\x00\x04\x00" * 800]
    expected = b"".join(chunks)

    client = TestClient(app)
    with client.websocket_connect(
        "/api/v1/recognize/stream", headers=AUTH_HEADERS,
    ) as ws:
        ws.send_json({
            "type": "start",
            "session_id": "sess-aud-1",
            "turn_id": 1,
            "scene_id": "example_bank",
        })
        for c in chunks:
            ws.send_bytes(c)
        ws.send_json({"type": "stop", "session_id": "sess-aud-1", "turn_id": 1})
        _drain_until(ws, {"result", "error"})

    assert len(recording_slu.calls) == 1
    received_pcm, scene_id = recording_slu.calls[0]
    assert received_pcm == expected
    assert scene_id == "example_bank"


def test_stream_overflow_drops_path_b_but_path_a_completes(
    recording_slu: RecordingSLU,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MAX_AUDIO_BYTES_ENV, "1024")  # 1 KiB cap
    big = b"\x00\x00" * 2000  # 4 KiB, four times the cap

    client = TestClient(app)
    with client.websocket_connect(
        "/api/v1/recognize/stream", headers=AUTH_HEADERS,
    ) as ws:
        ws.send_json({
            "type": "start",
            "session_id": "sess-aud-2",
            "turn_id": 1,
            "scene_id": "example_bank",
        })
        ws.send_bytes(big)
        ws.send_json({"type": "stop", "session_id": "sess-aud-2", "turn_id": 1})
        messages = _drain_until(ws, {"result", "error"})

    # Path A still completed: the turn produced a `result` frame.
    assert messages[-1]["type"] == "result"
    # Path B was dropped — SLU got an empty buffer (overflow guard zeroed
    # audio_accum and ``_finalize_turn`` then sent ``audio=None``, which
    # short-circuits the SLU call entirely inside Orchestrator).
    assert recording_slu.calls == []
