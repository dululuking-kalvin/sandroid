"""Phase 5d production hardening tests.

Covers:
- ``SANDROID_ENV=production`` gate: missing artifacts must raise, not silently
  downgrade to stub backends.
- ASR backend errors during streaming must surface as a non-fatal
  ``ASR_FAILED`` error frame without closing the socket on the server side
  (the turn is cancelled, not the session).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sandroid.api.app import app
from sandroid.api.deps import (
    ASR_MODEL_PATH_ENV,
    DEV_DEFAULT_API_KEY,
    ENV_ENV,
    NLU_MODEL_PATH_ENV,
    SILERO_PATH_ENV,
    get_asr,
    get_matcher,
    get_vad,
    reset_dependency_caches,
)
from sandroid.models.asr.base import AudioChunk, PartialTranscript
from sandroid.models.asr.paraformer import ParaformerError
from sandroid.models.asr.stub import StubASR
from sandroid.models.nlu.onnx_embedder import ONNXEmbedderError
from sandroid.vad.mock import MockVAD
from sandroid.vad.silero import SileroVADError

AUTH_HEADERS = {"x-api-key": DEV_DEFAULT_API_KEY}


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(ENV_ENV, raising=False)
    reset_dependency_caches()
    yield
    app.dependency_overrides.clear()
    reset_dependency_caches()


def _missing(tmp_path: Path) -> str:
    return str(tmp_path / "does-not-exist.bin")


# ---------- SANDROID_ENV=production gate ----------


def test_production_missing_asr_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_ENV, "production")
    monkeypatch.setenv(ASR_MODEL_PATH_ENV, _missing(tmp_path))
    with pytest.raises(ParaformerError):
        get_asr()


def test_production_missing_nlu_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_ENV, "production")
    monkeypatch.setenv(NLU_MODEL_PATH_ENV, _missing(tmp_path))
    with pytest.raises(ONNXEmbedderError):
        get_matcher()


def test_production_missing_vad_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_ENV, "production")
    monkeypatch.setenv(SILERO_PATH_ENV, _missing(tmp_path))
    with pytest.raises(SileroVADError):
        get_vad()


def test_dev_missing_asr_falls_back_to_stub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No SANDROID_ENV set -> dev mode -> must NOT raise.
    monkeypatch.setenv(ASR_MODEL_PATH_ENV, _missing(tmp_path))
    asr = get_asr()
    assert isinstance(asr, StubASR)


# ---------- ASR_FAILED non-fatal error frame ----------


class _ExplodingASR:
    """ASR backend whose stream() raises mid-iteration.

    transcribe_file is unused by the WS handler; stream() is the hot path.
    """

    async def transcribe_file(self, wav_bytes: bytes):  # type: ignore[no-untyped-def]
        raise AssertionError("WS handler should call stream(), not transcribe_file()")

    async def stream(
        self, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[PartialTranscript]:
        # Emit one partial so the test can prove the generator actually started
        # before the failure — exercises the try/except in _stream_segment.
        yielded = False
        async for _chunk in chunks:
            if not yielded:
                yielded = True
                yield PartialTranscript(text="", is_final=False, start_ms=0, end_ms=0)
            raise RuntimeError("synthetic backend failure")


def test_asr_failure_emits_non_fatal_error_frame() -> None:
    app.dependency_overrides[get_vad] = MockVAD
    app.dependency_overrides[get_asr] = _ExplodingASR

    client = TestClient(app)
    with client.websocket_connect("/api/v1/recognize/stream", headers=AUTH_HEADERS) as ws:
        ws.send_json(
            {
                "type": "start",
                "session_id": "sess-5d",
                "turn_id": 1,
                "scene_id": "example_bank",
            }
        )
        ws.send_bytes(b"\x00\x00" * 1600)
        ws.send_json({"type": "stop", "session_id": "sess-5d", "turn_id": 1})

        seen: list[dict] = []
        # Drain until error frame arrives. Socket stays open — we close it
        # client-side via the context manager.
        while True:
            msg = ws.receive_json()
            seen.append(msg)
            if msg.get("type") == "error":
                break

    kinds = [m["type"] for m in seen]
    assert "error" in kinds
    err = seen[-1]
    assert err["type"] == "error"
    assert err["code"] == "ASR_FAILED"
    assert err.get("fatal", False) is False
    assert err["session_id"] == "sess-5d"
    assert err["turn_id"] == 1
