"""Phase 5f-d: REST /recognize/file passes PCM16 to SLU.

Asserts that the bytes the SLU adapter receives equal the PCM16 produced by
``decode_wav`` on the uploaded WAV — proving the end-to-end audio plumbing
works at the file entry point. Also covers the SANDROID_MAX_AUDIO_BYTES
413 guard.
"""

from __future__ import annotations

import io
import wave
from collections.abc import Iterator

import httpx
import numpy as np
import pytest
from httpx import ASGITransport

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
from sandroid.core.audio import decode_wav
from sandroid.core.orchestrator import Orchestrator
from sandroid.models.asr.stub import StubASR
from sandroid.vad.mock import MockVAD
from tests._helpers.recording_slu import RecordingSLU

AUTH_HEADER = {"X-API-Key": DEV_DEFAULT_API_KEY}


def _silent_wav(duration_ms: int = 500, sample_rate: int = 16_000) -> bytes:
    samples = np.zeros(sample_rate * duration_ms // 1000, dtype=np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())
    return buffer.getvalue()


@pytest.fixture
def recording_slu() -> Iterator[RecordingSLU]:
    """Inject a RecordingSLU into the Orchestrator built by FastAPI Depends."""
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


@pytest.mark.asyncio
async def test_recognize_file_passes_pcm16_to_slu(
    recording_slu: RecordingSLU,
) -> None:
    wav = _silent_wav()
    expected_pcm = decode_wav(wav).pcm16

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recognize/file",
            headers=AUTH_HEADER,
            data={"scene_id": "example_bank"},
            files={"audio": ("hello.wav", wav, "audio/wav")},
        )
    assert resp.status_code == 200, resp.text
    assert len(recording_slu.calls) == 1
    received_pcm, scene_id = recording_slu.calls[0]
    assert received_pcm == expected_pcm
    assert scene_id == "example_bank"


@pytest.mark.asyncio
async def test_recognize_file_rejects_overlong_audio(
    recording_slu: RecordingSLU,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MAX_AUDIO_BYTES_ENV, "2048")  # 2 KiB ceiling
    wav = _silent_wav(duration_ms=500)  # ~16 KiB raw, well over the cap

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recognize/file",
            headers=AUTH_HEADER,
            data={"scene_id": "example_bank"},
            files={"audio": ("big.wav", wav, "audio/wav")},
        )
    assert resp.status_code == 413
    assert recording_slu.calls == []  # SLU never reached
