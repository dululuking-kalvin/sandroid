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
    get_asr,
    get_vad,
    reset_dependency_caches,
)
from sandroid.models.asr.stub import StubASR
from sandroid.vad.mock import MockVAD

AUTH_HEADER = {"X-API-Key": DEV_DEFAULT_API_KEY}


@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    reset_dependency_caches()
    # Silent WAVs are used as fixtures; MockVAD accepts them as speech.
    _mock = MockVAD()
    app.dependency_overrides[get_vad] = lambda: _mock
    yield
    app.dependency_overrides.clear()
    reset_dependency_caches()


def _scripted_asr_override(text: str) -> None:
    app.dependency_overrides[get_asr] = lambda: StubASR(scripted_transcript=text)


def _silent_wav(duration_ms: int = 500, sample_rate: int = 16_000) -> bytes:
    samples = np.zeros(sample_rate * duration_ms // 1000, dtype=np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_recognize_file_happy_path() -> None:
    _scripted_asr_override("你好")
    wav = _silent_wav()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recognize/file",
            headers=AUTH_HEADER,
            data={"scene_id": "example_bank"},
            files={"audio": ("hello.wav", wav, "audio/wav")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["top"]["intent_id"] == "greeting"
    assert body["current_category_id"] == "root"


@pytest.mark.asyncio
async def test_recognize_file_rejects_non_wav() -> None:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recognize/file",
            headers=AUTH_HEADER,
            data={"scene_id": "example_bank"},
            files={"audio": ("bogus.wav", b"not a wav", "audio/wav")},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_recognize_file_rejects_empty_audio() -> None:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recognize/file",
            headers=AUTH_HEADER,
            data={"scene_id": "example_bank"},
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_recognize_file_missing_scene_returns_422() -> None:
    wav = _silent_wav()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recognize/file",
            headers=AUTH_HEADER,
            files={"audio": ("hello.wav", wav, "audio/wav")},
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_recognize_file_unauthorized() -> None:
    wav = _silent_wav()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recognize/file",
            data={"scene_id": "example_bank"},
            files={"audio": ("hello.wav", wav, "audio/wav")},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_recognize_file_unknown_scene_404() -> None:
    _scripted_asr_override("你好")
    wav = _silent_wav()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recognize/file",
            headers=AUTH_HEADER,
            data={"scene_id": "ghost_scene"},
            files={"audio": ("hello.wav", wav, "audio/wav")},
        )
    assert resp.status_code == 404
