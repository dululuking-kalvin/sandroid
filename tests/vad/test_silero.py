"""Integration tests for the real Silero ONNX runtime.

Skipped automatically when ``deploy/models/silero_vad.onnx`` hasn't been
fetched — keeps CI green for contributors who haven't run
``scripts/fetch_models.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from sandroid.vad.silero import SileroVAD

MODEL_PATH = Path(__file__).resolve().parents[2] / "deploy" / "models" / "silero_vad.onnx"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"silero_vad.onnx missing at {MODEL_PATH}; run scripts/fetch_models.py",
)


def _silence(duration_ms: int) -> bytes:
    n = 16_000 * duration_ms // 1000
    return b"\x00\x00" * n


def _tone(duration_ms: int, hz: float = 220.0, amp: float = 0.4) -> bytes:
    """Low-frequency tone — not real speech, but far from silence and
    enough for Silero to emit non-zero probabilities. Used as a smoke
    signal, not as a speech-quality ground truth."""

    n = 16_000 * duration_ms // 1000
    t = np.arange(n, dtype=np.float32) / 16_000.0
    samples = (np.sin(2 * np.pi * hz * t) * amp * 32767).astype(np.int16)
    return samples.tobytes()


def test_detect_on_silence_returns_no_segments() -> None:
    vad = SileroVAD(model_path=MODEL_PATH)
    segments = vad.detect(_silence(1000), sample_rate=16_000)
    assert segments == []


def test_detect_rejects_non_16k() -> None:
    vad = SileroVAD(model_path=MODEL_PATH)
    with pytest.raises(Exception, match="16 kHz"):
        vad.detect(_silence(100), sample_rate=8_000)


def test_detect_empty_input_returns_no_segments() -> None:
    vad = SileroVAD(model_path=MODEL_PATH)
    assert vad.detect(b"", sample_rate=16_000) == []


@pytest.mark.asyncio
async def test_stream_events_on_silence_is_quiet() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(5):
            yield _silence(200)

    vad = SileroVAD(model_path=MODEL_PATH)
    events = [e async for e in vad.stream_events(chunks())]
    # With pure silence we expect zero speech_start events; any speech_end
    # without a matching start would be a segmenter bug.
    kinds = [e.kind for e in events]
    assert "speech_start" not in kinds


@pytest.mark.asyncio
async def test_stream_segments_on_silence_yields_nothing() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield _silence(800)

    vad = SileroVAD(model_path=MODEL_PATH)
    segments = [s async for s in vad.stream_segments(chunks())]
    assert segments == []


def test_detect_on_loud_tone_may_fire_but_does_not_crash() -> None:
    """Smoke test: a sustained tone won't necessarily classify as speech
    (Silero is trained on voice), but the pipeline must run without
    errors and return a well-formed list."""

    vad = SileroVAD(model_path=MODEL_PATH)
    result = vad.detect(_tone(1000), sample_rate=16_000)
    for seg in result:
        assert seg.end_ms >= seg.start_ms
