"""Integration tests for the offline Paraformer ASR adapter.

Skipped automatically when model artifacts are missing — mirrors the embedder
and Silero test patterns so contributors who haven't run
``scripts/fetch_models.py`` still get a green CI.
"""

from __future__ import annotations

import asyncio
import wave
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from sandroid.models.asr.base import AudioChunk
from sandroid.models.asr.paraformer import ParaformerASR, ParaformerError

_MODELS_DIR = Path(__file__).resolve().parents[3] / "deploy" / "models"
_MODEL_PATH = _MODELS_DIR / "paraformer_zh.int8.onnx"
_TOKENS_PATH = _MODELS_DIR / "paraformer_zh.tokens.txt"
_CMVN_PATH = _MODELS_DIR / "paraformer_zh.am.mvn"
_FIXTURE_WAV = Path(__file__).parent / "fixtures" / "hello_zh.wav"

pytestmark = pytest.mark.skipif(
    not (
        _MODEL_PATH.exists()
        and _TOKENS_PATH.exists()
        and _CMVN_PATH.exists()
        and _FIXTURE_WAV.exists()
    ),
    reason=(f"ASR model artifacts missing under {_MODELS_DIR}; run scripts/fetch_models.py"),
)


@pytest.fixture(scope="module")
def asr() -> ParaformerASR:
    return ParaformerASR(
        model_path=_MODEL_PATH,
        tokens_path=_TOKENS_PATH,
        cmvn_path=_CMVN_PATH,
    )


def _silence_wav(duration_s: float) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(int(16000 * duration_s), dtype=np.int16).tobytes())
    return buf.getvalue()


def test_transcribes_chinese_speech(asr: ParaformerASR) -> None:
    """Real 5.6s Mandarin clip must produce a non-empty Chinese transcript."""

    wav_bytes = _FIXTURE_WAV.read_bytes()
    result = asyncio.run(asr.transcribe_file(wav_bytes))

    assert result.text, "expected non-empty transcript"
    # Every character should be Chinese (CJK Unified Ideographs). The fixture
    # has no English / digits.
    for ch in result.text:
        assert "\u4e00" <= ch <= "\u9fff", f"unexpected non-Chinese char: {ch!r}"
    assert result.confidence == 1.0
    assert 5000 <= result.duration_ms <= 6000


def test_silence_returns_empty_or_short(asr: ParaformerASR) -> None:
    """Half a second of digital silence should not decode into meaningful text.

    Paraformer may emit stray high-freq tokens on silence; we tolerate a few
    characters but assert we didn't hallucinate a sentence.
    """

    result = asyncio.run(asr.transcribe_file(_silence_wav(0.5)))
    assert result.confidence == 1.0
    assert result.duration_ms == 500
    assert len(result.text) <= 3


def test_too_short_returns_empty(asr: ParaformerASR) -> None:
    """Sub-LFR-window audio (< 70 ms of frames) must not crash and must
    return empty text."""

    result = asyncio.run(asr.transcribe_file(_silence_wav(0.01)))
    assert result.text == ""
    assert result.duration_ms == 10


def test_rejects_non_16k_mono_pcm16() -> None:
    """WAV shape mismatches must raise ParaformerError, not crash downstream."""

    # Build an 8 kHz WAV.
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(np.zeros(8000, dtype=np.int16).tobytes())

    asr = ParaformerASR(
        model_path=_MODEL_PATH,
        tokens_path=_TOKENS_PATH,
        cmvn_path=_CMVN_PATH,
    )
    with pytest.raises(ParaformerError, match="16000"):
        asyncio.run(asr.transcribe_file(buf.getvalue()))


def test_missing_model_raises() -> None:
    with pytest.raises(ParaformerError, match="missing"):
        ParaformerASR(
            model_path=Path("/does/not/exist.onnx"),
            tokens_path=_TOKENS_PATH,
            cmvn_path=_CMVN_PATH,
        )


def test_stream_emits_partials_and_final(asr: ParaformerASR) -> None:
    """Pseudo-streaming: chunked PCM in -> partials + one final out."""

    # Strip the 44-byte WAV header; feed raw PCM16 in 200 ms slices.
    wav = _FIXTURE_WAV.read_bytes()
    with wave.open(BytesIO(wav), "rb") as r:
        pcm = r.readframes(r.getnframes())
    slice_bytes = (16000 // 5) * 2  # 200 ms

    async def _chunks() -> AsyncIterator[AudioChunk]:
        for i, start in enumerate(range(0, len(pcm), slice_bytes)):
            yield AudioChunk(pcm16=pcm[start : start + slice_bytes], sequence=i)

    async def _collect() -> list[tuple[bool, str]]:
        out: list[tuple[bool, str]] = []
        async for p in asr.stream(_chunks()):
            out.append((p.is_final, p.text))
        return out

    results = asyncio.run(_collect())
    assert results, "expected at least one frame"
    finals = [r for r in results if r[0]]
    partials = [r for r in results if not r[0]]
    assert len(finals) == 1, "exactly one is_final=True frame per stream"
    assert partials, "expected at least one partial before final"
    final_text = finals[0][1]
    assert final_text, "final transcript must be non-empty for the fixture"
    for ch in final_text:
        assert "\u4e00" <= ch <= "\u9fff", f"non-Chinese char in final: {ch!r}"
