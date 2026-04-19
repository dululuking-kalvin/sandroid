from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from sandroid.core.audio import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    AudioFormatError,
    decode_wav,
)


def _wav_bytes(
    samples: np.ndarray,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.astype(np.int16).tobytes())
    return buffer.getvalue()


def test_decode_wav_passthrough_16k_mono() -> None:
    samples = np.zeros(16_000, dtype=np.int16)  # 1 s of silence
    data = _wav_bytes(samples, sample_rate=16_000)
    buf = decode_wav(data)
    assert buf.sample_rate == TARGET_SAMPLE_RATE
    assert buf.channels == TARGET_CHANNELS
    assert buf.duration_ms == 1000
    assert len(buf.pcm16) == 16_000 * 2


def test_decode_wav_resamples_8k_to_16k() -> None:
    samples = np.zeros(8_000, dtype=np.int16)  # 1 s @ 8 kHz
    data = _wav_bytes(samples, sample_rate=8_000)
    buf = decode_wav(data)
    assert buf.sample_rate == TARGET_SAMPLE_RATE
    assert buf.duration_ms == 1000
    assert len(buf.pcm16) == 16_000 * 2  # upsampled to 16 kHz


def test_decode_wav_mixes_stereo_to_mono() -> None:
    stereo = np.zeros(16_000 * 2, dtype=np.int16)  # interleaved L/R, 1 s
    data = _wav_bytes(stereo, sample_rate=16_000, channels=2)
    buf = decode_wav(data)
    assert buf.channels == 1
    assert buf.duration_ms == 1000


def test_decode_wav_rejects_empty() -> None:
    with pytest.raises(AudioFormatError, match="empty"):
        decode_wav(b"")


def test_decode_wav_rejects_garbage() -> None:
    with pytest.raises(AudioFormatError, match="invalid WAV"):
        decode_wav(b"not a wav at all, not even close")


def test_decode_wav_rejects_24bit() -> None:
    samples = np.zeros(16_000 * 3, dtype=np.uint8)  # fake payload
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(3)
        writer.setframerate(16_000)
        writer.writeframes(samples.tobytes())
    with pytest.raises(AudioFormatError, match="sample width"):
        decode_wav(buffer.getvalue())
