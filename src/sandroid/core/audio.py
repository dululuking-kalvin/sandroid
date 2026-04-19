"""WAV → 16 kHz mono PCM16 front-end.

Uses only the stdlib (``wave`` + ``audioop``-style helpers implemented in
NumPy) so the audio path has no heavyweight dependencies before Phase 5.
The engine's canonical internal format is 16 kHz mono little-endian PCM16
— we normalize at this boundary so everything downstream can assume it.
"""

from __future__ import annotations

import io
import wave
from typing import Final

import numpy as np
from pydantic import BaseModel

TARGET_SAMPLE_RATE: Final[int] = 16_000
TARGET_CHANNELS: Final[int] = 1
TARGET_SAMPLE_WIDTH: Final[int] = 2  # bytes — PCM16


class AudioFormatError(ValueError):
    """Raised when a WAV payload is malformed or cannot be converted."""


class AudioBuffer(BaseModel):
    """Canonical form handed off to VAD / ASR."""

    pcm16: bytes
    sample_rate: int
    channels: int

    @property
    def duration_ms(self) -> int:
        frame_bytes = self.channels * TARGET_SAMPLE_WIDTH
        frames = len(self.pcm16) // frame_bytes
        return frames * 1000 // self.sample_rate


def decode_wav(data: bytes) -> AudioBuffer:
    """Parse a WAV file and normalize to 16 kHz mono PCM16."""

    if not data:
        raise AudioFormatError("empty audio payload")
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            frames = reader.readframes(reader.getnframes())
    except wave.Error as exc:
        raise AudioFormatError(f"invalid WAV payload: {exc}") from exc

    if sample_width != TARGET_SAMPLE_WIDTH:
        raise AudioFormatError(
            f"unsupported sample width {sample_width * 8}-bit; v0.1 accepts 16-bit PCM WAV only",
        )
    if channels not in (1, 2):
        raise AudioFormatError(f"unsupported channel count {channels}")

    samples = np.frombuffer(frames, dtype=np.int16)
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

    if sample_rate != TARGET_SAMPLE_RATE:
        samples = _resample_linear(samples, sample_rate, TARGET_SAMPLE_RATE)

    return AudioBuffer(
        pcm16=samples.tobytes(),
        sample_rate=TARGET_SAMPLE_RATE,
        channels=TARGET_CHANNELS,
    )


def _resample_linear(
    samples: np.ndarray,
    src_rate: int,
    dst_rate: int,
) -> np.ndarray:
    """Good-enough linear resampler for the front-end.

    We don't care about anti-aliasing fidelity here — Paraformer and wav2vec2
    front-ends will re-window the signal anyway, and the production path runs
    at the native 16 kHz most of the time.
    """

    if src_rate == dst_rate:
        return samples
    if samples.size == 0:
        return samples
    duration = samples.size / src_rate
    dst_count = max(1, round(duration * dst_rate))
    src_positions = np.linspace(0.0, samples.size - 1, num=dst_count)
    floors = np.floor(src_positions).astype(np.int64)
    fracs = src_positions - floors
    highs = np.minimum(floors + 1, samples.size - 1)
    resampled = (
        samples[floors].astype(np.float32) * (1.0 - fracs)
        + samples[highs].astype(np.float32) * fracs
    )
    return resampled.astype(np.int16)
