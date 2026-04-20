"""Silero VAD v5 ONNX wrapper.

Silero v5 takes a fixed 512-sample window at 16 kHz (32 ms), an LSTM
hidden state tensor, and the sample rate scalar; it returns a speech
probability in ``[0, 1]`` plus an updated state tensor that must be fed
back for the next window. State is per-session and must be reset between
calls.

All inference is synchronous CPU ONNX Runtime; a single window takes
~1 ms on our target hardware, so calling from an async handler is fine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import onnxruntime as ort

from sandroid.runtime import intra_op_threads as intra_op_threads_from_env
from sandroid.vad.base import (
    SegmentEvent,
    SegmentResult,
    SpeechSegment,
)
from sandroid.vad.segmenter import SegmentAggregator

# Silero v5 contract — do not change without retesting.
_WINDOW_SAMPLES = 512  # 32 ms at 16 kHz
_STATE_SHAPE = (2, 1, 128)
_FRAME_MS = _WINDOW_SAMPLES * 1000 // 16_000  # 32


class SileroVADError(RuntimeError):
    pass


class SileroVAD:
    """ONNX Runtime-backed Silero VAD implementing ``VoiceActivityDetector``."""

    def __init__(
        self,
        *,
        model_path: Path,
        threshold: float = 0.5,
        intra_op_threads: int | None = None,
    ) -> None:
        if not model_path.exists():
            raise SileroVADError(
                f"silero_vad.onnx not found at {model_path}. "
                "Run `python scripts/fetch_models.py` first.",
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = (
            intra_op_threads if intra_op_threads is not None else intra_op_threads_from_env()
        )
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._threshold = threshold
        self._sr = np.asarray(16_000, dtype=np.int64)

    def _infer_window(self, window: np.ndarray, state: np.ndarray) -> tuple[float, np.ndarray]:
        """Run one 512-sample window through Silero. Returns (prob, new_state)."""

        out, new_state = self._session.run(
            ["output", "stateN"],
            {
                "input": window.reshape(1, -1),
                "state": state,
                "sr": self._sr,
            },
        )
        return float(out[0, 0]), new_state

    @staticmethod
    def _pcm_to_float(pcm16: bytes) -> np.ndarray:
        if len(pcm16) % 2:
            pcm16 = pcm16[:-1]
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        return samples

    def _iter_windows(self, samples: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Split samples into 512-sample windows; pad the tail if needed."""

        total = samples.size
        if total == 0:
            return [], samples
        complete = total // _WINDOW_SAMPLES
        windows = [
            samples[i * _WINDOW_SAMPLES : (i + 1) * _WINDOW_SAMPLES] for i in range(complete)
        ]
        tail = samples[complete * _WINDOW_SAMPLES :]
        if tail.size:
            padded = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)
            padded[: tail.size] = tail
            windows.append(padded)
        return windows, tail

    def detect(self, pcm16: bytes, sample_rate: int) -> list[SpeechSegment]:
        if sample_rate != 16_000:
            raise SileroVADError(
                f"SileroVAD requires 16 kHz input; got {sample_rate}. "
                "Resample via sandroid.core.audio.decode_wav first.",
            )
        samples = self._pcm_to_float(pcm16)
        windows, _ = self._iter_windows(samples)
        if not windows:
            return []

        aggregator = SegmentAggregator(sample_rate=sample_rate, frame_ms=_FRAME_MS)
        state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        frame_bytes_per_window = _WINDOW_SAMPLES * 2
        segments: list[SpeechSegment] = []
        for i, window in enumerate(windows):
            prob, state = self._infer_window(window, state)
            start_byte = i * frame_bytes_per_window
            frame_bytes = pcm16[start_byte : start_byte + frame_bytes_per_window]
            # Pad tail for aggregator accounting when input was short.
            if len(frame_bytes) < frame_bytes_per_window:
                frame_bytes = frame_bytes + b"\x00" * (frame_bytes_per_window - len(frame_bytes))
            out = aggregator.feed(frame_bytes, prob >= self._threshold)
            for seg in out.segments:
                segments.append(SpeechSegment(start_ms=seg.start_ms, end_ms=seg.end_ms))
        for seg in aggregator.flush().segments:
            segments.append(SpeechSegment(start_ms=seg.start_ms, end_ms=seg.end_ms))
        return segments

    async def stream_events(
        self,
        chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[SegmentEvent]:
        async for out in self._drive_stream(chunks):
            for event in out.events:
                yield event

    async def stream_segments(
        self,
        chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[SegmentResult]:
        async for out in self._drive_stream(chunks):
            for seg in out.segments:
                yield seg

    async def _drive_stream(self, chunks: AsyncIterator[bytes]):  # type: ignore[no-untyped-def]
        """Shared backbone for stream_events / stream_segments.

        NOTE: callers typically use only one of the two; running both at
        once requires duplicating the source stream upstream (e.g. via
        ``asyncio.Queue`` fan-out). That is the WebSocket layer's job, not
        the VAD's.
        """

        aggregator = SegmentAggregator(sample_rate=16_000, frame_ms=_FRAME_MS)
        state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        residual = b""
        window_bytes = _WINDOW_SAMPLES * 2
        async for chunk in chunks:
            residual += chunk
            while len(residual) >= window_bytes:
                frame = residual[:window_bytes]
                residual = residual[window_bytes:]
                samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                prob, state = self._infer_window(samples, state)
                yield aggregator.feed(frame, prob >= self._threshold)
        if residual:
            padded = residual + b"\x00" * (window_bytes - len(residual))
            samples = np.frombuffer(padded, dtype=np.int16).astype(np.float32) / 32768.0
            prob, state = self._infer_window(samples, state)
            yield aggregator.feed(padded, prob >= self._threshold)
        yield aggregator.flush()
