"""Voice Activity Detection interface.

Pins the chunks-in → events + utterances out contract so the orchestrator
and API layers can depend on it regardless of backend (Silero, WebRTC,
Mock). Events and utterance audio are delivered on separate channels
(see ``stream_segments``) — event stream stays lightweight for real-time
status delivery, audio stream carries the complete utterance payloads
for ASR.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class SpeechSegment(BaseModel):
    """A contiguous stretch of detected speech — file-mode result."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


class SegmentEvent(BaseModel):
    """Streaming-mode event. ``kind`` marks segment boundaries or silence.

    Lightweight by design: carries only a timestamp so it's safe to emit
    at VAD cadence (~32 ms for Silero) and to forward to WebSocket clients
    as a status signal.
    """

    kind: Literal["speech_start", "speech_end", "silence"]
    at_ms: int = Field(ge=0)


class SegmentResult(BaseModel):
    """A complete utterance emitted once a speech boundary closes.

    Separate from ``SegmentEvent`` so lumpy PCM payloads don't ride the
    event stream. ASR consumes this; UI consumes events.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pcm16: bytes
    sample_rate: int = Field(gt=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class VoiceActivityDetector(Protocol):
    """File detection + dual-channel streaming detection.

    ``stream_events`` and ``stream_segments`` are async-generator factories:
    call them (no ``await``) and iterate the returned async iterator.
    Implementations typically drive a single internal state machine and
    fan-out into both channels; callers can subscribe to either or both.
    """

    def detect(self, pcm16: bytes, sample_rate: int) -> list[SpeechSegment]: ...

    def stream_events(
        self,
        chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[SegmentEvent]: ...

    def stream_segments(
        self,
        chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[SegmentResult]: ...
