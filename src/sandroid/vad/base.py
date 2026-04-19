"""Voice Activity Detection interface.

The Protocol pins the chunks-in / segments-out contract so the orchestrator
and API layers can depend on it before Silero (or any other implementation)
lands in Phase 5.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class SpeechSegment(BaseModel):
    """A contiguous stretch of detected speech."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


class SegmentEvent(BaseModel):
    """Streaming-mode event. ``kind`` marks segment boundaries or silence."""

    kind: Literal["speech_start", "speech_end", "silence"]
    at_ms: int = Field(ge=0)


class VoiceActivityDetector(Protocol):
    """Synchronous file detection + async streaming detection.

    ``stream`` is an async-generator function — call it (no ``await``) and
    iterate the returned async iterator.
    """

    def detect(self, pcm16: bytes, sample_rate: int) -> list[SpeechSegment]: ...

    def stream(
        self,
        chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[SegmentEvent]: ...
