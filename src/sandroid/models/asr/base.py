"""ASR adapter interface.

Pins the streaming contract before any backend exists. Implementations will
live under ``local.py`` (Paraformer ONNX) and ``cloud.py`` (Aliyun/Tencent/
iFlytek) in Phase 5.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import BaseModel, Field


class PartialTranscript(BaseModel):
    """Emitted during streaming. ``is_final`` flips on endpoint."""

    text: str
    is_final: bool
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


class FinalTranscript(BaseModel):
    """Emitted once the audio is fully consumed."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    duration_ms: int = Field(ge=0)


class AudioChunk(BaseModel):
    """16kHz mono PCM16, little-endian. The front-end guarantees this format."""

    pcm16: bytes
    sequence: int = Field(ge=0)


class ASRBackend(Protocol):
    """Streaming + file transcription contract. Chunks in → partials out.

    ``stream`` is an async-generator function: call it (no ``await``) and
    iterate the returned async iterator. mypy sees the return type below as
    the async-generator shape the implementations yield.
    """

    def stream(
        self,
        chunks: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[PartialTranscript]: ...

    async def transcribe_file(self, wav_bytes: bytes) -> FinalTranscript: ...
