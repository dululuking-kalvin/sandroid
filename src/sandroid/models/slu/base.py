"""End-to-end SLU adapter interface (Phase 5f Path B).

CLAUDE.md's dual-path architecture has Path A (ASR -> NLU) and Path B
(end-to-end PCM -> intent). This module pins Path B's streaming
contract before any real backend exists; ``stub.py`` ships an abstaining
implementation, and Phase 5f-c will land a wav2vec2-base-zh ONNX backend.

Design notes:

- Inputs reuse ``AudioChunk`` from ``models.asr.base`` so the same audio
  buffer can fan out to ASR (Path A) and SLU (Path B) without copying.
- Outputs are top-1 only (``intent_id`` + ``confidence``). N-best from
  Path B can't be merged cleanly with Path A's N-best (the candidate
  intent sets may diverge), so Fusion in Phase 5f-b consumes a single
  hypothesis per path.
- ``intent_id`` is ``None`` when the backend abstains (no hypothesis
  above its internal floor). This makes ``StubSLU`` a no-op contributor
  to Fusion: ``score = w_A * conf_A + w_B * 0.0`` collapses to Path A.
- ``scene_id`` is optional so adapters with a fully-general output head
  (no scene-conditioned classification) can ignore it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import BaseModel, Field

from sandroid.core.domain import IntentId
from sandroid.models.asr.base import AudioChunk


class PartialIntent(BaseModel):
    """Streaming hypothesis. ``is_final`` flips on EOS or hard endpoint."""

    intent_id: IntentId | None
    confidence: float = Field(ge=0.0, le=1.0)
    is_final: bool
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


class FinalIntent(BaseModel):
    """Emitted once the audio is fully consumed."""

    intent_id: IntentId | None
    confidence: float = Field(ge=0.0, le=1.0)
    duration_ms: int = Field(ge=0)


class SLUBackend(Protocol):
    """Streaming + file end-to-end SLU contract. PCM in, intent out.

    ``stream`` is an async-generator function: call it (no ``await``)
    and iterate the returned async iterator — same shape as
    ``ASRBackend.stream``.
    """

    def stream(
        self,
        chunks: AsyncIterator[AudioChunk],
        *,
        scene_id: str | None = None,
    ) -> AsyncIterator[PartialIntent]: ...

    async def recognize_file(
        self,
        wav_bytes: bytes,
        *,
        scene_id: str | None = None,
    ) -> FinalIntent: ...
