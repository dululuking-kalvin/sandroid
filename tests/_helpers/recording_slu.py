"""Test double that records the bytes handed to ``recognize_audio``.

Used by integration tests to prove every entry point (REST file, WS stream,
MRCP bridge) actually plumbs PCM16 through to the SLU adapter. Returns an
abstaining FinalIntent so fusion math is unaffected — we're testing wiring,
not scoring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sandroid.models.asr.base import AudioChunk
from sandroid.models.slu.base import FinalIntent, PartialIntent


class RecordingSLU:
    """Records every ``recognize_audio`` invocation. Always abstains."""

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str | None]] = []

    async def stream(
        self,
        chunks: AsyncIterator[AudioChunk],
        *,
        scene_id: str | None = None,
    ) -> AsyncIterator[PartialIntent]:
        async for _ in chunks:
            pass
        yield PartialIntent(
            intent_id=None,
            confidence=0.0,
            is_final=True,
            start_ms=0,
            end_ms=0,
        )

    async def recognize_audio(
        self,
        pcm16: bytes,
        *,
        scene_id: str | None = None,
    ) -> FinalIntent:
        self.calls.append((pcm16, scene_id))
        return FinalIntent(intent_id=None, confidence=0.0, duration_ms=0)
