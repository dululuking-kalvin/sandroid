"""StubSLU — Path B placeholder used in dev + tests.

Always abstains: emits ``intent_id=None, confidence=0.0``. This is the
correct degenerate behaviour for Fusion (the weighted sum collapses to
Path A only), so deployments without the wav2vec2 model installed run
identically to a Path-A-only system.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sandroid.models.asr.base import AudioChunk
from sandroid.models.slu.base import FinalIntent, PartialIntent


class StubSLU:
    """Deterministic no-op SLU. Drains audio, emits a single abstaining final."""

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

    async def recognize_file(
        self,
        wav_bytes: bytes,
        *,
        scene_id: str | None = None,
    ) -> FinalIntent:
        return FinalIntent(intent_id=None, confidence=0.0, duration_ms=0)
