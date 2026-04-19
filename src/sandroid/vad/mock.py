"""Pass-through VAD — marks the whole input as one speech segment.

Useful for wiring up the audio path before Silero lands; tests use it to
exercise orchestration without depending on a real VAD model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sandroid.vad.base import SegmentEvent, SpeechSegment


class MockVAD:
    def detect(self, pcm16: bytes, sample_rate: int) -> list[SpeechSegment]:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        duration_ms = (len(pcm16) // 2) * 1000 // sample_rate
        if duration_ms == 0:
            return []
        return [SpeechSegment(start_ms=0, end_ms=duration_ms)]

    async def stream(
        self,
        chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[SegmentEvent]:
        sent_start = False
        cursor_ms = 0
        async for chunk in chunks:
            if not sent_start:
                yield SegmentEvent(kind="speech_start", at_ms=0)
                sent_start = True
            cursor_ms += (len(chunk) // 2) * 1000 // 16000
        if sent_start:
            yield SegmentEvent(kind="speech_end", at_ms=cursor_ms)
