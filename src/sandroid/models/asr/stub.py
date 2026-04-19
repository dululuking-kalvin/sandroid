"""Deterministic ASR stub for end-to-end audio tests.

Two use modes:
- Constructed with ``scripted_transcript``: every call returns that text.
  Tests use this to drive the orchestrator through known intents without an
  actual ASR model.
- Constructed bare: returns a digest of the audio bytes (clearly fake, but
  stable) so integration tests can assert "something came back".
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

from sandroid.models.asr.base import AudioChunk, FinalTranscript, PartialTranscript


class StubASR:
    def __init__(self, *, scripted_transcript: str | None = None) -> None:
        self._scripted = scripted_transcript

    async def transcribe_file(self, wav_bytes: bytes) -> FinalTranscript:
        text = self._scripted or f"stub:{hashlib.sha1(wav_bytes).hexdigest()[:8]}"
        return FinalTranscript(text=text, confidence=1.0, duration_ms=0)

    async def stream(
        self,
        chunks: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[PartialTranscript]:
        accumulated = bytearray()
        last_emit_ms = 0
        async for chunk in chunks:
            accumulated += chunk.pcm16
            # Emit a partial every ~chunk; times are coarse, not load-bearing.
            last_emit_ms += (len(chunk.pcm16) // 2) * 1000 // 16000
            yield PartialTranscript(
                text=self._scripted or "",
                is_final=False,
                start_ms=0,
                end_ms=last_emit_ms,
            )
        final_text = self._scripted or f"stub:{hashlib.sha1(bytes(accumulated)).hexdigest()[:8]}"
        yield PartialTranscript(
            text=final_text,
            is_final=True,
            start_ms=0,
            end_ms=last_emit_ms,
        )
