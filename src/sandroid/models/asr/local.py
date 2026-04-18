"""Local ASR (Paraformer-small ONNX INT8).

Stub until Phase 5 wires Triton + the FunASR export. The class exists so the
orchestrator can type-check its dependency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sandroid.models.asr.base import AudioChunk, FinalTranscript, PartialTranscript


class LocalASR:
    def __init__(self, *, model_path: str | None = None) -> None:
        self._model_path = model_path

    async def stream(
        self,
        chunks: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[PartialTranscript]:
        del chunks
        raise NotImplementedError("LocalASR.stream lands in Phase 5 (Paraformer ONNX)")
        if False:  # pragma: no cover — satisfies AsyncIterator protocol shape
            yield PartialTranscript(text="", is_final=False, start_ms=0, end_ms=0)

    async def transcribe_file(self, wav_bytes: bytes) -> FinalTranscript:
        del wav_bytes
        raise NotImplementedError("LocalASR.transcribe_file lands in Phase 5")
