"""Cloud ASR (Aliyun / Tencent / iFlytek).

Stub until Phase 5. Dialect-heavy scenes will route through this adapter.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

from sandroid.models.asr.base import AudioChunk, FinalTranscript, PartialTranscript

CloudProvider = Literal["aliyun", "tencent", "iflytek"]


class CloudASR:
    def __init__(self, *, provider: CloudProvider, api_key: str) -> None:
        self._provider = provider
        self._api_key = api_key

    async def stream(
        self,
        chunks: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[PartialTranscript]:
        del chunks
        raise NotImplementedError(f"CloudASR[{self._provider}].stream lands in Phase 5")
        if False:  # pragma: no cover — satisfies AsyncIterator protocol shape
            yield PartialTranscript(text="", is_final=False, start_ms=0, end_ms=0)

    async def transcribe_file(self, wav_bytes: bytes) -> FinalTranscript:
        del wav_bytes
        raise NotImplementedError(
            f"CloudASR[{self._provider}].transcribe_file lands in Phase 5",
        )
