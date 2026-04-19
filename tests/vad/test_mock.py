from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from sandroid.vad.mock import MockVAD


def test_detect_returns_single_segment_for_one_second() -> None:
    vad = MockVAD()
    pcm = b"\x00\x00" * 16_000  # 1 s @ 16 kHz mono PCM16
    segments = vad.detect(pcm, sample_rate=16_000)
    assert len(segments) == 1
    assert segments[0].start_ms == 0
    assert segments[0].end_ms == 1000


def test_detect_empty_returns_no_segments() -> None:
    vad = MockVAD()
    assert vad.detect(b"", sample_rate=16_000) == []


def test_detect_rejects_bad_sample_rate() -> None:
    vad = MockVAD()
    with pytest.raises(ValueError, match="sample_rate"):
        vad.detect(b"\x00\x00", sample_rate=0)


@pytest.mark.asyncio
async def test_stream_emits_start_then_end() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield b"\x00\x00" * 8_000  # 0.5 s
        yield b"\x00\x00" * 8_000  # 0.5 s

    vad = MockVAD()
    events = [event async for event in vad.stream(chunks())]
    kinds = [e.kind for e in events]
    assert kinds == ["speech_start", "speech_end"]
    assert events[-1].at_ms == 1000


@pytest.mark.asyncio
async def test_stream_empty_emits_nothing() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        if False:  # pragma: no cover
            yield b""

    vad = MockVAD()
    events = [event async for event in vad.stream(chunks())]
    assert events == []
