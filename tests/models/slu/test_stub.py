"""Unit tests for StubSLU (Path B placeholder)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from sandroid.models.asr.base import AudioChunk
from sandroid.models.slu.base import FinalIntent, PartialIntent
from sandroid.models.slu.stub import StubSLU


async def _audio_chunks(n: int = 3) -> AsyncIterator[AudioChunk]:
    for i in range(n):
        yield AudioChunk(pcm16=b"\x00\x00" * 320, sequence=i)


async def test_stub_stream_drains_audio_and_abstains() -> None:
    """Stub consumes every chunk and yields exactly one abstaining final."""
    stub = StubSLU()
    finals: list[PartialIntent] = []
    async for partial in stub.stream(_audio_chunks(3)):
        finals.append(partial)
    assert len(finals) == 1
    only = finals[0]
    assert only.is_final is True
    assert only.intent_id is None
    assert only.confidence == 0.0


async def test_stub_stream_with_empty_audio_still_yields_final() -> None:
    """Even with zero audio chunks, stream emits a single abstain final."""
    stub = StubSLU()

    async def empty() -> AsyncIterator[AudioChunk]:
        if False:
            yield  # pragma: no cover — make this a generator

    finals = [p async for p in stub.stream(empty())]
    assert len(finals) == 1
    assert finals[0].intent_id is None
    assert finals[0].confidence == 0.0


async def test_stub_recognize_file_abstains() -> None:
    stub = StubSLU()
    out = await stub.recognize_file(b"")
    assert isinstance(out, FinalIntent)
    assert out.intent_id is None
    assert out.confidence == 0.0
    assert out.duration_ms == 0


async def test_stub_ignores_scene_id() -> None:
    """scene_id is part of the protocol but stub treats every scene the same."""
    stub = StubSLU()
    out_a = await stub.recognize_file(b"", scene_id="example_bank")
    out_b = await stub.recognize_file(b"", scene_id=None)
    assert out_a == out_b


@pytest.mark.parametrize("conf", [-0.1, 1.1])
def test_partial_intent_confidence_must_be_in_unit_range(conf: float) -> None:
    """The Pydantic Field(ge=0, le=1) keeps fusion-layer math sane."""
    with pytest.raises(ValidationError):
        PartialIntent(
            intent_id=None,
            confidence=conf,
            is_final=True,
            start_ms=0,
            end_ms=0,
        )
