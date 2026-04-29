"""Phase 5f-d: MRCP bridge passes per-channel PCM16 to SLU.

Spawns a real BridgeServer on a Unix socket, drives it with framed CBOR
START/AUDIO/EOS frames, and asserts the injected RecordingSLU saw the byte
sequence the client sent. Skipped on Windows — bridge transport is Unix
domain sockets only (matches ``test_bridge_server.py`` skip rule).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from collections.abc import AsyncIterator

import pytest

from sandroid.adapters.mrcp.bridge import (
    FRAME_AUDIO,
    FRAME_EOS,
    FRAME_RESULT,
    FRAME_START,
    BridgeServer,
    _cbor_encode_map,
    encode_frame,
    read_frame,
)
from sandroid.api.deps import (
    MAX_AUDIO_BYTES_ENV,
    get_matcher,
    get_registry,
    get_session_store,
    reset_dependency_caches,
)
from sandroid.core.orchestrator import Orchestrator
from sandroid.models.asr.base import AudioChunk, PartialTranscript
from tests._helpers.recording_slu import RecordingSLU

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="bridge server requires Unix domain sockets",
)


class _FixedASR:
    """Deterministic ASR that emits a fixed final transcript on EOS."""

    def __init__(self, text: str = "hello world") -> None:
        self._text = text

    async def stream(
        self, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[PartialTranscript]:
        async for _ in chunks:
            pass
        yield PartialTranscript(text=self._text, is_final=True, start_ms=0, end_ms=0)


def _orch_factory_with(rec: RecordingSLU):
    def _make() -> Orchestrator:
        return Orchestrator(
            registry=get_registry(),
            sessions=get_session_store(),
            matcher=get_matcher(),
            slu=rec,
        )
    return _make


async def _start_server(rec: RecordingSLU) -> tuple[BridgeServer, str]:
    sock = os.path.join(tempfile.mkdtemp(), "sandroid.sock")
    server = BridgeServer(
        socket_path=sock,
        asr_factory=_FixedASR,
        orchestrator_factory=_orch_factory_with(rec),
        default_scene="example_bank",
    )
    await server.start()
    return server, sock


async def _drive_turn(
    sock: str,
    audio_chunks: list[bytes],
) -> tuple[int, bytes]:
    """Open one channel, send START + AUDIO*N + EOS, return the RESULT frame."""
    reader, writer = await asyncio.open_unix_connection(path=sock)  # type: ignore[attr-defined]
    try:
        start_payload = _cbor_encode_map({
            "channel_id": "c1",
            "session_id": "s1",
            "sample_rate": 16000,
            "codec": "LPCM",
        })
        writer.write(encode_frame(FRAME_START, start_payload))
        for chunk in audio_chunks:
            writer.write(encode_frame(FRAME_AUDIO, chunk))
        writer.write(encode_frame(FRAME_EOS, b""))
        await writer.drain()
        ftype, payload = await asyncio.wait_for(read_frame(reader), timeout=5.0)
        return ftype, payload
    finally:
        writer.close()
        with __import__("contextlib").suppress(Exception):
            await writer.wait_closed()


@pytest.mark.asyncio
async def test_bridge_accumulates_pcm_and_passes_to_slu() -> None:
    reset_dependency_caches()
    rec = RecordingSLU()
    server, sock = await _start_server(rec)
    try:
        chunks = [b"\x01\x00" * 400, b"\x02\x00" * 400]
        expected = b"".join(chunks)

        ftype, _ = await _drive_turn(sock, chunks)
        assert ftype == FRAME_RESULT

        assert len(rec.calls) == 1
        received_pcm, scene_id = rec.calls[0]
        assert received_pcm == expected
        assert scene_id == "example_bank"
    finally:
        if server._server is not None:
            server._server.close()
            await server._server.wait_closed()
        reset_dependency_caches()


@pytest.mark.asyncio
async def test_bridge_overflow_drops_path_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MAX_AUDIO_BYTES_ENV, "1024")
    reset_dependency_caches()
    rec = RecordingSLU()
    server, sock = await _start_server(rec)
    try:
        big = b"\x00\x00" * 2000  # 4 KiB > 1 KiB cap
        ftype, _ = await _drive_turn(sock, [big])

        # Path A still produced NLSML.
        assert ftype == FRAME_RESULT
        # Path B was dropped — Orchestrator skips the SLU call when audio=None.
        assert rec.calls == []
    finally:
        if server._server is not None:
            server._server.close()
            await server._server.wait_closed()
        reset_dependency_caches()
