"""End-to-end test of BridgeServer over a real Unix socket with StubASR."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from sandroid.adapters.mrcp.bridge import (
    FRAME_AUDIO,
    FRAME_EOS,
    FRAME_ERROR,
    FRAME_RESULT,
    FRAME_START,
    BridgeServer,
    _cbor_encode_map,
    encode_frame,
    read_frame,
)
from sandroid.models.asr.base import AudioChunk, PartialTranscript

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


async def _start_server(path: str) -> BridgeServer:
    server = BridgeServer(socket_path=path, asr_factory=_FixedASR)
    await server.start()
    return server


async def _connect(path: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await asyncio.open_unix_connection(path=path)  # type: ignore[attr-defined]


async def test_full_recognition_flow() -> None:
    with tempfile.TemporaryDirectory() as d:
        sock = str(Path(d) / "bridge.sock")
        server = await _start_server(sock)
        try:
            reader, writer = await _connect(sock)
            start_payload = _cbor_encode_map(
                {
                    "channel_id": "ch-1",
                    "session_id": "sess-1",
                    "sample_rate": 16000,
                    "codec": "LPCM",
                }
            )
            writer.write(encode_frame(FRAME_START, start_payload))
            # 3 fake PCM chunks
            for _ in range(3):
                writer.write(encode_frame(FRAME_AUDIO, b"\x00\x00" * 320))
            writer.write(encode_frame(FRAME_EOS, b""))
            await writer.drain()

            ftype, payload = await asyncio.wait_for(read_frame(reader), timeout=5.0)
            assert ftype == FRAME_RESULT
            body = payload.decode()
            assert "hello world" in body
            assert "PLACEHOLDER_INTENT" in body
            writer.close()
            await writer.wait_closed()
        finally:
            if server._server is not None:
                server._server.close()
                await server._server.wait_closed()


async def test_audio_before_start_triggers_error() -> None:
    with tempfile.TemporaryDirectory() as d:
        sock = str(Path(d) / "bridge.sock")
        server = await _start_server(sock)
        try:
            reader, writer = await _connect(sock)
            writer.write(encode_frame(FRAME_AUDIO, b"\x00\x00"))
            await writer.drain()
            ftype, payload = await asyncio.wait_for(read_frame(reader), timeout=5.0)
            assert ftype == FRAME_ERROR
            assert b"UNEXPECTED_FRAME" in payload
            writer.close()
            await writer.wait_closed()
        finally:
            if server._server is not None:
                server._server.close()
                await server._server.wait_closed()


async def test_malformed_start_payload_triggers_error() -> None:
    with tempfile.TemporaryDirectory() as d:
        sock = str(Path(d) / "bridge.sock")
        server = await _start_server(sock)
        try:
            reader, writer = await _connect(sock)
            # Send START with garbage (not CBOR map)
            writer.write(encode_frame(FRAME_START, b"\xff\xff\xff"))
            await writer.drain()
            ftype, payload = await asyncio.wait_for(read_frame(reader), timeout=5.0)
            assert ftype == FRAME_ERROR
            assert b"BAD_START" in payload
            writer.close()
            await writer.wait_closed()
        finally:
            if server._server is not None:
                server._server.close()
                await server._server.wait_closed()


