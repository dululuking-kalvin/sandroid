"""End-to-end sanity check for the Step 0 MRCPv2 server.

Runs ``start_server`` with a scripted StubASR, fires a real RTP stream at it
via the client simulator, and asserts the NLSML body from RECOGNITION-COMPLETE
carries the scripted transcript. No network dependencies beyond loopback.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket

import pytest

from sandroid.adapters.mrcp.client_sim import recognize_once
from sandroid.adapters.mrcp.server import MrcpServerConfig, start_server
from sandroid.models.asr.stub import StubASR


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_recognize_roundtrip_returns_nlsml_with_transcript() -> None:
    mrcp_port = _free_tcp_port()
    rtp_port = _free_udp_port()
    config = MrcpServerConfig(
        host="127.0.0.1",
        mrcp_port=mrcp_port,
        rtp_port=rtp_port,
        turn_timeout_s=5.0,
    )
    asr = StubASR(scripted_transcript="你好")
    server = await start_server(asr=asr, config=config)
    serve_task = asyncio.create_task(server.serve_forever())
    try:
        # 500 ms of silence at 16 kHz mono PCM16 LE = 16000 bytes.
        pcm16_le = b"\x00\x00" * 8000
        result = await asyncio.wait_for(
            recognize_once(
                host="127.0.0.1",
                mrcp_port=mrcp_port,
                rtp_port=rtp_port,
                pcm16_le=pcm16_le,
            ),
            timeout=5.0,
        )
        assert result.start_of_input_seen is True
        assert result.complete.event_name == "RECOGNITION-COMPLETE"
        assert result.complete.request_state == "COMPLETE"
        assert "你好".encode() in result.complete.body
        assert b"application/nlsml+xml" in serialize_headers(result.complete.headers)
    finally:
        server.close()
        await server.wait_closed()
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task


def serialize_headers(headers: dict[str, str]) -> bytes:
    return b"\r\n".join(f"{k}: {v}".encode() for k, v in headers.items())
