"""Step 0 MRCPv2 server — validates the dialect, not a production endpoint.

One MRCP TCP connection = one recognition session. Control channel carries
RECOGNIZE + event frames; a sidecar UDP socket receives L16 RTP on a fixed
port. Real SDP negotiation is Step 1's job — here we hard-code the RTP port
so the client simulator and tests can tee up without a signaling round-trip.

Flow per turn:

    1. Client opens TCP, sends RECOGNIZE.
    2. Server replies with 200 IN-PROGRESS.
    3. Client streams RTP L16/8000 PCM to ``rtp_port`` until it decides the
       turn is done (silence / max duration).
    4. Client writes ``Q`` (a single byte) on the TCP channel to signal
       end-of-input. Non-standard, but keeps Step 0 free of SIP BYE / VAD.
    5. Server runs the PCM through the injected ASR backend, emits
       START-OF-INPUT and RECOGNITION-COMPLETE events, closes the TCP.

The StubASR injected by tests makes this deterministic; the same ``handle``
coroutine is later reused behind the real UniMRCP plugin in Step 1 — the
plugin just becomes the "client" for this same server.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import dataclass

from sandroid.adapters.mrcp.messages import (
    MrcpEvent,
    MrcpParseError,
    MrcpRequest,
    MrcpResponse,
    nlsml_transcript,
    parse_request,
    serialize_event,
    serialize_response,
)
from sandroid.adapters.mrcp.rtp import JitterBuffer, l16_be_to_pcm16_le, parse_rtp
from sandroid.models.asr.base import ASRBackend, AudioChunk, PartialTranscript

logger = logging.getLogger(__name__)

_END_OF_INPUT_SENTINEL = b"Q"  # see module docstring


@dataclass
class MrcpServerConfig:
    host: str = "127.0.0.1"
    mrcp_port: int = 1544  # UniMRCP default
    rtp_port: int = 6004
    # Upper bound on a single turn so a stuck client can't leak a socket.
    turn_timeout_s: float = 30.0


async def handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    asr: ASRBackend,
    config: MrcpServerConfig,
) -> None:
    """Serve one MRCPv2 session. See module docstring for the protocol shape."""

    peer = writer.get_extra_info("peername")
    logger.info("mrcp.connect", extra={"peer": peer})
    try:
        request = await _read_request(reader)
    except (MrcpParseError, ConnectionError) as exc:
        logger.warning("mrcp.parse_error", extra={"error": str(exc)})
        writer.close()
        return

    if request.method != "RECOGNIZE":
        await _send(writer, serialize_response(MrcpResponse(
            status_code=405,  # Method Not Allowed — we only do recog in Step 0
            request_id=request.request_id,
            request_state="COMPLETE",
        )))
        writer.close()
        return

    # Ack with IN-PROGRESS immediately; client may start sending RTP now.
    await _send(writer, serialize_response(MrcpResponse(
        status_code=200,
        request_id=request.request_id,
        request_state="IN-PROGRESS",
    )))

    rtp_socket = _bind_rtp_socket(config)
    try:
        pcm_bytes = await asyncio.wait_for(
            _collect_rtp_until_sentinel(reader, rtp_socket),
            timeout=config.turn_timeout_s,
        )
    except TimeoutError:
        logger.warning("mrcp.turn_timeout", extra={"request_id": request.request_id})
        pcm_bytes = b""
    finally:
        rtp_socket.close()

    if not pcm_bytes:
        await _emit_complete(writer, request.request_id, text="", confidence=0.0)
        writer.close()
        return

    # Fire START-OF-INPUT once we have any audio — matches UniMRCP timing.
    await _send(writer, serialize_event(MrcpEvent(
        event_name="START-OF-INPUT",
        request_id=request.request_id,
        request_state="IN-PROGRESS",
    )))

    text, confidence = await _run_asr(asr, pcm_bytes)
    await _emit_complete(writer, request.request_id, text=text, confidence=confidence)
    writer.close()


async def _emit_complete(
    writer: asyncio.StreamWriter,
    request_id: int,
    *,
    text: str,
    confidence: float,
) -> None:
    body = nlsml_transcript(text, confidence)
    headers = {
        "Completion-Cause": "000 success" if text else "001 no-input-timeout",
        "Content-Type": "application/nlsml+xml",
    }
    await _send(writer, serialize_event(MrcpEvent(
        event_name="RECOGNITION-COMPLETE",
        request_id=request_id,
        request_state="COMPLETE",
        headers=headers,
        body=body,
    )))


async def _read_request(reader: asyncio.StreamReader) -> MrcpRequest:
    """Read exactly one MRCPv2 request off the TCP stream."""

    start_line = await reader.readuntil(b"\r\n")
    tokens = start_line.strip().split()
    if len(tokens) < 2 or tokens[0] != b"MRCP/2.0":
        raise MrcpParseError(f"bad start line: {start_line!r}")
    message_length = int(tokens[1])
    remaining = message_length - len(start_line)
    rest = await reader.readexactly(remaining)
    return parse_request(start_line + rest)


async def _send(writer: asyncio.StreamWriter, payload: bytes) -> None:
    writer.write(payload)
    await writer.drain()


def _bind_rtp_socket(config: MrcpServerConfig) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((config.host, config.rtp_port))
    sock.setblocking(False)
    return sock


async def _collect_rtp_until_sentinel(
    reader: asyncio.StreamReader, rtp_socket: socket.socket
) -> bytes:
    """Read RTP until the client writes ``Q`` on the MRCP channel.

    The control-channel read and the RTP drain race; whichever finishes first
    wins. Using asyncio.wait lets us skip per-iteration busy-waits on the UDP
    socket.
    """

    loop = asyncio.get_running_loop()
    jitter = JitterBuffer()
    pcm = bytearray()

    async def drain_rtp() -> None:
        while True:
            data = await loop.sock_recv(rtp_socket, 4096)
            if not data:
                return
            try:
                packet = parse_rtp(data)
            except ValueError:
                continue  # skip malformed frame; logging too noisy at 50 pps
            for payload in jitter.push(packet):
                pcm.extend(l16_be_to_pcm16_le(payload))

    drain_task = asyncio.create_task(drain_rtp())
    sentinel_task = asyncio.create_task(reader.readexactly(1))
    done, pending = await asyncio.wait(
        {drain_task, sentinel_task}, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    # If the sentinel arrived, pull any buffered RTP still in the jitter.
    if sentinel_task in done:
        sentinel_byte = sentinel_task.result()
        if sentinel_byte != _END_OF_INPUT_SENTINEL:
            logger.warning(
                "mrcp.unexpected_sentinel", extra={"byte": repr(sentinel_byte)}
            )
        for payload in jitter.flush():
            pcm.extend(l16_be_to_pcm16_le(payload))
    return bytes(pcm)


async def _run_asr(asr: ASRBackend, pcm16: bytes) -> tuple[str, float]:
    """Drive the backend's streaming ASR with the collected PCM."""

    # 200 ms slices so the backend's pseudo-streaming path still emits partials
    # — we don't forward them out of MRCP (RFC 6787 has no partial event in
    # speechrecog), but we stay on the same code path the WS handler uses.
    slice_bytes = (16_000 // 5) * 2

    async def chunks():  # type: ignore[no-untyped-def]
        for seq, start in enumerate(range(0, len(pcm16), slice_bytes)):
            yield AudioChunk(pcm16=pcm16[start : start + slice_bytes], sequence=seq)

    final: PartialTranscript | None = None
    async for partial in asr.stream(chunks()):
        if partial.is_final:
            final = partial
    if final is None:
        return "", 0.0
    return final.text, 1.0


async def start_server(
    *, asr: ASRBackend, config: MrcpServerConfig | None = None
) -> asyncio.Server:
    """Bind the MRCP TCP listener and return the server.

    Caller owns the server's lifecycle — await ``server.serve_forever`` or
    close it explicitly. Tests typically use it as an async context:

        server = await start_server(asr=StubASR(), config=...)
        try:
            ...
        finally:
            server.close(); await server.wait_closed()
    """

    cfg = config or MrcpServerConfig()

    async def _handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await handle_connection(reader, writer, asr=asr, config=cfg)

    return await asyncio.start_server(_handler, cfg.host, cfg.mrcp_port)
