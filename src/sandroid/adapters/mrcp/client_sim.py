"""Python MRCPv2 client simulator — test fixture for the Step 0 server.

Speaks the same dialect our server expects: sends ``RECOGNIZE``, streams
L16/8000 RTP to the configured UDP port, writes the Step 0 end-of-input
sentinel on the control channel, and parses ``START-OF-INPUT`` +
``RECOGNITION-COMPLETE`` events off the same TCP socket.

Not suitable against a real UniMRCP server — that one uses SDP for RTP port
negotiation; Step 1 is where we replace this fixture with a real integration.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import struct
from dataclasses import dataclass, field

from sandroid.adapters.mrcp.messages import (
    MRCP_VERSION,
    MrcpEvent,
    MrcpParseError,
)

_RTP_HEADER = struct.Struct("!BBHII")  # matches rtp.parse_rtp layout
_DEFAULT_PAYLOAD_TYPE = 96  # dynamic; L16/8000 isn't in the static table


@dataclass
class RecognizeResult:
    start_of_input_seen: bool
    complete: MrcpEvent
    raw_events: list[MrcpEvent] = field(default_factory=list)


async def recognize_once(
    *,
    host: str,
    mrcp_port: int,
    rtp_port: int,
    pcm16_le: bytes,
    ssrc: int = 0x5A5A5A5A,
    packet_ms: int = 20,
) -> RecognizeResult:
    """Drive one recognition turn against the Step 0 server.

    ``pcm16_le`` is little-endian PCM16 @ 16 kHz mono. We swap to L16 BE on
    the wire and chunk into ``packet_ms`` RTP packets. Returns the parsed
    COMPLETE event.
    """

    reader, writer = await asyncio.open_connection(host, mrcp_port)
    try:
        request = _serialize_request(
            method="RECOGNIZE",
            request_id=1,
            headers={
                "Channel-Identifier": "sandroid-sim@speechrecog",
                "Content-Type": "text/uri-list",
                "Content-Length": "23",
            },
            body=b"builtin:dictation\r\n",
        )
        writer.write(request)
        await writer.drain()

        # Wait for 200 IN-PROGRESS so we know the server is ready for RTP.
        _in_progress_line = await reader.readuntil(b"\r\n")
        _in_progress_body = await _drain_message_tail(reader, _in_progress_line)

        # Ship RTP. We don't bother with pacing — the server's UDP socket
        # and our tiny jitter buffer don't care about wall-clock timing.
        _send_rtp(pcm16_le, host=host, rtp_port=rtp_port, ssrc=ssrc, packet_ms=packet_ms)

        # Signal end-of-input via the Step 0 sentinel.
        writer.write(b"Q")
        await writer.drain()

        # Read events until we see COMPLETE.
        events: list[MrcpEvent] = []
        start_seen = False
        complete: MrcpEvent | None = None
        while complete is None:
            line = await reader.readuntil(b"\r\n")
            tail = await _drain_message_tail(reader, line)
            event = _parse_event_frame(line + tail)
            events.append(event)
            if event.event_name == "START-OF-INPUT":
                start_seen = True
            elif event.event_name == "RECOGNITION-COMPLETE":
                complete = event
        return RecognizeResult(start_of_input_seen=start_seen, complete=complete, raw_events=events)
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_request(
    *, method: str, request_id: int, headers: dict[str, str], body: bytes
) -> bytes:
    """Build a MRCPv2 request message with message-length filled in.

    Mirrors the server-side serializer but keeps client code self-contained
    so the fixture doesn't depend on server internals.
    """

    header_block = b"".join(f"{n}: {v}".encode() + b"\r\n" for n, v in headers.items())
    # Build twice to stabilize message-length (see messages._serialize).
    template = b"%s %%d %s %d\r\n" % (MRCP_VERSION, method.encode(), request_id)
    total = len(template % 0 + b"\r\n" + header_block + b"\r\n" + body)
    while True:
        candidate = template % total + b"\r\n" + header_block + b"\r\n" + body
        if len(candidate) == total:
            return candidate
        total = len(candidate)


async def _drain_message_tail(reader: asyncio.StreamReader, start_line: bytes) -> bytes:
    """Given the start line, read the rest of the message."""

    tokens = start_line.strip().split()
    if len(tokens) < 2:
        raise MrcpParseError(f"bad MRCP start line: {start_line!r}")
    message_length = int(tokens[1])
    return await reader.readexactly(message_length - len(start_line))


def _parse_event_frame(raw: bytes) -> MrcpEvent:
    """Parse a server event. Mirrors messages.parse_request with an event shape."""

    head_end = raw.find(b"\r\n\r\n")
    if head_end < 0:
        raise MrcpParseError("event missing header/body separator")
    head = raw[:head_end].decode("utf-8")
    body = raw[head_end + 4 :]
    lines = head.split("\r\n")
    start_tokens = lines[0].split()
    if len(start_tokens) != 5:
        raise MrcpParseError(f"event start line expects 5 tokens, got: {lines[0]!r}")
    _version, _length, event_name, request_id, request_state = start_tokens
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, _, value = line.partition(":")
        headers[name.strip()] = value.strip()
    return MrcpEvent(
        event_name=event_name,
        request_id=int(request_id),
        request_state=request_state,
        headers=headers,
        body=body,
    )


def _send_rtp(
    pcm16_le: bytes,
    *,
    host: str,
    rtp_port: int,
    ssrc: int,
    packet_ms: int,
) -> None:
    """Fire-and-forget RTP packets from a fresh UDP socket."""

    if len(pcm16_le) % 2:
        raise ValueError("pcm16_le length must be even")
    samples_per_packet = (16_000 * packet_ms) // 1000
    bytes_per_packet = samples_per_packet * 2

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        timestamp = 0
        for seq, start in enumerate(range(0, len(pcm16_le), bytes_per_packet)):
            chunk_le = pcm16_le[start : start + bytes_per_packet]
            chunk_be = _swap_bytes(chunk_le)
            header = _RTP_HEADER.pack(
                0b10_0_0_0000,  # V=2, P=0, X=0, CC=0
                _DEFAULT_PAYLOAD_TYPE & 0x7F,
                seq & 0xFFFF,
                timestamp & 0xFFFFFFFF,
                ssrc & 0xFFFFFFFF,
            )
            sock.sendto(header + chunk_be, (host, rtp_port))
            timestamp += samples_per_packet
    finally:
        sock.close()


def _swap_bytes(pcm16_le: bytes) -> bytes:
    """LE → BE byte swap for L16 wire format."""

    return struct.pack(
        f">{len(pcm16_le) // 2}h",
        *struct.unpack(f"<{len(pcm16_le) // 2}h", pcm16_le),
    )


__all__ = ["RecognizeResult", "recognize_once"]
