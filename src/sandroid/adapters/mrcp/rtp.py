"""Minimal RTP L16/8000 depacketizer.

Scope (Step 0): extract raw PCM16 samples from RTP packets carrying L16 in
network byte order, handle out-of-order delivery with a small reorder buffer,
and expose the PCM stream as bytes ready for the existing ASR pipeline.

Out of scope: SRTP, CSRC mixing, header extensions, FEC, PCMU/PCMA/Opus. All
deliberately deferred — UniMRCP's default profile uses L16 which matches our
16 kHz mono internal format exactly (modulo 8 vs 16 kHz resample, see below).

RFC 3550 fixed header (12 bytes):

     0                   1                   2                   3
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |V=2|P|X|  CC   |M|     PT      |       sequence number         |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                           timestamp                           |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |           synchronization source (SSRC) identifier            |
    +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
    |            contributing source (CSRC) identifiers             |
    |                             ....                              |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
"""

from __future__ import annotations

import array
import struct
from dataclasses import dataclass, field

_RTP_HEADER = struct.Struct("!BBHII")  # 12 bytes
_RTP_VERSION_MASK = 0b11000000


class RtpParseError(ValueError):
    pass


@dataclass(frozen=True)
class RtpPacket:
    sequence: int
    timestamp: int
    ssrc: int
    payload_type: int
    marker: bool
    payload: bytes


def parse_rtp(raw: bytes) -> RtpPacket:
    """Parse one RTP packet — header + payload (no SRTP / extensions).

    Rejects malformed or non-version-2 packets immediately; CSRC list is
    skipped over but its contents are not exposed (we never mix).
    """

    if len(raw) < _RTP_HEADER.size:
        raise RtpParseError(f"RTP packet too short ({len(raw)}B)")
    byte0, byte1, seq, ts, ssrc = _RTP_HEADER.unpack_from(raw, 0)
    version = (byte0 & _RTP_VERSION_MASK) >> 6
    if version != 2:
        raise RtpParseError(f"unsupported RTP version {version}")
    csrc_count = byte0 & 0b1111
    marker = bool(byte1 & 0b10000000)
    payload_type = byte1 & 0b01111111

    payload_start = _RTP_HEADER.size + csrc_count * 4
    if payload_start > len(raw):
        raise RtpParseError("truncated CSRC list")
    payload = raw[payload_start:]
    return RtpPacket(
        sequence=seq,
        timestamp=ts,
        ssrc=ssrc,
        payload_type=payload_type,
        marker=marker,
        payload=payload,
    )


@dataclass
class JitterBuffer:
    """Tiny reorder window — holds ``capacity`` packets, emits in seq order.

    We keep the window very small (default 8 = 160 ms at 20 ms packets) because
    MRCP-to-ASR is an inside-datacenter hop; multi-second reordering would
    indicate a much worse problem than speech recognition can work around.

    L16 sequence numbers wrap at 2**16. We use a signed-difference compare
    (RFC 1982 serial-number arithmetic) so wrap is handled transparently.
    """

    capacity: int = 8
    _buffered: dict[int, bytes] = field(default_factory=dict)
    _next_seq: int | None = None

    def push(self, packet: RtpPacket) -> list[bytes]:
        """Feed one packet; return the PCM payloads now in order (possibly empty).

        Payloads are returned as raw bytes (L16 big-endian PCM16). Native
        (little-endian on every machine we ship to) is the adapter's
        responsibility when it hands this to the ASR pipeline.
        """

        if self._next_seq is None:
            self._next_seq = packet.sequence
        # Drop packets so old they've already wrapped past us.
        if _seq_less(packet.sequence, self._next_seq):
            return []
        self._buffered[packet.sequence] = packet.payload

        emitted: list[bytes] = []
        while self._next_seq in self._buffered:
            emitted.append(self._buffered.pop(self._next_seq))
            self._next_seq = (self._next_seq + 1) & 0xFFFF

        # Overflow guard: if the buffer grows past capacity, a packet we were
        # waiting on is probably lost — advance past the gap.
        if len(self._buffered) > self.capacity:
            base = self._next_seq or 0
            nearest = min(self._buffered, key=lambda s: _seq_distance(base, s))
            self._next_seq = nearest
            while self._next_seq in self._buffered:
                emitted.append(self._buffered.pop(self._next_seq))
                self._next_seq = (self._next_seq + 1) & 0xFFFF
        return emitted

    def flush(self) -> list[bytes]:
        """Drain anything still buffered, in sequence order — for stream end."""

        if not self._buffered:
            return []
        ordered_keys = sorted(self._buffered, key=lambda s: _seq_distance(self._next_seq or 0, s))
        emitted = [self._buffered[k] for k in ordered_keys]
        self._buffered.clear()
        return emitted


def l16_be_to_pcm16_le(payload: bytes) -> bytes:
    """Convert network-order (big-endian) L16 to little-endian PCM16.

    16 kHz mono assumed at the caller — this routine just does the byte
    swap. Output length equals input length.
    """

    if len(payload) % 2:
        raise RtpParseError(f"L16 payload length must be even, got {len(payload)}")
    # array.byteswap is a C loop — orders of magnitude faster than a Python
    # list comprehension for 20 ms packets (~640 bytes) at 50 pps.
    a = array.array("h")
    a.frombytes(payload)
    a.byteswap()
    return a.tobytes()


def _seq_less(a: int, b: int) -> bool:
    """True iff ``a`` is 'before' ``b`` in 16-bit wrap arithmetic."""

    return ((a - b) & 0xFFFF) > 0x8000


def _seq_distance(base: int, s: int) -> int:
    """Unsigned distance from base → s (wraps forward)."""

    return (s - base) & 0xFFFF
