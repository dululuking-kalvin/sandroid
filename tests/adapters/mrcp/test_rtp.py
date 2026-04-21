"""RTP depacketizer / jitter buffer unit tests."""

from __future__ import annotations

import struct

import pytest

from sandroid.adapters.mrcp.rtp import (
    JitterBuffer,
    RtpPacket,
    RtpParseError,
    l16_be_to_pcm16_le,
    parse_rtp,
)

_HEADER = struct.Struct("!BBHII")


def _build_packet(seq: int, ts: int, payload: bytes, payload_type: int = 96) -> bytes:
    return _HEADER.pack(
        0b10_0_0_0000, payload_type & 0x7F, seq & 0xFFFF, ts & 0xFFFFFFFF, 0xDEADBEEF
    ) + payload


def test_parse_rtp_extracts_payload() -> None:
    raw = _build_packet(seq=42, ts=1000, payload=b"\x12\x34\x56\x78")
    pkt = parse_rtp(raw)
    assert pkt.sequence == 42
    assert pkt.timestamp == 1000
    assert pkt.ssrc == 0xDEADBEEF
    assert pkt.payload == b"\x12\x34\x56\x78"
    assert pkt.payload_type == 96


def test_parse_rtp_rejects_wrong_version() -> None:
    # V=1, P=0, X=0, CC=0
    bad = bytes([0b01_0_0_0000]) + _HEADER.pack(
        0, 0, 0, 0, 0
    )[1:]
    with pytest.raises(RtpParseError):
        parse_rtp(bad)


def test_parse_rtp_rejects_short_packet() -> None:
    with pytest.raises(RtpParseError):
        parse_rtp(b"\x80\x60")


def test_l16_be_to_pcm16_le_swaps_pairs() -> None:
    be = b"\x00\x01\xFF\xFE"  # BE: 1, -2
    le = l16_be_to_pcm16_le(be)
    assert le == b"\x01\x00\xFE\xFF"


def test_l16_be_rejects_odd_length() -> None:
    with pytest.raises(RtpParseError):
        l16_be_to_pcm16_le(b"\x00\x01\x02")


def test_jitter_buffer_in_order() -> None:
    jb = JitterBuffer()
    assert jb.push(_pkt(1, b"a")) == [b"a"]
    assert jb.push(_pkt(2, b"b")) == [b"b"]
    assert jb.push(_pkt(3, b"c")) == [b"c"]


def test_jitter_buffer_holds_then_releases_ordered() -> None:
    jb = JitterBuffer()
    assert jb.push(_pkt(10, b"a")) == [b"a"]
    # Out-of-order: 12 arrives before 11 — hold 12, then 11 unlocks both.
    assert jb.push(_pkt(12, b"c")) == []
    assert jb.push(_pkt(11, b"b")) == [b"b", b"c"]


def test_jitter_buffer_overflow_advances_past_gap() -> None:
    jb = JitterBuffer(capacity=2)
    jb.push(_pkt(100, b"a"))  # establishes next=101
    # Stage three out-of-order future packets — capacity=2 means we must give up.
    jb.push(_pkt(103, b"d"))
    jb.push(_pkt(104, b"e"))
    emitted = jb.push(_pkt(105, b"f"))
    # Lost 101/102; the buffer advances and emits what's queued in order.
    assert emitted == [b"d", b"e", b"f"]


def test_jitter_buffer_flush_drains_remaining() -> None:
    jb = JitterBuffer()
    jb.push(_pkt(1, b"a"))  # emitted
    jb.push(_pkt(3, b"c"))  # held, waiting for 2
    assert jb.flush() == [b"c"]


def _pkt(seq: int, payload: bytes) -> RtpPacket:
    return RtpPacket(
        sequence=seq, timestamp=seq * 160, ssrc=0,
        payload_type=96, marker=False, payload=payload,
    )
