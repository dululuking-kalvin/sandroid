"""Unit tests for MRCP bridge wire protocol (framing + CBOR codec)."""

from __future__ import annotations

import asyncio
import struct

import pytest

from sandroid.adapters.mrcp.bridge import (
    FRAME_AUDIO,
    FRAME_START,
    MAX_FRAME_SIZE,
    CBORError,
    _cbor_decode,
    _cbor_encode_map,
    encode_frame,
    read_frame,
    render_nlsml,
)


def test_frame_roundtrip_empty_payload() -> None:
    raw = encode_frame(0x04, b"")
    assert raw == struct.pack(">I", 1) + bytes([0x04])


def test_frame_roundtrip_with_payload() -> None:
    payload = b"\x00\x01\x02hello"
    raw = encode_frame(FRAME_AUDIO, payload)
    total = struct.unpack(">I", raw[:4])[0]
    assert total == len(payload) + 1
    assert raw[4] == FRAME_AUDIO
    assert raw[5:] == payload


async def _reader_for(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


async def test_read_frame_parses_header_and_payload() -> None:
    payload = b"abc"
    reader = await _reader_for(encode_frame(FRAME_START, payload))
    ftype, out = await read_frame(reader)
    assert ftype == FRAME_START
    assert out == payload


async def test_read_frame_rejects_oversize() -> None:
    # total = MAX_FRAME_SIZE + 2 → payload length MAX_FRAME_SIZE+1 > cap
    header = struct.pack(">I", MAX_FRAME_SIZE + 2) + bytes([FRAME_AUDIO])
    reader = await _reader_for(header)
    with pytest.raises(ValueError, match="invalid frame size"):
        await read_frame(reader)


async def test_read_frame_rejects_zero_total() -> None:
    header = struct.pack(">I", 0) + bytes([FRAME_AUDIO])
    reader = await _reader_for(header)
    with pytest.raises(ValueError, match="invalid frame size"):
        await read_frame(reader)


def test_cbor_encode_decode_map_roundtrip() -> None:
    m = {"channel_id": "ch-1", "session_id": "sess-42", "sample_rate": 16000, "codec": "LPCM"}
    enc = _cbor_encode_map(m)
    decoded, n = _cbor_decode(enc)
    assert n == len(enc)
    assert decoded == m


def test_cbor_encode_uint_boundaries() -> None:
    # value 23 (fits in additional info directly)
    m = {"x": 23}
    dec, _ = _cbor_decode(_cbor_encode_map(m))
    assert dec == {"x": 23}
    # value 255 (ai=24, 1-byte)
    dec, _ = _cbor_decode(_cbor_encode_map({"x": 255}))
    assert dec == {"x": 255}
    # value 65535 (ai=25, 2-byte)
    dec, _ = _cbor_decode(_cbor_encode_map({"x": 65535}))
    assert dec == {"x": 65535}
    # value 70000 (ai=26, 4-byte)
    dec, _ = _cbor_decode(_cbor_encode_map({"x": 70000}))
    assert dec == {"x": 70000}


def test_cbor_decode_rejects_empty() -> None:
    with pytest.raises(CBORError):
        _cbor_decode(b"")


def test_cbor_decode_rejects_unsupported_major() -> None:
    # major 1 (negative int) — initial byte 0x20
    with pytest.raises(CBORError):
        _cbor_decode(bytes([0x20]))


def test_nlsml_escapes_xml_specials() -> None:
    out = render_nlsml("a <b> & c").decode()
    assert "&lt;b&gt;" in out
    assert "&amp;" in out
    assert "PLACEHOLDER_INTENT" in out
    assert 'confidence="0.90"' in out


def test_nlsml_custom_confidence() -> None:
    out = render_nlsml("hi", confidence=0.75).decode()
    assert 'confidence="0.75"' in out


def test_nlsml_emits_real_intent_id() -> None:
    out = render_nlsml("hello", confidence=0.42, intent_id="check_balance").decode()
    assert "<instance>check_balance</instance>" in out
    assert "PLACEHOLDER_INTENT" not in out
    assert 'confidence="0.42"' in out


def test_nlsml_escapes_intent_id_xml_specials() -> None:
    out = render_nlsml("t", intent_id="a&<b>").decode()
    assert "<instance>a&amp;&lt;b&gt;</instance>" in out


def test_large_frame_encoding_structure() -> None:
    payload = b"x" * 10_000
    raw = encode_frame(FRAME_AUDIO, payload)
    total = struct.unpack(">I", raw[:4])[0]
    assert total == 10_001
