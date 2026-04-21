"""Round-trip tests for the MRCPv2 codec."""

from __future__ import annotations

import pytest

from sandroid.adapters.mrcp.messages import (
    MrcpEvent,
    MrcpParseError,
    MrcpResponse,
    nlsml_transcript,
    parse_request,
    serialize_event,
    serialize_response,
)


def test_parse_recognize_request() -> None:
    body = b"builtin:dictation\r\n"
    raw = (
        b"MRCP/2.0 102 RECOGNIZE 1\r\n"
        b"Channel-Identifier: sandroid-sim@speechrecog\r\n"
        b"Content-Type: text/uri-list\r\n"
        b"Content-Length: 19\r\n"
        b"\r\n"
        + body
    )
    req = parse_request(raw)
    assert req.method == "RECOGNIZE"
    assert req.request_id == 1
    assert req.headers["Content-Type"] == "text/uri-list"
    assert req.body == body


def test_parse_request_rejects_wrong_version() -> None:
    raw = b"MRCP/1.0 42 RECOGNIZE 1\r\n\r\n"
    with pytest.raises(MrcpParseError):
        parse_request(raw)


def test_serialize_response_roundtrips_length() -> None:
    resp = MrcpResponse(
        status_code=200,
        request_id=1,
        request_state="IN-PROGRESS",
    )
    wire = serialize_response(resp)
    # message-length in the start line must equal the total wire length.
    declared = int(wire.split(b" ")[1])
    assert declared == len(wire)
    # Idempotent: re-parsing the start line should give back the same numbers.
    assert b"MRCP/2.0" in wire
    assert b"200 1 IN-PROGRESS" in wire


def test_serialize_event_with_body() -> None:
    body = nlsml_transcript("你好", confidence=0.87)
    ev = MrcpEvent(
        event_name="RECOGNITION-COMPLETE",
        request_id=7,
        request_state="COMPLETE",
        headers={
            "Completion-Cause": "000 success",
            "Content-Type": "application/nlsml+xml",
        },
        body=body,
    )
    wire = serialize_event(ev)
    declared = int(wire.split(b" ")[1])
    assert declared == len(wire)
    # Content-Length is derived, not manual.
    assert f"Content-Length: {len(body)}".encode() in wire
    # Chinese text survives serialization.
    assert "你好".encode() in wire


def test_nlsml_confidence_clamped_formatting() -> None:
    # Confidence is emitted with 2 decimals to match UniMRCP's convention.
    body = nlsml_transcript("hi", confidence=0.5)
    assert b'confidence="0.50"' in body
    # Special chars escape
    body2 = nlsml_transcript("5 < 7 & 7 > 5", confidence=1.0)
    assert b"&lt;" in body2 and b"&gt;" in body2 and b"&amp;" in body2
