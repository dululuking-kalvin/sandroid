"""MRCPv2 message codec — minimal subset sufficient for RECOGNIZE turnaround.

Scope (Step 0, strategy δ validation):
- Parse: client RECOGNIZE request (speechrecog resource).
- Serialize: server IN-PROGRESS / COMPLETE responses; START-OF-INPUT and
  RECOGNITION-COMPLETE events with NLSML body.

Out of scope: SPEAK / SYNTH, DTMF, SET-PARAMS (accepted but treated as no-op),
grammar bodies beyond ``builtin:dictation`` / ``session:*`` URIs. RFC 6787 is
the reference; we deliberately implement the dialect subset UniMRCP emits by
default rather than the full standard.

Wire format (text, CRLF line endings):

    MRCP/2.0 <message-length> <method|status-code> <request-id>\r\n
    Header-Name: value\r\n
    Header-Name: value\r\n
    \r\n
    <body — Content-Length bytes, or empty>

Events add a ``<request-state>`` token on the start line, e.g.::

    MRCP/2.0 112 RECOGNITION-COMPLETE 1 COMPLETE\r\n
"""

from __future__ import annotations

from dataclasses import dataclass, field

MRCP_VERSION = b"MRCP/2.0"
CRLF = b"\r\n"


class MrcpParseError(ValueError):
    """Raised when a byte stream is not a well-formed MRCPv2 message."""


@dataclass(frozen=True)
class MrcpRequest:
    method: str
    request_id: int
    headers: dict[str, str]
    body: bytes = b""


@dataclass(frozen=True)
class MrcpResponse:
    status_code: int
    request_id: int
    request_state: str  # IN-PROGRESS | COMPLETE | PENDING
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


@dataclass(frozen=True)
class MrcpEvent:
    event_name: str
    request_id: int
    request_state: str  # IN-PROGRESS | COMPLETE
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


def parse_request(raw: bytes) -> MrcpRequest:
    """Parse one MRCPv2 request message.

    ``raw`` must be the full message (start line + headers + body). Callers
    that read from a socket should first consume the start line to learn
    ``message-length``, then read exactly that many bytes and hand them here.
    """

    start_line, headers, body = _split_message(raw)
    tokens = start_line.split()
    if len(tokens) != 4:
        raise MrcpParseError(
            f"request start line needs 4 tokens, got {len(tokens)}: {start_line!r}"
        )
    version, _length, method, req_id = tokens
    if version != MRCP_VERSION.decode():
        raise MrcpParseError(f"unsupported MRCP version: {version!r}")
    try:
        request_id = int(req_id)
    except ValueError as exc:
        raise MrcpParseError(f"bad request-id {req_id!r}") from exc
    return MrcpRequest(
        method=method,
        request_id=request_id,
        headers=headers,
        body=body,
    )


def serialize_response(resp: MrcpResponse) -> bytes:
    """Serialize a response message — computes Content-Length + message-length."""

    return _serialize(
        third_token=str(resp.status_code),
        fourth_token=f"{resp.request_id} {resp.request_state}",
        headers=resp.headers,
        body=resp.body,
    )


def serialize_event(ev: MrcpEvent) -> bytes:
    """Serialize an event message."""

    return _serialize(
        third_token=ev.event_name,
        fourth_token=f"{ev.request_id} {ev.request_state}",
        headers=ev.headers,
        body=ev.body,
    )


def nlsml_transcript(text: str, confidence: float) -> bytes:
    """Build a minimal NLSML result body carrying just transcript + confidence.

    Matches what UniMRCP's speechrecog plugin produces when NLU is off: a
    single ``<interpretation>`` with ``<instance>`` = raw text, plus
    ``confidence`` on the interpretation element (0..1 per RFC 6787 §6.3.1).
    """

    safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    xml = (
        '<?xml version="1.0"?>\n'
        '<result xmlns="urn:ietf:params:xml:ns:nlsml:1.0" grammar="session:any">\n'
        f'  <interpretation confidence="{confidence:.2f}">\n'
        f"    <instance>{safe_text}</instance>\n"
        "  </interpretation>\n"
        "</result>\n"
    )
    return xml.encode("utf-8")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _serialize(
    *,
    third_token: str,
    fourth_token: str,
    headers: dict[str, str],
    body: bytes,
) -> bytes:
    """Assemble a message; fills Content-Length + message-length.

    MRCPv2's start line carries ``message-length`` — total bytes of the whole
    message including the start line itself. We assemble once with a
    placeholder length, compute the real length, and rewrite. Cheaper than
    double-buffering headers.
    """

    merged_headers = dict(headers)
    if body:
        merged_headers.setdefault("Content-Length", str(len(body)))

    header_block = b"".join(
        f"{name}: {value}".encode() + CRLF for name, value in merged_headers.items()
    )

    # Placeholder length; recomputed below.
    start_line_template = b"%s %%d %s %s" % (
        MRCP_VERSION,
        third_token.encode(),
        fourth_token.encode(),
    )
    provisional = start_line_template % 0 + CRLF + header_block + CRLF + body
    total_length = len(provisional)
    # Length field is ASCII decimal; widening the number may shift the total
    # by 1-2 bytes, so iterate until it's stable (usually 1 or 2 passes).
    while True:
        candidate = start_line_template % total_length + CRLF + header_block + CRLF + body
        if len(candidate) == total_length:
            return candidate
        total_length = len(candidate)


def _split_message(raw: bytes) -> tuple[str, dict[str, str], bytes]:
    """Return (start_line, headers, body) given a complete message."""

    head_end = raw.find(CRLF + CRLF)
    if head_end < 0:
        raise MrcpParseError("missing header/body separator (CRLF CRLF)")
    head = raw[:head_end].decode("utf-8", errors="strict")
    body = raw[head_end + 4 :]

    lines = head.split("\r\n")
    if not lines:
        raise MrcpParseError("empty message head")
    start_line = lines[0]
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep:
            raise MrcpParseError(f"malformed header line: {line!r}")
        headers[name.strip()] = value.strip()
    return start_line, headers, body
