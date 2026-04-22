"""Python half of the UniMRCP plugin bridge.

Speaks the protocol documented in ``docs/mrcp-bridge-protocol.md``:
5-byte frame header (4B BE length + 1B type) plus CBOR or raw payload.

Architecture: asyncio Unix-domain server. Each accepted connection
represents one MRCP channel. Audio arrives as AUDIO frames, ASR runs
via an injected adapter, and on EOS the final NLSML is sent back as
a single RESULT frame.

v0.1 embeds the transcript plus a placeholder intent so NLSML schema
is finalised from day one; real NLU plugs in at Task 4.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from sandroid.models.asr.base import AudioChunk, PartialTranscript
from sandroid.models.asr.stub import StubASR

logger = logging.getLogger(__name__)

# ---- Frame types (match bridge_client.h) ----
FRAME_START = 0x01
FRAME_AUDIO = 0x02
FRAME_EOS = 0x03
FRAME_STOP = 0x04
FRAME_RESULT = 0x10
FRAME_ERROR = 0x11

MAX_FRAME_SIZE = 1024 * 1024
DEFAULT_SOCKET_PATH = "/var/run/sandroid/bridge.sock"

# ---- Minimal CBOR decoder (matches what the C side encodes) ----


class CBORError(ValueError):
    """Raised when the START payload CBOR is malformed."""


def _cbor_decode(data: bytes) -> tuple[object, int]:
    """Decode one CBOR item. Returns (value, bytes_consumed).

    Supports: unsigned int (major 0), text string (major 3), map (major 5).
    Other major types raise CBORError.
    """
    if not data:
        raise CBORError("empty")
    ib = data[0]
    major = ib >> 5
    ai = ib & 0x1F
    pos = 1
    if ai < 24:
        val = ai
    elif ai == 24:
        val = data[pos]
        pos += 1
    elif ai == 25:
        val = struct.unpack_from(">H", data, pos)[0]
        pos += 2
    elif ai == 26:
        val = struct.unpack_from(">I", data, pos)[0]
        pos += 4
    elif ai == 27:
        val = struct.unpack_from(">Q", data, pos)[0]
        pos += 8
    else:
        raise CBORError(f"unsupported additional info {ai}")

    if major == 0:  # unsigned int
        return val, pos
    if major == 3:  # text string
        end = pos + val
        return data[pos:end].decode("utf-8"), end
    if major == 5:  # map
        out: dict[object, object] = {}
        for _ in range(val):
            k, kn = _cbor_decode(data[pos:])
            pos += kn
            v, vn = _cbor_decode(data[pos:])
            pos += vn
            out[k] = v
        return out, pos
    raise CBORError(f"unsupported major type {major}")


def _cbor_encode_map(m: dict[str, object]) -> bytes:
    """Encode a small map of str->(str|int). Enough for ERROR frames."""
    buf = bytearray()
    n = len(m)
    if n < 24:
        buf.append(0xA0 | n)
    else:
        buf.append(0xB8)
        buf.append(n)
    for k, v in m.items():
        _cbor_encode_str(buf, k)
        if isinstance(v, str):
            _cbor_encode_str(buf, v)
        elif isinstance(v, int):
            _cbor_encode_uint(buf, v)
        else:
            raise TypeError(f"unsupported value type {type(v).__name__}")
    return bytes(buf)


def _cbor_encode_str(buf: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    _cbor_encode_head(buf, 3, len(b))
    buf.extend(b)


def _cbor_encode_uint(buf: bytearray, v: int) -> None:
    _cbor_encode_head(buf, 0, v)


def _cbor_encode_head(buf: bytearray, major: int, val: int) -> None:
    if val < 24:
        buf.append((major << 5) | val)
    elif val <= 0xFF:
        buf.append((major << 5) | 24)
        buf.append(val)
    elif val <= 0xFFFF:
        buf.append((major << 5) | 25)
        buf.extend(struct.pack(">H", val))
    elif val <= 0xFFFFFFFF:
        buf.append((major << 5) | 26)
        buf.extend(struct.pack(">I", val))
    else:
        buf.append((major << 5) | 27)
        buf.extend(struct.pack(">Q", val))


# ---- Framing helpers ----


async def read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one frame. Returns (type, payload). Raises EOFError on clean close."""
    header = await reader.readexactly(5)
    total = struct.unpack(">I", header[:4])[0]
    ftype = header[4]
    if total < 1 or total - 1 > MAX_FRAME_SIZE:
        raise ValueError(f"invalid frame size {total}")
    payload = await reader.readexactly(total - 1) if total > 1 else b""
    return ftype, payload


def encode_frame(ftype: int, payload: bytes) -> bytes:
    total = len(payload) + 1
    return struct.pack(">I", total) + bytes([ftype]) + payload


# ---- ASR adapter Protocol compatible with StubASR / LocalASR ----


class _ASRFactory(Protocol):
    def __call__(self) -> object: ...  # returns something with .stream()


# ---- Per-channel state ----


@dataclass
class _Channel:
    channel_id: str = ""
    session_id: str = ""
    sample_rate: int = 16000
    codec: str = "LPCM"
    started: bool = False
    eos_seen: bool = False
    audio_queue: asyncio.Queue[bytes | None] = field(default_factory=asyncio.Queue)


@dataclass
class _HandlerCtx:
    channel: _Channel
    asr_task: asyncio.Task[str] | None = None


# ---- NLSML rendering ----


def render_nlsml(transcript: str, confidence: float = 0.9) -> bytes:
    """Produce NLSML with a placeholder intent so the schema is final.

    Real NLU replaces ``PLACEHOLDER_INTENT`` and per-intent confidence
    at Task 4 — the shape of the document does not change.
    """
    # escape minimal XML specials in transcript
    safe = (
        transcript.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    conf_str = f"{confidence:.2f}"
    return (
        '<?xml version="1.0"?>\n'
        '<result xmlns="http://www.ietf.org/xml/ns/mrcpv2"\n'
        '        grammar="session:grammar@sandroid">\n'
        f'  <interpretation confidence="{conf_str}">\n'
        f'    <input mode="speech">{safe}</input>\n'
        f'    <instance>PLACEHOLDER_INTENT</instance>\n'
        "  </interpretation>\n"
        "</result>\n"
    ).encode()


# ---- Connection handler ----


class BridgeServer:
    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        asr_factory: _ASRFactory | None = None,
    ) -> None:
        self._path = socket_path
        self._asr_factory = asr_factory or StubASR
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> asyncio.AbstractServer:
        parent = os.path.dirname(self._path) or "."
        os.makedirs(parent, exist_ok=True)
        # Remove stale socket from a previous run.
        if os.path.exists(self._path):
            with contextlib.suppress(OSError):
                os.unlink(self._path)
        self._server = await asyncio.start_unix_server(  # type: ignore[attr-defined]
            self._handle, path=self._path
        )
        logger.info("Bridge listening on %s", self._path)
        return self._server

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        ctx = _HandlerCtx(channel=_Channel())
        try:
            while True:
                try:
                    ftype, payload = await read_frame(reader)
                except (asyncio.IncompleteReadError, EOFError, ConnectionError):
                    logger.debug("Peer closed connection for %s", ctx.channel.channel_id or "?")
                    return
                if not await self._dispatch(ftype, payload, ctx, writer):
                    return
        finally:
            if ctx.asr_task is not None and not ctx.asr_task.done():
                ctx.asr_task.cancel()
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _dispatch(
        self,
        ftype: int,
        payload: bytes,
        ctx: _HandlerCtx,
        writer: asyncio.StreamWriter,
    ) -> bool:
        if ftype == FRAME_START:
            return await self._handle_start(payload, ctx, writer)
        if ftype == FRAME_AUDIO:
            return await self._handle_audio(payload, ctx, writer)
        if ftype == FRAME_EOS:
            return await self._handle_eos(ctx, writer)
        if ftype == FRAME_STOP:
            logger.debug("STOP received for %s", ctx.channel.channel_id)
            return False
        logger.debug("Ignoring unknown frame type 0x%02x", ftype)
        return True

    async def _handle_start(
        self,
        payload: bytes,
        ctx: _HandlerCtx,
        writer: asyncio.StreamWriter,
    ) -> bool:
        try:
            meta, _ = _cbor_decode(payload)
        except CBORError as e:
            await self._send_error(writer, "BAD_START", str(e))
            return False
        if not isinstance(meta, dict):
            await self._send_error(writer, "BAD_START", "not a map")
            return False
        ch = ctx.channel
        ch.channel_id = str(meta.get("channel_id", ""))
        ch.session_id = str(meta.get("session_id", ""))
        sr = meta.get("sample_rate", 16000)
        ch.sample_rate = int(sr) if isinstance(sr, int) else 16000
        ch.codec = str(meta.get("codec", "LPCM"))
        ch.started = True
        logger.info(
            "START channel=%s session=%s sr=%d codec=%s",
            ch.channel_id, ch.session_id, ch.sample_rate, ch.codec,
        )
        ctx.asr_task = asyncio.create_task(self._run_asr(ch))
        return True

    async def _handle_audio(
        self,
        payload: bytes,
        ctx: _HandlerCtx,
        writer: asyncio.StreamWriter,
    ) -> bool:
        if not ctx.channel.started:
            await self._send_error(writer, "UNEXPECTED_FRAME", "AUDIO before START")
            return False
        await ctx.channel.audio_queue.put(payload)
        return True

    async def _handle_eos(
        self,
        ctx: _HandlerCtx,
        writer: asyncio.StreamWriter,
    ) -> bool:
        ch = ctx.channel
        if not ch.started or ch.eos_seen:
            await self._send_error(writer, "UNEXPECTED_FRAME", "bad EOS")
            return False
        ch.eos_seen = True
        await ch.audio_queue.put(None)
        if ctx.asr_task is None:
            await self._send_error(writer, "INTERNAL", "no asr task")
            return False
        try:
            transcript = await asyncio.wait_for(ctx.asr_task, timeout=10.0)
        except TimeoutError:
            await self._send_error(writer, "ASR_FAILED", "asr timeout")
            return False
        except Exception as e:
            await self._send_error(writer, "ASR_FAILED", str(e))
            return False
        nlsml = render_nlsml(transcript)
        writer.write(encode_frame(FRAME_RESULT, nlsml))
        await writer.drain()
        logger.info("RESULT sent for %s (%d bytes)", ch.channel_id, len(nlsml))
        return True

    async def _send_error(
        self,
        writer: asyncio.StreamWriter,
        code: str,
        message: str,
    ) -> None:
        payload = _cbor_encode_map({"code": code, "message": message})
        writer.write(encode_frame(FRAME_ERROR, payload))
        with contextlib.suppress(Exception):
            await writer.drain()

    async def _run_asr(self, ch: _Channel) -> str:
        asr = self._asr_factory()

        async def chunks() -> AsyncIterator[AudioChunk]:
            seq = 0
            while True:
                item = await ch.audio_queue.get()
                if item is None:
                    return
                yield AudioChunk(pcm16=item, sequence=seq)
                seq += 1

        final_text = ""
        stream = asr.stream(chunks())  # type: ignore[attr-defined]
        async for partial in stream:
            assert isinstance(partial, PartialTranscript)
            if partial.is_final:
                final_text = partial.text
        return final_text


async def _main() -> None:
    logging.basicConfig(
        level=os.environ.get("SANDROID_BRIDGE_LOG", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    path = os.environ.get("SANDROID_BRIDGE_SOCK", DEFAULT_SOCKET_PATH)
    server = BridgeServer(socket_path=path)
    srv = await server.start()
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_main())
