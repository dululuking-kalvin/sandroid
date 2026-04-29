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
import json
import logging
import os
import struct
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from sandroid.models.asr.base import AudioChunk, PartialTranscript

# The default ASR backend is resolved lazily via sandroid.api.deps.get_asr
# (see _default_asr_factory) so that importing this module does not force
# the ONNX runtime or Paraformer weights to load. Tests and alternate
# entrypoints inject their own factory via BridgeServer(asr_factory=...).

logger = logging.getLogger(__name__)

# Dedicated traffic logger: one structured JSON line per completed recognition
# turn, for offline annotation and retraining data capture. Silent by default
# — _install_traffic_sink (called from _main) wires up the sink when the
# SANDROID_BRIDGE_TRAFFIC_LOG env var is set.
traffic_logger = logging.getLogger("sandroid.mrcp.traffic")
traffic_logger.propagate = False  # keep structured frames out of the main log


def _install_traffic_sink(target: str | None) -> None:
    """Attach a handler to traffic_logger based on env config.

    target = None -> no-op, traffic_logger stays silent.
    target = "stdout" / "stderr" -> stream to systemd journal.
    target = path -> append one JSON line per turn.
    """
    if not target:
        return
    if target in ("stdout", "stderr"):
        import sys  # noqa: PLC0415 — lazy, only when sink is enabled
        handler: logging.Handler = logging.StreamHandler(
            sys.stdout if target == "stdout" else sys.stderr
        )
    else:
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        handler = logging.FileHandler(target, encoding="utf-8")
    # JSON payload already rendered into record.msg; keep the formatter trivial.
    handler.setFormatter(logging.Formatter("%(message)s"))
    traffic_logger.addHandler(handler)
    traffic_logger.setLevel(logging.INFO)


def _emit_traffic(record: dict[str, object]) -> None:
    """Render one turn record as a single JSON line (no-op when no sink attached)."""
    if not traffic_logger.handlers:
        return
    traffic_logger.info(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


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
    # monotonic seconds at START for end-to-end turn duration
    start_monotonic: float = 0.0
    # Path B (SLU) input: mirrors the LPCM stream. ``audio_overflow`` flips
    # if the accumulator would exceed ``max_audio_bytes`` — Path A keeps
    # running so the bridge still emits NLSML.
    audio_accum: bytearray = field(default_factory=bytearray)
    audio_overflow: bool = False
    max_audio_bytes: int = 0  # 0 means uninitialised; set on START


@dataclass
class _HandlerCtx:
    channel: _Channel
    asr_task: asyncio.Task[str] | None = None


# ---- NLSML rendering ----


PLACEHOLDER_INTENT = "PLACEHOLDER_INTENT"


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_nlsml(
    transcript: str,
    confidence: float = 0.9,
    intent_id: str = PLACEHOLDER_INTENT,
) -> bytes:
    """Produce NLSML with a matched intent or placeholder when NLU yields nothing.

    ``intent_id`` defaults to ``PLACEHOLDER_INTENT`` so callers that have no
    matcher (pure ASR mode, or matcher returned empty N-best) still produce
    a schema-stable document.
    """
    conf_str = f"{confidence:.2f}"
    return (
        '<?xml version="1.0"?>\n'
        '<result xmlns="http://www.ietf.org/xml/ns/mrcpv2"\n'
        '        grammar="session:grammar@sandroid">\n'
        f'  <interpretation confidence="{conf_str}">\n'
        f'    <input mode="speech">{_xml_escape(transcript)}</input>\n'
        f'    <instance>{_xml_escape(intent_id)}</instance>\n'
        "  </interpretation>\n"
        "</result>\n"
    ).encode()


# ---- Connection handler ----


def _default_asr_factory() -> object:
    """Resolve the configured ASR backend via the API layer's dep factory.

    Respects SANDROID_ASR_BACKEND ("paraformer" default, "stub" fallback)
    and the same SANDROID_ENV production-gate used by the REST service, so
    the bridge and the HTTP API share one ASR instance-construction path.

    If the api.deps import fails (e.g. FastAPI / onnxruntime not installed
    on the host running only the bridge), silently fall back to StubASR
    so a bare bridge deployment is still usable. Production can force the
    full path by installing the full sandroid runtime dependencies.
    """
    try:
        from sandroid.api.deps import get_asr  # noqa: PLC0415 — lazy
    except ImportError as e:
        logger.warning("api.deps unavailable (%s); using StubASR", e)
        from sandroid.models.asr.stub import StubASR  # noqa: PLC0415

        return StubASR()
    return get_asr()


def _resolve_max_audio_bytes() -> int:
    """Mirror the REST/WS size cap; fall back to a hard-coded 8 MiB if the
    api.deps helper isn't importable on a bare-bridge host."""
    try:
        from sandroid.api.deps import get_max_audio_bytes  # noqa: PLC0415
        return get_max_audio_bytes()
    except ImportError:
        return 8 * 1024 * 1024


class _OrchestratorFactory(Protocol):
    def __call__(self) -> object: ...  # returns something with .recognize()


def _default_orchestrator_factory() -> object | None:
    """Resolve the recognition Orchestrator via api.deps dep factories.

    Mirrors ``_default_asr_factory``: if api.deps or its transitive imports
    (fastapi, tokenizers, scenes directory, NLU model artifacts, ...) are
    unavailable, fall back to None — the bridge will then skip matcher
    invocation and emit NLSML with ``PLACEHOLDER_INTENT`` as it did before.
    """
    try:
        # Lazy imports so bare bridge hosts (no fastapi/onnxruntime) still boot.
        from sandroid.api.deps import (  # noqa: PLC0415
            get_matcher,
            get_registry,
            get_session_store,
        )
        from sandroid.core.orchestrator import Orchestrator  # noqa: PLC0415
    except ImportError as e:
        logger.warning("orchestrator deps unavailable (%s); NLU disabled", e)
        return None

    try:
        return Orchestrator(
            registry=get_registry(),
            sessions=get_session_store(),
            matcher=get_matcher(),
        )
    except Exception as e:  # artefact-missing / scene-loader errors -> placeholder
        logger.warning("orchestrator construction failed (%s); NLU disabled", e)
        return None


class BridgeServer:
    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        asr_factory: _ASRFactory | None = None,
        orchestrator_factory: _OrchestratorFactory | None = None,
        default_scene: str | None = None,
    ) -> None:
        self._path = socket_path
        self._asr_factory = asr_factory or _default_asr_factory
        self._orch_factory = orchestrator_factory or _default_orchestrator_factory
        self._default_scene = default_scene
        self._orchestrator: object | None = None
        self._orchestrator_built = False
        self._server: asyncio.AbstractServer | None = None

    def _get_orchestrator(self) -> object | None:
        """Lazily resolve the orchestrator so import-time costs stay out of boot."""
        if not self._orchestrator_built:
            self._orchestrator = self._orch_factory()
            self._orchestrator_built = True
        return self._orchestrator

    async def start(self) -> asyncio.AbstractServer:
        parent = os.path.dirname(self._path) or "."
        os.makedirs(parent, exist_ok=True)
        # Remove stale socket from a previous run.
        if os.path.exists(self._path):
            with contextlib.suppress(OSError):
                os.unlink(self._path)
        self._server = await asyncio.start_unix_server(  # type: ignore[attr-defined,unused-ignore]
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
        ch.start_monotonic = time.monotonic()
        ch.max_audio_bytes = _resolve_max_audio_bytes()
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
        ch = ctx.channel
        if not ch.started:
            await self._send_error(writer, "UNEXPECTED_FRAME", "AUDIO before START")
            return False
        # Mirror into Path B's buffer; overflow disables further accumulation
        # so memory stays bounded under malformed clients. Path A is unaffected.
        if not ch.audio_overflow:
            if len(ch.audio_accum) + len(payload) > ch.max_audio_bytes:
                ch.audio_overflow = True
                ch.audio_accum = bytearray()
            else:
                ch.audio_accum.extend(payload)
        await ch.audio_queue.put(payload)
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
        eos_t0 = time.monotonic()
        try:
            transcript = await asyncio.wait_for(ctx.asr_task, timeout=10.0)
        except TimeoutError:
            await self._send_error(writer, "ASR_FAILED", "asr timeout")
            return False
        except Exception as e:
            await self._send_error(writer, "ASR_FAILED", str(e))
            return False
        asr_t = time.monotonic()
        intent_id, confidence = await self._match_intent(transcript, ch)
        nlu_t = time.monotonic()
        nlsml = render_nlsml(transcript, confidence=confidence, intent_id=intent_id)
        writer.write(encode_frame(FRAME_RESULT, nlsml))
        await writer.drain()
        logger.info(
            "RESULT sent for %s (%d bytes) intent=%s conf=%.2f",
            ch.channel_id, len(nlsml), intent_id, confidence,
        )
        _emit_traffic(
            {
                "ts": time.time(),
                "channel_id": ch.channel_id,
                "session_id": ch.session_id,
                "scene_id": self._default_scene,
                "transcript": transcript,
                "intent_id": intent_id,
                "confidence": round(confidence, 4),
                "asr_ms": int((asr_t - eos_t0) * 1000),
                "nlu_ms": int((nlu_t - asr_t) * 1000),
                "turn_ms": int((nlu_t - ch.start_monotonic) * 1000)
                if ch.start_monotonic
                else None,
            }
        )
        return True

    async def _match_intent(self, transcript: str, ch: _Channel) -> tuple[str, float]:
        """Route the transcript through the configured Orchestrator.

        Returns (intent_id, confidence). Falls back to (PLACEHOLDER_INTENT, 0.9)
        when the orchestrator isn't available or produces no top match — the
        NLSML contract stays stable in both cases.
        """
        orch = self._get_orchestrator()
        if orch is None or self._default_scene is None:
            return PLACEHOLDER_INTENT, 0.9
        try:
            from sandroid.core.orchestrator import RecognitionRequest  # noqa: PLC0415
            req = RecognitionRequest(
                scene_id=self._default_scene,
                text=transcript,
                audio=None if ch.audio_overflow else bytes(ch.audio_accum),
                session_id=ch.session_id or None,
            )
            resp = await orch.recognize(req)  # type: ignore[attr-defined]
        except Exception as e:  # matcher failures shouldn't kill the turn
            logger.warning("matcher failed (%s); emitting placeholder", e)
            return PLACEHOLDER_INTENT, 0.9
        if resp.top is None:
            return PLACEHOLDER_INTENT, 0.9
        return resp.top.intent_id, resp.top.confidence

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
    backend = os.environ.get("SANDROID_ASR_BACKEND", "paraformer").lower()
    default_scene = os.environ.get("SANDROID_BRIDGE_DEFAULT_SCENE") or None
    traffic_sink = os.environ.get("SANDROID_BRIDGE_TRAFFIC_LOG") or None
    _install_traffic_sink(traffic_sink)
    logger.info(
        "Starting bridge: asr_backend=%s socket=%s default_scene=%s traffic=%s",
        backend, path, default_scene or "(none, NLU disabled)",
        traffic_sink or "(disabled)",
    )
    server = BridgeServer(socket_path=path, default_scene=default_scene)
    srv = await server.start()
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_main())
