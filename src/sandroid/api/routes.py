"""Public HTTP routes."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import time
import wave
from collections.abc import AsyncIterator

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from sandroid.api.deps import (
    API_KEY_ENV,
    DEV_DEFAULT_API_KEY,
    get_asr,
    get_orchestrator,
    get_vad,
    require_api_key,
)
from sandroid.api.metrics import (
    recognize_duration_seconds,
    recognize_requests_total,
    ws_active_turns,
)
from sandroid.core.audio import AudioFormatError, decode_wav
from sandroid.core.orchestrator import (
    Orchestrator,
    RecognitionRequest,
    RecognitionResponse,
)
from sandroid.models.asr.base import ASRBackend, AudioChunk
from sandroid.vad.base import SegmentResult, VoiceActivityDetector

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])
logger = logging.getLogger(__name__)


def _scene_label(scene_id: str | None, category_path: str | None) -> str:
    """Bounded-cardinality label for metrics — never the raw path string."""

    if scene_id:
        return scene_id
    if category_path:
        return "path"  # collapse all path-based calls into a single bucket
    return "unknown"


def _observe_recognize_result(
    path: str, scene: str, started_at: float, result: str
) -> None:
    duration = time.monotonic() - started_at
    recognize_requests_total.labels(path=path, scene=scene, result=result).inc()
    recognize_duration_seconds.labels(path=path).observe(duration)
    logger.info(
        "recognize.complete",
        extra={
            "path": path,
            "scene": scene,
            "result": result,
            "duration_ms": round(duration * 1000, 2),
        },
    )


@router.post("/recognize/text", response_model=RecognitionResponse)
async def recognize_text(
    req: RecognitionRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> RecognitionResponse:
    started = time.monotonic()
    scene = _scene_label(req.scene_id, req.category_path)
    if req.scene_id is None and req.category_path is None:
        _observe_recognize_result("text", scene, started, "error")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="either scene_id or category_path is required",
        )
    try:
        response = await orchestrator.recognize(req)
    except KeyError as exc:
        _observe_recognize_result("text", scene, started, "error")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        _observe_recognize_result("text", scene, started, "error")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    _observe_recognize_result("text", scene, started, "ok")
    return response


@router.post("/recognize/file", response_model=RecognitionResponse)
async def recognize_file(
    audio: UploadFile = File(...),
    scene_id: str | None = Form(default=None),
    category_path: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    n_best: int = Form(default=5, ge=1, le=20),
    orchestrator: Orchestrator = Depends(get_orchestrator),
    asr: ASRBackend = Depends(get_asr),
    vad: VoiceActivityDetector = Depends(get_vad),
) -> RecognitionResponse:
    started = time.monotonic()
    scene = _scene_label(scene_id, category_path)
    if scene_id is None and category_path is None:
        _observe_recognize_result("file", scene, started, "error")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="either scene_id or category_path is required",
        )
    raw = await audio.read()
    try:
        buffer = decode_wav(raw)
    except AudioFormatError as exc:
        _observe_recognize_result("file", scene, started, "error")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    segments = vad.detect(buffer.pcm16, buffer.sample_rate)
    if not segments:
        _observe_recognize_result("file", scene, started, "no_speech")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="no speech detected in audio",
        )

    transcript = await asr.transcribe_file(raw)

    req = RecognitionRequest(
        scene_id=scene_id,
        category_path=category_path,
        text=transcript.text,
        session_id=session_id,
        n_best=n_best,
    )
    try:
        response = await orchestrator.recognize(req)
    except KeyError as exc:
        _observe_recognize_result("file", scene, started, "error")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        _observe_recognize_result("file", scene, started, "error")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    _observe_recognize_result("file", scene, started, "ok")
    return response


# Top-level WebSocket router without the HTTP API-key dep (WS auth is manual).
ws_router = APIRouter(prefix="/api/v1")


def _check_ws_api_key(websocket: WebSocket) -> bool:
    expected = os.environ.get(API_KEY_ENV, DEV_DEFAULT_API_KEY)
    provided = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
    return provided is not None and provided == expected


class _TurnContext:
    """Per-turn mutable state owned by `recognize_stream`.

    A new `start` frame replaces the active context; the prior context's task
    is cancelled so the client can barge-in without ceremony.
    """

    __slots__ = (
        "cancelled", "category_path", "metric_recorded", "n_best",
        "queue", "scene_id", "started_at", "task",
    )

    def __init__(
        self,
        scene_id: str | None,
        category_path: str | None,
        n_best: int,
    ) -> None:
        self.scene_id = scene_id
        self.category_path = category_path
        self.n_best = n_best
        # PCM chunks from the client; ``None`` is the end-of-stream sentinel.
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.task: asyncio.Task[str] | None = None
        self.cancelled = False
        self.started_at = time.monotonic()
        self.metric_recorded = False


async def _send_error(
    websocket: WebSocket,
    session_id: str,
    turn_id: int,
    code: str,
    message: str,
    *,
    fatal: bool = False,
) -> None:
    await websocket.send_json(
        {
            "type": "error",
            "code": code,
            "message": message,
            "fatal": fatal,
            "session_id": session_id,
            "turn_id": turn_id,
        }
    )


def _wrap_pcm_as_wav(pcm16: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16 kHz mono PCM16 bytes in a minimal WAV header.

    Paraformer's ``transcribe_file`` parses WAV; this is the cheapest bridge
    from VAD's raw-PCM segments to the ASR backend contract.
    """

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


async def _pump_vad_asr(
    websocket: WebSocket,
    asr: ASRBackend,
    vad: VoiceActivityDetector,
    ctx: _TurnContext,
    session_id: str,
    turn_id: int,
) -> str:
    """VAD-gated pump (Step B, 方案 A).

    Streams PCM chunks through Silero VAD, forwards ``speech_start`` /
    ``speech_end`` events to the client, and runs offline ASR
    (``transcribe_file``) on the first closed segment. Returns the recognized
    text; empty string means no speech was captured before end-of-stream.
    """

    ws_active_turns.inc()
    try:
        return await _pump_vad_asr_inner(
            websocket, asr, vad, ctx, session_id, turn_id
        )
    finally:
        ws_active_turns.dec()


async def _pump_vad_asr_inner(
    websocket: WebSocket,
    asr: ASRBackend,
    vad: VoiceActivityDetector,
    ctx: _TurnContext,
    session_id: str,
    turn_id: int,
) -> str:
    events_q: asyncio.Queue[bytes | None] = asyncio.Queue()
    segments_q: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def fan_out() -> None:
        while True:
            item = await ctx.queue.get()
            await events_q.put(item)
            await segments_q.put(item)
            if item is None:
                return

    async def events_source() -> AsyncIterator[bytes]:
        while True:
            item = await events_q.get()
            if item is None:
                return
            yield item

    async def segments_source() -> AsyncIterator[bytes]:
        while True:
            item = await segments_q.get()
            if item is None:
                return
            yield item

    async def forward_events() -> None:
        async for ev in vad.stream_events(events_source()):
            if ctx.cancelled:
                return
            await websocket.send_json(
                {
                    "type": ev.kind,
                    "at_ms": ev.at_ms,
                    "session_id": session_id,
                    "turn_id": turn_id,
                }
            )

    async def first_segment() -> SegmentResult | None:
        async for seg in vad.stream_segments(segments_source()):
            if ctx.cancelled:
                return None
            return seg
        return None

    fan_task = asyncio.create_task(fan_out())
    events_task = asyncio.create_task(forward_events())
    try:
        seg = await first_segment()
    finally:
        # Drain the fan-out / events pipeline so they don't leak tasks.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await fan_task
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await events_task

    if ctx.cancelled or seg is None or not seg.pcm16:
        return ""
    return await _stream_segment_through_asr(
        websocket, asr, seg, ctx, session_id, turn_id,
    )


async def _stream_segment_through_asr(
    websocket: WebSocket,
    asr: ASRBackend,
    seg: SegmentResult,
    ctx: _TurnContext,
    session_id: str,
    turn_id: int,
) -> str:
    """Feed one VAD segment into the ASR stream and forward partials/final."""

    # Feed the segment PCM into the ASR stream in ~200 ms slices so the
    # pseudo-streaming backend has room to emit mid-segment partials.
    slice_bytes = max(2, (seg.sample_rate // 5) * 2)  # 200 ms of PCM16

    async def segment_chunks() -> AsyncIterator[AudioChunk]:
        pcm = seg.pcm16
        for seq, start in enumerate(range(0, len(pcm), slice_bytes)):
            yield AudioChunk(pcm16=pcm[start : start + slice_bytes], sequence=seq)

    final_text = ""
    try:
        async for partial in asr.stream(segment_chunks()):
            if ctx.cancelled:
                return ""
            await websocket.send_json(
                {
                    "type": "final" if partial.is_final else "partial",
                    "text": partial.text,
                    "start_ms": seg.start_ms,
                    "end_ms": seg.start_ms + partial.end_ms,
                    "session_id": session_id,
                    "turn_id": turn_id,
                }
            )
            if partial.is_final:
                final_text = partial.text
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if ctx.cancelled:
            return ""
        await _send_error(
            websocket, session_id, turn_id,
            "ASR_FAILED", f"ASR backend error: {exc}",
        )
        ctx.cancelled = True
        return ""
    return final_text


async def _finalize_turn(
    websocket: WebSocket,
    orchestrator: Orchestrator,
    ctx: _TurnContext,
    session_id: str,
    turn_id: int,
) -> None:
    """Drain ASR, run the orchestrator, emit `result` or a non-fatal error."""

    assert ctx.task is not None
    scene = _scene_label(ctx.scene_id, ctx.category_path)
    result = "error"
    try:
        try:
            final_text = await ctx.task
        except asyncio.CancelledError:
            result = "cancelled"
            raise

        if ctx.cancelled:
            result = "cancelled"
            return

        if not final_text.strip():
            await _send_error(
                websocket, session_id, turn_id, "NO_SPEECH", "no transcript produced"
            )
            result = "no_speech"
            return

        req = RecognitionRequest(
            scene_id=ctx.scene_id,
            category_path=ctx.category_path,
            text=final_text,
            session_id=session_id,
            n_best=ctx.n_best,
        )
        try:
            response = await orchestrator.recognize(req)
        except KeyError as exc:
            await _send_error(websocket, session_id, turn_id, "UNKNOWN_SCENE", str(exc))
            return
        except ValueError as exc:
            await _send_error(websocket, session_id, turn_id, "INVALID_REQUEST", str(exc))
            return

        payload = response.model_dump()
        payload["session_id"] = session_id
        payload["turn_id"] = turn_id
        await websocket.send_json({"type": "result", **payload})
        result = "ok"
    finally:
        if not ctx.metric_recorded:
            ctx.metric_recorded = True
            if result != "cancelled":
                _observe_recognize_result("stream", scene, ctx.started_at, result)


async def _cancel_turn(ctx: _TurnContext) -> None:
    ctx.cancelled = True
    await ctx.queue.put(None)
    if ctx.task is not None and not ctx.task.done():
        ctx.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await ctx.task


class _StreamSession:
    """Owns a single WS connection's turn state machine."""

    __slots__ = (
        "_asr", "_ctx", "_current_turn_id", "_finalize",
        "_orchestrator", "_session_id", "_vad",
    )

    def __init__(
        self,
        orchestrator: Orchestrator,
        asr: ASRBackend,
        vad: VoiceActivityDetector,
    ) -> None:
        self._orchestrator = orchestrator
        self._asr = asr
        self._vad = vad
        self._session_id: str | None = None
        self._current_turn_id: int = -1
        self._ctx: _TurnContext | None = None
        self._finalize: asyncio.Task[None] | None = None

    async def handle_pcm(self, data: bytes) -> None:
        if self._ctx is None or self._ctx.cancelled:
            return
        await self._ctx.queue.put(data)

    async def handle_stop(self) -> None:
        if self._ctx is None or self._ctx.cancelled:
            return
        await self._ctx.queue.put(None)

    async def handle_start(self, websocket: WebSocket, control: dict) -> bool:
        """Return False if the connection should be closed as fatal."""

        new_session_id = control.get("session_id")
        new_turn_id = control.get("turn_id")
        scene_id = control.get("scene_id")
        category_path = control.get("category_path")
        if not isinstance(new_session_id, str) or not isinstance(new_turn_id, int):
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "BAD_START",
                    "message": "start requires string session_id and int turn_id",
                    "fatal": True,
                }
            )
            return False
        if scene_id is None and category_path is None:
            await _send_error(
                websocket, new_session_id, new_turn_id,
                "BAD_START", "either scene_id or category_path is required", fatal=True,
            )
            return False
        if self._session_id is not None and new_session_id != self._session_id:
            await _send_error(
                websocket, new_session_id, new_turn_id,
                "SESSION_MISMATCH", "session_id changed mid-connection", fatal=True,
            )
            return False
        if new_turn_id <= self._current_turn_id:
            await _send_error(
                websocket, new_session_id, new_turn_id,
                "STALE_TURN",
                f"turn_id {new_turn_id} is not greater than current {self._current_turn_id}",
            )
            return True

        await self._abandon_active()
        self._session_id = new_session_id
        self._current_turn_id = new_turn_id
        ctx = _TurnContext(
            scene_id=scene_id, category_path=category_path,
            n_best=int(control.get("n_best", 5)),
        )
        ctx.task = asyncio.create_task(
            _pump_vad_asr(
                websocket, self._asr, self._vad, ctx, new_session_id, new_turn_id
            )
        )
        self._ctx = ctx
        self._finalize = asyncio.create_task(
            _finalize_turn(websocket, self._orchestrator, ctx, new_session_id, new_turn_id)
        )
        return True

    async def _abandon_active(self) -> None:
        if self._ctx is not None and not self._ctx.cancelled:
            await _cancel_turn(self._ctx)
        if self._finalize is not None and not self._finalize.done():
            self._finalize.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._finalize

    async def close(self) -> None:
        await self._abandon_active()


@ws_router.websocket("/recognize/stream")
async def recognize_stream(
    websocket: WebSocket,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    asr: ASRBackend = Depends(get_asr),
    vad: VoiceActivityDetector = Depends(get_vad),
) -> None:
    """Streaming recognition over WebSocket. See ``docs/ws-protocol.md``."""

    if not _check_ws_api_key(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()

    session = _StreamSession(orchestrator, asr, vad)
    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break
            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                await session.handle_pcm(data)
                continue

            text = message.get("text")
            if text is None:
                continue
            try:
                control = json.loads(text)
            except ValueError:
                continue
            frame_type = control.get("type")
            if frame_type == "start":
                if not await session.handle_start(websocket, control):
                    await websocket.close(code=4422)
                    return
            elif frame_type in {"stop", "end"}:
                await session.handle_stop()
    finally:
        await session.close()
