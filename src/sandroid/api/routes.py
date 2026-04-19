"""Public HTTP routes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
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
from sandroid.core.audio import AudioFormatError, decode_wav
from sandroid.core.orchestrator import (
    Orchestrator,
    RecognitionRequest,
    RecognitionResponse,
)
from sandroid.models.asr.base import ASRBackend, AudioChunk
from sandroid.vad.base import VoiceActivityDetector

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


@router.post("/recognize/text", response_model=RecognitionResponse)
async def recognize_text(
    req: RecognitionRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> RecognitionResponse:
    if req.scene_id is None and req.category_path is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="either scene_id or category_path is required",
        )
    try:
        return await orchestrator.recognize(req)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


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
    if scene_id is None and category_path is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="either scene_id or category_path is required",
        )
    raw = await audio.read()
    try:
        buffer = decode_wav(raw)
    except AudioFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    segments = vad.detect(buffer.pcm16, buffer.sample_rate)
    if not segments:
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
        return await orchestrator.recognize(req)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


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

    __slots__ = ("cancelled", "category_path", "n_best", "queue", "scene_id", "task")

    def __init__(
        self,
        scene_id: str | None,
        category_path: str | None,
        n_best: int,
    ) -> None:
        self.scene_id = scene_id
        self.category_path = category_path
        self.n_best = n_best
        self.queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        self.task: asyncio.Task[str] | None = None
        self.cancelled = False


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


async def _pump_asr(
    websocket: WebSocket,
    asr: ASRBackend,
    ctx: _TurnContext,
    session_id: str,
    turn_id: int,
) -> str:
    async def source() -> AsyncIterator[AudioChunk]:
        while True:
            item = await ctx.queue.get()
            if item is None:
                return
            yield item

    final_text = ""
    async for partial in asr.stream(source()):
        if ctx.cancelled:
            break
        await websocket.send_json(
            {
                "type": "final" if partial.is_final else "partial",
                "text": partial.text,
                "start_ms": partial.start_ms,
                "end_ms": partial.end_ms,
                "session_id": session_id,
                "turn_id": turn_id,
            }
        )
        if partial.is_final:
            final_text = partial.text
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
    try:
        final_text = await ctx.task
    except asyncio.CancelledError:
        return

    if ctx.cancelled:
        return

    if not final_text.strip():
        await _send_error(websocket, session_id, turn_id, "NO_SPEECH", "no transcript produced")
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


async def _cancel_turn(ctx: _TurnContext) -> None:
    ctx.cancelled = True
    await ctx.queue.put(None)
    if ctx.task is not None and not ctx.task.done():
        ctx.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await ctx.task


class _StreamSession:
    """Owns a single WS connection's turn state machine."""

    __slots__ = ("_asr", "_ctx", "_current_turn_id", "_finalize", "_orchestrator", "_session_id")

    def __init__(self, orchestrator: Orchestrator, asr: ASRBackend) -> None:
        self._orchestrator = orchestrator
        self._asr = asr
        self._session_id: str | None = None
        self._current_turn_id: int = -1
        self._ctx: _TurnContext | None = None
        self._finalize: asyncio.Task[None] | None = None

    async def handle_pcm(self, data: bytes) -> None:
        if self._ctx is None or self._ctx.cancelled:
            return
        await self._ctx.queue.put(AudioChunk(pcm16=data, sequence=0))

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
            _pump_asr(websocket, self._asr, ctx, new_session_id, new_turn_id)
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
) -> None:
    """Streaming recognition over WebSocket. See ``docs/ws-protocol.md``."""

    if not _check_ws_api_key(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()

    session = _StreamSession(orchestrator, asr)
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
