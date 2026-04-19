"""Public HTTP routes."""

from __future__ import annotations

import asyncio
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


async def _drain_socket(
    websocket: WebSocket,
    queue: asyncio.Queue[AudioChunk | None],
) -> None:
    """Read binary PCM frames / JSON control until ``end`` or disconnect."""

    sequence = 0
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is not None:
                await queue.put(AudioChunk(pcm16=data, sequence=sequence))
                sequence += 1
                continue
            text = message.get("text")
            if text is None:
                continue
            try:
                control = json.loads(text)
            except ValueError:
                continue
            if control.get("type") == "end":
                break
    except WebSocketDisconnect:
        pass
    await queue.put(None)


async def _pump_asr(
    websocket: WebSocket,
    asr: ASRBackend,
    queue: asyncio.Queue[AudioChunk | None],
) -> str:
    async def source() -> AsyncIterator[AudioChunk]:
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item

    final_text = ""
    async for partial in asr.stream(source()):
        await websocket.send_json(
            {
                "type": "final" if partial.is_final else "partial",
                "text": partial.text,
                "start_ms": partial.start_ms,
                "end_ms": partial.end_ms,
            }
        )
        if partial.is_final:
            final_text = partial.text
    return final_text


@ws_router.websocket("/recognize/stream")
async def recognize_stream(
    websocket: WebSocket,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    asr: ASRBackend = Depends(get_asr),
) -> None:
    """Streaming recognition over WebSocket.

    Protocol:
    - Client connects; first frame is a JSON control message:
        {"scene_id"|"category_path": ..., "session_id"?: ..., "n_best"?: int}
    - Subsequent binary frames are raw 16 kHz mono PCM16 chunks.
    - A JSON {"type": "end"} frame flips the stream to final.
    - Server emits JSON partials / final and a RecognitionResponse at the end.
    """

    if not _check_ws_api_key(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()

    try:
        control = await websocket.receive_json()
    except WebSocketDisconnect:
        return
    scene_id = control.get("scene_id")
    category_path = control.get("category_path")
    if scene_id is None and category_path is None:
        await websocket.send_json(
            {"type": "error", "detail": "either scene_id or category_path is required"},
        )
        await websocket.close(code=4422)
        return

    queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
    asr_task = asyncio.create_task(_pump_asr(websocket, asr, queue))
    await _drain_socket(websocket, queue)
    final_text = await asr_task

    if not final_text.strip():
        await websocket.send_json({"type": "error", "detail": "no transcript produced"})
        await websocket.close()
        return

    req = RecognitionRequest(
        scene_id=scene_id,
        category_path=category_path,
        text=final_text,
        session_id=control.get("session_id"),
        n_best=int(control.get("n_best", 5)),
    )
    try:
        response = await orchestrator.recognize(req)
    except (KeyError, ValueError) as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
        await websocket.close()
        return

    await websocket.send_json({"type": "result", **response.model_dump()})
    await websocket.close()
