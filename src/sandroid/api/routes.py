"""Public HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from sandroid.api.deps import get_orchestrator, require_api_key
from sandroid.core.orchestrator import (
    Orchestrator,
    RecognitionRequest,
    RecognitionResponse,
)

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
