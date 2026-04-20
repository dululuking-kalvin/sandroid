from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from sandroid import __version__
from sandroid.api.deps import (
    get_asr,
    get_matcher,
    get_registry,
    get_session_store,
    get_vad,
)
from sandroid.api.routes import router as v1_router
from sandroid.api.routes import ws_router as v1_ws_router


class HealthResponse(BaseModel):
    status: str
    version: str


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm up all singletons so artifact errors fail the boot, not a request.

    In SANDROID_ENV=production this turns missing ONNX files into an
    immediate startup failure (which orchestration can detect via its
    liveness check); in dev it just eagerly resolves the stub fallbacks.
    """

    # Skip warmup when tests have overridden the deps. Otherwise calling the
    # real provider would bypass the override and load real models in CI.
    if not app.dependency_overrides:
        get_registry()
        get_session_store()
        get_matcher()
        get_vad()
        get_asr()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="sandroid", version=__version__, lifespan=_lifespan)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    app.include_router(v1_router)
    app.include_router(v1_ws_router)
    return app


app = create_app()
