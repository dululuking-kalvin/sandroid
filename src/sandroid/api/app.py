from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from sandroid import __version__
from sandroid.api.deps import (
    get_asr,
    get_matcher,
    get_registry,
    get_session_store,
    get_vad,
)
from sandroid.api.logging_config import configure_logging
from sandroid.api.metrics import registry as metrics_registry
from sandroid.api.routes import router as v1_router
from sandroid.api.routes import ws_router as v1_ws_router


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    version: str
    checks: dict[str, str]


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


_READINESS_PROBES: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("registry", get_registry),
    ("session_store", get_session_store),
    ("matcher", get_matcher),
    ("vad", get_vad),
    ("asr", get_asr),
)


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="sandroid", version=__version__, lifespan=_lifespan)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        """Liveness — process is up. Does not exercise dependencies."""

        return HealthResponse(status="ok", version=__version__)

    @app.get("/readyz")
    async def readyz() -> Response:
        """Readiness — every dependency resolves.

        In production (SANDROID_ENV=production) this is how orchestration
        decides to route traffic; in dev missing artifacts fall back to
        stubs so the check will typically still report ``ok`` (the
        per-check detail still identifies which backend is in use).
        """

        checks: dict[str, str] = {}
        overall_ok = True
        for name, probe in _READINESS_PROBES:
            try:
                probe()
                checks[name] = "ok"
            except Exception as exc:
                checks[name] = f"error: {type(exc).__name__}: {exc}"
                overall_ok = False
        body = ReadinessResponse(
            status="ok" if overall_ok else "error",
            version=__version__,
            checks=checks,
        ).model_dump()
        return JSONResponse(status_code=200 if overall_ok else 503, content=body)

    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus scrape target. Unauthenticated — restrict at network layer."""

        return Response(
            content=generate_latest(metrics_registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    app.include_router(v1_router)
    app.include_router(v1_ws_router)
    return app


app = create_app()
