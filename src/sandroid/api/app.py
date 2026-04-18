from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from sandroid import __version__


class HealthResponse(BaseModel):
    status: str
    version: str


def create_app() -> FastAPI:
    app = FastAPI(title="sandroid", version=__version__)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    return app


app = create_app()
