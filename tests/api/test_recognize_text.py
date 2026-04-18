from __future__ import annotations

import httpx
import pytest
from httpx import ASGITransport

from sandroid.api.app import app
from sandroid.api.deps import DEV_DEFAULT_API_KEY, reset_dependency_caches

AUTH_HEADER = {"X-API-Key": DEV_DEFAULT_API_KEY}


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    reset_dependency_caches()


@pytest.mark.asyncio
async def test_recognize_text_happy_path() -> None:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recognize/text",
            json={"scene_id": "example_bank", "text": "你好"},
            headers=AUTH_HEADER,
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["top"]["intent_id"] == "greeting"
    assert 0.0 < body["top"]["confidence"] <= 1.0
    assert body["current_category_id"] == "root"
    assert body["session_id"]


@pytest.mark.asyncio
async def test_recognize_text_rejects_missing_api_key() -> None:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recognize/text",
            json={"scene_id": "example_bank", "text": "你好"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_recognize_text_requires_scene_or_path() -> None:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recognize/text",
            json={"text": "你好"},
            headers=AUTH_HEADER,
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_recognize_text_unknown_scene_404() -> None:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recognize/text",
            json={"scene_id": "ghost", "text": "你好"},
            headers=AUTH_HEADER,
        )
    assert resp.status_code == 404
