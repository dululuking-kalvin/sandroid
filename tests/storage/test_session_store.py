from __future__ import annotations

import asyncio

import pytest

from sandroid.storage.session_store import InMemorySessionStore


@pytest.mark.asyncio
async def test_create_and_get_roundtrip() -> None:
    store = InMemorySessionStore()
    created = await store.create(scene_id="example_bank", current_category_id="root")
    loaded = await store.get(created.session_id)
    assert loaded.session_id == created.session_id
    assert loaded.current_category_id == "root"


@pytest.mark.asyncio
async def test_get_unknown_raises() -> None:
    store = InMemorySessionStore()
    with pytest.raises(KeyError):
        await store.get("ghost")


@pytest.mark.asyncio
async def test_put_replaces_session() -> None:
    store = InMemorySessionStore()
    session = await store.create(scene_id="s", current_category_id="root")
    session.current_category_id = "other"
    await store.put(session)
    loaded = await store.get(session.session_id)
    assert loaded.current_category_id == "other"


@pytest.mark.asyncio
async def test_get_returns_copy_not_reference() -> None:
    store = InMemorySessionStore()
    session = await store.create(scene_id="s", current_category_id="root")
    loaded = await store.get(session.session_id)
    loaded.current_category_id = "mutated"
    again = await store.get(session.session_id)
    assert again.current_category_id == "root"


@pytest.mark.asyncio
async def test_concurrent_creates_have_unique_ids() -> None:
    store = InMemorySessionStore()
    results = await asyncio.gather(
        *(store.create(scene_id="s", current_category_id="root") for _ in range(20)),
    )
    assert len({s.session_id for s in results}) == 20


@pytest.mark.asyncio
async def test_delete_removes_session() -> None:
    store = InMemorySessionStore()
    session = await store.create(scene_id="s", current_category_id="root")
    await store.delete(session.session_id)
    with pytest.raises(KeyError):
        await store.get(session.session_id)
