"""Session state persistence.

v0.1 ships an in-memory store. The Protocol is the seam; swap to Redis or
Postgres later without touching the orchestrator.
"""

from __future__ import annotations

import asyncio
from typing import Protocol
from uuid import uuid4

from sandroid.core.domain import CategoryId, Session


class SessionStore(Protocol):
    """Storage seam for call-level dialogue state."""

    async def create(self, *, scene_id: str, current_category_id: CategoryId) -> Session: ...
    async def get(self, session_id: str) -> Session: ...
    async def put(self, session: Session) -> None: ...
    async def delete(self, session_id: str) -> None: ...


class InMemorySessionStore:
    """Dict-backed store. Not durable — fine for v0.1 single-node deploys."""

    def __init__(self) -> None:
        self._by_id: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        scene_id: str,
        current_category_id: CategoryId,
    ) -> Session:
        del scene_id  # reserved for future multi-scene indexing
        session = Session(session_id=uuid4().hex, current_category_id=current_category_id)
        async with self._lock:
            self._by_id[session.session_id] = session
        return session

    async def get(self, session_id: str) -> Session:
        async with self._lock:
            if session_id not in self._by_id:
                raise KeyError(f"unknown session: {session_id!r}")
            return self._by_id[session_id].model_copy(deep=True)

    async def put(self, session: Session) -> None:
        async with self._lock:
            self._by_id[session.session_id] = session.model_copy(deep=True)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._by_id.pop(session_id, None)
