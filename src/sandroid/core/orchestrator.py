"""Recognition orchestrator — glues registry + scope + matcher + session.

This is the hot path: given a scene pointer and a transcript, return the
top intent + N-best + optional ``follow_up``. Pure Python for now; the audio
path lives behind the ``text`` input until ASR/NLU/SLU adapters land.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from sandroid.core.domain import CategoryId, CategoryPath, Intent, IntentId, Session
from sandroid.core.matcher import IntentMatcher, MatchCandidate
from sandroid.core.scope import scope
from sandroid.storage.scene_registry import SceneRegistry
from sandroid.storage.session_store import SessionStore


class RecognitionRequest(BaseModel):
    scene_id: str | None = None
    category_path: CategoryPath | None = None
    text: str
    session_id: str | None = None
    n_best: int = Field(default=5, ge=1, le=20)


class TopHit(BaseModel):
    intent_id: IntentId
    confidence: float
    answer: str | None
    follow_up: CategoryPath | None


class RecognitionResponse(BaseModel):
    session_id: str
    top: TopHit | None
    n_best: list[MatchCandidate]
    follow_up: CategoryPath | None = None
    current_category_id: CategoryId


class Orchestrator:
    def __init__(
        self,
        *,
        registry: SceneRegistry,
        sessions: SessionStore,
        matcher: IntentMatcher,
    ) -> None:
        self._registry = registry
        self._sessions = sessions
        self._matcher = matcher

    async def recognize(self, req: RecognitionRequest) -> RecognitionResponse:
        scene_id, category_id = self._registry.resolve(
            scene_id=req.scene_id,
            category_path=req.category_path,
        )
        tree = self._registry.get(scene_id)

        session = await self._load_or_create_session(
            req.session_id,
            scene_id=scene_id,
            category_id=category_id,
        )
        # Caller's current category_path / scene overrides stored pointer if
        # the caller explicitly passed one — lets external dialog managers
        # drive the scene.
        if req.session_id is None or req.category_path is not None or req.scene_id is not None:
            session.current_category_id = category_id

        candidates: list[Intent] = scope(tree, session.current_category_id)
        ranked = self._matcher.score(req.text, candidates, n_best=req.n_best)

        top_hit, follow_up = self._resolve_top(ranked, candidates)

        if follow_up is not None:
            try:
                target = tree.resolve_path(follow_up)
            except KeyError:
                target = None
            if target is not None:
                session.current_category_id = target.id
                session.switching_history.append(target.id)

        session.history.append(req.text)
        await self._sessions.put(session)

        return RecognitionResponse(
            session_id=session.session_id,
            top=top_hit,
            n_best=ranked,
            follow_up=follow_up,
            current_category_id=session.current_category_id,
        )

    async def _load_or_create_session(
        self,
        session_id: str | None,
        *,
        scene_id: str,
        category_id: CategoryId,
    ) -> Session:
        if session_id is None:
            return await self._sessions.create(
                scene_id=scene_id,
                current_category_id=category_id,
            )
        try:
            return await self._sessions.get(session_id)
        except KeyError:
            return await self._sessions.create(
                scene_id=scene_id,
                current_category_id=category_id,
            )

    @staticmethod
    def _resolve_top(
        ranked: list[MatchCandidate],
        candidates: list[Intent],
    ) -> tuple[TopHit | None, CategoryPath | None]:
        if not ranked:
            return None, None
        top = ranked[0]
        intent_by_id = {i.id: i for i in candidates}
        intent = intent_by_id[top.intent_id]
        follow_up = intent.switching or intent.follow_up
        return (
            TopHit(
                intent_id=intent.id,
                confidence=top.confidence,
                answer=intent.answer,
                follow_up=follow_up,
            ),
            follow_up,
        )
