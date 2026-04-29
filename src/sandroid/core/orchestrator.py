"""Recognition orchestrator — glues registry + scope + matcher + session + fusion.

This is the hot path: given a scene pointer and a transcript (and optionally
raw audio for Path B), return the top intent + N-best + optional ``follow_up``.

Phase 5f-b adds Path B end-to-end SLU: when ``RecognitionRequest.audio`` is
provided, ``Orchestrator`` runs the matcher (Path A) and the SLU adapter
(Path B) in parallel and combines them via the Fusion layer using per-scene
weights. With ``audio=None``, Path B abstains and the result is
algebraically identical to the pre-5f-b matcher-only path.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from sandroid.core.domain import CategoryId, CategoryPath, Intent, IntentId, Session
from sandroid.core.fusion import DEFAULT_CONFIG, FusionConfig, fuse
from sandroid.core.matcher import IntentMatcher, MatchCandidate
from sandroid.core.scope import scope
from sandroid.models.slu.base import FinalIntent, SLUBackend
from sandroid.storage.scene_registry import SceneRegistry
from sandroid.storage.session_store import SessionStore


class RecognitionRequest(BaseModel):
    scene_id: str | None = None
    category_path: CategoryPath | None = None
    text: str
    # Raw audio for Path B (E2E SLU). Format: PCM16 little-endian, 16 kHz,
    # mono. No container header. Front-end is responsible for transcoding
    # before passing — sandroid's REST/WS/MRCP entry points all do this.
    # None -> Path B abstains; algebraically equivalent to the pre-5f-b
    # matcher-only result.
    audio: bytes | None = None
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
        slu: SLUBackend | None = None,
        fusion_config: FusionConfig | None = None,
    ) -> None:
        self._registry = registry
        self._sessions = sessions
        self._matcher = matcher
        # slu/fusion default to None/DEFAULT_CONFIG so existing call sites
        # (and tests) that haven't migrated to 5f-b still work — Path B
        # is silently disabled when slu is None.
        self._slu = slu
        self._fusion_config = fusion_config or DEFAULT_CONFIG

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

        # Path A + Path B run concurrently. Both ONNX paths are CPU-bound;
        # asyncio.to_thread releases the event loop and lets them overlap.
        ranked, slu_result = await self._run_dual_paths(
            text=req.text,
            audio=req.audio,
            scene_id=scene_id,
            candidates=candidates,
            n_best=req.n_best,
        )

        weights = self._fusion_config.for_scene(scene_id)
        fused = fuse(ranked, slu_result, candidates, weights, n_best=req.n_best)

        top_hit, follow_up = self._resolve_top(fused, candidates)

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
            n_best=fused,
            follow_up=follow_up,
            current_category_id=session.current_category_id,
        )

    async def _run_dual_paths(
        self,
        *,
        text: str,
        audio: bytes | None,
        scene_id: str,
        candidates: list[Intent],
        n_best: int,
    ) -> tuple[list[MatchCandidate], FinalIntent | None]:
        """Run Path A (matcher) + Path B (SLU) concurrently. Returns both raw outputs.

        Fusion happens upstream; this layer is just about parallel execution
        and Path B's "abstain when not configured" contract.
        """
        # Path A: matcher.score is sync + CPU-bound. asyncio.to_thread keeps
        # the event loop responsive so Path B can run alongside.
        path_a = asyncio.to_thread(self._matcher.score, text, candidates, n_best=n_best)

        if self._slu is not None and audio is not None:
            path_b: asyncio.Future[FinalIntent | None] = asyncio.ensure_future(
                self._slu.recognize_audio(audio, scene_id=scene_id)
            )
            ranked, slu_result = await asyncio.gather(path_a, path_b)
        else:
            ranked = await path_a
            slu_result = None

        return ranked, slu_result

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
