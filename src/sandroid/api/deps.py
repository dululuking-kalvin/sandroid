"""FastAPI dependency wiring.

Holds the process-lifetime singletons (registry, session store, matcher,
orchestrator) and the API-key auth dependency. Lazy-constructed so tests
can override.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, Header, HTTPException, status

from sandroid.core.matcher import IntentMatcher, StubMatcher
from sandroid.core.orchestrator import Orchestrator
from sandroid.models.asr.base import ASRBackend
from sandroid.models.asr.stub import StubASR
from sandroid.storage.scene_registry import SceneRegistry
from sandroid.storage.session_store import InMemorySessionStore, SessionStore
from sandroid.vad.base import VoiceActivityDetector
from sandroid.vad.mock import MockVAD

DEFAULT_SCENES_DIR = Path(__file__).resolve().parents[3] / "configs" / "scenes"
API_KEY_ENV = "SANDROID_API_KEY"
DEV_DEFAULT_API_KEY = "dev-insecure-change-me"


@lru_cache(maxsize=1)
def get_registry() -> SceneRegistry:
    directory = os.environ.get("SANDROID_SCENES_DIR", str(DEFAULT_SCENES_DIR))
    return SceneRegistry.from_directory(directory)


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    return InMemorySessionStore()


@lru_cache(maxsize=1)
def get_matcher() -> IntentMatcher:
    return StubMatcher()


@lru_cache(maxsize=1)
def get_asr() -> ASRBackend:
    return StubASR()


@lru_cache(maxsize=1)
def get_vad() -> VoiceActivityDetector:
    return MockVAD()


def get_orchestrator(
    registry: SceneRegistry = Depends(get_registry),
    sessions: SessionStore = Depends(get_session_store),
    matcher: IntentMatcher = Depends(get_matcher),
) -> Orchestrator:
    return Orchestrator(registry=registry, sessions=sessions, matcher=matcher)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get(API_KEY_ENV, DEV_DEFAULT_API_KEY)
    if x_api_key is None or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-API-Key",
        )


def reset_dependency_caches() -> None:
    """Used by tests to drop cached singletons between runs."""

    get_registry.cache_clear()
    get_session_store.cache_clear()
    get_matcher.cache_clear()
    get_asr.cache_clear()
    get_vad.cache_clear()
