from __future__ import annotations

from pathlib import Path

import pytest

from sandroid.core.matcher import StubMatcher
from sandroid.core.orchestrator import Orchestrator, RecognitionRequest
from sandroid.storage.scene_registry import SceneRegistry
from sandroid.storage.session_store import InMemorySessionStore

SCENES_DIR = Path(__file__).resolve().parents[2] / "configs" / "scenes"


def _make_orchestrator() -> tuple[Orchestrator, InMemorySessionStore]:
    registry = SceneRegistry.from_directory(SCENES_DIR)
    sessions = InMemorySessionStore()
    matcher = StubMatcher()
    return Orchestrator(registry=registry, sessions=sessions, matcher=matcher), sessions


@pytest.mark.asyncio
async def test_recognize_returns_top_intent_and_n_best() -> None:
    orch, _ = _make_orchestrator()
    resp = await orch.recognize(
        RecognitionRequest(scene_id="example_bank", text="你好"),
    )
    assert resp.top is not None
    assert resp.top.intent_id == "greeting"
    assert resp.n_best[0].intent_id == "greeting"
    assert resp.current_category_id == "root"


@pytest.mark.asyncio
async def test_recognize_follow_up_moves_session_pointer() -> None:
    # At ROOT (STANDARD), BRANCHING children expose only their special hooks.
    # Matching "跨境汇款" should fire the cross-border hook's follow_up and
    # move the session pointer there.
    orch, store = _make_orchestrator()
    resp = await orch.recognize(
        RecognitionRequest(scene_id="example_bank", text="跨境汇款"),
    )
    assert resp.top is not None
    assert resp.top.intent_id == "transfer_cross_border_hook"
    assert resp.follow_up == "ROOT->转账->跨境转账"
    loaded = await store.get(resp.session_id)
    assert loaded.current_category_id == "transfer_cross_border"


@pytest.mark.asyncio
async def test_recognize_persists_session_across_turns() -> None:
    orch, _ = _make_orchestrator()
    first = await orch.recognize(
        RecognitionRequest(scene_id="example_bank", text="你好"),
    )
    second = await orch.recognize(
        RecognitionRequest(session_id=first.session_id, scene_id="example_bank", text="你好"),
    )
    assert second.session_id == first.session_id


@pytest.mark.asyncio
async def test_recognize_no_match_returns_empty() -> None:
    orch, _ = _make_orchestrator()
    resp = await orch.recognize(
        RecognitionRequest(scene_id="example_bank", text="zzzzzzzz"),
    )
    assert resp.top is None
    assert resp.n_best == []
