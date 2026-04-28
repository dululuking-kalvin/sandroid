from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from sandroid.core.fusion import FusionConfig, FusionWeights
from sandroid.core.matcher import StubMatcher
from sandroid.core.orchestrator import Orchestrator, RecognitionRequest
from sandroid.models.asr.base import AudioChunk
from sandroid.models.slu.base import FinalIntent, PartialIntent, SLUBackend
from sandroid.models.slu.stub import StubSLU
from sandroid.storage.scene_registry import SceneRegistry
from sandroid.storage.session_store import InMemorySessionStore

SCENES_DIR = Path(__file__).resolve().parents[2] / "configs" / "scenes"


def _make_orchestrator(
    *,
    slu: SLUBackend | None = None,
    fusion_config: FusionConfig | None = None,
) -> tuple[Orchestrator, InMemorySessionStore]:
    registry = SceneRegistry.from_directory(SCENES_DIR)
    sessions = InMemorySessionStore()
    matcher = StubMatcher()
    orch = Orchestrator(
        registry=registry,
        sessions=sessions,
        matcher=matcher,
        slu=slu,
        fusion_config=fusion_config,
    )
    return orch, sessions


class _FixedSLU:
    """SLU double that returns a pre-set FinalIntent regardless of input."""

    def __init__(self, intent_id: str | None, confidence: float) -> None:
        self._intent_id = intent_id
        self._confidence = confidence

    async def stream(
        self,
        chunks: AsyncIterator[AudioChunk],
        *,
        scene_id: str | None = None,
    ) -> AsyncIterator[PartialIntent]:
        async for _ in chunks:
            pass
        yield PartialIntent(
            intent_id=self._intent_id,
            confidence=self._confidence,
            is_final=True,
            start_ms=0,
            end_ms=0,
        )

    async def recognize_file(
        self,
        wav_bytes: bytes,
        *,
        scene_id: str | None = None,
    ) -> FinalIntent:
        return FinalIntent(
            intent_id=self._intent_id,
            confidence=self._confidence,
            duration_ms=1000,
        )


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


# ---------- Phase 5f-b: Path B + Fusion ----------


def _bank_fusion(w_a: float, w_b: float) -> FusionConfig:
    """Per-scene weights for example_bank, default everything else to A-only."""
    return FusionConfig(
        default=FusionWeights(w_a=1.0, w_b=0.0),
        scenes={"example_bank": FusionWeights(w_a=w_a, w_b=w_b)},
    )


@pytest.mark.asyncio
async def test_recognize_with_audio_and_stub_slu_matches_path_a_only() -> None:
    """StubSLU abstains -> fused result equals Path A even with non-zero w_b."""
    orch, _ = _make_orchestrator(slu=StubSLU(), fusion_config=_bank_fusion(0.6, 0.4))
    a_only = await _make_orchestrator()[0].recognize(
        RecognitionRequest(scene_id="example_bank", text="你好"),
    )
    fused = await orch.recognize(
        RecognitionRequest(scene_id="example_bank", text="你好", audio=b"\x00" * 320),
    )
    assert fused.top is not None
    assert fused.top.intent_id == a_only.top.intent_id
    # B abstained (stub) -> fusion is a no-op, A's score passes through.
    assert fused.top.confidence == pytest.approx(a_only.top.confidence)


@pytest.mark.asyncio
async def test_recognize_no_audio_skips_path_b_entirely() -> None:
    """audio=None -> Path B not invoked, weights ignored, raw matcher score returned."""
    # If B were invoked, we'd see scaled scores. With audio=None we want raw.
    orch, _ = _make_orchestrator(slu=StubSLU(), fusion_config=_bank_fusion(0.5, 0.5))
    resp = await orch.recognize(
        RecognitionRequest(scene_id="example_bank", text="你好"),
    )
    a_only_resp = await _make_orchestrator()[0].recognize(
        RecognitionRequest(scene_id="example_bank", text="你好"),
    )
    assert resp.top is not None
    # No fusion scaling — confidence matches the matcher's raw score.
    assert resp.top.confidence == pytest.approx(a_only_resp.top.confidence)


@pytest.mark.asyncio
async def test_recognize_b_consensus_boosts_confidence() -> None:
    """B votes for the same intent A picked top -> fused conf > either alone."""
    orch, _ = _make_orchestrator(
        slu=_FixedSLU(intent_id="greeting", confidence=0.9),
        fusion_config=_bank_fusion(0.5, 0.5),
    )
    a_only_resp = await _make_orchestrator()[0].recognize(
        RecognitionRequest(scene_id="example_bank", text="你好"),
    )
    fused = await orch.recognize(
        RecognitionRequest(scene_id="example_bank", text="你好", audio=b"\x00" * 320),
    )
    assert fused.top is not None
    assert fused.top.intent_id == "greeting"
    # 0.5 * conf_A + 0.5 * 0.9 should exceed pure-A scaled by 0.5 alone.
    expected = 0.5 * a_only_resp.top.confidence + 0.5 * 0.9
    assert fused.top.confidence == pytest.approx(expected)


@pytest.mark.asyncio
async def test_recognize_b_out_of_scope_intent_ignored() -> None:
    """B votes for an off-scope intent_id -> silently dropped, A wins."""
    orch, _ = _make_orchestrator(
        slu=_FixedSLU(intent_id="not_a_real_intent", confidence=0.99),
        fusion_config=_bank_fusion(0.5, 0.5),
    )
    fused = await orch.recognize(
        RecognitionRequest(scene_id="example_bank", text="你好", audio=b"\x00" * 320),
    )
    assert fused.top is not None
    assert fused.top.intent_id == "greeting"  # A's pick stands


@pytest.mark.asyncio
async def test_recognize_b_can_flip_top_when_w_b_dominates() -> None:
    """High w_b + strong B vote on a non-top A intent -> B's choice wins."""
    # transfer_card has a hook 'transfer_in_country_hook' attached as special;
    # pick an in-scope intent the matcher won't surface for "你好".
    orch, _ = _make_orchestrator(
        slu=_FixedSLU(intent_id="card_lost_report", confidence=0.95),
        fusion_config=_bank_fusion(0.1, 0.9),
    )
    fused = await orch.recognize(
        RecognitionRequest(scene_id="example_bank", text="你好", audio=b"\x00" * 320),
    )
    assert fused.top is not None
    assert fused.top.intent_id == "card_lost_report"


@pytest.mark.asyncio
async def test_recognize_unconfigured_scene_uses_fusion_default() -> None:
    """Scenes not listed in fusion config inherit the default (1.0, 0.0)."""
    orch, _ = _make_orchestrator(
        slu=_FixedSLU(intent_id="greeting", confidence=0.99),
        fusion_config=FusionConfig(default=FusionWeights(w_a=1.0, w_b=0.0)),
    )
    fused = await orch.recognize(
        RecognitionRequest(scene_id="example_bank", text="你好", audio=b"\x00" * 320),
    )
    a_only = await _make_orchestrator()[0].recognize(
        RecognitionRequest(scene_id="example_bank", text="你好"),
    )
    assert fused.top is not None
    # w_b=0 means B's vote contributes nothing — same confidence as pure A.
    assert fused.top.confidence == pytest.approx(a_only.top.confidence)
