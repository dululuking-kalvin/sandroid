"""Unit tests for the Fusion layer (Phase 5f-b)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sandroid.core.domain import Intent
from sandroid.core.fusion import (
    DEFAULT_CONFIG,
    FusionConfig,
    FusionWeights,
    fuse,
    load_fusion_config,
)
from sandroid.core.matcher import MatchCandidate
from sandroid.models.slu.base import FinalIntent


def _intent(iid: str) -> Intent:
    return Intent(id=iid, category_id="c", question=["q"])


def _candidates() -> list[Intent]:
    return [_intent("a"), _intent("b"), _intent("c")]


# ---------- FusionWeights ----------


def test_weights_reject_both_zero() -> None:
    with pytest.raises(ValidationError, match="cannot both be zero"):
        FusionWeights(w_a=0.0, w_b=0.0)


def test_weights_reject_out_of_range() -> None:
    with pytest.raises(ValidationError):
        FusionWeights(w_a=1.5, w_b=0.5)


def test_weights_normalize_to_unit_sum() -> None:
    w = FusionWeights(w_a=0.6, w_b=0.6)
    a, b = w.normalized()
    assert a == pytest.approx(0.5)
    assert b == pytest.approx(0.5)


def test_weights_normalize_path_a_only() -> None:
    a, b = FusionWeights(w_a=1.0, w_b=0.0).normalized()
    assert (a, b) == (1.0, 0.0)


# ---------- FusionConfig.for_scene ----------


def test_config_for_scene_falls_back_to_default() -> None:
    cfg = FusionConfig(
        default=FusionWeights(w_a=1.0, w_b=0.0),
        scenes={"bank": FusionWeights(w_a=0.7, w_b=0.3)},
    )
    assert cfg.for_scene("bank").w_b == 0.3
    assert cfg.for_scene("unknown").w_b == 0.0
    assert cfg.for_scene(None).w_b == 0.0


# ---------- load_fusion_config ----------


def test_load_missing_file_returns_default() -> None:
    cfg = load_fusion_config("/path/that/does/not/exist.yaml")
    assert cfg is DEFAULT_CONFIG


def test_load_none_returns_default() -> None:
    assert load_fusion_config(None) is DEFAULT_CONFIG


def test_load_real_yaml(tmp_path: Path) -> None:
    p = tmp_path / "fusion.yaml"
    p.write_text(
        "default:\n  w_a: 1.0\n  w_b: 0.0\n"
        "scenes:\n  bank:\n    w_a: 0.7\n    w_b: 0.3\n",
        encoding="utf-8",
    )
    cfg = load_fusion_config(p)
    assert cfg.default.w_a == 1.0
    assert cfg.scenes["bank"].w_b == 0.3


def test_load_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_fusion_config(p)


def test_load_empty_file_uses_default(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    cfg = load_fusion_config(p)
    assert cfg.default.w_a == 1.0
    assert cfg.default.w_b == 0.0


def test_repo_fusion_yaml_loads() -> None:
    """The shipped configs/fusion.yaml must parse cleanly — guards against typos."""
    repo_yaml = Path(__file__).resolve().parents[2] / "configs" / "fusion.yaml"
    assert repo_yaml.exists(), "configs/fusion.yaml is missing"
    cfg = load_fusion_config(repo_yaml)
    # Ship-time defaults: Path A only.
    assert cfg.default.w_a == 1.0
    assert cfg.default.w_b == 0.0


# ---------- fuse() — algorithmic cases ----------


def test_fuse_a_only_when_b_abstains() -> None:
    """B=None means Path B is unavailable; weights MUST NOT scale A's scores.

    Otherwise text-only requests (audio=None, B silent) would silently get
    halved confidences whenever per-scene fusion configures w_b > 0. The
    fused result is identical to A's input.
    """
    ranked_a = [MatchCandidate(intent_id="a", confidence=0.8)]
    weights = FusionWeights(w_a=0.6, w_b=0.4)
    out = fuse(ranked_a, None, _candidates(), weights)
    assert len(out) == 1
    assert out[0].intent_id == "a"
    assert out[0].confidence == pytest.approx(0.8)


def test_fuse_consensus_boosts_winning_intent() -> None:
    """A and B both like 'a' -> fused score is the weighted sum."""
    ranked_a = [
        MatchCandidate(intent_id="a", confidence=0.8),
        MatchCandidate(intent_id="b", confidence=0.4),
    ]
    slu = FinalIntent(intent_id="a", confidence=0.7, duration_ms=1000)
    weights = FusionWeights(w_a=0.6, w_b=0.4)
    out = fuse(ranked_a, slu, _candidates(), weights)
    assert out[0].intent_id == "a"
    # 0.6*0.8 + 0.4*0.7 = 0.76
    assert out[0].confidence == pytest.approx(0.76)


def test_fuse_disagreement_can_flip_top() -> None:
    """A picks 'a' weakly, B picks 'b' strongly with high w_b -> b wins."""
    ranked_a = [
        MatchCandidate(intent_id="a", confidence=0.5),
        MatchCandidate(intent_id="b", confidence=0.3),
    ]
    slu = FinalIntent(intent_id="b", confidence=0.95, duration_ms=1000)
    weights = FusionWeights(w_a=0.3, w_b=0.7)
    out = fuse(ranked_a, slu, _candidates(), weights)
    assert out[0].intent_id == "b"
    # b: 0.3*0.3 + 0.7*0.95 = 0.755 ; a: 0.3*0.5 = 0.15
    assert out[0].confidence == pytest.approx(0.755)
    assert out[1].intent_id == "a"


def test_fuse_drops_b_intent_outside_scope() -> None:
    """B suggests an intent not in the scene's candidate pool -> equivalent to B abstain."""
    ranked_a = [MatchCandidate(intent_id="a", confidence=0.8)]
    slu = FinalIntent(intent_id="off_scope", confidence=0.99, duration_ms=1000)
    weights = FusionWeights(w_a=0.5, w_b=0.5)
    out = fuse(ranked_a, slu, _candidates(), weights)
    # Off-scope B means Path B effectively abstained — weights don't apply.
    assert len(out) == 1
    assert out[0].intent_id == "a"
    assert out[0].confidence == pytest.approx(0.8)


def test_fuse_drops_b_when_intent_id_is_none() -> None:
    """B abstains via intent_id=None -> equivalent to B=None: A passes through."""
    ranked_a = [MatchCandidate(intent_id="a", confidence=0.8)]
    slu = FinalIntent(intent_id=None, confidence=0.0, duration_ms=1000)
    weights = FusionWeights(w_a=0.6, w_b=0.4)
    out = fuse(ranked_a, slu, _candidates(), weights)
    assert out[0].intent_id == "a"
    assert out[0].confidence == pytest.approx(0.8)


def test_fuse_promotes_b_only_intent_into_n_best() -> None:
    """A doesn't rank 'c' at all but B votes for it -> 'c' appears in fused list."""
    ranked_a = [MatchCandidate(intent_id="a", confidence=0.6)]
    slu = FinalIntent(intent_id="c", confidence=0.9, duration_ms=1000)
    weights = FusionWeights(w_a=0.3, w_b=0.7)
    out = fuse(ranked_a, slu, _candidates(), weights)
    ids = [c.intent_id for c in out]
    # c: 0.7*0.9 = 0.63 ; a: 0.3*0.6 = 0.18
    assert ids[0] == "c"
    assert out[0].confidence == pytest.approx(0.63)


def test_fuse_empty_a_with_abstain_b_returns_nothing() -> None:
    out = fuse([], None, _candidates(), FusionWeights(w_a=1.0, w_b=0.0))
    assert out == []


def test_fuse_caps_n_best() -> None:
    """N-best parameter trims even if more intents have non-zero score."""
    ranked_a = [
        MatchCandidate(intent_id="a", confidence=0.9),
        MatchCandidate(intent_id="b", confidence=0.6),
        MatchCandidate(intent_id="c", confidence=0.3),
    ]
    out = fuse(ranked_a, None, _candidates(), FusionWeights(w_a=1.0, w_b=0.0), n_best=2)
    assert len(out) == 2
    assert [c.intent_id for c in out] == ["a", "b"]


def test_fuse_stable_order_on_ties() -> None:
    """Equal scores break ties by intent_id ascending."""
    ranked_a = [
        MatchCandidate(intent_id="a", confidence=0.5),
        MatchCandidate(intent_id="b", confidence=0.5),
    ]
    out = fuse(ranked_a, None, _candidates(), FusionWeights(w_a=1.0, w_b=0.0))
    assert [c.intent_id for c in out] == ["a", "b"]
