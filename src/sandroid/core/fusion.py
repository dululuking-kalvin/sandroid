"""Fusion layer for Path A (ASR+NLU) and Path B (E2E SLU) — Phase 5f-b.

CLAUDE.md's dual-path architecture has two recognizers running in parallel:

- Path A: ASR -> NLU. Transcript-grounded, explainable, weak under noise.
- Path B: end-to-end PCM -> intent. Robust to ASR errors, weak on rare
  intents and harder to debug.

Fusion combines them per scene with a weighted sum:

    score(intent) = w_A * conf_A(intent) + w_B * conf_B(intent)

Weights live in ``configs/fusion.yaml``: a global ``default`` plus optional
``scenes`` overrides. The default ships as ``(w_a=1.0, w_b=0.0)`` — Path A
only — so adding the fusion layer is a no-op until per-scene weights are
explicitly configured. This is deliberate: while Path B is StubSLU
(abstain), any non-zero w_b would silently halve every confidence.

Algorithm contract (see ``fuse``):

- ``ranked_a`` is the matcher's existing scoped N-best (already restricted
  to the scene's candidate pool).
- ``slu_b`` is Path B's single hypothesis. ``None`` means Path B abstained
  (no audio available, or the backend chose not to commit).
- If Path B picks an intent that isn't in the scene's candidate pool
  (e.g. a global E2E head suggesting something off-scope) it's silently
  dropped — Path A's ranking wins by default.
- Output is sorted descending; zero-scored entries are filtered out so
  the response shape stays clean.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from sandroid.core.domain import Intent, IntentId
    from sandroid.core.matcher import MatchCandidate
    from sandroid.models.slu.base import FinalIntent


class FusionWeights(BaseModel):
    """Per-scene Path A / Path B weights. Must be in [0, 1]; not required to sum to 1."""

    model_config = {"frozen": True}

    w_a: float = Field(ge=0.0, le=1.0)
    w_b: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _at_least_one_nonzero(self) -> FusionWeights:
        if self.w_a == 0.0 and self.w_b == 0.0:
            raise ValueError("FusionWeights: w_a and w_b cannot both be zero")
        return self

    def normalized(self) -> tuple[float, float]:
        """Return (w_a, w_b) rescaled so they sum to 1.0.

        Lets users specify e.g. (0.6, 0.6) without the engine over-confident-ing
        — the fused score stays in [0, 1].
        """
        total = self.w_a + self.w_b
        return self.w_a / total, self.w_b / total


class FusionConfig(BaseModel):
    """Loaded fusion.yaml: a default for unknown scenes plus per-scene overrides."""

    default: FusionWeights
    scenes: dict[str, FusionWeights] = Field(default_factory=dict)

    def for_scene(self, scene_id: str | None) -> FusionWeights:
        if scene_id is not None and scene_id in self.scenes:
            return self.scenes[scene_id]
        return self.default


# Conservative default used when fusion.yaml is missing entirely.
# (1.0, 0.0) means "Path A only" — a safe no-op until scenes opt in.
DEFAULT_CONFIG = FusionConfig(default=FusionWeights(w_a=1.0, w_b=0.0))


def load_fusion_config(path: Path | str | None) -> FusionConfig:
    """Load fusion.yaml. ``None`` or a missing file yields ``DEFAULT_CONFIG``."""
    if path is None:
        return DEFAULT_CONFIG
    p = Path(path)
    if not p.exists():
        return DEFAULT_CONFIG
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"fusion.yaml must be a mapping at top level, got {type(raw).__name__}")
    default_block = raw.get("default") or {"w_a": 1.0, "w_b": 0.0}
    scenes_block = raw.get("scenes") or {}
    return FusionConfig(
        default=FusionWeights(**default_block),
        scenes={
            name: FusionWeights(**weights)
            for name, weights in scenes_block.items()
        },
    )


def fuse(
    ranked_a: list[MatchCandidate],
    slu_b: FinalIntent | None,
    candidates: list[Intent],
    weights: FusionWeights,
    *,
    n_best: int = 5,
) -> list[MatchCandidate]:
    """Combine Path A's ranked list with Path B's single hypothesis.

    Out-of-scope Path B intents are silently dropped (CLAUDE.md: ``followUp``
    handles cross-scene jumps; Fusion is purely about decision *within* a
    scene's candidate pool).
    """
    # When Path B has nothing to contribute (no audio, no SLU configured, or
    # the backend abstained), Fusion must be a no-op on Path A's scores —
    # otherwise per-scene weights would silently shrink confidence even on
    # text-only requests. Early-return preserves the pre-5f-b behaviour
    # algebraically.
    b_active = (
        slu_b is not None
        and slu_b.intent_id is not None
        and any(c.id == slu_b.intent_id for c in candidates)
    )
    if not b_active:
        return list(ranked_a)[:n_best]

    # Local imports to keep this module's import graph small.
    from sandroid.core.matcher import MatchCandidate  # noqa: PLC0415

    w_a, w_b = weights.normalized()
    candidate_ids = {c.id for c in candidates}

    scores: dict[IntentId, float] = {iid: 0.0 for iid in candidate_ids}
    for cand in ranked_a:
        if cand.intent_id in scores:
            scores[cand.intent_id] = w_a * cand.confidence

    # b_active implies slu_b and its intent_id are not None, and the intent
    # is in scope; mypy needs the explicit narrow.
    assert slu_b is not None and slu_b.intent_id is not None
    scores[slu_b.intent_id] += w_b * slu_b.confidence

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    fused = [
        MatchCandidate(intent_id=iid, confidence=score)
        for iid, score in ranked
        if score > 0.0
    ]
    return fused[:n_best]
