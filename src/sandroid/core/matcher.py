"""Intent matching.

The Protocol is what the orchestrator depends on. Real ASR/NLU/SLU backends
(Phase 5+) will implement it. Until then, ``StubMatcher`` does character-n-gram
Jaccard over the transcript vs. each Intent's training questions — enough to
drive end-to-end integration tests without ML infrastructure.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from sandroid.core.domain import Intent, IntentId


class MatchCandidate(BaseModel):
    intent_id: IntentId
    confidence: float = Field(ge=0.0, le=1.0)


class IntentMatcher(Protocol):
    """Score candidates against a transcript and return them sorted desc."""

    def score(
        self,
        transcript: str,
        candidates: list[Intent],
        *,
        n_best: int = 5,
    ) -> list[MatchCandidate]: ...


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    text = text.strip()
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class StubMatcher:
    """Character bigram Jaccard. Deterministic, dependency-free, good enough.

    - Confidence is ``max_over_questions(jaccard(transcript, question))``.
    - Returns candidates with non-zero confidence, sorted descending.
    - Ties broken by ``intent_id`` for stable ordering in tests.
    """

    def __init__(self, *, ngram: int = 2) -> None:
        self._n = ngram

    def score(
        self,
        transcript: str,
        candidates: list[Intent],
        *,
        n_best: int = 5,
    ) -> list[MatchCandidate]:
        t_grams = _char_ngrams(transcript, self._n)
        scored: list[MatchCandidate] = []
        for intent in candidates:
            best = 0.0
            for question in intent.question:
                q_grams = _char_ngrams(question, self._n)
                score = _jaccard(t_grams, q_grams)
                best = max(best, score)
            if best > 0.0:
                scored.append(MatchCandidate(intent_id=intent.id, confidence=best))
        scored.sort(key=lambda c: (-c.confidence, c.intent_id))
        return scored[:n_best]
