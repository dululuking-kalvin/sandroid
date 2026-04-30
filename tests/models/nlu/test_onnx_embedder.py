"""Integration tests for the ONNX embedder matcher.

Skipped automatically when model artifacts are missing — mirrors the Silero
integration test pattern so contributors who haven't run
``scripts/fetch_models.py`` still get a green CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sandroid.core.domain import Intent
from sandroid.models.nlu.onnx_embedder import ONNXEmbedderError, ONNXEmbedderMatcher

_MODELS_DIR = Path(__file__).resolve().parents[3] / "deploy" / "models"
_MODEL_PATH = _MODELS_DIR / "nlu_embedder_quantized.onnx"
_TOKENIZER_PATH = _MODELS_DIR / "nlu_tokenizer.json"

pytestmark = pytest.mark.skipif(
    not (_MODEL_PATH.exists() and _TOKENIZER_PATH.exists()),
    reason=(f"NLU model artifacts missing under {_MODELS_DIR}; run scripts/fetch_models.py"),
)


@pytest.fixture(scope="module")
def matcher() -> ONNXEmbedderMatcher:
    return ONNXEmbedderMatcher(model_path=_MODEL_PATH, tokenizer_path=_TOKENIZER_PATH)


def _intent(intent_id: str, questions: list[str]) -> Intent:
    return Intent(id=intent_id, category_id="root", question=questions, answer=None)


def test_semantic_similarity_beats_unrelated(matcher: ONNXEmbedderMatcher) -> None:
    """A paraphrase of a training question should rank above unrelated intents.

    The transcript never appears verbatim in any ``question`` list — this is
    the property the character-n-gram ``StubMatcher`` can't deliver.
    """

    candidates = [
        _intent("greeting", ["你好", "您好，请问有什么可以帮您"]),
        _intent("balance", ["查询余额", "我想知道我卡里还剩多少钱"]),
        _intent("transfer", ["我要转账", "境内转账"]),
    ]
    results = matcher.score("帮我看下账户里有多少钱", candidates, n_best=3)

    assert results, "expected at least one candidate"
    assert results[0].intent_id == "balance"
    # Every confidence must be in [0, 1].
    for r in results:
        assert 0.0 <= r.confidence <= 1.0


def test_returns_sorted_descending(matcher: ONNXEmbedderMatcher) -> None:
    candidates = [
        _intent("a", ["苹果"]),
        _intent("b", ["香蕉"]),
        _intent("c", ["橘子"]),
    ]
    results = matcher.score("我想吃水果", candidates, n_best=3)
    confidences = [r.confidence for r in results]
    assert confidences == sorted(confidences, reverse=True)


def test_empty_transcript_returns_empty(matcher: ONNXEmbedderMatcher) -> None:
    candidates = [_intent("greeting", ["你好"])]
    assert matcher.score("", candidates) == []
    assert matcher.score("   ", candidates) == []


def test_no_candidates_returns_empty(matcher: ONNXEmbedderMatcher) -> None:
    assert matcher.score("你好", []) == []


def test_intents_without_questions_are_skipped(matcher: ONNXEmbedderMatcher) -> None:
    """MEMORY-mode intents (per configs/scenes/example_bank.yaml) have
    ``question: []`` and must not crash or receive a score."""

    candidates = [
        _intent("memory_terminal", []),
        _intent("greeting", ["你好"]),
    ]
    results = matcher.score("你好", candidates, n_best=5)
    intent_ids = {r.intent_id for r in results}
    assert "memory_terminal" not in intent_ids
    assert "greeting" in intent_ids


def test_n_best_truncates(matcher: ONNXEmbedderMatcher) -> None:
    candidates = [_intent(f"i{k}", [f"问题{k}"]) for k in range(10)]
    results = matcher.score("问题1", candidates, n_best=3)
    assert len(results) <= 3


def test_missing_model_raises() -> None:
    with pytest.raises(ONNXEmbedderError, match="missing"):
        ONNXEmbedderMatcher(
            model_path=Path("/does/not/exist.onnx"),
            tokenizer_path=_TOKENIZER_PATH,
        )
