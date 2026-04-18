from __future__ import annotations

from sandroid.core.domain import Intent
from sandroid.core.matcher import StubMatcher


def _intent(intent_id: str, questions: list[str]) -> Intent:
    return Intent(id=intent_id, category_id="c", question=questions)


def test_matcher_returns_candidates_sorted_desc() -> None:
    matcher = StubMatcher()
    intents = [
        _intent("a", ["今天天气怎么样"]),
        _intent("b", ["你好"]),
        _intent("c", ["我要转账"]),
    ]
    ranked = matcher.score("今天天气如何", intents, n_best=5)
    assert ranked
    assert ranked[0].intent_id == "a"
    confidences = [c.confidence for c in ranked]
    assert confidences == sorted(confidences, reverse=True)
    assert all(0.0 < c.confidence <= 1.0 for c in ranked)


def test_matcher_drops_zero_matches() -> None:
    matcher = StubMatcher()
    intents = [_intent("a", ["完全不相关的问题"])]
    ranked = matcher.score("xyz", intents)
    assert ranked == []


def test_matcher_respects_n_best() -> None:
    matcher = StubMatcher()
    intents = [_intent(f"i{i}", ["你好世界"]) for i in range(10)]
    ranked = matcher.score("你好", intents, n_best=3)
    assert len(ranked) == 3


def test_matcher_exact_match_scores_one() -> None:
    matcher = StubMatcher()
    intents = [_intent("a", ["你好世界"])]
    ranked = matcher.score("你好世界", intents)
    assert ranked[0].confidence == 1.0


def test_matcher_ties_break_by_intent_id() -> None:
    matcher = StubMatcher()
    intents = [_intent("z_intent", ["你好"]), _intent("a_intent", ["你好"])]
    ranked = matcher.score("你好", intents)
    assert [c.intent_id for c in ranked] == ["a_intent", "z_intent"]
