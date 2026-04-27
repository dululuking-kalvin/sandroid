"""get_slu() factory tests (Phase 5f-a).

Mirrors the backend-selection tests for get_asr / get_matcher in test_phase5d.
StubSLU has no model artifacts so we don't need a SANDROID_ENV production
gate test — there's nothing to fail-loud on yet.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from sandroid.api.deps import (
    SLU_BACKEND_ENV,
    get_slu,
    reset_dependency_caches,
)
from sandroid.models.slu.stub import StubSLU


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(SLU_BACKEND_ENV, raising=False)
    reset_dependency_caches()
    yield
    reset_dependency_caches()


def test_get_slu_default_is_stub() -> None:
    assert isinstance(get_slu(), StubSLU)


def test_get_slu_explicit_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SLU_BACKEND_ENV, "stub")
    assert isinstance(get_slu(), StubSLU)


def test_get_slu_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SLU_BACKEND_ENV, "STUB")
    assert isinstance(get_slu(), StubSLU)


def test_get_slu_wav2vec2_not_yet_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 5f-c will add the wav2vec2 branch; until then it's a config error."""
    monkeypatch.setenv(SLU_BACKEND_ENV, "wav2vec2")
    with pytest.raises(ValueError, match="unknown SLU backend"):
        get_slu()


def test_get_slu_unknown_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SLU_BACKEND_ENV, "garbage")
    with pytest.raises(ValueError, match="unknown SLU backend"):
        get_slu()


def test_get_slu_is_singleton() -> None:
    """@lru_cache means repeated calls return the same instance."""
    a = get_slu()
    b = get_slu()
    assert a is b
