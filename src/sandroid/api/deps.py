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
from sandroid.models.asr.paraformer import ParaformerASR, ParaformerError
from sandroid.models.asr.stub import StubASR
from sandroid.models.nlu.onnx_embedder import ONNXEmbedderError, ONNXEmbedderMatcher
from sandroid.models.slu.base import SLUBackend
from sandroid.models.slu.stub import StubSLU
from sandroid.storage.scene_registry import SceneRegistry
from sandroid.storage.session_store import InMemorySessionStore, SessionStore
from sandroid.vad.base import VoiceActivityDetector
from sandroid.vad.mock import MockVAD
from sandroid.vad.silero import SileroVAD, SileroVADError

_MODELS_ROOT = Path(__file__).resolve().parents[3] / "deploy" / "models"
DEFAULT_SCENES_DIR = Path(__file__).resolve().parents[3] / "configs" / "scenes"
DEFAULT_SILERO_PATH = _MODELS_ROOT / "silero_vad.onnx"
DEFAULT_NLU_MODEL_PATH = _MODELS_ROOT / "nlu_embedder_quantized.onnx"
DEFAULT_NLU_TOKENIZER_PATH = _MODELS_ROOT / "nlu_tokenizer.json"
DEFAULT_ASR_MODEL_PATH = _MODELS_ROOT / "paraformer_zh.int8.onnx"
DEFAULT_ASR_TOKENS_PATH = _MODELS_ROOT / "paraformer_zh.tokens.txt"
DEFAULT_ASR_CMVN_PATH = _MODELS_ROOT / "paraformer_zh.am.mvn"
API_KEY_ENV = "SANDROID_API_KEY"
DEV_DEFAULT_API_KEY = "dev-insecure-change-me"
ENV_ENV = "SANDROID_ENV"  # "dev" (default) | "production"
VAD_BACKEND_ENV = "SANDROID_VAD_BACKEND"  # "silero" (default) | "mock"
SILERO_PATH_ENV = "SANDROID_SILERO_PATH"
NLU_BACKEND_ENV = "SANDROID_NLU_BACKEND"  # "embedder" (default) | "stub"
NLU_MODEL_PATH_ENV = "SANDROID_NLU_MODEL_PATH"
NLU_TOKENIZER_PATH_ENV = "SANDROID_NLU_TOKENIZER_PATH"
ASR_BACKEND_ENV = "SANDROID_ASR_BACKEND"  # "paraformer" (default) | "stub"
ASR_MODEL_PATH_ENV = "SANDROID_ASR_MODEL_PATH"
ASR_TOKENS_PATH_ENV = "SANDROID_ASR_TOKENS_PATH"
ASR_CMVN_PATH_ENV = "SANDROID_ASR_CMVN_PATH"
SLU_BACKEND_ENV = "SANDROID_SLU_BACKEND"  # "stub" (default); "wav2vec2" lands in Phase 5f-c
MAX_AUDIO_BYTES_ENV = "SANDROID_MAX_AUDIO_BYTES"  # entry-point ceiling, default 8 MiB
DEFAULT_MAX_AUDIO_BYTES = 8 * 1024 * 1024  # ~4 minutes of PCM16 16k mono


def get_max_audio_bytes() -> int:
    """Maximum audio buffer size accepted by REST/WS/MRCP entry points.

    Caps memory pressure under malicious or accidental long uploads. Tunable
    via the SANDROID_MAX_AUDIO_BYTES env var. Each ~960 KB of PCM16 buffer
    is roughly 30 seconds of phone audio at 16 kHz mono, so the default
    accommodates ordinary IVR turns and rejects anything that looks like
    an extended recording attack.
    """
    raw = os.environ.get(MAX_AUDIO_BYTES_ENV)
    if raw is None:
        return DEFAULT_MAX_AUDIO_BYTES
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"{MAX_AUDIO_BYTES_ENV} must be an integer; got {raw!r}"
        ) from e
    if value <= 0:
        raise ValueError(f"{MAX_AUDIO_BYTES_ENV} must be positive; got {value}")
    return value


def _is_production() -> bool:
    """Phase 5d gate: dev default falls back to stubs, production must fail loud.

    Dev ergonomics (artifacts not fetched -> boot cleanly with stubs) are
    preserved; production environments set SANDROID_ENV=production to turn
    every artifact error into a boot failure so missing models never silently
    downgrade serving quality.
    """

    return os.environ.get(ENV_ENV, "dev").lower() == "production"


@lru_cache(maxsize=1)
def get_registry() -> SceneRegistry:
    directory = os.environ.get("SANDROID_SCENES_DIR", str(DEFAULT_SCENES_DIR))
    return SceneRegistry.from_directory(directory)


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    return InMemorySessionStore()


@lru_cache(maxsize=1)
def get_matcher() -> IntentMatcher:
    backend = os.environ.get(NLU_BACKEND_ENV, "embedder").lower()
    if backend == "stub":
        return StubMatcher()
    if backend != "embedder":
        raise ValueError(f"unknown NLU backend {backend!r}; expected 'embedder' or 'stub'")
    model_path = Path(os.environ.get(NLU_MODEL_PATH_ENV, str(DEFAULT_NLU_MODEL_PATH)))
    tokenizer_path = Path(
        os.environ.get(NLU_TOKENIZER_PATH_ENV, str(DEFAULT_NLU_TOKENIZER_PATH))
    )
    try:
        return ONNXEmbedderMatcher(model_path=model_path, tokenizer_path=tokenizer_path)
    except ONNXEmbedderError:
        if _is_production():
            raise
        # Dev fallback: boot cleanly when artifacts aren't fetched yet.
        return StubMatcher()


@lru_cache(maxsize=1)
def get_asr() -> ASRBackend:
    backend = os.environ.get(ASR_BACKEND_ENV, "paraformer").lower()
    if backend == "stub":
        return StubASR()
    if backend != "paraformer":
        raise ValueError(f"unknown ASR backend {backend!r}; expected 'paraformer' or 'stub'")
    model_path = Path(os.environ.get(ASR_MODEL_PATH_ENV, str(DEFAULT_ASR_MODEL_PATH)))
    tokens_path = Path(os.environ.get(ASR_TOKENS_PATH_ENV, str(DEFAULT_ASR_TOKENS_PATH)))
    cmvn_path = Path(os.environ.get(ASR_CMVN_PATH_ENV, str(DEFAULT_ASR_CMVN_PATH)))
    try:
        return ParaformerASR(
            model_path=model_path,
            tokens_path=tokens_path,
            cmvn_path=cmvn_path,
        )
    except ParaformerError:
        if _is_production():
            raise
        # Dev fallback: boot cleanly when artifacts aren't fetched yet.
        return StubASR()


@lru_cache(maxsize=1)
def get_slu() -> SLUBackend:
    """End-to-end SLU (Path B). Default ``stub`` abstains so Fusion in 5f-b
    degenerates to Path A on hosts without the wav2vec2 model installed.
    The ``wav2vec2`` backend lands in Phase 5f-c — until then, configuring it
    is a hard configuration error rather than a silent fallback.
    """

    backend = os.environ.get(SLU_BACKEND_ENV, "stub").lower()
    if backend == "stub":
        return StubSLU()
    raise ValueError(f"unknown SLU backend {backend!r}; expected 'stub'")


@lru_cache(maxsize=1)
def get_vad() -> VoiceActivityDetector:
    backend = os.environ.get(VAD_BACKEND_ENV, "silero").lower()
    if backend == "mock":
        return MockVAD()
    if backend != "silero":
        raise ValueError(f"unknown VAD backend {backend!r}; expected 'silero' or 'mock'")
    model_path = Path(os.environ.get(SILERO_PATH_ENV, str(DEFAULT_SILERO_PATH)))
    try:
        return SileroVAD(model_path=model_path)
    except SileroVADError:
        if _is_production():
            raise
        # Dev fallback: boot cleanly when the ONNX file isn't fetched yet.
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
    get_slu.cache_clear()
    get_vad.cache_clear()
