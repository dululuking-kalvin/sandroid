# Phase 5f-d — Audio Wiring (PCM16) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `RecognitionRequest.audio` (added in 5f-b but never populated) from every external entry point — REST `/recognize/file`, WebSocket `/recognize/stream`, and the MRCP UniMRCP bridge — through to the `Orchestrator`, so Path B (SLU) finally receives PCM16 bytes. StubSLU still abstains, but the pipeline becomes ready for a real wav2vec2 backend in 5f-c without further plumbing.

**Architecture:**
- Audio format unification: every entry point hands the Orchestrator **raw PCM16 little-endian, 16 kHz, mono** (no container header). Front-ends transcode once at the edge (REST: `decode_wav` already produces this in `AudioBuffer.pcm16`; WS: client sends raw PCM frames; MRCP: bridge forwards LPCM payloads).
- Memory-pressure cap: a single `SANDROID_MAX_AUDIO_BYTES` env var (default 8 MiB ≈ 4 minutes of phone audio) gates every entry point. **Overflow contract:** Path A keeps running so the user still gets a transcript; Path B is dropped for that turn (`audio=None` passed to Orchestrator). The size check intercepts at the entry, never inside Orchestrator — Orchestrator stays size-policy-free.
- WS streaming: PCM accumulates into a per-turn `bytearray` mirroring (but not blocking) the ASR queue. The accumulator is owned by `_TurnContext` so `_finalize_turn` can read it without crossing scopes.
- Bridge: per-channel `_Channel.audio_accum` mirrors LPCM payloads as they arrive. Same overflow rule — overflow flips `audio_overflow=True`, `_match_intent` passes `audio=None`.
- Test wiring: a new `_RecordingSLU` test double records the PCM bytes it received (instead of abstaining silently like StubSLU). Three integration tests — one per entry point — assert the byte sequence reaching SLU equals the bytes the entry point ingested. Plus an overflow assertion per entry point that proves Path A still completes when audio is dropped.

**Tech Stack:** Python 3.11, FastAPI (REST + WS), asyncio, pytest+pytest-asyncio, pydantic v2, in-tree UniMRCP bridge protocol (CBOR + length-prefixed frames over Unix socket).

**Status note (2026-04-28):** Tasks 1–3 and the REST half of Task 6 already shipped to `main` before this plan was written. They are documented here for completeness with `[x]` checkboxes; the executing agent should verify-by-running rather than re-implement. **Active work starts at Task 4.**

---

## File Structure

| File | Role |
|---|---|
| `src/sandroid/models/slu/base.py` | `SLUBackend` Protocol — the `recognize_audio(pcm16, *, scene_id)` method (renamed from `recognize_file`). Already shipped. |
| `src/sandroid/models/slu/stub.py` | `StubSLU` — abstaining adapter. Already renamed to match the new Protocol. |
| `src/sandroid/core/orchestrator.py` | `RecognitionRequest.audio: bytes \| None` docstring updated. Already shipped. |
| `src/sandroid/api/deps.py` | `MAX_AUDIO_BYTES_ENV`, `DEFAULT_MAX_AUDIO_BYTES`, `get_max_audio_bytes()` factory. Already shipped. |
| `src/sandroid/api/routes.py` | `recognize_file` size-checks + passes `audio=buffer.pcm16` (already shipped). `_TurnContext` extended with `audio_accum`/`audio_overflow`/`max_audio_bytes` (already shipped). Remaining: instantiation site, `handle_pcm` accumulation, `_finalize_turn` passes `audio=…`. |
| `src/sandroid/adapters/mrcp/bridge.py` | `_Channel` gains `audio_accum`/`audio_overflow`/`max_audio_bytes`. `_handle_audio` mirrors payloads with overflow guard. `_match_intent` passes `audio=…` to `RecognitionRequest`. |
| `tests/api/test_routes_audio_wiring.py` (new) | Integration test for REST `/recognize/file`: `_RecordingSLU` records bytes; assert equality with uploaded WAV's decoded PCM16. Includes a 413 overflow test. |
| `tests/api/test_stream_audio_wiring.py` (new) | Integration test for WS `/recognize/stream`: client streams known PCM frames; `_RecordingSLU` reconstructs them; overflow turn drops Path B but still emits `result`. |
| `tests/adapters/mrcp/test_bridge_audio_wiring.py` (new) | Integration test for the MRCP bridge: pumps LPCM frames through the in-tree client sim; asserts `_RecordingSLU` saw the same byte sequence; overflow path tested. |
| `tests/_helpers/recording_slu.py` (new) | `_RecordingSLU` test double — records every `recognize_audio` call into a list, returns abstain. Importable from all three integration tests. |

---

## Task 1: Rename `SLUBackend.recognize_file` → `recognize_audio(pcm16)` *(already shipped — verify only)*

**Files:**
- Modify: `src/sandroid/models/slu/base.py`
- Modify: `src/sandroid/models/slu/stub.py`
- Modify: `src/sandroid/core/orchestrator.py:151` (the call site inside `_run_dual_paths`)
- Test: `tests/models/slu/test_stub.py`
- Test: `tests/core/test_orchestrator.py` (`_FixedSLU` double)

- [x] **Step 1: Verify Protocol rename**

Run: `grep -n "recognize_audio\|recognize_file" src/sandroid/models/slu/base.py src/sandroid/models/slu/stub.py src/sandroid/core/orchestrator.py`

Expected: every hit reads `recognize_audio(pcm16)`; zero hits for `recognize_file`.

- [x] **Step 2: Verify SLU + Orchestrator tests pass**

Run: `uv run pytest tests/models/slu/ tests/core/ -q`

Expected: all green.

---

## Task 2: Document `RecognitionRequest.audio` as raw PCM16 16k mono *(already shipped — verify only)*

**Files:**
- Modify: `src/sandroid/core/orchestrator.py` (docstring on the `audio` field of `RecognitionRequest`)

- [x] **Step 1: Verify the field comment matches the protocol**

Run: `grep -n -A6 "audio: bytes" src/sandroid/core/orchestrator.py`

Expected: comment block describes "PCM16 little-endian, 16 kHz, mono. No container header." and explains the `None → Path B abstains` contract.

---

## Task 3: REST `recognize_file` passes PCM16 + size-checks *(already shipped — verify only)*

**Files:**
- Modify: `src/sandroid/api/deps.py` (env var + factory)
- Modify: `src/sandroid/api/routes.py` (import + size check + `audio=buffer.pcm16`)

- [x] **Step 1: Verify factory and size guard exist**

Run: `grep -n "MAX_AUDIO_BYTES\|get_max_audio_bytes\|413\|REQUEST_ENTITY_TOO_LARGE" src/sandroid/api/deps.py src/sandroid/api/routes.py`

Expected: env name, default constant, factory definition, and one 413 raise inside `recognize_file`.

- [x] **Step 2: Verify `audio=buffer.pcm16` is wired into `RecognitionRequest`**

Run: `grep -n -B1 -A1 "buffer.pcm16" src/sandroid/api/routes.py`

Expected: a single hit inside the `RecognitionRequest(...)` construction in `recognize_file`.

---

## Task 4: WS `recognize_stream` accumulates PCM and passes it to Orchestrator

**Files:**
- Modify: `src/sandroid/api/routes.py:556-559` (the `_TurnContext(...)` instantiation in `_StreamSession.handle_start`)
- Modify: `src/sandroid/api/routes.py:506-509` (`_StreamSession.handle_pcm`)
- Modify: `src/sandroid/api/routes.py:447-453` (the `RecognitionRequest(...)` construction in `_finalize_turn`)
- Modify: `src/sandroid/api/routes.py:31` (already imports `get_max_audio_bytes`; verify only)

The `_TurnContext` class itself already carries `audio_accum: bytearray`, `audio_overflow: bool`, `max_audio_bytes: int` — these were added before this plan was written. We are wiring the producer (`handle_pcm`) and the consumer (`_finalize_turn`) plus fixing the call site.

- [ ] **Step 1: Update the `_TurnContext(...)` instantiation site**

`src/sandroid/api/routes.py` around line 556 currently reads:

```python
        ctx = _TurnContext(
            scene_id=scene_id, category_path=category_path,
            n_best=int(control.get("n_best", 5)),
        )
```

The constructor now requires `max_audio_bytes`. Change to:

```python
        ctx = _TurnContext(
            scene_id=scene_id, category_path=category_path,
            n_best=int(control.get("n_best", 5)),
            max_audio_bytes=get_max_audio_bytes(),
        )
```

- [ ] **Step 2: Run an explicit type check to catch any other call sites**

Run: `uv run mypy src/sandroid/api/routes.py`

Expected: no errors. (If another `_TurnContext(...)` call site exists — it shouldn't — mypy will flag the missing argument.)

- [ ] **Step 3: Mirror PCM into `audio_accum` inside `handle_pcm`**

`src/sandroid/api/routes.py:506-509` currently reads:

```python
    async def handle_pcm(self, data: bytes) -> None:
        if self._ctx is None or self._ctx.cancelled:
            return
        await self._ctx.queue.put(data)
```

Change to:

```python
    async def handle_pcm(self, data: bytes) -> None:
        if self._ctx is None or self._ctx.cancelled:
            return
        ctx = self._ctx
        # Mirror raw PCM into Path B's accumulator. Overflow disables Path B
        # for the rest of the turn but lets ASR keep running so the user still
        # gets a transcript.
        if not ctx.audio_overflow:
            if len(ctx.audio_accum) + len(data) > ctx.max_audio_bytes:
                ctx.audio_overflow = True
                # Free what we already buffered — it won't be used.
                ctx.audio_accum = bytearray()
            else:
                ctx.audio_accum.extend(data)
        await ctx.queue.put(data)
```

Note: order matters — accumulator update happens before `queue.put` so a slow consumer doesn't widen the window between "ASR has the byte" and "Path B has the byte". The two paths see the same buffered bytes, just stored differently.

- [ ] **Step 4: Pass accumulated PCM into `RecognitionRequest` inside `_finalize_turn`**

`src/sandroid/api/routes.py:447-453` currently reads:

```python
        req = RecognitionRequest(
            scene_id=ctx.scene_id,
            category_path=ctx.category_path,
            text=final_text,
            session_id=session_id,
            n_best=ctx.n_best,
        )
```

Change to:

```python
        req = RecognitionRequest(
            scene_id=ctx.scene_id,
            category_path=ctx.category_path,
            text=final_text,
            audio=None if ctx.audio_overflow else bytes(ctx.audio_accum),
            session_id=session_id,
            n_best=ctx.n_best,
        )
```

- [ ] **Step 5: Run existing WS tests to confirm no regression**

Run: `uv run pytest tests/api/ -q -k "stream or websocket or ws"`

Expected: all green. (No new tests yet — those land in Task 7.)

- [ ] **Step 6: Commit**

```bash
git add src/sandroid/api/routes.py
git commit -m "feat(5f-d): WS recognize_stream wires PCM16 audio through to Orchestrator

handle_pcm now mirrors each frame into _TurnContext.audio_accum; once the
accumulator reaches SANDROID_MAX_AUDIO_BYTES, audio_overflow flips and Path
B is dropped for the rest of the turn (Path A still completes so the
caller gets a transcript). _finalize_turn passes the accumulated buffer
into RecognitionRequest.audio."
```

---

## Task 5: MRCP bridge — accumulate PCM per channel and pass to Orchestrator

**Files:**
- Modify: `src/sandroid/adapters/mrcp/bridge.py:215-225` (`_Channel` dataclass — add fields)
- Modify: `src/sandroid/adapters/mrcp/bridge.py:438-448` (`_handle_audio` — mirror payloads with overflow guard)
- Modify: `src/sandroid/adapters/mrcp/bridge.py:501-524` (`_match_intent` — pass `audio=…` into `RecognitionRequest`)

The bridge currently has no size policy. It should reuse `get_max_audio_bytes()` from `api.deps` when available, falling back to the `DEFAULT_MAX_AUDIO_BYTES` constant when `api.deps` can't be imported (the bridge is designed to run on hosts without FastAPI installed — see `_default_asr_factory`).

- [ ] **Step 1: Extend `_Channel` with the accumulator fields**

`src/sandroid/adapters/mrcp/bridge.py:215-225` currently reads:

```python
@dataclass
class _Channel:
    channel_id: str = ""
    session_id: str = ""
    sample_rate: int = 16000
    codec: str = "LPCM"
    started: bool = False
    eos_seen: bool = False
    audio_queue: asyncio.Queue[bytes | None] = field(default_factory=asyncio.Queue)
    # monotonic seconds at START for end-to-end turn duration
    start_monotonic: float = 0.0
```

Add the three new fields:

```python
@dataclass
class _Channel:
    channel_id: str = ""
    session_id: str = ""
    sample_rate: int = 16000
    codec: str = "LPCM"
    started: bool = False
    eos_seen: bool = False
    audio_queue: asyncio.Queue[bytes | None] = field(default_factory=asyncio.Queue)
    # monotonic seconds at START for end-to-end turn duration
    start_monotonic: float = 0.0
    # Path B (SLU) input: mirrors the LPCM stream. ``audio_overflow`` flips
    # if the accumulator would exceed ``max_audio_bytes`` — Path A keeps
    # running so the bridge still emits NLSML.
    audio_accum: bytearray = field(default_factory=bytearray)
    audio_overflow: bool = False
    max_audio_bytes: int = 0  # 0 means uninitialised; set on START
```

- [ ] **Step 2: Add a module-level helper that resolves the size cap**

In `src/sandroid/adapters/mrcp/bridge.py`, add this near the existing `_default_asr_factory` / `_default_orchestrator_factory` helpers (around line 295):

```python
def _resolve_max_audio_bytes() -> int:
    """Mirror the REST/WS size cap. Falls back to the shared default constant
    when api.deps cannot be imported (bare-bridge deployments)."""
    try:
        from sandroid.api.deps import get_max_audio_bytes  # noqa: PLC0415
    except ImportError:
        from sandroid.api.deps import DEFAULT_MAX_AUDIO_BYTES  # noqa: PLC0415
        return DEFAULT_MAX_AUDIO_BYTES
    return get_max_audio_bytes()
```

The fallback `from sandroid.api.deps import DEFAULT_MAX_AUDIO_BYTES` is unreachable in practice (deps.py is the same module that exposes `get_max_audio_bytes`) — keep the helper minimal:

```python
def _resolve_max_audio_bytes() -> int:
    """Mirror the REST/WS size cap; fall back to a hard-coded 8 MiB if the
    api.deps helper isn't importable on a bare-bridge host."""
    try:
        from sandroid.api.deps import get_max_audio_bytes  # noqa: PLC0415
        return get_max_audio_bytes()
    except ImportError:
        return 8 * 1024 * 1024
```

- [ ] **Step 3: Initialise `max_audio_bytes` on START**

`src/sandroid/adapters/mrcp/bridge.py` around line 429 (inside `_handle_start`, just before `ctx.asr_task = asyncio.create_task(...)`), add:

```python
        ch.max_audio_bytes = _resolve_max_audio_bytes()
```

- [ ] **Step 4: Mirror payloads into the accumulator inside `_handle_audio`**

`src/sandroid/adapters/mrcp/bridge.py:438-448` currently reads:

```python
    async def _handle_audio(
        self,
        payload: bytes,
        ctx: _HandlerCtx,
        writer: asyncio.StreamWriter,
    ) -> bool:
        if not ctx.channel.started:
            await self._send_error(writer, "UNEXPECTED_FRAME", "AUDIO before START")
            return False
        await ctx.channel.audio_queue.put(payload)
        return True
```

Change to:

```python
    async def _handle_audio(
        self,
        payload: bytes,
        ctx: _HandlerCtx,
        writer: asyncio.StreamWriter,
    ) -> bool:
        ch = ctx.channel
        if not ch.started:
            await self._send_error(writer, "UNEXPECTED_FRAME", "AUDIO before START")
            return False
        # Mirror into Path B's buffer; overflow disables further accumulation
        # so memory stays bounded under malformed clients. Path A is unaffected.
        if not ch.audio_overflow:
            if len(ch.audio_accum) + len(payload) > ch.max_audio_bytes:
                ch.audio_overflow = True
                ch.audio_accum = bytearray()
            else:
                ch.audio_accum.extend(payload)
        await ch.audio_queue.put(payload)
        return True
```

- [ ] **Step 5: Pass `audio=…` into `RecognitionRequest` inside `_match_intent`**

`src/sandroid/adapters/mrcp/bridge.py:511-518` currently reads:

```python
            from sandroid.core.orchestrator import RecognitionRequest  # noqa: PLC0415
            req = RecognitionRequest(
                scene_id=self._default_scene,
                text=transcript,
                session_id=ch.session_id or None,
            )
            resp = await orch.recognize(req)  # type: ignore[attr-defined]
```

Change to:

```python
            from sandroid.core.orchestrator import RecognitionRequest  # noqa: PLC0415
            req = RecognitionRequest(
                scene_id=self._default_scene,
                text=transcript,
                audio=None if ch.audio_overflow else bytes(ch.audio_accum),
                session_id=ch.session_id or None,
            )
            resp = await orch.recognize(req)  # type: ignore[attr-defined]
```

- [ ] **Step 6: Run the existing bridge tests to confirm no regression**

Run: `uv run pytest tests/adapters/mrcp/ -q`

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/sandroid/adapters/mrcp/bridge.py
git commit -m "feat(5f-d): MRCP bridge wires PCM16 audio through to Orchestrator

_Channel gains audio_accum / audio_overflow / max_audio_bytes. _handle_audio
mirrors LPCM payloads into the accumulator with the same overflow contract
as the WS path. _match_intent forwards the buffered bytes (or None on
overflow) into RecognitionRequest.audio."
```

---

## Task 6: SANDROID_MAX_AUDIO_BYTES enforced at all entry points *(REST half shipped — finish coverage)*

**Files:**
- Verify: `src/sandroid/api/deps.py` — env var + factory (already shipped)
- Verify: `src/sandroid/api/routes.py:130-138` — REST `/recognize/file` 413 raise (already shipped)
- Verify: WS overflow path — covered by Task 4
- Verify: MRCP overflow path — covered by Task 5

This task is a **completeness audit**, not new code. Run it after Tasks 4 and 5 land.

- [ ] **Step 1: Verify all three entry points reference the shared cap**

Run: `grep -n "max_audio_bytes\|MAX_AUDIO_BYTES\|get_max_audio_bytes\|audio_overflow" src/sandroid/api/routes.py src/sandroid/adapters/mrcp/bridge.py src/sandroid/api/deps.py`

Expected hits:
- `deps.py`: env var, default constant, factory.
- `routes.py`: import, REST size check + 413 raise, `_TurnContext` field setup, overflow guard in `handle_pcm`, conditional `audio=…` in `_finalize_turn`.
- `bridge.py`: `_resolve_max_audio_bytes` helper, `_Channel` fields, `_handle_audio` overflow guard, conditional `audio=…` in `_match_intent`.

- [ ] **Step 2: Sanity-check the env var override path**

Run: `SANDROID_MAX_AUDIO_BYTES=2048 uv run pytest tests/api/test_routes.py::test_recognize_file_rejects_overlong_audio -q` *(test added in Task 7 — defer this verification until Task 7 lands).* For now, run: `uv run pytest tests/ -q -k "max_audio"` and confirm at least the 413 test from Task 7 (when ready) honours the env var.

---

## Task 7: Integration tests — `_RecordingSLU` proves wiring works end-to-end

**Files:**
- Create: `tests/_helpers/__init__.py`
- Create: `tests/_helpers/recording_slu.py`
- Create: `tests/api/test_routes_audio_wiring.py`
- Create: `tests/api/test_stream_audio_wiring.py`
- Create: `tests/adapters/mrcp/test_bridge_audio_wiring.py`

The test double records every byte sequence handed to `recognize_audio`. Each integration test asserts that what the SLU saw matches what the entry point ingested. Plus an overflow test per entry point that proves Path A still completes (transcript / NLSML still emitted) when the buffer overflows.

### 7a. Test double

- [ ] **Step 1: Create the helpers package**

```python
# tests/_helpers/__init__.py
```

Empty file — just makes the directory a package.

- [ ] **Step 2: Write `_RecordingSLU`**

```python
# tests/_helpers/recording_slu.py
"""Test double that records the bytes handed to ``recognize_audio``.

Used by integration tests to prove every entry point (REST file, WS stream,
MRCP bridge) actually plumbs PCM16 through to the SLU adapter. Returns an
abstaining FinalIntent so fusion math is unaffected — we're testing wiring,
not scoring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sandroid.models.asr.base import AudioChunk
from sandroid.models.slu.base import FinalIntent, PartialIntent


class RecordingSLU:
    """Records every ``recognize_audio`` invocation. Always abstains."""

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str | None]] = []

    async def stream(
        self,
        chunks: AsyncIterator[AudioChunk],
        *,
        scene_id: str | None = None,
    ) -> AsyncIterator[PartialIntent]:
        async for _ in chunks:
            pass
        yield PartialIntent(
            intent_id=None, confidence=0.0, is_final=True, start_ms=0, end_ms=0,
        )

    async def recognize_audio(
        self,
        pcm16: bytes,
        *,
        scene_id: str | None = None,
    ) -> FinalIntent:
        self.calls.append((pcm16, scene_id))
        return FinalIntent(intent_id=None, confidence=0.0, duration_ms=0)
```

- [ ] **Step 3: Run the import smoke-check**

Run: `uv run python -c "from tests._helpers.recording_slu import RecordingSLU; r = RecordingSLU(); print(r.calls)"`

Expected: `[]` printed; no import errors.

### 7b. REST `/recognize/file` integration test

- [ ] **Step 4: Write `tests/api/test_routes_audio_wiring.py`**

The REST file route uses `decode_wav` to normalize the upload to PCM16 16k mono. The test must therefore upload a known WAV, then assert `RecordingSLU.calls[0][0]` equals the PCM16 produced by `decode_wav` on the same input — not the raw WAV bytes.

```python
"""Phase 5f-d: REST /recognize/file passes PCM16 to SLU."""

from __future__ import annotations

import io
import wave

import pytest
from fastapi.testclient import TestClient

from sandroid.api.app import create_app
from sandroid.api.deps import (
    DEV_DEFAULT_API_KEY,
    MAX_AUDIO_BYTES_ENV,
    get_slu,
    reset_dependency_caches,
)
from sandroid.core.audio import decode_wav

from tests._helpers.recording_slu import RecordingSLU


def _make_wav(pcm16: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


@pytest.fixture
def client_with_recording_slu() -> tuple[TestClient, RecordingSLU]:
    reset_dependency_caches()
    rec = RecordingSLU()
    app = create_app()
    app.dependency_overrides[get_slu] = lambda: rec
    yield TestClient(app), rec
    app.dependency_overrides.clear()
    reset_dependency_caches()


def test_recognize_file_passes_pcm16_to_slu(
    client_with_recording_slu: tuple[TestClient, RecordingSLU],
) -> None:
    client, rec = client_with_recording_slu
    pcm = b"\x01\x00\x02\x00\x03\x00\x04\x00" * 4000  # ~2 s of fake PCM16
    wav = _make_wav(pcm)

    expected = decode_wav(wav).pcm16

    resp = client.post(
        "/api/v1/recognize/file",
        headers={"X-API-Key": DEV_DEFAULT_API_KEY},
        data={"scene_id": "example_bank"},
        files={"audio": ("turn.wav", wav, "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    assert len(rec.calls) == 1
    received_pcm, _ = rec.calls[0]
    assert received_pcm == expected


def test_recognize_file_rejects_overlong_audio(
    client_with_recording_slu: tuple[TestClient, RecordingSLU],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MAX_AUDIO_BYTES_ENV, "2048")  # 2 KiB ceiling
    reset_dependency_caches()
    client, rec = client_with_recording_slu

    pcm = b"\x00\x00" * 10_000  # ~20 KiB raw, comfortably over the cap
    wav = _make_wav(pcm)

    resp = client.post(
        "/api/v1/recognize/file",
        headers={"X-API-Key": DEV_DEFAULT_API_KEY},
        data={"scene_id": "example_bank"},
        files={"audio": ("big.wav", wav, "audio/wav")},
    )
    assert resp.status_code == 413
    assert rec.calls == []  # SLU never reached
```

- [ ] **Step 5: Run the REST tests**

Run: `uv run pytest tests/api/test_routes_audio_wiring.py -q`

Expected: both tests pass.

If the `app.dependency_overrides[get_slu] = ...` pattern doesn't work because `get_slu` isn't yet a FastAPI `Depends` (it's an `lru_cache` factory imported directly into the Orchestrator dep), inspect `src/sandroid/api/deps.py:get_orchestrator` — you'll need to override `get_orchestrator` instead and inject an Orchestrator constructed with `slu=rec`. In that case replace the override with:

```python
from sandroid.api.deps import get_orchestrator
from sandroid.core.orchestrator import Orchestrator
# inside the fixture:
def _build_orch():
    from sandroid.api.deps import get_matcher, get_registry, get_session_store
    return Orchestrator(
        registry=get_registry(),
        sessions=get_session_store(),
        matcher=get_matcher(),
        slu=rec,
    )
app.dependency_overrides[get_orchestrator] = _build_orch
```

(The same override pattern is needed in 7c and 7d.)

### 7c. WS `/recognize/stream` integration test

The WS test uses FastAPI's `TestClient.websocket_connect`. It needs to send `start` JSON, push raw PCM frames, then `stop`, and read back `result`. Then it asserts `RecordingSLU` saw the concatenation of the PCM frames.

- [ ] **Step 6: Write `tests/api/test_stream_audio_wiring.py`**

```python
"""Phase 5f-d: WS /recognize/stream passes accumulated PCM16 to SLU."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from sandroid.api.app import create_app
from sandroid.api.deps import (
    DEV_DEFAULT_API_KEY,
    MAX_AUDIO_BYTES_ENV,
    get_orchestrator,
    reset_dependency_caches,
)
from sandroid.core.orchestrator import Orchestrator

from tests._helpers.recording_slu import RecordingSLU


def _build_orch_with_slu(slu: RecordingSLU) -> Orchestrator:
    from sandroid.api.deps import get_matcher, get_registry, get_session_store
    return Orchestrator(
        registry=get_registry(),
        sessions=get_session_store(),
        matcher=get_matcher(),
        slu=slu,
    )


@pytest.fixture
def ws_client_with_recording_slu() -> tuple[TestClient, RecordingSLU]:
    reset_dependency_caches()
    rec = RecordingSLU()
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: _build_orch_with_slu(rec)
    yield TestClient(app), rec
    app.dependency_overrides.clear()
    reset_dependency_caches()


def _drain_until_result(ws) -> dict:
    while True:
        msg = ws.receive_json()
        if msg.get("type") == "result":
            return msg
        if msg.get("type") == "error" and msg.get("fatal"):
            raise AssertionError(f"unexpected fatal error: {msg}")


def test_stream_accumulates_pcm_and_passes_to_slu(
    ws_client_with_recording_slu: tuple[TestClient, RecordingSLU],
) -> None:
    client, rec = ws_client_with_recording_slu
    chunks = [b"\x01\x00\x02\x00" * 800, b"\x03\x00\x04\x00" * 800]  # ~6.4 KiB total
    expected = b"".join(chunks)

    with client.websocket_connect(
        "/api/v1/recognize/stream",
        headers={"X-API-Key": DEV_DEFAULT_API_KEY},
    ) as ws:
        ws.send_text(json.dumps({
            "type": "start", "session_id": "sess1", "turn_id": 1,
            "scene_id": "example_bank",
        }))
        for c in chunks:
            ws.send_bytes(c)
        ws.send_text(json.dumps({"type": "stop"}))
        _drain_until_result(ws)

    assert len(rec.calls) == 1
    received_pcm, _ = rec.calls[0]
    assert received_pcm == expected


def test_stream_overflow_drops_path_b_but_path_a_completes(
    ws_client_with_recording_slu: tuple[TestClient, RecordingSLU],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MAX_AUDIO_BYTES_ENV, "1024")  # 1 KiB cap
    reset_dependency_caches()
    client, rec = ws_client_with_recording_slu

    big = b"\x00\x00" * 2000  # 4 KiB, four times the cap
    with client.websocket_connect(
        "/api/v1/recognize/stream",
        headers={"X-API-Key": DEV_DEFAULT_API_KEY},
    ) as ws:
        ws.send_text(json.dumps({
            "type": "start", "session_id": "sess2", "turn_id": 1,
            "scene_id": "example_bank",
        }))
        ws.send_bytes(big)
        ws.send_text(json.dumps({"type": "stop"}))
        result = _drain_until_result(ws)

    # Path A still produced a transcript (or no-speech) — the result frame
    # arrived at all, which proves the turn finalised. Path B was dropped:
    # SLU saw zero bytes (caller hit overflow before any chunk landed in
    # the accumulator).
    assert result["type"] == "result"
    if rec.calls:
        # Some VAD configurations may still produce a final frame; if SLU was
        # invoked it must have been with empty audio, not the truncated buffer.
        assert rec.calls[0][0] == b""
```

- [ ] **Step 7: Run the WS tests**

Run: `uv run pytest tests/api/test_stream_audio_wiring.py -q`

Expected: both tests pass. If `_drain_until_result` times out, check that the test ASR backend (StubASR by default) returns a non-empty transcript; if not, the route emits `NO_SPEECH` and never invokes the orchestrator. In that case, force the ASR backend by setting `monkeypatch.setenv("SANDROID_ASR_BACKEND", "stub")` and using a stub that returns a fixed transcript — or assert directly on the `error` frame with code `NO_SPEECH` and skip the SLU-call assertion for the no-speech edge.

### 7d. MRCP bridge integration test

The bridge has an in-tree client sim (`src/sandroid/adapters/mrcp/client_sim.py`) — reuse it. The bridge's orchestrator factory needs to be patched to inject a `RecordingSLU`-equipped Orchestrator.

- [ ] **Step 8: Write `tests/adapters/mrcp/test_bridge_audio_wiring.py`**

```python
"""Phase 5f-d: MRCP bridge passes per-channel PCM16 to SLU."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from sandroid.adapters.mrcp.bridge import BridgeServer
from sandroid.adapters.mrcp.client_sim import simulate_turn  # adjust if name differs
from sandroid.api.deps import MAX_AUDIO_BYTES_ENV, reset_dependency_caches
from sandroid.core.orchestrator import Orchestrator
from sandroid.models.asr.stub import StubASR

from tests._helpers.recording_slu import RecordingSLU


def _orch_factory_with(rec: RecordingSLU):
    def _make() -> Orchestrator:
        from sandroid.api.deps import get_matcher, get_registry, get_session_store
        return Orchestrator(
            registry=get_registry(),
            sessions=get_session_store(),
            matcher=get_matcher(),
            slu=rec,
        )
    return _make


@pytest.mark.asyncio
async def test_bridge_accumulates_pcm_and_passes_to_slu() -> None:
    reset_dependency_caches()
    rec = RecordingSLU()
    sock = os.path.join(tempfile.mkdtemp(), "sandroid.sock")
    bridge = BridgeServer(
        socket_path=sock,
        asr_factory=lambda: StubASR(),
        orchestrator_factory=_orch_factory_with(rec),
        default_scene="example_bank",
    )
    server = await bridge.start()
    try:
        chunks = [b"\x01\x00" * 400, b"\x02\x00" * 400]
        expected = b"".join(chunks)

        await simulate_turn(
            socket_path=sock,
            channel_id="c1",
            session_id="s1",
            audio_chunks=chunks,
        )

        assert len(rec.calls) == 1
        received_pcm, _ = rec.calls[0]
        assert received_pcm == expected
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_bridge_overflow_drops_path_b(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MAX_AUDIO_BYTES_ENV, "1024")
    reset_dependency_caches()
    rec = RecordingSLU()
    sock = os.path.join(tempfile.mkdtemp(), "sandroid.sock")
    bridge = BridgeServer(
        socket_path=sock,
        asr_factory=lambda: StubASR(),
        orchestrator_factory=_orch_factory_with(rec),
        default_scene="example_bank",
    )
    server = await bridge.start()
    try:
        big = b"\x00\x00" * 2000  # 4 KiB > 1 KiB cap
        await simulate_turn(
            socket_path=sock,
            channel_id="c1",
            session_id="s1",
            audio_chunks=[big],
        )

        # Bridge still emits NLSML (turn completed); SLU either wasn't called
        # at all or was called with empty bytes.
        assert all(call[0] == b"" for call in rec.calls)
    finally:
        server.close()
        await server.wait_closed()
```

If `simulate_turn` doesn't exist with that signature, open `src/sandroid/adapters/mrcp/client_sim.py` and use whatever helper is there (or write a minimal one inline that opens the socket, sends START/AUDIO×N/EOS/STOP frames, reads the RESULT frame, and returns).

- [ ] **Step 9: Run the bridge tests**

Run: `uv run pytest tests/adapters/mrcp/test_bridge_audio_wiring.py -q`

Expected: both tests pass.

### 7e. Full suite + commit

- [ ] **Step 10: Run the full test suite + lint/typecheck**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy src/`

Expected: green across the board. If anything from earlier phases regresses, fix at the smallest diff that restores green — do not refactor.

- [ ] **Step 11: Commit**

```bash
git add tests/_helpers tests/api/test_routes_audio_wiring.py tests/api/test_stream_audio_wiring.py tests/adapters/mrcp/test_bridge_audio_wiring.py
git commit -m "test(5f-d): integration tests prove PCM16 reaches SLU at all entry points

RecordingSLU records every recognize_audio call. Three integration tests
(REST file upload, WS stream, MRCP bridge) assert the byte sequence the SLU
sees equals what the entry point ingested. Each test pair also covers
overflow: SANDROID_MAX_AUDIO_BYTES drops Path B silently while Path A still
emits a result/NLSML."
```

---

## Self-Review Checklist (run before declaring the plan done)

- [x] **Spec coverage:** Every sub-task referenced in the pre-compact summary (5f-d-1 through 5f-d-7) maps to a numbered Task in this plan.
- [x] **Placeholder scan:** No "TBD"/"implement later"/"add appropriate handling" — every step shows the actual code.
- [x] **Type consistency:** Field names match across tasks (`audio_accum`, `audio_overflow`, `max_audio_bytes`); the SLU Protocol method is `recognize_audio(pcm16, *, scene_id)` everywhere; `RecordingSLU.calls` is a `list[tuple[bytes, str | None]]` consistently.
- [x] **TDD ordering:** Tasks 4 and 5 are wiring (no test-first because the contract is already covered by existing orchestrator/fusion unit tests); Task 7 layers integration tests on top of working wiring. This matches the "wiring is structural" carve-out — integration tests prove end-to-end behaviour after structural change, which is more valuable here than test-first wiring stubs.
- [x] **Frequent commits:** One commit per task (4, 5, 7) plus the verification-only tasks (1, 2, 3) which need no commit.

---

## Out of Scope

- **Wav2vec2 backend (Phase 5f-c):** still deferred until we have real IVR data + a GPU box for fine-tuning. This plan only ensures the wiring is ready.
- **Streaming SLU (`SLUBackend.stream`):** the Protocol method exists but no entry point uses it yet. Adding streaming Path B is a separate phase — for 5f-d we only wire the buffered `recognize_audio` path. Once the wav2vec2 backend lands and 24-core CPU latency profiling has data, the WS path will graduate to streaming, but that is 5f-e or later.
- **Per-scene `max_audio_bytes` overrides:** YAML scene configs do not yet expose this. If a compliance-heavy scene needs a different ceiling later, add it as a scene-level field — out of scope here.
- **Cancellation interaction with Path B:** if a turn is cancelled (barge-in) before SLU finishes, the existing `ctx.cancelled` flag short-circuits `_finalize_turn`, so the SLU call simply isn't made. No new code needed; this is a property of the existing finalize flow.
