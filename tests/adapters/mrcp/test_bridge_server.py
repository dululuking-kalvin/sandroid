"""End-to-end test of BridgeServer over a real Unix socket with StubASR."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from sandroid.adapters.mrcp.bridge import (
    FRAME_AUDIO,
    FRAME_EOS,
    FRAME_ERROR,
    FRAME_RESULT,
    FRAME_START,
    BridgeServer,
    _cbor_encode_map,
    _install_traffic_sink,
    encode_frame,
    read_frame,
    traffic_logger,
)
from sandroid.core.matcher import MatchCandidate
from sandroid.core.orchestrator import (
    RecognitionRequest,
    RecognitionResponse,
    TopHit,
)
from sandroid.models.asr.base import AudioChunk, PartialTranscript

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="bridge server requires Unix domain sockets",
)


class _FixedASR:
    """Deterministic ASR that emits a fixed final transcript on EOS."""

    def __init__(self, text: str = "hello world") -> None:
        self._text = text

    async def stream(self, chunks: AsyncIterator[AudioChunk]) -> AsyncIterator[PartialTranscript]:
        async for _ in chunks:
            pass
        yield PartialTranscript(text=self._text, is_final=True, start_ms=0, end_ms=0)


class _FakeOrchestrator:
    """Orchestrator-shaped double that ignores the transcript and returns a fixed top hit."""

    def __init__(self, intent_id: str, confidence: float) -> None:
        self._intent_id = intent_id
        self._confidence = confidence

    async def recognize(self, req: RecognitionRequest) -> RecognitionResponse:
        return RecognitionResponse(
            session_id=req.session_id or "sess-test",
            top=TopHit(
                intent_id=self._intent_id,
                confidence=self._confidence,
                answer=None,
                follow_up=None,
            ),
            n_best=[MatchCandidate(intent_id=self._intent_id, confidence=self._confidence)],
            follow_up=None,
            current_category_id="ROOT",
        )


async def _start_server(
    path: str,
    *,
    orchestrator: object | None = None,
    default_scene: str | None = None,
) -> BridgeServer:
    # Always inject an explicit orchestrator_factory so tests don't resolve
    # the real api.deps chain (which would pick up whatever artefacts happen
    # to be installed on the test host).
    server = BridgeServer(
        socket_path=path,
        asr_factory=_FixedASR,
        orchestrator_factory=lambda: orchestrator,
        default_scene=default_scene,
    )
    await server.start()
    return server


async def _connect(path: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await asyncio.open_unix_connection(path=path)  # type: ignore[attr-defined,unused-ignore]


async def test_full_recognition_flow() -> None:
    with tempfile.TemporaryDirectory() as d:
        sock = str(Path(d) / "bridge.sock")
        server = await _start_server(sock)
        try:
            reader, writer = await _connect(sock)
            start_payload = _cbor_encode_map(
                {
                    "channel_id": "ch-1",
                    "session_id": "sess-1",
                    "sample_rate": 16000,
                    "codec": "LPCM",
                }
            )
            writer.write(encode_frame(FRAME_START, start_payload))
            # 3 fake PCM chunks
            for _ in range(3):
                writer.write(encode_frame(FRAME_AUDIO, b"\x00\x00" * 320))
            writer.write(encode_frame(FRAME_EOS, b""))
            await writer.drain()

            ftype, payload = await asyncio.wait_for(read_frame(reader), timeout=5.0)
            assert ftype == FRAME_RESULT
            body = payload.decode()
            assert "hello world" in body
            assert "PLACEHOLDER_INTENT" in body
            writer.close()
            await writer.wait_closed()
        finally:
            if server._server is not None:
                server._server.close()
                await server._server.wait_closed()


async def _run_recog_turn(server: BridgeServer, sock: str) -> tuple[int, bytes]:
    """Drive one full START -> AUDIO*3 -> EOS turn; return (ftype, payload)."""
    reader, writer = await _connect(sock)
    start_payload = _cbor_encode_map(
        {
            "channel_id": "ch-1",
            "session_id": "sess-1",
            "sample_rate": 16000,
            "codec": "LPCM",
        }
    )
    writer.write(encode_frame(FRAME_START, start_payload))
    for _ in range(3):
        writer.write(encode_frame(FRAME_AUDIO, b"\x00\x00" * 320))
    writer.write(encode_frame(FRAME_EOS, b""))
    await writer.drain()
    ftype, payload = await asyncio.wait_for(read_frame(reader), timeout=5.0)
    writer.close()
    await writer.wait_closed()
    return ftype, payload


async def test_orchestrator_intent_lands_in_nlsml() -> None:
    """When a scene + orchestrator are configured, NLSML carries the matched intent."""
    orch = _FakeOrchestrator(intent_id="check_balance", confidence=0.83)
    with tempfile.TemporaryDirectory() as d:
        sock = str(Path(d) / "bridge.sock")
        server = await _start_server(sock, orchestrator=orch, default_scene="example_bank")
        try:
            ftype, payload = await _run_recog_turn(server, sock)
            assert ftype == FRAME_RESULT
            body = payload.decode()
            assert "<instance>check_balance</instance>" in body
            assert 'confidence="0.83"' in body
            assert "PLACEHOLDER_INTENT" not in body
        finally:
            if server._server is not None:
                server._server.close()
                await server._server.wait_closed()


async def test_traffic_log_emits_json_line_per_turn() -> None:
    """Enabling the traffic sink writes exactly one JSON line with expected fields."""
    orch = _FakeOrchestrator(intent_id="transfer", confidence=0.65)
    with tempfile.TemporaryDirectory() as d:
        sock = str(Path(d) / "bridge.sock")
        log_path = str(Path(d) / "traffic.log")
        # Clean slate: remove any handler left over from a prior test run.
        for h in list(traffic_logger.handlers):
            traffic_logger.removeHandler(h)
        _install_traffic_sink(log_path)
        try:
            server = await _start_server(sock, orchestrator=orch, default_scene="example_bank")
            try:
                ftype, _ = await _run_recog_turn(server, sock)
                assert ftype == FRAME_RESULT
            finally:
                if server._server is not None:
                    server._server.close()
                    await server._server.wait_closed()

            # Flush all traffic handlers so the file on disk is complete.
            for h in traffic_logger.handlers:
                h.flush()
            lines = Path(log_path).read_text(encoding="utf-8").splitlines()
            assert len(lines) == 1
            rec = json.loads(lines[0])
            assert rec["intent_id"] == "transfer"
            assert rec["confidence"] == 0.65
            assert rec["scene_id"] == "example_bank"
            assert rec["transcript"] == "hello world"
            assert rec["channel_id"] == "ch-1"
            assert rec["session_id"] == "sess-1"
            assert isinstance(rec["asr_ms"], int)
            assert isinstance(rec["nlu_ms"], int)
            assert isinstance(rec["turn_ms"], int)
        finally:
            for h in list(traffic_logger.handlers):
                traffic_logger.removeHandler(h)
                h.close()


async def test_orchestrator_none_falls_back_to_placeholder() -> None:
    """Explicit orchestrator_factory returning None -> NLSML keeps the placeholder."""
    with tempfile.TemporaryDirectory() as d:
        sock = str(Path(d) / "bridge.sock")
        # No orchestrator injected AND default_scene unset -> _match_intent takes the fallback.
        server = await _start_server(sock)
        try:
            ftype, payload = await _run_recog_turn(server, sock)
            assert ftype == FRAME_RESULT
            body = payload.decode()
            assert "PLACEHOLDER_INTENT" in body
            assert 'confidence="0.90"' in body
        finally:
            if server._server is not None:
                server._server.close()
                await server._server.wait_closed()


async def test_audio_before_start_triggers_error() -> None:
    with tempfile.TemporaryDirectory() as d:
        sock = str(Path(d) / "bridge.sock")
        server = await _start_server(sock)
        try:
            reader, writer = await _connect(sock)
            writer.write(encode_frame(FRAME_AUDIO, b"\x00\x00"))
            await writer.drain()
            ftype, payload = await asyncio.wait_for(read_frame(reader), timeout=5.0)
            assert ftype == FRAME_ERROR
            assert b"UNEXPECTED_FRAME" in payload
            writer.close()
            await writer.wait_closed()
        finally:
            if server._server is not None:
                server._server.close()
                await server._server.wait_closed()


async def test_malformed_start_payload_triggers_error() -> None:
    with tempfile.TemporaryDirectory() as d:
        sock = str(Path(d) / "bridge.sock")
        server = await _start_server(sock)
        try:
            reader, writer = await _connect(sock)
            # Send START with garbage (not CBOR map)
            writer.write(encode_frame(FRAME_START, b"\xff\xff\xff"))
            await writer.drain()
            ftype, payload = await asyncio.wait_for(read_frame(reader), timeout=5.0)
            assert ftype == FRAME_ERROR
            assert b"BAD_START" in payload
            writer.close()
            await writer.wait_closed()
        finally:
            if server._server is not None:
                server._server.close()
                await server._server.wait_closed()
