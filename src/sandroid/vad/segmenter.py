"""Speech segmentation state machine.

Ported from the dual-layer VAD reference project. Decouples the state
model from any particular speech detector — the caller feeds ``(frame,
is_speech_bool)`` pairs and gets back a stream of ``SegmentEvent`` and
``SegmentResult`` objects. SileroVAD (and any future WebRTC/Silero
combo) drives this.

State model:
    SILENT    — no confirmed speech yet. Holds up to ``speech_start_frames``
                tentative speech frames in a pre-roll buffer so we don't
                lose the leading edge once the state flips.
    SPEAKING  — confirmed speech in progress. Any silence flips to
                TRAILING; ``max_speech_duration`` forces a cut.
    TRAILING  — currently silent but might resume. ``silence_end_frames``
                of continuous silence closes the segment; any speech
                frame flips back to SPEAKING.

Emits:
    - ``SegmentEvent(kind="speech_start")`` when SILENT → SPEAKING
    - ``SegmentResult(pcm16=...)`` plus ``SegmentEvent(kind="speech_end")``
      when SPEAKING/TRAILING → SILENT (natural boundary or force cut)
    - Short segments (below ``min_speech_duration_ms``) are suppressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from sandroid.vad.base import SegmentEvent, SegmentResult


class _State(Enum):
    SILENT = "silent"
    SPEAKING = "speaking"
    TRAILING = "trailing"


@dataclass
class SegmenterOutput:
    """Everything produced by one ``feed`` call. Either list may be empty."""

    events: list[SegmentEvent] = field(default_factory=list)
    segments: list[SegmentResult] = field(default_factory=list)


class SegmentAggregator:
    """Hysteresis-based segmenter.

    All durations are expressed in **milliseconds**; the caller provides a
    fixed ``frame_ms`` (e.g. 32 ms for Silero at 16 kHz / 512 samples) and
    the aggregator converts to/from frame counts internally.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_ms: int = 32,
        speech_start_ms: int = 300,
        silence_end_ms: int = 600,
        min_speech_duration_ms: int = 250,
        max_speech_duration_ms: int = 30_000,
    ) -> None:
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._speech_start_frames = max(1, speech_start_ms // frame_ms)
        self._silence_end_frames = max(1, silence_end_ms // frame_ms)
        self._min_speech_ms = min_speech_duration_ms
        self._max_speech_ms = max_speech_duration_ms

        self._state: _State = _State.SILENT
        self._speech_count = 0
        self._silence_count = 0
        self._pre_buffer: list[bytes] = []
        self._segment_buffer = bytearray()
        self._segment_start_ms = 0
        self._cursor_ms = 0

    @property
    def state(self) -> str:
        return self._state.value

    def feed(self, frame: bytes, is_speech: bool) -> SegmenterOutput:
        """Feed one frame. Returns any events + closed segments produced."""

        out = SegmenterOutput()
        self._cursor_ms += self._frame_ms
        if self._state is _State.SILENT:
            self._handle_silent(frame, is_speech, out)
        elif self._state is _State.SPEAKING:
            self._handle_speaking(frame, is_speech, out)
        else:  # TRAILING
            self._handle_trailing(frame, is_speech, out)
        self._maybe_force_cut(out)
        return out

    def flush(self) -> SegmenterOutput:
        """Close any in-flight segment. Call on stream end."""

        out = SegmenterOutput()
        if self._state in (_State.SPEAKING, _State.TRAILING):
            self._close_segment(out)
        return out

    def _handle_silent(self, frame: bytes, is_speech: bool, out: SegmenterOutput) -> None:
        if not is_speech:
            self._speech_count = 0
            self._pre_buffer.clear()
            return
        self._pre_buffer.append(frame)
        self._speech_count += 1
        if self._speech_count >= self._speech_start_frames:
            start_ms = self._cursor_ms - len(self._pre_buffer) * self._frame_ms
            self._segment_start_ms = max(0, start_ms)
            self._segment_buffer.clear()
            for f in self._pre_buffer:
                self._segment_buffer += f
            self._pre_buffer.clear()
            self._speech_count = 0
            self._state = _State.SPEAKING
            out.events.append(
                SegmentEvent(kind="speech_start", at_ms=self._segment_start_ms),
            )

    def _handle_speaking(self, frame: bytes, is_speech: bool, out: SegmenterOutput) -> None:
        self._segment_buffer += frame
        if is_speech:
            return
        self._state = _State.TRAILING
        self._silence_count = 1
        if self._silence_count >= self._silence_end_frames:
            self._close_segment(out)

    def _handle_trailing(self, frame: bytes, is_speech: bool, out: SegmenterOutput) -> None:
        self._segment_buffer += frame
        if is_speech:
            self._state = _State.SPEAKING
            self._silence_count = 0
            return
        self._silence_count += 1
        if self._silence_count >= self._silence_end_frames:
            self._close_segment(out)

    def _maybe_force_cut(self, out: SegmenterOutput) -> None:
        if self._state is _State.SILENT:
            return
        duration = self._cursor_ms - self._segment_start_ms
        if duration < self._max_speech_ms:
            return
        # Force-cut: close the current segment but stay in SPEAKING so the
        # next frame continues a fresh segment rather than dropping back to
        # SILENT (which would require another full speech_start window).
        self._emit_segment(out, end_ms=self._cursor_ms, kind_after=_State.SPEAKING)
        self._segment_start_ms = self._cursor_ms

    def _close_segment(self, out: SegmenterOutput) -> None:
        self._emit_segment(out, end_ms=self._cursor_ms, kind_after=_State.SILENT)
        self._silence_count = 0

    def _emit_segment(
        self,
        out: SegmenterOutput,
        *,
        end_ms: int,
        kind_after: _State,
    ) -> None:
        duration = end_ms - self._segment_start_ms
        if duration >= self._min_speech_ms and self._segment_buffer:
            out.segments.append(
                SegmentResult(
                    pcm16=bytes(self._segment_buffer),
                    sample_rate=self._sample_rate,
                    start_ms=self._segment_start_ms,
                    end_ms=end_ms,
                ),
            )
            out.events.append(SegmentEvent(kind="speech_end", at_ms=end_ms))
        self._segment_buffer.clear()
        self._state = kind_after
