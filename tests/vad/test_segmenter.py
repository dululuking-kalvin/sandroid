from __future__ import annotations

from sandroid.vad.segmenter import SegmentAggregator

FRAME_MS = 32
FRAME_BYTES = 512 * 2  # 512 samples @ 16 kHz PCM16


def _frame(value: int = 0) -> bytes:
    return bytes([value & 0xFF]) * FRAME_BYTES


def _collect(agg: SegmentAggregator, pattern: list[bool]) -> tuple[list[str], list[int]]:
    """Feed ``pattern`` (True=speech) and return (event kinds, segment durations)."""

    kinds: list[str] = []
    durations: list[int] = []
    for is_speech in pattern:
        out = agg.feed(_frame(1 if is_speech else 0), is_speech)
        kinds.extend(e.kind for e in out.events)
        durations.extend(s.duration_ms for s in out.segments)
    tail = agg.flush()
    kinds.extend(e.kind for e in tail.events)
    durations.extend(s.duration_ms for s in tail.segments)
    return kinds, durations


def test_short_blip_suppressed() -> None:
    agg = SegmentAggregator(
        frame_ms=FRAME_MS,
        speech_start_ms=96,  # 3 frames
        silence_end_ms=96,
        min_speech_duration_ms=200,
    )
    # 3 speech frames to trigger SPEAKING, then immediate silence — total
    # duration is under the 200 ms minimum so the segment is dropped.
    kinds, durations = _collect(agg, [True] * 3 + [False] * 3)
    assert durations == []
    # The speech_start event still fires; it's just not matched by a result.
    assert "speech_start" in kinds
    assert "speech_end" not in kinds


def test_normal_utterance_emits_segment() -> None:
    agg = SegmentAggregator(
        frame_ms=FRAME_MS,
        speech_start_ms=96,
        silence_end_ms=96,
        min_speech_duration_ms=100,
    )
    pattern = [True] * 20 + [False] * 3  # ~640 ms speech, 96 ms silence
    kinds, durations = _collect(agg, pattern)
    assert kinds.count("speech_start") == 1
    assert kinds.count("speech_end") == 1
    assert len(durations) == 1
    assert durations[0] >= 640


def test_pre_roll_is_preserved() -> None:
    agg = SegmentAggregator(
        frame_ms=FRAME_MS,
        speech_start_ms=3 * FRAME_MS,
        silence_end_ms=96,
        min_speech_duration_ms=50,
    )
    pattern = [True] * 10 + [False] * 3
    _, durations = _collect(agg, pattern)
    # 10 speech frames plus trailing silence into SPEAKING means the
    # segment should span at least the 10 speech frames from the start —
    # pre-roll must restore the first 3.
    assert durations
    assert durations[0] >= 10 * FRAME_MS


def test_force_cut_on_max_duration() -> None:
    agg = SegmentAggregator(
        frame_ms=FRAME_MS,
        speech_start_ms=FRAME_MS,
        silence_end_ms=1000,
        min_speech_duration_ms=32,
        max_speech_duration_ms=320,  # 10 frames
    )
    # Sustained speech for 20 frames — should force-cut around frame 10.
    pattern = [True] * 20
    kinds, durations = _collect(agg, pattern)
    assert kinds.count("speech_start") == 1
    # Expect at least one force-cut emission plus the flush at the end.
    assert len(durations) >= 2


def test_flush_closes_in_flight_segment() -> None:
    agg = SegmentAggregator(
        frame_ms=FRAME_MS,
        speech_start_ms=FRAME_MS,
        silence_end_ms=1000,
        min_speech_duration_ms=32,
    )
    # Speech never terminates via silence_end_ms; flush() must close it.
    pattern = [True] * 5
    kinds, durations = _collect(agg, pattern)
    assert durations
    assert kinds.count("speech_end") == 1
