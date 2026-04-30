"""Offline Paraformer-zh ASR adapter (Phase 5c Step A).

Loads the INT8 ONNX export from
``csukuangfj/paraformer-onnxruntime-python-example`` plus its ``tokens.txt``
and ``am.mvn`` (CMVN statistics). Audio goes in as a 16 kHz mono PCM16 WAV;
transcript comes out as UTF-8 text.

Pipeline (mirrors the HF example's 60-line script, with sandroid conventions
for resource management and type safety):

    WAV -> int16 -> float32                        (wave module)
    float32 samples * 32768 -> 80-dim fbank        (kaldi-native-fbank)
    LFR stacking window=7 / shift=6 → [T', 560]
    (x + neg_mean) * inv_std using am.mvn          (element-wise CMVN)
    ONNX run: speech/speech_lengths → logits/token_num
    argmax per frame, drop <blank>=0 + </s>=2, look up tokens.txt

Streaming is out of scope for Step A — use the stub until Step C lands.
"""

from __future__ import annotations

import io
import logging
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Lock

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort

from sandroid.models.asr.base import AudioChunk, FinalTranscript, PartialTranscript
from sandroid.runtime import intra_op_threads

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_FBANK_NUM_BINS = 80
_LFR_WINDOW = 7
_LFR_SHIFT = 6
_STACKED_DIM = _FBANK_NUM_BINS * _LFR_WINDOW  # 560
_BLANK_ID = 0
_EOS_ID = 2
# Re-decode the accumulated PCM every _PARTIAL_EMIT_MS of fresh audio.
# 800 ms balances CPU (we re-run the full non-autoregressive decoder each
# time) against partial responsiveness for the IVR use case.
_PARTIAL_EMIT_MS = 800


class ParaformerError(RuntimeError):
    """Raised when the Paraformer adapter cannot initialize or decode."""


class ParaformerASR:
    """Offline Paraformer-zh ASR over onnxruntime. Streaming raises NotImplementedError.

    ``confidence`` is hardcoded to 1.0: Paraformer's non-autoregressive decoder
    doesn't emit a calibrated per-utterance confidence and the fusion layer
    doesn't consume ASR confidence in v0.1. Reconsider in Phase 5d.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        tokens_path: Path,
        cmvn_path: Path,
        partial_emit_ms: int = _PARTIAL_EMIT_MS,
    ) -> None:
        self._partial_emit_ms = partial_emit_ms
        for p, label in (
            (model_path, "model"),
            (tokens_path, "tokens"),
            (cmvn_path, "cmvn"),
        ):
            if not p.exists():
                raise ParaformerError(f"Paraformer {label} file missing: {p}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads()
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._lock = Lock()

        self._tokens: dict[int, str] = {}
        with tokens_path.open(encoding="utf-8") as fp:
            for i, line in enumerate(fp):
                token = line.strip().split()[0] if line.strip() else ""
                self._tokens[i] = token

        self._neg_mean, self._inv_std = self._load_cmvn(cmvn_path)
        if self._neg_mean.shape != (_STACKED_DIM,) or self._inv_std.shape != (_STACKED_DIM,):
            raise ParaformerError(
                f"CMVN shape mismatch: expected ({_STACKED_DIM},) "
                f"got neg_mean {self._neg_mean.shape}, inv_std {self._inv_std.shape}"
            )

        logger.info(
            "paraformer-zh loaded: vocab=%d, stacked_dim=%d",
            len(self._tokens),
            _STACKED_DIM,
        )

    @staticmethod
    def _load_cmvn(path: Path) -> tuple[np.ndarray, np.ndarray]:
        neg_mean: np.ndarray | None = None
        inv_std: np.ndarray | None = None
        with path.open() as fp:
            for line in fp:
                if not line.startswith("<LearnRateCoef>"):
                    continue
                vals = np.array(
                    [float(x) for x in line.split()[3:-1]],
                    dtype=np.float32,
                )
                if neg_mean is None:
                    neg_mean = vals
                else:
                    inv_std = vals
                    break
        if neg_mean is None or inv_std is None:
            raise ParaformerError(f"am.mvn missing <LearnRateCoef> rows: {path}")
        return neg_mean, inv_std

    async def transcribe_file(self, wav_bytes: bytes) -> FinalTranscript:
        pcm = self._decode_wav(wav_bytes)
        duration_ms = round(len(pcm) / _SAMPLE_RATE * 1000)

        feats = self._features(pcm)
        if feats.shape[0] == 0:
            return FinalTranscript(text="", confidence=1.0, duration_ms=duration_ms)

        text = self._run_and_decode(feats)
        return FinalTranscript(text=text, confidence=1.0, duration_ms=duration_ms)

    async def stream(
        self,
        chunks: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[PartialTranscript]:
        """Pseudo-streaming: accumulate PCM, re-decode periodically, emit partials.

        Paraformer is a non-autoregressive offline decoder — there is no
        incremental state to advance. We trade CPU for responsiveness: every
        ``partial_emit_ms`` of fresh audio we re-run the full pipeline on the
        accumulated buffer and emit a ``PartialTranscript``. On stream end we
        emit one ``is_final=True`` frame. Step D (if we need sub-200ms tail
        latency) is swapping in a real streaming model behind this interface.
        """

        accumulated = bytearray()
        emitted_ms = 0
        last_text = ""
        async for chunk in chunks:
            if not chunk.pcm16:
                continue
            accumulated += chunk.pcm16
            buffered_ms = len(accumulated) // 2 * 1000 // _SAMPLE_RATE
            if buffered_ms - emitted_ms < self._partial_emit_ms:
                continue
            text = self._decode_pcm_buffer(bytes(accumulated))
            emitted_ms = buffered_ms
            last_text = text
            yield PartialTranscript(
                text=text,
                is_final=False,
                start_ms=0,
                end_ms=buffered_ms,
            )

        buffered_ms = len(accumulated) // 2 * 1000 // _SAMPLE_RATE
        final_text = self._decode_pcm_buffer(bytes(accumulated)) if accumulated else last_text
        yield PartialTranscript(
            text=final_text,
            is_final=True,
            start_ms=0,
            end_ms=buffered_ms,
        )

    def _decode_pcm_buffer(self, pcm16: bytes) -> str:
        """Run features + ONNX + argmax over a raw-PCM buffer. Used by stream()."""

        if not pcm16:
            return ""
        pcm_float = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        feats = self._features(pcm_float)
        if feats.shape[0] == 0:
            return ""
        return self._run_and_decode(feats)

    @staticmethod
    def _decode_wav(wav_bytes: bytes) -> np.ndarray:
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as w:
                if w.getnchannels() != 1:
                    raise ParaformerError(f"expected mono WAV, got {w.getnchannels()} channels")
                if w.getsampwidth() != 2:
                    raise ParaformerError(f"expected 16-bit PCM, got {w.getsampwidth() * 8}-bit")
                if w.getframerate() != _SAMPLE_RATE:
                    raise ParaformerError(f"expected {_SAMPLE_RATE} Hz, got {w.getframerate()} Hz")
                frames = w.readframes(w.getnframes())
        except wave.Error as exc:
            raise ParaformerError(f"failed to parse WAV: {exc}") from exc
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    def _features(self, pcm_float: np.ndarray) -> np.ndarray:
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = _SAMPLE_RATE
        opts.mel_opts.num_bins = _FBANK_NUM_BINS

        fb = knf.OnlineFbank(opts)
        # kaldi-native-fbank expects int16-range floats, per the HF example.
        fb.accept_waveform(_SAMPLE_RATE, (pcm_float * 32768.0).tolist())
        fb.input_finished()

        n = fb.num_frames_ready
        if n < _LFR_WINDOW:
            return np.zeros((0, _STACKED_DIM), dtype=np.float32)

        feats = np.stack([fb.get_frame(i) for i in range(n)])

        t_out = (feats.shape[0] - _LFR_WINDOW) // _LFR_SHIFT + 1
        stacked = np.lib.stride_tricks.as_strided(
            feats,
            shape=(t_out, _STACKED_DIM),
            strides=((_LFR_SHIFT * _FBANK_NUM_BINS) * 4, 4),
        ).copy()  # defensive: as_strided views are fragile
        return (stacked + self._neg_mean) * self._inv_std

    def _run_and_decode(self, feats: np.ndarray) -> str:
        speech = feats[None, :, :].astype(np.float32)
        lengths = np.array([speech.shape[1]], dtype=np.int32)
        with self._lock:
            logits, _ = self._session.run(
                ["logits", "token_num"],
                {"speech": speech, "speech_lengths": lengths},
            )
        ids = logits[0].argmax(axis=-1).tolist()
        pieces = [
            self._tokens[i] for i in ids if i not in (_BLANK_ID, _EOS_ID) and i in self._tokens
        ]
        return "".join(pieces)
