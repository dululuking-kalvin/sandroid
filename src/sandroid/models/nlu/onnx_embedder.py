"""Zero-training NLU baseline using a Chinese sentence embedder.

Implements ``IntentMatcher`` by encoding the transcript and each candidate
Intent's training questions with a pretrained BGE-style model (CLS pooling,
L2-normalized), then ranking by max cosine similarity to the transcript.

This is a **cold-start baseline**, not a learned classifier — it works for any
scene without scene-specific training data and stays in the codebase as a
fallback even after Phase 5b-bis lands per-scene fine-tuned classifiers.

Model artifacts are fetched by ``scripts/fetch_models.py``:
- ``deploy/models/nlu_embedder_quantized.onnx`` (Xenova/bge-small-zh-v1.5 INT8)
- ``deploy/models/nlu_tokenizer.json``

Both are pinned by upstream commit SHA + SHA-256 of the file.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from sandroid.core.domain import Intent
from sandroid.core.matcher import MatchCandidate
from sandroid.runtime import intra_op_threads

_MAX_SEQ_LEN = 512
_EMBED_DIM = 512


class ONNXEmbedderError(RuntimeError):
    """Raised when the embedder model or tokenizer is missing / unloadable."""


class ONNXEmbedderMatcher:
    """Cosine-similarity matcher over sentence embeddings.

    Construction is eager (loads the model + tokenizer) so startup fails loud
    rather than the first request hitting a missing file. Inference is
    thread-safe via a single lock around the ONNX session; intra-op thread
    count is resolved from ``SANDROID_ONNX_INTRA_OP_THREADS`` so operators can
    tune CPU usage against the 10-concurrent-call budget.
    """

    def __init__(self, model_path: Path, tokenizer_path: Path) -> None:
        if not model_path.exists():
            raise ONNXEmbedderError(
                f"NLU embedder model missing at {model_path}; run scripts/fetch_models.py"
            )
        if not tokenizer_path.exists():
            raise ONNXEmbedderError(
                f"NLU tokenizer missing at {tokenizer_path}; run scripts/fetch_models.py"
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads()
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=_MAX_SEQ_LEN)
        self._tokenizer.enable_padding()
        self._lock = Lock()

    def score(
        self,
        transcript: str,
        candidates: list[Intent],
        *,
        n_best: int = 5,
    ) -> list[MatchCandidate]:
        transcript = transcript.strip()
        if not transcript or not candidates:
            return []

        # Flatten: [transcript, q1, q2, ..., qN] — one forward pass per call.
        # Intents with many questions blow this up, but scenes are small in
        # practice (≤ a few hundred questions total) and a single ONNX run is
        # far cheaper than the per-call tokenize + round-trip overhead.
        texts: list[str] = [transcript]
        offsets: list[tuple[int, int]] = []  # (start, end) in `texts` per intent
        for intent in candidates:
            start = len(texts)
            questions = [q.strip() for q in intent.question if q.strip()]
            texts.extend(questions)
            offsets.append((start, start + len(questions)))

        vectors = self._embed(texts)
        transcript_vec = vectors[0]

        scored: list[MatchCandidate] = []
        for intent, (start, end) in zip(candidates, offsets, strict=True):
            if end <= start:
                continue
            question_vecs = vectors[start:end]
            similarities = question_vecs @ transcript_vec
            best_cosine = float(similarities.max())
            confidence = max(0.0, min(1.0, (best_cosine + 1.0) / 2.0))
            scored.append(MatchCandidate(intent_id=intent.id, confidence=confidence))

        scored.sort(key=lambda c: (-c.confidence, c.intent_id))
        return scored[:n_best]

    def _embed(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        with self._lock:
            (last_hidden_state,) = self._session.run(
                ["last_hidden_state"],
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )

        # BGE convention: use [CLS] (index 0) as the sentence representation,
        # then L2-normalize so dot product equals cosine similarity.
        cls = last_hidden_state[:, 0, :]
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return cls / norms
