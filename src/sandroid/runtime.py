"""CPU thread tuning for ONNX Runtime adapters (Phase 5d).

All adapters (Silero VAD, Paraformer ASR, NLU embedder) used to hard-code
``intra_op_num_threads=1``. That is fine on a laptop but burns latency on
the deployment target (24C / 10 concurrent calls) because each request is
single-threaded. Operators can now tune it via ``SANDROID_ONNX_INTRA_OP_THREADS``.

Equally important: we also set ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` at
package import time so BLAS / OpenMP libraries loaded by numpy / onnxruntime
respect the same budget. They read those envs **during dlopen**, so we
cannot set them from the adapter constructors — too late. ``sandroid``'s
``__init__`` calls us before any heavyweight import path.
"""

from __future__ import annotations

import os

INTRA_OP_THREADS_ENV = "SANDROID_ONNX_INTRA_OP_THREADS"
_DEFAULT_INTRA_OP_THREADS = 2


def intra_op_threads() -> int:
    """Resolve the per-session intra-op thread count from env with a safe default."""

    raw = os.environ.get(INTRA_OP_THREADS_ENV)
    if raw is None:
        return _DEFAULT_INTRA_OP_THREADS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{INTRA_OP_THREADS_ENV} must be a positive integer, got {raw!r}"
        ) from exc
    if value < 1:
        raise ValueError(
            f"{INTRA_OP_THREADS_ENV} must be >= 1, got {value}"
        )
    return value


def apply_process_thread_caps() -> None:
    """Set OMP_NUM_THREADS / MKL_NUM_THREADS if the operator left them unset.

    We mirror the per-session intra-op value so BLAS-level parallelism does
    not compete with ONNX-level parallelism. Respect any explicit operator
    setting — a server tuned by hand wins over our defaults.
    """

    try:
        threads = intra_op_threads()
    except ValueError:
        # Bad env value — defer the hard failure to the adapter constructor
        # where the stack trace pins the blame.
        return
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))
