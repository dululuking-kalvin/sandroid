"""sandroid — Chinese-first Spoken Language Understanding engine."""

from sandroid.runtime import apply_process_thread_caps

__version__ = "0.1.0.dev0"

# Set OMP_NUM_THREADS / MKL_NUM_THREADS before any downstream import pulls
# in numpy / onnxruntime — those libs read the envs at dlopen time.
apply_process_thread_caps()
