"""Generate a deterministic placeholder WAV for the MRCP smoke test.

Produces ``deploy/umc/canned/hello.wav`` as a 1-second 16 kHz mono PCM16 file
containing a low-amplitude tone burst that any energy-based VAD treats as
speech. **This is not a real "你好" recording** — it exists so the smoke
scenario, umc XML, and pre-flight checks can be validated end-to-end on hosts
without microphones. Operators are expected to replace it on the sandroid-dev
VM with a real recording before the first real smoke run, e.g.:

    arecord -f S16_LE -r 16000 -c 1 -d 1 deploy/umc/canned/hello.wav

Run from the repo root:

    python scripts/smoke/_make_placeholder_wav.py
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "deploy" / "umc" / "canned" / "hello.wav"
SAMPLE_RATE = 16_000
DURATION_S = 1.0
FREQ_HZ = 440.0
AMPLITUDE = 0.3  # well below clipping; loud enough for VAD energy gate


def main() -> None:
    n_samples = int(SAMPLE_RATE * DURATION_S)
    frames = bytearray()
    for i in range(n_samples):
        sample = int(32767 * AMPLITUDE * math.sin(2 * math.pi * FREQ_HZ * i / SAMPLE_RATE))
        frames.extend(struct.pack("<h", sample))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(frames))

    size = OUT.stat().st_size
    print(f"wrote {OUT} ({size} bytes, {n_samples} samples, {DURATION_S}s @ {SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()
