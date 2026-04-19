"""Download serving-time model artifacts into ``deploy/models/``.

Models are pinned by URL + SHA-256 so different machines produce the same
runtime. The directory is gitignored — the script is the source of truth
for what's expected to be there.

Usage:
    python scripts/fetch_models.py          # fetch all missing / mismatched
    python scripts/fetch_models.py --force  # re-download even if present
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "deploy" / "models"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    dest: Path


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="silero_vad",
        url="https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx",
        sha256="2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f",
        dest=MODELS_DIR / "silero_vad.onnx",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(spec: ModelSpec) -> None:
    spec.dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = spec.dest.with_suffix(spec.dest.suffix + ".part")
    print(f"[fetch] {spec.name}: {spec.url}", file=sys.stderr)
    with urllib.request.urlopen(spec.url) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
    got = _sha256(tmp)
    if got != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"[fetch] {spec.name}: sha256 mismatch\n  expected {spec.sha256}\n  got      {got}"
        )
    tmp.replace(spec.dest)
    print(f"[fetch] {spec.name}: ok → {spec.dest.relative_to(ROOT)}", file=sys.stderr)


def _needs_fetch(spec: ModelSpec, force: bool) -> bool:
    if force or not spec.dest.exists():
        return True
    return _sha256(spec.dest) != spec.sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    for spec in MODELS:
        if _needs_fetch(spec, args.force):
            _download(spec)
        else:
            print(f"[fetch] {spec.name}: up-to-date", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
