# sandroid

Chinese-first Spoken Language Understanding (SLU) engine for FreeSWITCH-based
contact centers. Audio in (WAV or streaming PCM) + scene identifier out ⇒
top intent with a confidence score in `[0, 1]`.

Architecture, domain model, and all pinned design decisions live in
[`CLAUDE.md`](./CLAUDE.md). Read that first.

## Requirements

- WSL2 Ubuntu (dev) or bare Ubuntu (deploy)
- Python **3.11** (managed automatically by `uv`)
- [`uv`](https://docs.astral.sh/uv/) for dependency and environment management
- Docker + `docker compose` (for Triton and Postgres, added in later phases)

## Quickstart

```bash
# 1. Install uv if you don't have it
pip install --user uv       # or: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync dependencies (creates .venv, installs runtime + dev extras)
uv sync --extra dev

# 3. Run the dev server
uv run uvicorn sandroid.api.app:app --reload --host 0.0.0.0 --port 8000

# 4. Smoke-check
curl http://localhost:8000/healthz
```

## Development commands

```bash
uv run pytest                 # run tests
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy                   # type-check
```

## Project status

Pre-code through 2026-04-19. Phase 1 scaffolding lands with this commit:
`pyproject.toml` + src-layout package + `/healthz` endpoint + CI. Phase 2
(Category tree loader + scene YAML schema) is next.
