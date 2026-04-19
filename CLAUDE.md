# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

`sandroid` is a **Chinese-first Spoken Language Understanding (SLU) engine** deployed as a service on Ubuntu. It accepts audio (WAV file or realtime PCM stream) plus a **scene identifier (a Category node in the global tree, referenced by `category_id` or `category_path`)**, and returns the most-likely intent along with a confidence score in `[0, 1]`.

Primary integration target is **FreeSWITCH** for contact-center / IVR use cases, via two parallel channels: **MRCPv2** (industry-standard) and **ESL + WebSocket bridge** (lightweight).

The file `docs/NLP Voice Engine API Manual v1.0.0 rev1 (CN).pdf` is a **reference specification** from a prior commercial engine (TSINGNOVA). Treat it as design inspiration only — sandroid deliberately simplifies its model. See "Concept mapping vs. reference PDF" below.

## Core Architecture

### Dual-path recognition + fusion

Two recognition paths run in parallel; a **Fusion** layer arbitrates between them with scene-configurable weights. Users tune weights based on per-scene training results.

```
Audio (WAV / PCM stream)
  │
  ▼
Front-End: VAD + resample to 16kHz mono PCM16
  │
  ├──► Path A: ASR → NLU
  │      ASR (Paraformer-small ONNX INT8, or pluggable cloud ASR)
  │        → transcript
  │      NLU (RoBERTa-wwm-ext-small ONNX INT8, intent classifier)
  │        → (intent_A, conf_A)
  │
  └──► Path B: End-to-End SLU
         wav2vec2-base-zh + intent head (ONNX INT8)
           → (intent_B, conf_B)           [runs in parallel with Path A]
  │
  ▼
Fusion: score = w_A·conf_A + w_B·conf_B (+ scene rules)
  │
  ▼
Top intent + confidence ∈ [0,1]  (N-best list available)
```

**Why both paths?**
- Path A is transcript-grounded and explainable; benefits from huge text NLU pretraining; weaker under ASR errors and noisy audio.
- Path B is end-to-end and more robust to ASR noise; weaker with low-frequency intents and harder to debug.
- Fusion weights let each scene strike its own balance.

### Layering

- **Edge**: FastAPI REST (WAV upload + management) + WebSocket (streaming PCM) + MRCPv2 server (FreeSWITCH) + ESL/WS bridge (FreeSWITCH alternate).
- **Service**: Session Manager, Scene Router, Recognition Orchestrator (drives A/B in parallel), Context state machine (handles category-switching / follow-up).
- **Model**: All models served via **Triton Inference Server (CPU backend, ONNX + INT8 quantization)**. GPU-free deployment is a hard constraint.
- **Training (offline)**: Data ingestion → augmentation (noise, speed perturb, channel simulation for telephony) → NLU finetune → E2E SLU finetune → per-scene eval.
- **Data**: Scene/Intent config in YAML + PostgreSQL; audio training data in WebDataset/LMDB; session logs in PostgreSQL (or ClickHouse if volume grows).

### Domain model — global Category tree

The entire system shares **one Category tree rooted at `ROOT`**. A "scene" is not a top-level container — it is the **Category node the caller points at** (`category_id` or `category_path`, never `category_name` because names can collide).

**Core objects:**

- **Category**: a node in the tree. Has `id`, `path` (e.g. `ROOT->银行业务->信用卡`), a **mode** (see V3 mode table), optional `extendsEnumCategory` reference, and can hold:
  - **Ordinary Intents** — normal knowledge points the node itself may answer
  - **Special Intents** — "hooks" whose purpose is to be matched **by the parent** and trigger a `followUp` that pulls the user into this Category for the next turn
  - **Child Categories** — sub-topics
  A Category may hold both Intents and child Categories simultaneously. Default mode is `STANDARD`.
- **Intent**: a knowledge point attached to a Category. Fields: id, `question` (training utterances), `answer`, optional `slots`, `is_special` flag, optional `switching` override (per-intent category switch).
- **Session**: per-call context. Tracks current Category (scene pointer), accumulated slots (from SEQUENTIAL turns), dialogue history, category-switching history.
- **Confidence**: always `[0.0, 1.0]`. Do **not** use the `[0, 100]` scale from the reference PDF — normalize at the boundary.

### Category mode — V3 orthogonal property model

**A Category's mode is defined by three groups of orthogonal properties.** Every mode is just an assignment of switch values; do not read mode names as rule bundles.

| Group | Property | Meaning |
|---|---|---|
| **A — structure** | A1 | May hold ordinary Intents? |
|  | A2 | May (must) hold special Intents? |
| **S — search (when this node IS the scene)** | S1 | Search downward into children? |
|  | S2 | Depth: immediate children only / all descendants |
| **E — expose (when this node IS a child reached by an ancestor's search)** | E1 | Are our ordinary Intents visible to the ancestor? |
|  | E2 | Are our special Intents visible to the ancestor? |
|  | E3 | May the ancestor **penetrate through us** and continue into our children? |

**The search scope when Category X is the scene** is computed recursively:

```
scope(X) = X.ordinary_intents (always — querying yourself opens yourself)
         + for each child C of X, collect_via_ancestor_view(C)
         + extendsEnumCategory pool if any
         [child recursion only happens if X.S1=yes, bounded by X.S2]

collect_via_ancestor_view(C):
    result = []
    if C.E1: result += C.ordinary_intents
    if C.E2: result += C.special_intents
    if C.E3: result += recursively collect_via_ancestor_view for each child of C
    return result
```

**Key insight:** "Can X's Intent be matched by an upper-layer query?" is determined jointly by:
1. The ancestor's search behavior (S1/S2 — does it look down at all, and how far)
2. X's exposure behavior (E1/E2 — what X shows outward)
3. Every intermediate node's E3 (can the search penetrate through them)

**V3 mode table:**

| mode | A1 ordinary | A2 special | S1 search down | S2 depth | E1 ord→up | E2 spec→up | E3 allow penetration |
|---|---|---|---|---|---|---|---|
| **STANDARD** | ✅ | ❌ no concept | ✅ | all descendants | ✅ | — | ✅ |
| **ENUM** | ✅ | ❌ no concept | ⚠️ not used as scene | — | ⚠️ only when referenced by `extendsEnumCategory` | — | ✅ (penetrates when pulled in via extends) |
| **SEQUENTIAL** | ✅ | ✅ required | ❌ | — | ❌ | ✅ | ❌ |
| **BRANCHING** | ❌ forbidden | ✅ required | ✅ | immediate children only | — | ✅ | ❌ |
| **DRILLDOWN** | ✅ | ✅ required | ✅ | all descendants | ✅ | ✅ | ✅ |
| **MEMORY** | ✅ | ❌ no concept | ❌ | — | ❌ | — | ❌ |

**Derived behaviors (do not re-state elsewhere — derive from the table):**

- A STANDARD ancestor that reaches a BRANCHING/SEQUENTIAL child **only sees the child's special Intents** (E2=✅, E3=❌) — the child's internal flow is reached only by matching the hook and following `followUp`.
- A STANDARD ancestor that reaches a DRILLDOWN child **sees the child's ordinary Intents, its special Intents, AND penetrates further** (E1=E2=E3=✅). So DRILLDOWN is a full citizen of the STANDARD tree + acts as a hook.
- Ordinary vs special Intents in the candidate pool **compete by confidence** — special Intents are not prioritized artificially. If the user asks a clear question, the ordinary Intent wins and answers directly; if the user is vague, the special Intent wins and `followUp` guides them deeper.
- ENUM is never a scene itself. It only contributes when a Category references it via `extendsEnumCategory`, at which point its entire subtree (subject to its own E3=✅) merges into the referencing node's candidate pool.

**Configuration-time validation** (enforced by the scene loader; fail fast, not at runtime):
- BRANCHING with any ordinary Intent → error
- SEQUENTIAL / BRANCHING / DRILLDOWN without at least one special Intent → error
- SEQUENTIAL with multiple child Categories at the same depth → error (must be a single chain)
- MEMORY with any child Category → error
- Category referenced as `extendsEnumCategory` but mode ≠ ENUM → error
- `category_path` referring to a name collision → error (names must be unique per parent)

### SEQUENTIAL flow termination (replaces PDF's MEMORY mechanism)

The PDF's MEMORY mode synthesizes a composite question from prior SEQUENTIAL answers and runs it through a classifier. We **keep the mode name** but **replace the mechanism** with an `on_complete` strategy declared on the terminal node. This decouples the flow-termination concept from any single implementation and avoids the combinatorial annotation cost of the original design.

`on_complete.type` (configured per MEMORY-terminal Category, or the last SEQUENTIAL node when MEMORY isn't used):

- **`handoff`** (v0.1, default): engine returns `{slots, state: COMPLETED}` to the caller. Business backend decides next step. Covers flight-booking-like flows where decisions depend on external data (DBs, APIs).
- **`rules`** (v0.1): YAML rule list evaluated against accumulated slots. Example:
  ```yaml
  on_complete:
    type: rules
    rules:
      - when: "slots.cabin == '经济舱' and slots.days_until <= 7"
        intent: RECOMMEND_QUICK_ECONOMY
      - default: RECOMMEND_GENERIC
  ```
  Deterministic, inspectable, no training data needed.
- **`intent_classifier`** (v0.2): a small classifier trained on serialized slot sequences → Intent. Equivalent to PDF's MEMORY mechanism but behind a strategy flag. Only build when a concrete scene demands it.

**v0.1 ships `handoff` and `rules` only.** Do not implement `intent_classifier` until a real scene requires it.

### Category Switching (per-intent override)

Any Intent may carry an optional `switching` field pointing to a different `category_path`. When that Intent is the top-1 match, the engine returns `followUp = <that path>` and the session pointer moves there on the next turn. This covers the PDF's 分类跳转 use cases without introducing a separate mechanism.

### N-best semantics

The recognition response includes an N-best list of candidates scoped to **the current decision point only** (i.e. the current scene Category's candidate pool after recursive collection). `followUp` (if any) is returned as a separate field, not mixed into N-best. Rationale: keep "what could the user have meant now" orthogonal to "where should they go next".

## Deployment Constraints (MUST respect)

Target machine: **24-core CPU, 32 GB RAM, no GPU, 10 concurrent calls, end-to-end latency ≤ 500 ms**.

This drives several hard rules:

1. **No GPU models.** All inference runs on CPU via ONNX Runtime inside Triton.
2. **INT8 quantization mandatory** for every model (ASR, NLU, E2E SLU).
3. **Use "small/base" model sizes, not "large"**:
   - ASR: Paraformer-small (local) or cloud ASR (Aliyun/Tencent/iFlytek) via pluggable adapter.
   - NLU: RoBERTa-wwm-ext-small or distilbert-wwm.
   - E2E SLU: wav2vec2-base-zh (not large). If latency budget is exceeded in first-version tests, defer Path B and ship Path A only.
4. **Streaming ASR required.** Partial transcripts let NLU/SLU start before final endpoint, keeping p95 latency under budget.
5. **Latency budget (per call, sequential path)**:
   - VAD endpoint: ≤ 50 ms
   - Streaming ASR tail: ≤ 200 ms
   - NLU inference: ≤ 80 ms
   - Fusion + serialization: ≤ 20 ms
   - Path B runs in parallel with Path A — does not enter sequential budget, but must finish within 200 ms from audio-end or be dropped from fusion.
   - Total target: ≤ 350 ms; 150 ms buffer for network/FreeSWITCH.
6. **Concurrency model**: asyncio-based FastAPI + Triton dynamic batching. 10 concurrent streams is the design point; CPU threads for ONNX Runtime must be tuned (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`) against concurrency — do not let each request claim all 24 cores.

## Pluggable ASR (dialect & flexibility)

ASR is behind an adapter interface. Two implementations ship initially:
- `LocalASR`: Paraformer-small ONNX (FunASR export), covers Mandarin + mild regional accents via data augmentation (scheme 6a).
- `CloudASR`: Aliyun / Tencent / iFlytek. Pick per-scene for dialects or quality-critical scenes.

Scene config chooses which ASR to use. The fusion layer and NLU do **not** care which ASR produced the transcript.

## FreeSWITCH Integration

Both channels must work from day one:
- **MRCPv2** server (port 1544, UniMRCP-compatible), consumed by FreeSWITCH `mod_unimrcp`. Standard, interoperable with other PBXes.
- **ESL + WebSocket bridge**: a small FreeSWITCH-side script (Lua/JS in FS) pipes audio to our `/api/v1/recognize/stream` WS endpoint. Lighter to deploy when MRCP is overkill.

Both ultimately call into the same internal Recognition Orchestrator — adapters are thin.

## Concept mapping vs. reference PDF

| Reference PDF term | sandroid term | Notes |
|---|---|---|
| 知识库 (global tree) | Global Category tree | Kept — single tree rooted at `ROOT`, shared by the whole system |
| 知识库分类 (category) | Category node | Kept. Scene = the Category node a caller points at (by `category_id` or `category_path`) |
| 知识点 (faq) | Intent (ordinary) | One intent per "faq"; optional slots; optional per-intent `switching` override |
| 特殊/依附知识点 | Intent (special, `is_special=true`) | Hook Intents. Exposed upward but do not match the node itself |
| 分类模式 STANDARD | STANDARD | Same semantics; now formally defined via V3 properties (A1=✅/A2=∅/S1=✅/S2=all/E1=✅/E3=✅) |
| 分类模式 ENUM | ENUM | Kept. Only participates via `extendsEnumCategory` references |
| 分类模式 QUESTIONNAIRE_STEPPING | SEQUENTIAL | Kept. Single-child chain constraint enforced by loader |
| 分类模式 QUESTIONNAIRE_BRANCHING | BRANCHING | Kept. **No ordinary Intents allowed** — hook-only node |
| 分类模式 DRILLDOWN | DRILLDOWN | Kept. **Can hold ordinary Intents AND penetrate to all descendants** (upgraded from the PDF, which forbade ordinary hits on self; we allow both, let confidence decide) |
| 分类模式 MEMORY | MEMORY | Mode name kept; **mechanism replaced** with `on_complete` strategy (`handoff` / `rules` in v0.1, `intent_classifier` in v0.2). No more "composite question" string synthesis + hand-labeling in Report |
| extendsEnumCategory | extendsEnumCategory | Kept. Any non-ENUM Category may reference exactly one ENUM Category to pull its subtree into the search pool |
| 分类跳转 (switching) | Per-Intent `switching` field + Session pointer | Same behavior, declared on the Intent rather than parsed from a JSON blob in CSV imports |
| followUp | followUp (separate response field) | Returned alongside — not inside — the N-best list |
| confidence ∈ [0, 100] | confidence ∈ [0, 1] | **Always normalize at the model adapter boundary** |
| N-Best-List-Length / Confidence-Threshold | Same semantics, exposed as WS config + scene-level defaults | Scoped to the current decision point only (see N-best semantics above) |
| /api/uaudio/file + /api/uaudio/websocket | /api/v1/recognize/file + /api/v1/recognize/stream | Versioned paths |
| questionId (三种格式) | `category_id` OR `category_path` | **`category_name` dropped** — ambiguous when names repeat across branches |
| VAD multi-segment params | Silero VAD + optional multi-segment wrapper | Keep the PDF's multi-segment idea for long-form recordings |
| Report-based MEMORY training | `on_complete.type: intent_classifier` (v0.2 only) | Replaces the manual Expected-Faqid annotation workflow; deferred until a real scene needs it |

## Resolved design decisions (ADR log)

Recorded here so future sessions do not re-litigate. Date stamps use the project timezone.

- **2026-04-17** — Deployment profile: 24C/32G/no-GPU, 10 concurrent calls, ≤500 ms E2E latency (scheme C from the infra discussion).
- **2026-04-17** — Dialect handling: scheme 6a (Mandarin + accent via data augmentation); dialects routed through `CloudASR` adapters, not via forking NLU models.
- **2026-04-17** — Stack: Python 3.11, FastAPI, Triton CPU, Paraformer/BERT-wwm/wav2vec2-zh (all small/base + ONNX INT8).
- **2026-04-17** — FreeSWITCH: MRCPv2 AND ESL+WS bridge ship together from v0.1.
- **2026-04-17** — Confidence range: `[0, 1]` everywhere; normalize at every model adapter boundary.
- **2026-04-17** — Scene model: **single global Category tree**, default mode STANDARD. Scene reference uses `category_id` or `category_path` only (no `category_name` — collision risk).
- **2026-04-17** — All six modes kept: STANDARD, ENUM, SEQUENTIAL, BRANCHING, DRILLDOWN, MEMORY. Rule semantics formalized via the V3 orthogonal property table (A/S/E groups).
- **2026-04-17** — DRILLDOWN upgraded vs. the PDF: may hold ordinary Intents AND allows upstream penetration (E1=E2=E3=✅). Confidence — not Intent kind — decides top-1.
- **2026-04-17** — MEMORY mechanism replaced by `on_complete` strategy pattern. v0.1 ships `handoff` + `rules`. `intent_classifier` (the PDF's original scheme) deferred until a concrete scene requires it.
- **2026-04-17** — N-best scoped to current decision point only; `followUp` returned as a separate response field.
- **2026-04-17** — Per-Intent `switching` field replaces the PDF's CSV-embedded SWITCHING JSON for category switching.
- **2026-04-19** — Package manager: **uv** (Python 3.11). Dev environment: WSL2 Ubuntu on Windows host; deployment target remains bare Ubuntu.
- **2026-04-19** — Default slot schema: pragmatic minimal set — `{name: str, type: str, value: Any, confidence: float, source: "asr"|"rule"|"user"}`. Extend per-scene as needed; do not over-design upfront.
- **2026-04-19** — Scene YAML tree encoding: **adjacency list** (each Category declares `parent_id` or is implicit by nesting in a single YAML file per scene root). Rejected: path-string encoding (rename pain), materialized-path (duplication).
- **2026-04-19** — Auth: **API Key via `X-API-Key` header**, validated per request. Keys stored hashed in Postgres; rotation via admin endpoint. OAuth/JWT deferred until multi-tenant need arises.
- **2026-04-19** — Audio retention: **two-tier policy**.
  - **Labeled audio** (any recording that has been human-annotated or auto-tagged with ground-truth intent/slots) → **retained indefinitely** for retraining and model tuning.
  - **Unlabeled audio** → **retained 180 days** (6-month buffer window), giving the annotation team time to pull samples before auto-deletion.
  - Storage path: `/var/sandroid/audit/{labeled|unlabeled}/{scene_id}/{YYYY-MM-DD}/`. Format: Opus (re-decode to PCM before training). A nightly job sweeps expired unlabeled files.
  - Per-scene YAML may override retention (e.g. compliance-heavy scenes set `unlabeled_retention_days: 1095`).
- **2026-04-19** — MRCPv2 implementation: **strategy α — wrap UniMRCP C library** via Cython/pybind11 bindings. Rationale: long-term controllability and protocol conformance; acceptable upfront cost given FreeSWITCH is a day-one integration target. Strategies β (fork community Python MRCP), γ (from-scratch subset), δ (UniMRCP subprocess + event bridge) rejected.
- **2026-04-20** — WS `/recognize/stream` protocol (spec: `docs/ws-protocol.md`). **Implicit handshake** — client may push PCM immediately after `start`; VAD `speech_start_ms` absorbs any lead-in race. No server `ready` frame (would add a needless RTT and diverge from MRCPv2 timing). Every frame (both directions) carries `session_id` + `turn_id` for correlation. **Barge-in via new `turn_id`** — a higher-id `start` implicitly terminates the previous turn; no explicit `stop` required. **Errors are async non-fatal frames** (`{type:"error", fatal:false}`); socket only closes on auth/shape failures or `fatal:true`. Rationale: the real speaker is an IVR caller with no visibility into server state, so the protocol optimizes for latency + observability + barge-in, not for handshake ceremony.

## Project Status

**Pre-code**. Only `docs/` exists. The plan and domain model above are agreed with the user (2026-04-17). When scaffolding, follow this structure:

```
sandroid/
├── CLAUDE.md                  (this file)
├── README.md
├── docs/                      (keep PDF + future design docs)
├── pyproject.toml             (managed via uv or poetry)
├── configs/
│   ├── scenes/*.yaml          (one file per scene)
│   └── fusion.yaml            (default fusion weights)
├── src/sandroid/
│   ├── api/                   (FastAPI routes, WS handlers)
│   ├── core/
│   │   ├── session.py
│   │   ├── orchestrator.py    (drives Path A + Path B in parallel)
│   │   └── fusion.py
│   ├── models/
│   │   ├── asr/               (LocalASR, CloudASR adapters)
│   │   ├── nlu/
│   │   └── slu/               (end-to-end)
│   ├── vad/                   (Silero wrapper)
│   ├── adapters/
│   │   ├── mrcp/              (MRCPv2 server)
│   │   ├── esl_ws/            (FreeSWITCH ESL + WS bridge)
│   │   └── triton/
│   └── storage/               (Postgres models, migrations)
├── training/
│   ├── data/                  (raw + processed)
│   ├── pipelines/             (data augmentation, export)
│   └── recipes/               (per-scene configs & scripts)
├── scripts/                   (deploy, migration, data mgmt)
├── deploy/
│   ├── docker-compose.yml     (app + Triton + Postgres)
│   └── triton/model_repository (exported ONNX models + configs)
└── tests/                     (pytest; unit + integration)
```

## Commands (to be filled as scaffolding lands)

Nothing is buildable yet. As each piece is added, update this section with:
- How to install deps (`uv sync` / `poetry install`)
- How to run the dev server (FastAPI)
- How to run Triton locally (`docker compose up triton`)
- How to run a single test (`pytest tests/path::test_name`)
- How to export a model to ONNX + INT8 (per-model script)
- How to start MRCP server and ESL/WS bridge in dev mode

**Rule:** only add commands here after they actually work locally. No placeholder commands.

## Guardrails for future work

- **Never** introduce GPU-only dependencies (torch with CUDA at runtime, TensorRT, FlashAttention, etc.) in the serving path. Training is separate and may use GPU — keep training code under `training/` and do not let it leak into `src/sandroid/`.
- **Normalize confidence at every boundary** (ASR/NLU/SLU outputs → `[0,1]` immediately). Bugs here silently corrupt fusion.
- **Streaming is a first-class concern**, not an afterthought. Design model wrappers and the orchestrator for chunk-in / partial-out from the start.
- When adding a new scene, always create a corresponding YAML in `configs/scenes/` and a training recipe in `training/recipes/`. Do not hardcode scene logic in Python.
- When touching the MRCP/ESL adapters, verify with a real FreeSWITCH instance (docker image exists) — protocol bugs are silent and only surface in production.
