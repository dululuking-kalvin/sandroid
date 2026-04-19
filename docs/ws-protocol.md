# sandroid WebSocket protocol — `/api/v1/recognize/stream`

**Status:** design-locked 2026-04-20. Implementation catches up in Phase 5c.

## Design intent

sandroid is the backend for a **turn-based IVR**. The caller (FreeSWITCH or a
business process) plays a prompt, then streams the user's audio into this
endpoint. VAD closes the utterance; the engine returns the intent; the caller
picks the next scene and opens a new turn. The human user never sees the
protocol — only the orchestration layer does.

Given that, the protocol follows four rules:

1. **No handshake.** The client may start pushing PCM immediately after `start`.
   VAD's `speech_start_ms` absorbs any lead-in silence or micro-race.
2. **Correlation by id, not by ordering.** Every frame (client → server and
   server → client) carries `session_id` + `turn_id`. Latency, re-ordering, and
   barge-in stop being ambiguous.
3. **Barge-in via new turn.** A new `start` with a higher `turn_id` implicitly
   terminates the previous turn. No explicit `stop` is required, though the
   server still accepts it for graceful shutdowns.
4. **Errors are async frames, not closes.** The server reports problems over
   the same channel as `{type: "error", fatal: bool, ...}`. Only truly
   unrecoverable states close the socket.

Together these give the low-latency profile of the "implicit handshake" option
plus the observability of an explicit one, and align with MRCPv2 timing so the
two adapters stay coherent.

## Connection lifecycle

```
client                                    server
  │── WS connect + X-API-Key header ────▶  │
  │◀── accept / 4401 close ───────────────│
  │
  │── {type:"start", session_id, turn_id, scene_id, ...} ──▶
  │── PCM frame (binary) ──▶
  │── PCM frame ──▶
  │── PCM frame ──▶                        │──▶ {type:"partial", text, session_id, turn_id}
  │── PCM frame ──▶                        │──▶ {type:"speech_start", ...}
  │── PCM frame ──▶                        │──▶ {type:"speech_end", ...}
  │   (VAD cuts; client may keep          │──▶ {type:"final", text, ...}
  │   pushing — discarded after final)    │──▶ {type:"result", intent, followUp, ...}
  │
  │── {type:"start", turn_id: n+1, ...} ──▶ (barge-in — new turn begins)
```

Auth is a single check at connect time: header `X-API-Key` or query param
`api_key`. Closing codes reserved for auth/shape failures only:

| Code | Meaning |
|------|---------|
| `4401` | Unauthorized (bad / missing API key) |
| `4422` | Malformed `start` frame (server never entered a turn) |
| `1000` | Normal close initiated by client |

Recognition-level failures do **not** close the socket — they are reported as
`{type: "error", fatal: false}` and the socket stays open for the next turn.

## Frame catalogue

All JSON frames are sent as text messages. PCM audio is sent as binary.

### Client → server

#### `start` (JSON)

Opens a new turn. Sent before any PCM for this turn.

```json
{
  "type": "start",
  "session_id": "c0ffee-...",          // required, stable across turns in this session
  "turn_id": 3,                        // required, strictly increasing per session
  "scene_id": "example_bank",          // one of scene_id | category_path required
  "category_path": null,
  "n_best": 5,                         // optional, default 5
  "sample_rate": 16000                 // optional, default 16000; must be 16000 today
}
```

A `start` with a `turn_id` less than or equal to the current turn is rejected
with an async `error` frame (non-fatal) and ignored. A `start` with a higher
`turn_id` implicitly terminates the current turn (no `final`/`result` is
emitted for the abandoned turn — the client has already moved on).

#### PCM frame (binary)

Raw PCM16, little-endian, 16 kHz mono, any chunk size. Frames received
**before** a `start` or **after** a `final` for the current turn are silently
discarded and counted in a `dropped_frames` metric — they do not produce
errors, because in real IVR setups the client may be slightly ahead of the
server or sending tail audio while the server is finalizing.

#### `stop` (JSON, optional)

```json
{ "type": "stop", "session_id": "...", "turn_id": 3 }
```

Forces the server to close the current utterance (as if VAD had fired
`speech_end`). Useful for graceful shutdowns. Barge-in does **not** need
`stop` — a new `start` is enough.

### Server → client

Every server frame includes `session_id` and `turn_id` (omitted here for
brevity — treat them as always present).

#### `speech_start` / `speech_end` (JSON)

VAD boundary events. Purely informational — the client does not need to react.

```json
{ "type": "speech_start", "at_ms": 240 }
{ "type": "speech_end",   "at_ms": 1680 }
```

#### `partial` (JSON)

Streaming ASR partial. May be emitted multiple times per turn. `text` is
cumulative (replaces previous partial).

```json
{ "type": "partial", "text": "我想查询", "start_ms": 240, "end_ms": 900 }
```

#### `final` (JSON)

Streaming ASR final transcript for the turn. Exactly one per turn.

```json
{ "type": "final", "text": "我想查询余额", "start_ms": 240, "end_ms": 1680 }
```

#### `result` (JSON)

Orchestrator output — top intent, N-best, and next-turn pointer. Exactly one
per turn, after `final`.

```json
{
  "type": "result",
  "top": { "intent_id": "balance_query", "score": 0.87 },
  "candidates": [...],
  "current_category_id": "bank/balance",
  "followUp": null
}
```

#### `error` (JSON)

```json
{ "type": "error", "code": "UNKNOWN_SCENE", "message": "...", "fatal": false }
```

| `code` | Fatal | Meaning |
|--------|-------|---------|
| `UNKNOWN_SCENE` | false | `scene_id` / `category_path` not in the catalog |
| `NO_SPEECH` | false | VAD produced no segments before `stop` |
| `ASR_FAILED` | false | Backend failed; client may retry with a new `turn_id` |
| `STALE_TURN` | false | `start` frame's `turn_id` was ≤ current turn |
| `INTERNAL` | true | Unrecoverable; server closes after sending |

The server closes the socket only when `fatal=true`.

## Invariants the server enforces

- Exactly one `final` and one `result` per successful turn.
- `partial` never appears after `final` for the same `turn_id`.
- `turn_id` is strictly increasing within a session.
- Audio frames before `start` or after `final` are dropped, not errored.
- A new turn started before the previous one finalizes cancels the previous
  turn — the server emits no `final`/`result` for it.

## What this buys us

- **Latency parity with MRCPv2** — no extra roundtrip before streaming audio.
- **Barge-in is free** — a new `start` terminates the old turn, no choreography
  needed from the caller.
- **Observability** — every frame is id-correlated, so late partials from a
  cancelled turn are trivially filtered by the client.
- **Failure mode simplicity** — errors don't tear down the socket, so the next
  turn proceeds without reconnect overhead.

## What is deliberately not in v0.1

- Server-initiated `ready` handshake — unnecessary under the IVR model.
- Dynamic sample-rate renegotiation — 16 kHz is the sole supported rate.
- Multi-channel / stereo audio.
- Resumable turns after network drop — a reconnect starts fresh with the next
  `turn_id`.
