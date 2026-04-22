# MRCP Plugin ↔ Python Bridge Protocol (v1)

Unix-domain socket protocol that connects the UniMRCP C plugin
(`sandroidrecog.so`) to the sandroid Python ASR/NLU pipeline.

- **Transport**: `AF_UNIX` `SOCK_STREAM`, one socket per MRCP channel.
  Default path: `/var/run/sandroid/bridge.sock` (override via
  `SANDROID_BRIDGE_SOCK` env var on both sides).
- **Direction**: bidirectional. C plugin is always the client; Python
  bridge is always the server.
- **Encoding**: framed binary. No TLS (localhost only), no auth (FS
  permissions guard the socket).

## Frame format

Every frame on the wire is:

```
  ┌─────────────────┬────────┬──────────────────────────┐
  │ length (4B, BE) │ type   │ payload (length-1 bytes) │
  │ uint32 big-end  │ uint8  │ CBOR or raw bytes        │
  └─────────────────┴────────┴──────────────────────────┘
```

- `length` covers `type` + `payload` (i.e. total frame = `length + 4`).
- `length` max 1 MiB. Larger frames → `ERROR(code=FRAME_TOO_LARGE)`
  and connection MUST be closed.
- `payload` interpretation depends on `type`.

## Type codes

| Code | Name       | Direction | Payload                                  |
|------|------------|-----------|------------------------------------------|
| 0x01 | `START`    | C → Py    | CBOR map (see below)                     |
| 0x02 | `AUDIO`    | C → Py    | raw PCM16-LE bytes (20 ms frames)        |
| 0x03 | `EOS`      | C → Py    | empty — end-of-speech signal from VAD    |
| 0x04 | `STOP`     | C → Py    | empty — caller STOP or channel close     |
| 0x10 | `RESULT`   | Py → C    | NLSML XML bytes (UTF-8)                  |
| 0x11 | `ERROR`    | either    | CBOR map: `{code: str, message: str}`    |
| 0x12 | `PARTIAL`  | Py → C    | CBOR map: `{transcript: str, stable: bool}` — **reserved for v2**, not used in v1 |

Unknown `type` codes MUST be ignored by the receiver (forward-compat).

### `START` payload (CBOR map)

```cbor
{
  "channel_id":  str,   # MRCP channel identifier (e.g. "<session>@speechrecog")
  "session_id":  str,   # MRCP session ID
  "sample_rate": uint,  # 8000 or 16000
  "codec":       str,   # "LPCM" (v1 only supports PCM16-LE)
  "scene":       str,   # optional — scene hint from MRCP headers (reserved, may be empty)
}
```

Unknown keys MUST be ignored.

## Lifecycle

```
C plugin                                    Python bridge
   │                                              │
   ├── connect() ──────────────────────────────►  │  accept()
   │                                              │
   ├── START (channel_id, sr, codec) ──────────►  │  init ASR session
   │                                              │
   ├── AUDIO (PCM frame) ──────────────────────►  │  feed ASR
   ├── AUDIO (PCM frame) ──────────────────────►  │  feed ASR
   ├── AUDIO (...) ────────────────────────────►  │  ...
   │                                              │
   ├── EOS ─────────────────────────────────────► │  finalize ASR
   │                                              │
   │  ◄─── RESULT (NLSML) ─────────────────────── │
   │                                              │
   ├── STOP ───────────────────────────────────►  │  cleanup
   │                                              │
   └── close()                                    │  close()
```

### Normal flow

1. C plugin connects on MRCP `channel_open`.
2. C sends `START` **before** any `AUDIO`. Python allocates ASR state.
3. C streams `AUDIO` frames as they arrive from RTP (20 ms each).
4. C's `mpf_activity_detector` detects speech end → C sends `EOS`.
5. Python finalizes ASR → renders NLSML → sends `RESULT`.
6. C plugin embeds NLSML body in `RECOGNITION-COMPLETE` event.
7. C plugin sends `STOP` and closes the socket on `channel_close`.

### Barge-in / early termination

If MRCPv2 caller sends `STOP`, C plugin sends `STOP` frame and closes
the socket immediately. Python must tolerate connection drop at any
point after `START` and release ASR resources.

### Error semantics

- **Python-side failure** (ASR exception, OOM, etc.) → Python sends
  `ERROR` frame and closes connection. C plugin then completes MRCP
  request with `Completion-Cause: 001 no-input-timeout` or 004 `error`
  depending on when it occurred.
- **C-side framing error** → C plugin closes socket (no ERROR frame
  attempted — assume protocol is broken).
- **Timeout**: if Python does not respond with `RESULT` within
  `recognition-timeout` (header from RECOGNIZE request, default 10 s),
  C plugin completes MRCP with `Completion-Cause: 001 no-match` and
  sends `STOP`.

### Error codes

| Code                 | Meaning                                          |
|----------------------|--------------------------------------------------|
| `FRAME_TOO_LARGE`    | Length field exceeds 1 MiB                       |
| `BAD_START`          | `START` payload missing required field           |
| `UNEXPECTED_FRAME`   | Received AUDIO before START, EOS after STOP, etc |
| `ASR_FAILED`         | Upstream ASR adapter raised (mirrors HTTP code)  |
| `INTERNAL`           | Any other server-side exception                  |

## Graceful degradation

The C plugin MUST survive a bridge outage:

- If `connect()` fails at `channel_open` → log warning, fall back to
  the legacy canned `result.xml` path. MRCPv2 contract still holds
  (caller gets a `RECOGNITION-COMPLETE` with `Completion-Cause 000
  success` and the canned NLSML). This is the production safety net.
- If the bridge closes the socket mid-stream → C plugin falls back to
  canned result for that channel and logs a warning. Next channel
  retries the connection.
- There is no automatic reconnect during an active MRCP channel
  (simpler state machine; connection is short-lived anyway).

## Reconnect policy (across channels)

Every new MRCP channel opens a fresh socket. There is no persistent
connection pool. Python bridge accepts connections indefinitely.

## Versioning

This is **v1**. Future versions bump the TCP port *and* the default
socket path (`/var/run/sandroid/bridge.v2.sock`) so old and new can
coexist during a rollout. `PARTIAL` (0x12) is reserved for v2
streaming partial transcripts.
