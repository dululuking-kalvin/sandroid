# Phase 6a — MRCP Smoke Regression Gate

One-shot regression check that the **`umc → UniMRCP → sandroid-recog plugin →
Python bridge → Orchestrator → NLSML`** pipeline is end-to-end functional on
the sandroid-dev VM. This is the cheapest layer of FreeSWITCH-stack
verification: it does not boot FreeSWITCH itself (that is Phase 6b), but it
does exercise every sandroid-owned component using a real MRCPv2 conversation.

Run it after any change to `src/sandroid/adapters/mrcp/`, `src/sandroid/core/`,
or the UniMRCP plugin under `native/unimrcp-plugin/`.

## Prerequisites

The target host must already have:

- UniMRCP 1.8.0 installed under `/opt/unimrcp/` per
  [`deploy/ubuntu/unimrcp-setup.md`](../deploy/ubuntu/unimrcp-setup.md)
- The `sandroid-recog` plugin installed and `Sandroid-Recog-1` registered
  in `unimrcpserver.xml` (run `deploy/unimrcp/install-plugin.sh`)
- `sandroid-bridge.service` active and listening on `/var/run/sandroid/bridge.sock`
  (set `INSTALL_BRIDGE=1` when running `install-plugin.sh`)
- `unimrcpserver` daemonized and binding to `1544` (MRCPv2) and `8060` (SIP):

  ```bash
  LD_LIBRARY_PATH=/opt/unimrcp/lib /opt/unimrcp/bin/unimrcpserver -r /opt/unimrcp -d
  ```

## First-time setup: replace the placeholder WAV

The repo ships `deploy/umc/canned/hello.wav` as a 1-second 440 Hz tone burst
generated from `scripts/smoke/_make_placeholder_wav.py`. **It is not a real
"你好" recording** — the energy gate in `MockVAD` accepts it, but the real
Paraformer ASR will not produce a usable transcript from it. Before the first
real run on sandroid-dev, replace it with a real recording:

```bash
# Run on the dev VM (or any Linux host with a microphone, then scp).
arecord -f S16_LE -r 16000 -c 1 -d 1 /tmp/hello.wav
# Inspect briefly to confirm the recording captured speech.
aplay /tmp/hello.wav
# Copy into the repo and commit.
cp /tmp/hello.wav $SANDROID_REPO/deploy/umc/canned/hello.wav
git -C $SANDROID_REPO add deploy/umc/canned/hello.wav
git -C $SANDROID_REPO commit -m "chore(6a): real \"你好\" canned WAV for smoke test"
```

Until that swap happens, the smoke test will only pass with the **stub matcher
+ stub ASR** combination — the real ASR will transcribe the tone burst as
empty/garbage and the matcher will return `PLACEHOLDER_INTENT`.

## Running

From a clone of the repo on the dev VM:

```bash
cd $SANDROID_REPO     # e.g. /opt/sandroid (cloned by deploy/unimrcp/install-plugin.sh)
./scripts/smoke/mrcp_smoke.sh
# expect:  [smoke] OK: matched intent='greeting'
```

For verbose output (full umc log + last 30 lines of bridge journal on
failure):

```bash
./scripts/smoke/mrcp_smoke.sh --verbose
```

Override defaults via env vars when paths differ:

```bash
UNIMRCP_PREFIX=/opt/unimrcp \
SANDROID_REPO=/opt/sandroid \
BRIDGE_SOCK=/var/run/sandroid/bridge.sock \
EXPECTED_INTENT=greeting \
./scripts/smoke/mrcp_smoke.sh
```

## Exit codes

| Code | Meaning | First thing to check |
|------|---------|----------------------|
| 0 | Pass — `Interpretation[0].instance[0]: greeting` matched | — |
| 2 | Pre-flight failed | Pre-flight messages name the missing piece (binary / socket / port) |
| 3 | `umc` invocation failed or timed out (15 s) | `journalctl -u sandroid-bridge -n 50`; `tail /opt/unimrcp/log/unimrcpserver_current.log` |
| 4 | `umc` ran but the expected intent did not appear | Check the verbose log — does `RECOGNITION-COMPLETE` even appear? Was `PLACEHOLDER_INTENT` returned (matcher misconfigured)? |

## Common failure modes

### `bridge socket missing`

The bridge service is down or has not been installed. Re-run
`INSTALL_BRIDGE=1 sudo -E deploy/unimrcp/install-plugin.sh` from the repo.
Verify with `systemctl status sandroid-bridge` and `ls -l /var/run/sandroid/bridge.sock`.

### `no listener on tcp/1544 (MRCPv2)`

`unimrcpserver` is not running or was started without `-r /opt/unimrcp`. The
`-r` flag is mandatory — without it, the server boots but never binds any
ports (`dirlayout.xml` resolves `../conf` against CWD; daemon mode CWD is `/`).
Re-launch with the full command from
[`deploy/ubuntu/unimrcp-setup.md`](../deploy/ubuntu/unimrcp-setup.md#running-the-server).

### `bridge returned PLACEHOLDER_INTENT`

The bridge reached the matcher, but the matcher returned no top hit (or the
ASR returned an empty transcript). Likely causes:

1. The placeholder WAV is still in place — see "First-time setup" above.
2. `SANDROID_NLU_BACKEND` is `embedder` but the ONNX model artifact isn't
   fetched (`deploy/models/nlu_embedder_quantized.onnx`). The bridge falls back
   to `StubMatcher` in dev mode (`SANDROID_ENV=dev`), but stub matching needs
   meaningful transcript text — silent / tone-burst input fails it.
3. The bridge service was started without a `--default-scene`. Confirm by
   reading the `Environment=` lines in `/etc/systemd/system/sandroid-bridge.service`.

### `umc invocation failed or timed out`

Either UniMRCP took longer than 15 s to handle the RECOGNIZE (very likely
MRCPv2 SETUP / ANNOUNCE wedged), or `umc` itself segfaulted. Two diagnostics:

1. Run `umc` interactively with the same scenario:
   ```bash
   LD_LIBRARY_PATH=/opt/unimrcp/lib /opt/unimrcp/bin/umc -r /opt/unimrcp
   # at the umc> prompt:
   run SandroidRecogSmoke
   # observe the SIP/RTSP exchange
   ```
2. Check UniMRCP's own log — `tail -F /opt/unimrcp/log/unimrcpserver_current.log`
   while the smoke script runs.

### `Interpretation[0].instance[0]: PLACEHOLDER_INTENT`

The bridge returned NLSML with the placeholder. Same root causes as
"bridge returned PLACEHOLDER_INTENT" above — the smoke script just calls
this out explicitly when it sees that exact string.

## What this test does *not* cover

- **FreeSWITCH integration** — `mod_unimrcp` from a real FreeSWITCH instance
  speaks a slightly different MRCPv2 dialect than `umc` does. That is Phase 6b.
- **SIP-side regressions** — `umc` makes a SIP call to UniMRCP, but no
  external SIP client (Zoiper, pjsua) is in the loop. Phase 6c.
- **Path B (SLU)** — `RecognitionRequest.audio` is wired (Phase 5f-d), but
  StubSLU still abstains. Path B is exercised structurally but not
  functionally until Phase 5f-c lands.
- **Latency budget** — the script enforces a 15 s timeout but does not
  measure p50/p95. Add Prometheus scraping of `/metrics` if you need a
  latency gate.

## CI integration

`.github/workflows/mrcp-smoke.yml` ships a disabled workflow stub. To enable:

1. Provision a self-hosted GitHub Actions runner labeled `linux-mrcp` with
   the prerequisites above already installed.
2. Edit the workflow's `on:` trigger from `workflow_dispatch` to a `pull_request`
   paths-filter on `src/sandroid/adapters/mrcp/**`.
3. Push and verify the workflow runs against a fresh PR.

Until a runner exists, the workflow is intentionally manual-only — do not
flip the trigger speculatively, GitHub will fail the job on every PR with
"no runner available."
