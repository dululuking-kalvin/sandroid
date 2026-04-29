# Phase 6a — First-Run Checklist for sandroid-dev

Paste-able instructions for the **first** real execution of the MRCP smoke
gate on the sandroid-dev VM. After the first successful run, normal usage
falls back to the one-liner in [`phase-6a-smoke.md`](./phase-6a-smoke.md).
Use this document only when:

1. You have never run `scripts/smoke/mrcp_smoke.sh` on this host before, or
2. The smoke script has just landed and you want to discover its surprises
   while a human is watching, or
3. The bridge / UniMRCP / plugin install has been wiped and re-bootstrapped.

The checklist is divided into stages. **Stop at the first stage that fails**
— each stage's diagnostic block tells you what to look at before retrying.

---

## Stage 0 — assumed baseline

Before starting, confirm the dev VM has the layout from
[`deploy/ubuntu/unimrcp-setup.md`](../deploy/ubuntu/unimrcp-setup.md):

```bash
ssh sandroid-dev
[[ -x /opt/unimrcp/bin/umc ]]                  && echo "umc OK"
[[ -x /opt/unimrcp/bin/unimrcpserver ]]        && echo "server OK"
[[ -d /opt/unimrcp/conf/client-profiles ]]     && echo "client-profiles OK"
[[ -d /opt/unimrcp/data ]]                     && echo "data dir OK"
```

If any line silently produces no output, follow the install guide before
continuing — the smoke gate cannot run on a host without UniMRCP.

---

## Stage 1 — refresh the repo + bridge

Pull `main` (the smoke artifacts landed in commit `9618a78` —
`feat(6a): MRCP smoke regression gate`) and re-deploy the bridge so the
service is running the latest Python sources:

```bash
ssh sandroid-dev
cd /opt/sandroid                # or wherever the working tree lives
git fetch origin main
git checkout main && git pull --ff-only

# Re-run the installer with INSTALL_BRIDGE=1 to re-stage src/sandroid/ into
# /opt/sandroid/src and restart sandroid-bridge.service.
INSTALL_BRIDGE=1 \
UNIMRCP_SRC=/opt/unimrcp-src \
UNIMRCP_PREFIX=/opt/unimrcp \
SANDROID_PREFIX=/opt/sandroid \
sudo -E deploy/unimrcp/install-plugin.sh
```

**Verify:**

```bash
systemctl is-active sandroid-bridge      # → active
ls -l /var/run/sandroid/bridge.sock      # → srwxrwxrwx
```

**If the bridge fails to start:** `journalctl -u sandroid-bridge -n 50 --no-pager`.

---

## Stage 2 — replace the placeholder WAV with a real "你好"

The repo ships `deploy/umc/canned/hello.wav` as a synthetic 440 Hz tone.
Real Paraformer ASR will not transcribe it as anything useful, so the
matcher returns `PLACEHOLDER_INTENT` and the smoke gate fails with exit 4.
Capture a real one second:

```bash
# Record on the dev VM (or any Linux host with a microphone, then scp).
arecord -f S16_LE -r 16000 -c 1 -d 1 /tmp/hello.wav

# Verify format — must read `Stereo` / `Mono`, `Bit Rate`, etc.
file /tmp/hello.wav
soxi /tmp/hello.wav 2>/dev/null || aplay --dump-hw-params /tmp/hello.wav

# Listen briefly to confirm you actually said "你好".
aplay /tmp/hello.wav
```

**Required format:** WAV, 16 kHz, mono, PCM16 little-endian, ≤ 2 s. If
`arecord` defaults differ, force them with `-f S16_LE -r 16000 -c 1`.

Stage the recording into the repo and commit:

```bash
cp /tmp/hello.wav /opt/sandroid/deploy/umc/canned/hello.wav
cd /opt/sandroid
git status deploy/umc/canned/hello.wav     # → modified
git add deploy/umc/canned/hello.wav
git commit -m "chore(6a): real \"你好\" canned WAV for smoke test"
git push origin main
```

(Future smoke runs on other hosts pull this same WAV — no need to
re-record.)

---

## Stage 3 — first dry run

Run the smoke script in verbose mode so the full umc transcript is on
screen:

```bash
cd /opt/sandroid
./scripts/smoke/mrcp_smoke.sh --verbose
```

**Three possible first-run outcomes:**

### 3a. `[smoke] OK: matched intent='greeting'` — ship it

The whole stack works. Every subsequent run can use the quiet form
(`./scripts/smoke/mrcp_smoke.sh`). Move on to "After the first run" below.

### 3b. Pre-flight failure (exit 2)

The script tells you which file/socket/port is missing. Most common:

| Message | Fix |
|---------|-----|
| `bridge socket missing: /var/run/sandroid/bridge.sock` | `sudo systemctl restart sandroid-bridge` |
| `no listener on tcp/1544` | UniMRCP isn't running — `LD_LIBRARY_PATH=/opt/unimrcp/lib /opt/unimrcp/bin/unimrcpserver -r /opt/unimrcp -d` |
| `umc binary not executable` | UniMRCP install incomplete — re-run setup guide |

### 3c. umc invocation failed / assertion failed (exit 3 or 4)

The smoke script's umc XML scenario is **plausibly correct** but has not
been validated against a real UniMRCP 1.8.0 client until now. First-run
tweaks are likely. Investigate in this order:

**Step 1: did umc parse the scenario at all?**

```bash
grep -E "Load Scenario|SandroidRecogSmoke|scenario" \
    /opt/unimrcp/log/unimrcpclient_current.log | tail -20
```

If you see `Failed to load scenario` or `Unknown scenario`, the XML schema
in `deploy/umc/sandroid-recog.xml` does not match what this build of umc
expects. Compare against the bundled scenarios:

```bash
ls /opt/unimrcp/conf/client-profiles/
diff <(xmllint --format /opt/unimrcp/conf/client-profiles/scenarios.xml) \
     <(xmllint --format /opt/unimrcp/conf/client-profiles/sandroid-scenarios.xml)
```

Adjust attributes (`class`, `profile`, child elements like `<recog>`,
`<player>`) to mirror a working scenario from the bundled file. Commit
the fix back to `deploy/umc/sandroid-recog.xml`.

**Step 2: did the bridge get a START frame?**

```bash
journalctl -u sandroid-bridge -n 50 --no-pager
# Look for: `START channel=… session=… sr=16000 codec=LPCM`
```

If START never arrives, the umc → UniMRCP → plugin path is broken before
the bridge — likely a plugin registration issue. Verify
`Sandroid-Recog-1` is loaded:

```bash
grep -E "Load Plugin|Sandroid-Recog|sandroidrecog" \
    /opt/unimrcp/log/unimrcpserver_current.log | tail -10
```

Expected: `Load Plugin [Sandroid-Recog-1] [/opt/unimrcp/plugin/sandroidrecog.so]`.

**Step 3: did the bridge return NLSML but with the wrong intent?**

```bash
journalctl -u sandroid-bridge -n 50 --no-pager | grep -E "RESULT|intent|confidence"
# Look for: `RESULT sent for … intent=greeting conf=…`
```

If the bridge returned `intent=PLACEHOLDER_INTENT`, see "bridge returned
PLACEHOLDER_INTENT" in [`phase-6a-smoke.md`](./phase-6a-smoke.md#bridge-returned-placeholder_intent).

If the bridge returned `intent=<something else>`, the matcher chose a
different scene/intent. Check the bridge service unit for the
`--default-scene` flag and confirm the scene's training utterances match
"你好":

```bash
grep -E "Environment=|ExecStart=" /etc/systemd/system/sandroid-bridge.service
grep -A20 "id: greeting" /opt/sandroid/configs/scenes/example_bank.yaml
```

---

## After the first run

Once Stage 3a is reached:

1. **Pin the working state.** If you adjusted the scenario XML, the patches,
   or the systemd unit, commit the changes to the repo so the next host
   bootstrapping from main reproduces the same green run.
2. **Add a regression entry.** Drop a one-liner under "Known-good runs" in
   `docs/phase-6a-smoke.md` (date + commit hash + who ran it). Five entries
   are enough to spot environmental drift.
3. **Plan 6b / 6c.** With the bridge proven healthy, the cheapest next gate
   is FreeSWITCH-as-MRCP-client (Phase 6b). Open a sub-project spec when
   you're ready.

If first-run tweaks were needed, also:

4. **Refresh this document.** Replace the speculative "likely tweaks
   needed" hedges with the concrete diff that ended up working. Future
   first-runs on other dev hosts deserve the cleaner narrative.

---

## Why this exists separately

The smoke script and its runbook describe the **steady state** — what to do
once the gate is green. This document describes the **bootstrap state** —
the messy first run where the umc scenario XML is unverified, the
placeholder WAV is in place, and the bridge service may need a kick.
After 6a stabilises, this doc can shrink to a "see [`phase-6a-smoke.md`]"
redirect; until then it's the authoritative path through the surprises.
