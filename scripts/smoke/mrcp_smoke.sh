#!/usr/bin/env bash
#
# Phase 6a — MRCP smoke regression gate.
#
# Drives `umc` through one RECOGNIZE turn against the sandroid-recog UniMRCP
# plugin and the Python bridge, then asserts the bridge returned NLSML with
# `Interpretation[0].instance[0]: greeting`. Designed to run on the
# sandroid-dev VM after deploy/unimrcp/install-plugin.sh has been applied.
#
# Pass criterion (strict, deliberate):
#   - umc log contains `Interpretation[0].instance[0]: greeting`
#   - umc log does NOT contain `PLACEHOLDER_INTENT` (would mean the matcher
#     never received a usable transcript — bridge or ASR misconfigured)
#
# Usage:
#   ./scripts/smoke/mrcp_smoke.sh                  # quiet: prints OK/FAIL
#   ./scripts/smoke/mrcp_smoke.sh --verbose        # also tails bridge journal on failure
#   UNIMRCP_PREFIX=/opt/unimrcp \
#   SANDROID_REPO=/opt/sandroid \
#   ./scripts/smoke/mrcp_smoke.sh
#
# Exit codes:
#   0  — pass (greeting intent matched)
#   2  — pre-flight failure (umc missing / bridge socket missing / port not listening)
#   3  — umc invocation failed (timeout, network, MRCP-level error)
#   4  — assertion failed (umc ran, but result text didn't match)

set -euo pipefail

VERBOSE=0
for arg in "$@"; do
    case "$arg" in
        -v|--verbose) VERBOSE=1 ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

UNIMRCP_PREFIX="${UNIMRCP_PREFIX:-/opt/unimrcp}"
SANDROID_REPO="${SANDROID_REPO:-$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel 2>/dev/null || echo "")}"
BRIDGE_SOCK="${BRIDGE_SOCK:-/var/run/sandroid/bridge.sock}"
EXPECTED_INTENT="${EXPECTED_INTENT:-greeting}"
SCENARIO_NAME="SandroidRecogSmoke"

UMC_BIN="$UNIMRCP_PREFIX/bin/umc"
UMC_CONF_DIR="$UNIMRCP_PREFIX/conf/client-profiles"
UMC_DATA_DIR="$UNIMRCP_PREFIX/data"
SCENARIO_SRC="$SANDROID_REPO/deploy/umc/sandroid-recog.xml"
WAV_SRC="$SANDROID_REPO/deploy/umc/canned/hello.wav"
WAV_DEST="$UMC_DATA_DIR/sandroid/hello.wav"

log()  { printf '[smoke] %s\n' "$*"; }
fail() { printf '[smoke] FAIL: %s\n' "$*" >&2; }

#--------------------------------------------------------------- pre-flight ---
preflight() {
    local err=0

    [[ -x "$UMC_BIN" ]] || { fail "umc binary not executable at $UMC_BIN"; err=1; }
    [[ -d "$UMC_CONF_DIR" ]] || { fail "client-profiles dir missing: $UMC_CONF_DIR"; err=1; }
    [[ -d "$UMC_DATA_DIR" ]] || { fail "data dir missing: $UMC_DATA_DIR"; err=1; }
    [[ -n "$SANDROID_REPO" && -d "$SANDROID_REPO" ]] || { fail "SANDROID_REPO unset/missing"; err=1; }
    [[ -f "$SCENARIO_SRC" ]] || { fail "scenario XML missing in repo: $SCENARIO_SRC"; err=1; }
    [[ -f "$WAV_SRC" ]] || { fail "canned WAV missing in repo: $WAV_SRC"; err=1; }

    # Bridge must be listening on its Unix socket — without it, umc still
    # connects to UniMRCP but the plugin gets ECONNREFUSED on every START.
    [[ -S "$BRIDGE_SOCK" ]] || { fail "bridge socket missing: $BRIDGE_SOCK (is sandroid-bridge.service running?)"; err=1; }

    # MRCPv2 (1544), SIP (8060) — UniMRCP must be listening.
    if command -v ss >/dev/null 2>&1; then
        ss -tln 2>/dev/null | grep -qE ':1544\b' || { fail "no listener on tcp/1544 (MRCPv2) — is unimrcpserver up?"; err=1; }
        ss -tln 2>/dev/null | grep -qE ':8060\b' || { fail "no listener on tcp/8060 (SIP) — is unimrcpserver up?"; err=1; }
    else
        log "warn: ss(8) unavailable, skipping port checks"
    fi

    return "$err"
}

#--------------------------------------------------------------- staging -----
stage() {
    # Install the scenario file alongside the bundled scenarios. We use a
    # distinct filename so we never collide with the upstream defaults.
    install -m 0644 "$SCENARIO_SRC" "$UMC_CONF_DIR/sandroid-scenarios.xml"
    install -d -m 0755 "$UMC_DATA_DIR/sandroid"
    install -m 0644 "$WAV_SRC" "$WAV_DEST"
    log "staged scenario -> $UMC_CONF_DIR/sandroid-scenarios.xml"
    log "staged audio    -> $WAV_DEST"
}

#--------------------------------------------------------------- run umc -----
run_umc() {
    local logfile
    logfile="$(mktemp -t mrcp_smoke.XXXXXX.log)"
    trap 'rm -f "$logfile"' RETURN

    log "running umc scenario=$SCENARIO_NAME (timeout 15s)"
    # Drive umc non-interactively: `run <scenario>` then `quit`.
    if ! timeout 15 bash -c "
        echo 'run $SCENARIO_NAME'
        sleep 8
        echo 'quit'
    " | LD_LIBRARY_PATH="$UNIMRCP_PREFIX/lib" "$UMC_BIN" \
        -r "$UNIMRCP_PREFIX" \
        > "$logfile" 2>&1
    then
        fail "umc invocation failed or timed out"
        cat "$logfile" >&2
        return 3
    fi

    if [[ "$VERBOSE" -eq 1 ]]; then
        log "----- umc log -----"
        cat "$logfile"
        log "-------------------"
    fi

    if grep -q "Interpretation\[0\].instance\[0\]: $EXPECTED_INTENT" "$logfile"; then
        log "OK: matched intent='$EXPECTED_INTENT'"
        return 0
    fi

    fail "expected intent '$EXPECTED_INTENT' not found in umc output"
    grep -E 'Interpretation|RECOGNITION-COMPLETE|FAILURE|error' "$logfile" >&2 || true

    if grep -q "PLACEHOLDER_INTENT" "$logfile"; then
        fail "bridge returned PLACEHOLDER_INTENT — matcher saw no usable transcript"
    fi

    if [[ "$VERBOSE" -eq 1 ]] && command -v journalctl >/dev/null 2>&1; then
        log "----- last 30 lines of sandroid-bridge journal -----"
        journalctl -u sandroid-bridge -n 30 --no-pager >&2 || true
        log "----------------------------------------------------"
    fi

    return 4
}

#----------------------------------------------------------------- main ------
log "phase 6a smoke — UNIMRCP_PREFIX=$UNIMRCP_PREFIX SANDROID_REPO=$SANDROID_REPO"

if ! preflight; then
    exit 2
fi

stage
run_umc
